import torch
import torch.optim as optim
from torch import autograd
import numpy as np
from tqdm import trange, tqdm
import trimesh
from src.utils import libmcubes, common_util
from src.common import make_3d_grid, normalize_coord, add_key, coord2index
from src.utils.libmise import MISE
import time
import math

counter = 0


class Generator3D(object):
    '''  Generator class for Occupancy Networks.

    It provides functions to generate the final mesh as well refining options.

    Args:
        model (nn.Module): trained Occupancy Network model
        points_batch_size (int): batch size for points evaluation
        threshold (float): threshold value
        refinement_step (int): number of refinement steps
        device (device): pytorch device
        resolution0 (int): start resolution for MISE
        upsampling steps (int): number of upsampling steps
        with_normals (bool): whether normals should be estimated
        padding (float): how much padding should be used for MISE
        sample (bool): whether z should be sampled
        input_type (str): type of input
        vol_info (dict): volume infomation
        vol_bound (dict): volume boundary
        simplify_nfaces (int): number of faces the mesh should be simplified to
    '''

    def __init__(self, model, points_batch_size=100000,
                 threshold=0.5, refinement_step=0, device=None,
                 resolution0=16, upsampling_steps=3,
                 with_normals=False, padding=0.1, sample=False,
                 input_type = None,
                 vol_info = None,
                 vol_bound = None,
                 simplify_nfaces=None):
        self.model = model.to(device)
        self.points_batch_size = points_batch_size
        self.refinement_step = refinement_step
        self.threshold = threshold
        self.device = device
        self.resolution0 = resolution0
        self.upsampling_steps = upsampling_steps
        self.with_normals = with_normals
        self.input_type = input_type
        self.padding = padding
        self.sample = sample
        self.simplify_nfaces = simplify_nfaces
        
        # for pointcloud_crop
        self.vol_bound = vol_bound
        if vol_info is not None:
            self.input_vol, _, _ = vol_info
        
    def generate_mesh(self, data, return_stats=True):
        ''' Generates the output mesh.

        Args:
            data (tensor): data tensor
            return_stats (bool): whether stats should be returned
        '''
        self.model.eval()
        device = self.device
        for k,v in data.items():
            data[k] = v.to(device)
        stats_dict = {}

        t0 = time.time()
        
        # obtain features for all crops
        with torch.no_grad():
            c = self.model.encode_inputs(data)
        if type(c) is tuple:
            for cs in c:
                for k,v in cs.items():
                    cs[k] = v[0].unsqueeze(0)
        else:
            for k,v in c.items():
                c[k] = v[0].unsqueeze(0)
        stats_dict['time (encode inputs)'] = time.time() - t0
        
        mesh = self.generate_from_latent(c, stats_dict=stats_dict)

        if return_stats:
            return mesh, stats_dict
        else:
            return mesh

    def generate_from_latent(self, c=None, stats_dict={}, **kwargs):
        ''' Generates mesh from latent.
            Works for shapes normalized to a unit cube

        Args:
            c (tensor): latent conditioned code c
            stats_dict (dict): stats dictionary
        '''
        threshold = np.log(self.threshold) - np.log(1. - self.threshold)

        t0 = time.time()
        # Compute bounding box size
        box_size = 1 + self.padding
        
        # Shortcut
        if self.upsampling_steps == 0:
            nx = self.resolution0
            pointsf = box_size * make_3d_grid(
                (-0.5,)*3, (0.5,)*3, (nx,)*3
            )

            values = self.eval_points(pointsf, c, **kwargs).cpu().numpy()
            value_grid = values.reshape(nx, nx, nx)
        else:
            mesh_extractor = MISE(
                self.resolution0, self.upsampling_steps, threshold)

            points = mesh_extractor.query()
            while points.shape[0] != 0:
                # Query points
                pointsf = points / mesh_extractor.resolution
                # Normalize to bounding box
                pointsf = box_size * (pointsf - 0.5)
                pointsf = torch.FloatTensor(pointsf).to(self.device)
                # Evaluate model and update
                values = self.eval_points(pointsf, c, **kwargs).cpu().numpy()
                values = values.astype(np.float64)
                mesh_extractor.update(points, values)
                points = mesh_extractor.query()

            value_grid = mesh_extractor.to_dense()
        

        # Extract mesh
        stats_dict['time (eval points)'] = time.time() - t0

        mesh = self.extract_mesh(value_grid, c, stats_dict=stats_dict)
        return mesh

    def eval_points(self, p, c=None, vol_bound=None, **kwargs):
        ''' Evaluates the occupancy values for the points.

        Args:
            p (tensor): points 
            c (tensor): encoded feature volumes
        '''
        p_split = torch.split(p, self.points_batch_size)
        occ_hats = []
        for pi in p_split:
            pi = pi.unsqueeze(0).to(self.device)
            with torch.no_grad():
                occ_hat = self.model.eval_points(pi, c, **kwargs)['occ'].logits
            occ_hats.append(occ_hat.squeeze(0).detach().cpu())
        
        occ_hat = torch.cat(occ_hats, dim=0)
        return occ_hat

    def extract_mesh(self, occ_hat, c=None, stats_dict=dict()):
        ''' Extracts the mesh from the predicted occupancy grid.

        Args:
            occ_hat (tensor): value grid of occupancies
            c (tensor): encoded feature volumes
            stats_dict (dict): stats dictionary
        '''
        # Some short hands
        n_x, n_y, n_z = occ_hat.shape
        box_size = 1 + self.padding
        threshold = np.log(self.threshold) - np.log(1. - self.threshold)
        # Make sure that mesh is watertight
        t0 = time.time()
        occ_hat_padded = np.pad(
            occ_hat, 1, 'constant', constant_values=-1e6)
        vertices, triangles = libmcubes.marching_cubes(
            occ_hat_padded, threshold)
        stats_dict['time (marching cubes)'] = time.time() - t0
        # Strange behaviour in libmcubes: vertices are shifted by 0.5
        vertices -= 0.5
        # # Undo padding
        vertices -= 1
        
        if self.vol_bound is not None:
            # Scale the mesh back to its original metric
            bb_min = self.vol_bound['query_vol'][:, 0].min(axis=0)
            bb_max = self.vol_bound['query_vol'][:, 1].max(axis=0)
            mc_unit = max(bb_max - bb_min) / (self.vol_bound['axis_n_crop'].max() * self.resolution0*2**self.upsampling_steps)
            vertices = vertices * mc_unit + bb_min
        else: 
            # Normalize to bounding box
            vertices /= np.array([n_x-1, n_y-1, n_z-1])
            vertices = box_size * (vertices - 0.5)
        
        # Create mesh
        mesh = trimesh.Trimesh(vertices / (1., 1., 3), triangles,
                               vertex_normals=None,
                               process=False)

        # Directly return if mesh is empty
        if vertices.shape[0] == 0:
            return mesh

        # TODO: normals are lost here
        if self.simplify_nfaces is not None:
            t0 = time.time()
            from src.utils.libsimplify import simplify_mesh
            mesh = simplify_mesh(mesh, self.simplify_nfaces, 5.)
            stats_dict['time (simplify)'] = time.time() - t0

        # Refine mesh
        if self.refinement_step > 0:
            t0 = time.time()
            self.refine_mesh(mesh, occ_hat, c)
            stats_dict['time (refine)'] = time.time() - t0

        return mesh

    def generate_pointcloud(self, data, threshold=0.75, use_gt_occ=False):
        self.model.eval()
        device = self.device
        self.model.eval()
        device = self.device
        for k,v in data.items():
            data[k] = v.to(device)
        stats_dict = {}

        t0 = time.time()
        
        # obtain features for all crops
        with torch.no_grad():
            c = self.model.encode_inputs(data)
        pts = data['sampled_pts']
        B,_,N,C = pts.shape
        pts = pts.reshape([B*2,N,C])
        p_split = torch.split(pts, self.points_batch_size, dim=-1)
        occ_hats = []
        features = []
        flows = []
        for pi in p_split:
            with torch.no_grad():
                outputs = self.model.eval_points(pi, c)
                occ_hats.append((outputs['occ'].probs > threshold).detach().cpu())
                if 'corr' in outputs:
                    features.append(outputs['corr'].detach().cpu())
                if 'flow' in outputs:
                    flows.append(outputs['flow'].detach().cpu())
        pts = pts.cpu().numpy()
        occ_hat = torch.cat(occ_hats, dim=1).numpy()
        if use_gt_occ:
            occ_hat = data['sampled_occ'].reshape([B*2, N]).cpu().numpy()
        pos_pts0 = pts[0][occ_hat[0] == 1.].reshape((-1,3))
        pos_idx0 = common_util.subsample_points(pos_pts0, resolution=0.013)
        pos_pts0 = pos_pts0[pos_idx0]
        pos_pts1 = pts[1][occ_hat[1] == 1.].reshape((-1,3))
        pos_idx1 = common_util.subsample_points(pos_pts1, resolution=0.013)
        pos_pts1 = pos_pts1[pos_idx1]
        pos_pts = np.concatenate([pos_pts0, pos_pts1], axis=0) / (1.,1.,3.)
        if len(features) != 0:
            feature = torch.cat(features, dim=1).numpy()
            f_dim = feature.shape[-1]
            pos_f0 = feature[0][occ_hat[0] == 1.].reshape((-1,f_dim))
            pos_f1 = feature[1][occ_hat[1] == 1.].reshape((-1,f_dim))
            pos_f0 = pos_f0[pos_idx0]
            pos_f1 = pos_f1[pos_idx1]
            pos_f = np.concatenate([pos_f0, pos_f1], axis=0)
            if pos_f.shape[0] < 100:
                pcloud_both = pos_pts
            else:
                tsne_result = common_util.embed_tsne(pos_f)
                colors = common_util.get_color_map(tsne_result)
                pcloud_both = np.concatenate([pos_pts, colors], axis=1)
        else:
            pcloud_both = pos_pts
        pcloud0 = pcloud_both[:pos_pts0.shape[0]]
        pcloud1 = pcloud_both[pos_pts0.shape[0]:]
        if len(flows) != 0:
            flow = torch.cat(flows, dim=1).numpy() / 10. 
            pos_f0 = flow[0][occ_hat[0] == 1.].reshape((-1,3))
            pos_f1 = flow[1][occ_hat[1] == 1.].reshape((-1,3))
            pos_f0 = pos_f0[pos_idx0]
            pos_f1 = pos_f1[pos_idx1]
            pcloud_unroll_0 = pcloud0.copy()
            pcloud_unroll_0[:,:3] += pos_f0 / (1.,1.,3.)
            pcloud_unroll_1 = pcloud1.copy()
            pcloud_unroll_1[:,:3] += pos_f1 / (1.,1.,3.)
            return pcloud0, pcloud1,pcloud_unroll_0,pcloud_unroll_1
        return pcloud0, pcloud1


    def refine_mesh(self, mesh, occ_hat, c=None):
        ''' Refines the predicted mesh.

        Args:   
            mesh (trimesh object): predicted mesh
            occ_hat (tensor): predicted occupancy grid
            c (tensor): latent conditioned code c
        '''

        self.model.eval()

        # Some shorthands
        n_x, n_y, n_z = occ_hat.shape
        assert(n_x == n_y == n_z)
        # threshold = np.log(self.threshold) - np.log(1. - self.threshold)
        threshold = self.threshold

        # Vertex parameter
        v0 = torch.FloatTensor(mesh.vertices).to(self.device)
        v = torch.nn.Parameter(v0.clone())

        # Faces of mesh
        faces = torch.LongTensor(mesh.faces).to(self.device)

        # Start optimization
        optimizer = optim.RMSprop([v], lr=1e-4)

        for it_r in trange(self.refinement_step):
            optimizer.zero_grad()

            # Loss
            face_vertex = v[faces]
            eps = np.random.dirichlet((0.5, 0.5, 0.5), size=faces.shape[0])
            eps = torch.FloatTensor(eps).to(self.device)
            face_point = (face_vertex * eps[:, :, None]).sum(dim=1)

            face_v1 = face_vertex[:, 1, :] - face_vertex[:, 0, :]
            face_v2 = face_vertex[:, 2, :] - face_vertex[:, 1, :]
            face_normal = torch.cross(face_v1, face_v2)
            face_normal = face_normal / \
                (face_normal.norm(dim=1, keepdim=True) + 1e-10)
            face_value = torch.sigmoid(
                self.model.eval_points(face_point.unsqueeze(0), c)['occ'].logits
            )
            normal_target = -autograd.grad(
                [face_value.sum()], [face_point], create_graph=True)[0]

            normal_target = \
                normal_target / \
                (normal_target.norm(dim=1, keepdim=True) + 1e-10)
            loss_target = (face_value - threshold).pow(2).mean()
            loss_normal = \
                (face_normal - normal_target).pow(2).sum(dim=1).mean()

            loss = loss_target + 0.01 * loss_normal

            # Update
            loss.backward()
            optimizer.step()

        mesh.vertices = v.data.cpu().numpy()

        return mesh

    def generate_occ_grid(self, c=None, stats_dict={}, **kwargs):
        ''' Generates mesh from latent.
            Works for shapes normalized to a unit cube

        Args:
            c (tensor): latent conditioned code c
            stats_dict (dict): stats dictionary
        '''
        threshold = np.log(self.threshold) - np.log(1. - self.threshold)

        t0 = time.time()
        # Compute bounding box size
        box_size = 1 + self.padding
        
        # Shortcut
        if self.upsampling_steps == 0:
            nx = self.resolution0
            pointsf = box_size * make_3d_grid(
                (-0.5,)*3, (0.5,)*3, (nx,)*3
            )

            values = self.eval_points(pointsf, c, **kwargs).cpu().numpy()
            value_grid = values.reshape(nx, nx, nx)
        else:
            mesh_extractor = MISE(
                self.resolution0, self.upsampling_steps, threshold)

            points = mesh_extractor.query()
            while points.shape[0] != 0:
                # Query points
                pointsf = points / mesh_extractor.resolution
                # Normalize to bounding box
                pointsf = box_size * (pointsf - 0.5)
                pointsf = torch.FloatTensor(pointsf).to(self.device)
                # Evaluate model and update
                values = self.eval_points(pointsf, c, **kwargs).cpu().numpy()
                values = values.astype(np.float64)
                mesh_extractor.update(points, values)
                points = mesh_extractor.query()

            value_grid = mesh_extractor.to_dense()
        
        return value_grid
    