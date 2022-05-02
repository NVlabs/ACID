import os
import yaml
import pickle
import torch
import logging
import numpy as np
from torch.utils import data
from torch.utils.data.dataloader import default_collate 

from src.utils import plushsim_util, common_util

scene_range = plushsim_util.SCENE_RANGE.copy()
to_range = np.array([[-1.1,-1.1,-1.1],[1.1,1.1,1.1]]) * 0.5

logger = logging.getLogger(__name__)

def collate_remove_none(batch):
    ''' Collater that puts each data field into a tensor with outer dimension
        batch size.

    Args:
        batch: batch
    '''

    batch = list(filter(lambda x: x is not None, batch))
    return data.dataloader.default_collate(batch)


def worker_init_fn(worker_id):
    ''' Worker init function to ensure true randomness.
    '''
    def set_num_threads(nt):
        try: 
            import mkl; mkl.set_num_threads(nt)
        except: 
            pass
            torch.set_num_threads(1)
            os.environ['IPC_ENABLE']='1'
            for o in ['OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:
                os.environ[o] = str(nt)

    random_data = os.urandom(4)
    base_seed = int.from_bytes(random_data, byteorder="big")
    np.random.seed(base_seed + worker_id)

def collate_pair_fn(batch):
    num_points = batch[0]['sampled_pts'].shape[1]
    collated = {}
    for key in batch[0]:
        if key == 'geo_dists':
            collated[key] = torch.as_tensor(np.concatenate([d[key] for d in batch]))
        elif key == 'num_pairs':
            indices = []
            for i,d in enumerate(batch):
                indices.append(np.arange(d['num_pairs']) + i * num_points) 
            collated["pair_indices"] = torch.as_tensor(np.concatenate(indices))
        else:
            collated[key] = default_collate([d[key] for d in batch])
    return collated

class PlushEnvBoth(data.Dataset):

    def __init__(self, flow_root, pair_root, num_points, 
                 split="train", transform={}, pos_ratio=2):
        # Attributes
        self.flow_root = flow_root 
        self.num_points = num_points
        self.split = split
        if split != "train":
            self.num_points = -1
        self.pair_root = pair_root
        self.transform = transform
        self.pos_ratio = pos_ratio 

        if split == 'train':
            with open(os.path.join(flow_root, 'train.pkl'), 'rb') as fp:
                self.models = pickle.load(fp)
        else:
            with open(os.path.join(flow_root, 'test.pkl'), 'rb') as fp:
                self.models = pickle.load(fp)
            
    def __len__(self):
        ''' Returns the length of the dataset.
        '''
        return len(self.models)

    def __getitem__(self, idx):
        ''' Returns an item of the dataset.

        Args:
            idx (int): ID of data point
        '''
        data = {}
        split_id, model_id, reset_id, int_id = self.models[idx]

        # load frame and get partial observation 
        points_dict = np.load(
            plushsim_util.get_flow_data_file(
                self.flow_root,split_id, model_id, reset_id, int_id))
        obj_pcloud, env_pcloud = self._prepare_partial_obs(points_dict)

        # load pair frame info
        pair_info = np.load(
            plushsim_util.get_flow_pair_data_file(
                self.pair_root,split_id, model_id, reset_id, int_id))
        pair_reset_id, pair_int_id = self._get_pair_id(pair_info)

        # load pair frame and get partial observation 
        points_dict2 = np.load(
            plushsim_util.get_flow_data_file(
                self.flow_root,split_id, model_id, pair_reset_id, pair_int_id))
        obj_pcloud2, env_pcloud2 = self._prepare_partial_obs(points_dict2)

        if self.split == 'train':
            # if training, load random points
            # implicit network sampled points            
            pts, occs, sampled_pts, sampled_occ, sampled_flow, sampled_inds = self._prepare_points(
                points_dict)
            # get which occupied points are sampled (index is in the occupied subset)
            occed = occs != 0
            num_occed = occed.sum()
            total_to_occs = np.zeros(pts.shape[0], dtype=np.uint32)
            total_to_occs[occed] = np.arange(num_occed)
            sampled_occs_ids = total_to_occs[sampled_inds[sampled_occ == 1.]]
            # basically sampled_positive ids is used to index the pairs in pair info npz
            # reorganize sampled_pts
            sampled_pts = np.concatenate([sampled_pts[sampled_occ == 1.], sampled_pts[sampled_occ == 0.]])
            sampled_occ = np.concatenate([sampled_occ[sampled_occ == 1.], sampled_occ[sampled_occ == 0.]])
            sampled_flow = np.concatenate([sampled_flow[sampled_occ == 1.], sampled_flow[sampled_occ == 0.]])
            geo_dists, tgtids = self._prepare_pair_data(pair_info, sampled_occs_ids)
            _,_, sampled_pts2, sampled_occ2, sampled_flow2, _ = self._prepare_points(points_dict2, chosen=tgtids)            
        else:
            # if not training, load matched points
            sampled_pts, sampled_pts2, \
                sampled_occ, sampled_occ2, \
                sampled_flow, sampled_flow2, geo_dists = self._prepare_matched_unique(points_dict, points_dict2)

        data = {               
            "obj_obs":np.stack([obj_pcloud,obj_pcloud2]),
            "env_obs":np.stack([env_pcloud,env_pcloud2]),
            "sampled_pts":np.stack([sampled_pts,sampled_pts2]),
            "sampled_occ":np.stack([sampled_occ,sampled_occ2]),
            "sampled_flow":np.stack([sampled_flow,sampled_flow2]),
            "geo_dists":geo_dists.astype(np.float32),
            "num_pairs":len(geo_dists),
            "idx":idx,
            "start_frame":int(points_dict['start_frame']),
            "end_frame":int(points_dict['end_frame']),
        }
        return data

    def _get_pts_related_info(self, points_dict):
        pts = points_dict['pts'].astype(np.float32)
        occs = np.unpackbits(points_dict['occ'])
        inds = points_dict['ind']
        flow = np.zeros((len(pts), 3), dtype=np.float32)
        flow[occs != 0] = points_dict['flow'].astype(np.float32) * 10.
        return pts, occs, inds, flow
    
    def _prepare_matched_unique(self, points_dict, points_dict2):
        pts1,occs1,inds1,flow1 = self._get_pts_related_info(points_dict)
        pts2,occs2,inds2,flow2 = self._get_pts_related_info(points_dict2)

        cls1, id1 = np.unique(inds1, return_index=True)
        cls2, id2 = np.unique(inds2, return_index=True)
        int_cls, int_id1, int_id2 = np.intersect1d(cls1, cls2, 
                        assume_unique=True, return_indices=True)
        geo_dists = np.zeros_like(int_cls)

        unique_pts_1 = pts1[occs1==1][id1[int_id1]]
        unique_flow_1 = flow1[occs1==1][id1[int_id1]]
        unique_occ_1 = np.ones(geo_dists.shape[0], dtype=occs1.dtype)

        sub_inds = common_util.subsample_points(unique_pts_1, resolution=0.03, return_index=True)
        unique_pts_1 = unique_pts_1[sub_inds]
        unique_flow_1 = unique_flow_1[sub_inds]
        unique_occ_1 = unique_occ_1[sub_inds]

        sample_others1 = np.random.randint(pts1.shape[0], size=pts1.shape[0] - unique_pts_1.shape[0])
        pts_others1 = pts1[sample_others1]
        occ_others1 = occs1[sample_others1]
        flow_others1 = flow1[sample_others1]
        sampled_pts1 = np.concatenate([unique_pts_1, pts_others1])
        sampled_occ1 = np.concatenate([unique_occ_1, occ_others1])
        sampled_flow1 = np.concatenate([unique_flow_1, flow_others1])

        unique_pts_2 = pts2[occs2==1][id2[int_id2]]
        unique_flow_2 = flow2[occs2==1][id2[int_id2]]
        unique_occ_2 = np.ones(geo_dists.shape[0], dtype=occs2.dtype)

        unique_pts_2 = unique_pts_2[sub_inds]
        unique_flow_2 = unique_flow_2[sub_inds]
        unique_occ_2 = unique_occ_2[sub_inds]

        sample_others2 = np.random.randint(pts2.shape[0], size=pts2.shape[0] - unique_pts_2.shape[0])
        pts_others2 = pts2[sample_others2]
        occ_others2 = occs2[sample_others2]
        flow_others2 = flow2[sample_others2]
        sampled_pts2 = np.concatenate([unique_pts_2, pts_others2])
        sampled_occ2 = np.concatenate([unique_occ_2, occ_others2])
        sampled_flow2 = np.concatenate([unique_flow_2, flow_others2])

        geo_dists = geo_dists[sub_inds]
        return sampled_pts1, sampled_pts2,\
             sampled_occ1, sampled_occ2, \
             sampled_flow1, sampled_flow2, geo_dists

    
    def _prepare_partial_obs(self, info_dict):
        # obj partial observation 
        obj_pcloud = info_dict['obj_pcloud_obs'].astype(np.float32) 
        grasp_loc = common_util.transform_points(info_dict['grasp_loc'], scene_range, to_range)
        target_loc = common_util.transform_points(info_dict['target_loc'], scene_range, to_range)
        tiled_grasp_loc = np.tile(grasp_loc, (len(obj_pcloud), 1)).astype(np.float32)
        tiled_target_loc = np.tile(target_loc, (len(obj_pcloud), 1)).astype(np.float32)
        obj_pcloud= np.concatenate([obj_pcloud, tiled_target_loc, obj_pcloud[:,:3] - tiled_grasp_loc], axis=-1)
        if 'obj_pcloud' in self.transform:
            obj_pcloud = self.transform['obj_pcloud'](obj_pcloud)

        # scene partial observation 
        env_pcloud = info_dict['env_pcloud'].astype(np.float32) 
        env_pcloud += 1e-4 * np.random.randn(*env_pcloud.shape)
        if 'env_pcloud' in self.transform:
            env_pcloud = self.transform['env_pcloud'](env_pcloud)
        return obj_pcloud, env_pcloud
    
    # chosen is the set of positive points that's preselected
    def _prepare_points(self, points_dict, chosen=None):
        pts,occs,inds,flow = self._get_pts_related_info(points_dict)
        if chosen is None:
            if self.num_points == -1:
                sampled_pts = pts
                sampled_occ = occs
                sampled_flow = flow
                sampled_inds = np.arange(len(pts))
            else:
                sampled_inds = np.random.randint(pts.shape[0], size=self.num_points)
                sampled_pts = pts[sampled_inds]
                sampled_occ = occs[sampled_inds]
                sampled_flow = flow[sampled_inds]
        else:
            pts_chosen = pts[occs!= 0][chosen]
            occ_chosen = np.ones(chosen.shape[0], dtype=occs.dtype)
            flow_chosen = flow[occs!= 0][chosen]

            if self.num_points == -1:
                sample_others = np.random.randint(pts.shape[0], size=pts.shape[0] - chosen.shape[0])
            else:
                sample_others = np.random.randint(pts.shape[0], size=self.num_points - chosen.shape[0])
            pts_others = pts[sample_others]
            occ_others = occs[sample_others]
            flow_others = flow[sample_others]

            sampled_inds = np.concatenate([chosen, sample_others])
            sampled_pts = np.concatenate([pts_chosen, pts_others])
            sampled_occ = np.concatenate([occ_chosen, occ_others])
            sampled_flow= np.concatenate([flow_chosen, flow_others])
        return pts, occs, sampled_pts, sampled_occ.astype(np.float32), sampled_flow, sampled_inds

    def _get_pair_id(self, pair_info):
        pair_filename = os.path.splitext(str(pair_info["target_file"]))[0]
        pair_reset_id, pair_frame_id = (int(f) for f in pair_filename.split('_'))
        return pair_reset_id, pair_frame_id

    def _prepare_pair_data(self, pair_info, sampled_occs_ids):
        # load pair info
        dists_sampled = pair_info['dists'][sampled_occs_ids]
        tgtid_sampled = pair_info['inds'][sampled_occs_ids]
        # draw samples, 
        # for half of the points, we draw from their three closests,
        # for the other half, we draw from the further points
        H,W = dists_sampled.shape
        draw_pair_ids = np.random.randint(3, size=H) 
        draw_pair_ids[H // self.pos_ratio:] = np.random.randint(3, high=W, size=H - H // self.pos_ratio)

        tgtids = tgtid_sampled[np.arange(H), draw_pair_ids]
        geo_dists = dists_sampled[np.arange(H), draw_pair_ids]
        # contrastive_mask = geo_dists > self.contrastive_threshold
        return geo_dists, tgtids
    
    def get_model_dict(self, idx):
        return self.models[idx]

class PlushEnvGeom(data.Dataset):

    def __init__(self, geom_root, pair_root, num_points, 
                 split="train", transform={}, pos_ratio=2):
        # Attributes
        self.geom_root = geom_root
        self.num_points = num_points
        self.split = split
        if split != "train":
            self.num_points = -1
        self.pair_root = pair_root
        self.transform = transform
        self.pos_ratio = pos_ratio 

        if split == 'train':
            with open(os.path.join(geom_root, 'train.pkl'), 'rb') as fp:
                self.models = pickle.load(fp)
        else:
            with open(os.path.join(geom_root, 'test.pkl'), 'rb') as fp:
                self.models = pickle.load(fp)
            
    def __len__(self):
        ''' Returns the length of the dataset.
        '''
        return len(self.models)

    def __getitem__(self, idx):
        ''' Returns an item of the dataset.

        Args:
            idx (int): ID of data point
        '''
        data = {}
        split_id, model_id, reset_id, frame_id = self.models[idx]

        # load frame and get partial observation 
        points_dict = np.load(
            plushsim_util.get_geom_data_file(
                self.geom_root,split_id, model_id, reset_id, frame_id))
        obj_pcloud, env_pcloud = self._prepare_partial_obs(points_dict)

        # load pair frame info
        pair_info = np.load(
            plushsim_util.get_pair_data_file(
                self.pair_root,split_id, model_id, reset_id, frame_id))
        pair_reset_id, pair_frame_id = self._get_pair_id(pair_info)

        # load pair frame and get partial observation 
        points_dict2 = np.load(
            plushsim_util.get_geom_data_file(
                self.geom_root,split_id, model_id, pair_reset_id, pair_frame_id))
        obj_pcloud2, env_pcloud2 = self._prepare_partial_obs(points_dict2)

        if self.split == 'train':
            # if training, load random points
            # implicit network sampled points            
            pts, occs, sampled_pts, sampled_occ, sampled_inds = self._prepare_points(points_dict)
            # get which occupied points are sampled (index is in the occupied subset)
            occed = occs != 0
            num_occed = occed.sum()
            total_to_occs = np.zeros(pts.shape[0], dtype=np.uint32)
            total_to_occs[occed] = np.arange(num_occed)
            sampled_occs_ids = total_to_occs[sampled_inds[sampled_occ == 1.]]
            # basically sampled_positive ids is used to index the pairs in pair info npz
            # reorganize sampled_pts
            sampled_pts = np.concatenate([sampled_pts[sampled_occ == 1.], sampled_pts[sampled_occ == 0.]])
            sampled_occ = np.concatenate([sampled_occ[sampled_occ == 1.], sampled_occ[sampled_occ == 0.]])
            geo_dists, tgtids = self._prepare_pair_data(pair_info, sampled_occs_ids)
            _,_, sampled_pts2, sampled_occ2, _ = self._prepare_points(points_dict2, chosen=tgtids)            
        else:
            # if not training, load matched points
            sampled_pts, sampled_pts2, sampled_occ, sampled_occ2, geo_dists = self._prepare_matched_unique(points_dict, points_dict2)

        data = {               
            "obj_obs":np.stack([obj_pcloud,obj_pcloud2]),
            "env_obs":np.stack([env_pcloud,env_pcloud2]),
            "sampled_pts":np.stack([sampled_pts,sampled_pts2]),
            "sampled_occ":np.stack([sampled_occ,sampled_occ2]),
            "geo_dists":geo_dists.astype(np.float32),
            "num_pairs":len(geo_dists),
            "idx":idx,
        }
        return data
    
    def _prepare_matched_unique(self, points_dict, points_dict2):
        pts1 = points_dict['pts'].astype(np.float32)
        occs1 = np.unpackbits(points_dict['occ'])
        inds1 = points_dict['ind']
        pts2 = points_dict2['pts'].astype(np.float32)
        occs2 = np.unpackbits(points_dict2['occ'])
        inds2 = points_dict2['ind']

        cls1, id1 = np.unique(inds1, return_index=True)
        cls2, id2 = np.unique(inds2, return_index=True)
        int_cls, int_id1, int_id2 = np.intersect1d(cls1, cls2, assume_unique=True, return_indices=True)
        geo_dists = np.zeros_like(int_cls)

        unique_pts_1 = pts1[occs1==1][id1[int_id1]]
        unique_pts_2 = pts2[occs2==1][id2[int_id2]]
        unique_occ_1 = np.ones(geo_dists.shape[0], dtype=occs1.dtype)
        unique_occ_2 = np.ones(geo_dists.shape[0], dtype=occs2.dtype)

        sample_others1 = np.random.randint(pts1.shape[0], size=pts1.shape[0] - unique_pts_1.shape[0])
        sample_others2 = np.random.randint(pts2.shape[0], size=pts2.shape[0] - unique_pts_2.shape[0])
        pts_others1 = pts1[sample_others1]
        occ_others1 = occs1[sample_others1]
        pts_others2 = pts2[sample_others2]
        occ_others2 = occs2[sample_others2]

        sampled_pts1 = np.concatenate([unique_pts_1, pts_others1])
        sampled_occ1 = np.concatenate([unique_occ_1, occ_others1])
        sampled_pts2 = np.concatenate([unique_pts_2, pts_others2])
        sampled_occ2 = np.concatenate([unique_occ_2, occ_others2])
        return sampled_pts1, sampled_pts2, sampled_occ1, sampled_occ2, geo_dists

    
    def _prepare_partial_obs(self, info_dict):
        # obj partial observation 
        obj_pcloud = info_dict['obj_pcloud'].astype(np.float32) 
        obj_pcloud += 1e-4 * np.random.randn(*obj_pcloud.shape)
        if 'obj_pcloud' in self.transform:
            obj_pcloud = self.transform['obj_pcloud'](obj_pcloud)

        # scene partial observation 
        env_pcloud = info_dict['env_pcloud'].astype(np.float32) 
        env_pcloud += 1e-4 * np.random.randn(*env_pcloud.shape)
        if 'env_pcloud' in self.transform:
            env_pcloud = self.transform['env_pcloud'](env_pcloud)
        return obj_pcloud, env_pcloud
    
    # chosen is the set of positive points that's preselected
    def _prepare_points(self, points_dict, chosen=None):
        pts = points_dict['pts'].astype(np.float32)
        occs = points_dict['occ']
        occs = np.unpackbits(occs)#[:points.shape[0]]
        if chosen is None:
            if self.num_points == -1:
                sampled_pts = pts
                sampled_occ = occs
                sampled_inds = np.arange(len(pts))
            else:
                sampled_inds = np.random.randint(pts.shape[0], size=self.num_points)
                sampled_pts = pts[sampled_inds]
                sampled_occ = occs[sampled_inds]
        else:
            pts_chosen = pts[occs!= 0][chosen]
            occ_chosen = np.ones(chosen.shape[0], dtype=occs.dtype)

            if self.num_points == -1:
                sample_others = np.random.randint(pts.shape[0], size=pts.shape[0] - chosen.shape[0])
            else:
                sample_others = np.random.randint(pts.shape[0], size=self.num_points - chosen.shape[0])
            pts_others = pts[sample_others]
            occ_others = occs[sample_others]

            sampled_inds = np.concatenate([chosen, sample_others])
            sampled_pts = np.concatenate([pts_chosen, pts_others])
            sampled_occ = np.concatenate([occ_chosen, occ_others])
        return pts, occs, sampled_pts, sampled_occ.astype(np.float32), sampled_inds

    def _get_pair_id(self, pair_info):
        pair_filename = os.path.splitext(str(pair_info["target_file"]))[0]
        pair_reset_id, pair_frame_id = (int(f) for f in pair_filename.split('_'))
        return pair_reset_id, pair_frame_id

    def _prepare_pair_data(self, pair_info, sampled_occs_ids):
        # load pair info
        dists_sampled = pair_info['dists'][sampled_occs_ids]
        tgtid_sampled = pair_info['inds'][sampled_occs_ids]
        # draw samples, 
        # for half of the points, we draw from their three closests,
        # for the other half, we draw from the further points
        H,W = dists_sampled.shape
        draw_pair_ids = np.random.randint(3, size=H) 
        draw_pair_ids[H // self.pos_ratio:] = np.random.randint(3, high=W, size=H - H // self.pos_ratio)

        tgtids = tgtid_sampled[np.arange(H), draw_pair_ids]
        geo_dists = dists_sampled[np.arange(H), draw_pair_ids]
        # contrastive_mask = geo_dists > self.contrastive_threshold
        return geo_dists, tgtids
    
    def get_model_dict(self, idx):
        return self.models[idx]

def build_transform_geom(cfg):
    from . import transforms as tsf
    from torchvision import transforms
    transform = {}
    transform['obj_pcloud'] = transforms.Compose([
        tsf.SubsamplePointcloud(cfg['data']['pointcloud_n_obj']),
        tsf.PointcloudNoise(cfg['data']['pointcloud_noise'])
    ])
    transform['env_pcloud'] = transforms.Compose([
        tsf.SubsamplePointcloud(cfg['data']['pointcloud_n_env']),
        tsf.PointcloudNoise(cfg['data']['pointcloud_noise'])
    ])
    return transform

def get_geom_dataset(cfg, split='train', transform='build'):
    geom_root = cfg['data']['geom_path'] 
    pair_root = cfg['data']['pair_path'] 
    num_points = cfg['data']['points_subsample']
    pos_ratio = cfg['data'].get('pos_ratio', 2)
    if transform == 'build':
        transform = build_transform_geom(cfg)
    return PlushEnvGeom(geom_root, pair_root, num_points, split=split, transform=transform, pos_ratio=pos_ratio) 

def get_combined_dataset(cfg, split='train', transform='build'):
    flow_root = cfg['data']['flow_path'] 
    pair_root = cfg['data']['pair_path'] 
    num_points = cfg['data']['points_subsample']
    pos_ratio = cfg['data'].get('pos_ratio', 2)
    if transform == 'build':
        transform = build_transform_geom(cfg)
    return PlushEnvBoth(flow_root, pair_root, num_points, split=split, transform=transform, pos_ratio=pos_ratio) 


def get_plush_loader(cfg, mode, split='train', transform='build', test_shuffle=False, num_workers=None):
    if mode == 'geom':
        dataset = get_geom_dataset(cfg, split, transform)
    elif mode == 'combined':
        dataset = get_combined_dataset(cfg, split, transform)
    if split == 'train':
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=cfg['training']['batch_size'], 
            num_workers=cfg['training']['n_workers'], 
            shuffle=True,
            collate_fn=collate_pair_fn,
            worker_init_fn=worker_init_fn)
    else:
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=1, 
            num_workers=cfg['training']['n_workers_val'] if num_workers is None else num_workers, 
            shuffle=test_shuffle,
            collate_fn=collate_pair_fn)
    return loader

def get_plan_loader(cfg, transform='build', category="teddy",num_workers=None):
    transform = build_transform_geom(cfg)
    dataset = PlushEnvPlan(cfg['data']['plan_path'], category=category, transform=transform)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, 
        num_workers=cfg['training']['n_workers_val'] if num_workers is None else num_workers, 
        shuffle=False,)
    return loader

class PlushEnvPlan(data.Dataset):

    def __init__(self, plan_root, category="teddy",transform={}):
        # Attributes
        self.plan_root = plan_root 
        self.transform = transform
        self.category = category

        import glob
        self.scenarios = glob.glob(f'{plan_root}/**/*.npz', recursive=True)
        self.scenarios = [x for x in self.scenarios if category in x][:-1]
        self.scenarios.sort()
            
    def __len__(self):
        ''' Returns the length of the dataset.
        '''
        return len(self.scenarios)

    def __getitem__(self, idx):
        ''' Returns an item of the dataset.

        Args:
            idx (int): ID of data point
        '''
        data = {}

        # load frame and get partial observation 
        infos = np.load(self.scenarios[idx])
        obj_pcloud_start, env_pcloud_start = self._prepare_partial_obs(infos, "start")
        obj_pcloud_end, env_pcloud_end = self._prepare_partial_obs(infos, "end")
        action = infos['actions'].astype(np.float32)            
        pts_start, occ_start, ind_start  = self._get_pts_related_info(infos, 'start')
        pts_end, occ_end, ind_end  = self._get_pts_related_info(infos, 'end')

        data = {               
            "obj_obs_start":obj_pcloud_start,
            "env_obs_start":env_pcloud_start,
            "obj_obs_end":obj_pcloud_end,
            "env_obs_end":env_pcloud_end,
            'gt_pts_start': infos['sim_pts_start'].astype(np.float32),
            'gt_pts_end': infos['sim_pts_end'].astype(np.float32),
            'sampled_pts_start': pts_start,
            'sampled_occ_start': occ_start,
            'sampled_ind_start': ind_start,
            'sampled_pts_end': pts_end,
            'sampled_occ_end': occ_end,
            'sampled_ind_end': ind_end,
            "actions": action,
            "sequence_ids":infos['sequence_ids'],
            "fname":self.scenarios[idx],
            "idx":idx,
        }
        return data

    def _prepare_partial_obs(self, info_dict, key):
        # obj partial observation 
        obj_pcloud = info_dict[f'obj_pcloud_{key}'].astype(np.float32) 
        if 'obj_pcloud' in self.transform:
            obj_pcloud = self.transform['obj_pcloud'](obj_pcloud)

        # scene partial observation 
        env_pcloud = info_dict[f'env_pcloud_{key}'].astype(np.float32) 
        env_pcloud += 1e-4 * np.random.randn(*env_pcloud.shape)
        if 'env_pcloud' in self.transform:
            env_pcloud = self.transform['env_pcloud'](env_pcloud)
        return obj_pcloud, env_pcloud
    
    def _get_pts_related_info(self, points_dict, key):
        pts = points_dict[f'pts_{key}'].astype(np.float32)
        occs = np.unpackbits(points_dict[f'occ_{key}']).astype(np.float32)
        inds = points_dict[f'ind_{key}'].astype(np.int32)
        return pts, occs, inds 