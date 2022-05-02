import torch
import numpy as np
import torch.nn as nn
from torch import distributions as dist
from src.conv_onet.models import decoder
from src.utils import plushsim_util

# Decoder dictionary
decoder_dict = {
    'geom_decoder': decoder.GeomDecoder,
    'combined_decoder': decoder.CombinedDecoder,
}
class ConvImpDyn(nn.Module):
    def __init__(self, obj_per_encoder, obj_act_encoder, env_encoder, decoder, device=None, env_scale_factor=2.):
        super().__init__()
        
        self.decoder = decoder.to(device)

        self.obj_per_encoder = obj_per_encoder.to(device)
        self.obj_act_encoder = obj_act_encoder.to(device)

        if env_encoder is None:
            self.env_encoder = env_encoder
        else:
            self.env_encoder = env_encoder.to(device)
            self.env_upsample = torch.nn.UpsamplingBilinear2d(scale_factor=env_scale_factor)
        self._device = device

    def forward(self, inputs, sample=True, **kwargs):
        ''' Performs a forward pass through the network.

        Args:
            p (tensor): sampled points
            inputs (tensor): conditioning input
            sample (bool): whether to sample for z
        '''
        #############
        c_per, c_act = self.encode_inputs(inputs)
        return self.decode(inputs, c_per, c_act, **kwargs)

    def forward_perception(self, inputs, filter=True,):
        c_per, c_env = self.encode_perception(inputs, merge_env_feature=False)
        for k in c_per.keys():
            env_f = self.env_upsample(c_env[k])
            c_env[k] = env_f
            c_per[k] = torch.cat([c_per[k], env_f], dim=1)
        # get curr observation state and features
        p = inputs['sampled_pts']
        if len(p.shape) > 3:
            B,_,N,C = p.shape
            curr_p = p.reshape([B*2,N,C])
        else:
            curr_p = p
        curr_state, per_features = self.decoder.decode_perception(curr_p, c_per)
        occ_pred = dist.Bernoulli(logits=curr_state['occ']).probs >= 0.5
        curr_state['occ'] = occ_pred
        if filter:
            curr_p = curr_p[occ_pred]
            if 'corr' in curr_state:
                curr_state['corr'] = curr_state['corr'][occ_pred]
            for i,p in enumerate(per_features):
                per_features[i] = p[occ_pred]
        return c_per, c_env, curr_p, curr_state, per_features 

    def rollout(self, pts, per_features, c_env, actions):
        actions = actions.squeeze()
        num_sequence = actions.shape[0]
        num_actions = actions.shape[-2]
        all_traj = []
        total_time_act_render = 0
        total_time_act_decode = 0
        import time
        # from functools import partial
        # render_pts_func = partial(plushsim_util.render_points, return_index=True)
        curr_pts = [pts for _ in range(num_sequence)]
        for j in range(num_actions):
            act_traj = []
            points_world = [p.cpu().numpy().squeeze() 
                            * (1200, 1200, 400) 
                            / (1.1,1.1,1.1) 
                            + (0, 0, 180) for p in curr_pts]
            for i in range(num_sequence):
                g,t = actions[i,0,j], actions[i,1,j]
                start_time = time.time()
                c_act, act_partial = self.get_action_encoding(curr_pts[i], g, t, c_env)
                total_time_act_render += time.time() - start_time
                act_traj.append(act_partial)
                start_time = time.time()
                flow = self.decoder.decode_action(curr_pts[i], c_act, per_features)['flow']
                curr_pts[i] = curr_pts[i] + flow / 10.
                total_time_act_decode += time.time() - start_time
            all_traj.append((curr_pts.copy(), act_traj))
        print("total time render: ",total_time_act_render)
        print("total time decode: ",total_time_act_decode)
        return all_traj
    
    def rollout_async(self, pts, per_features, c_env, actions):
        actions = actions.squeeze()
        num_sequence = actions.shape[0]
        num_actions = actions.shape[-2]
        all_traj = []
        total_time_act_render = 0
        total_time_act_decode = 0
        total_async_time_act_render = 0
        import time
        from functools import partial
        render_pts_func = partial(plushsim_util.render_points, return_index=True)
        curr_pts = [pts for _ in range(num_sequence)]
        for j in range(num_actions):
            start_time = time.time()
            points_world = [p.cpu().numpy().squeeze() 
                            * (1200, 1200, 400) 
                            / (1.1,1.1,1.1) 
                            + (0, 0, 180) for p in curr_pts]
            from multiprocessing import Pool
            with Pool(16) as p: 
                vis_idxes = p.map(render_pts_func, points_world)
            xyzs, acts = [],[]
            for i in range(num_sequence):
                g,t = actions[i,0,j], actions[i,1,j]
                # c_act, act_partial = self.get_action_encoding(
                #         curr_pts[i], g, t, c_env, vis_idx=vis_idxes[i])
                obj_xyz, obj_act = self.get_action_encoding_new(
                        curr_pts[i], g, t, c_env, vis_idx=vis_idxes[i])
                xyzs.append(obj_xyz)
                acts.append(obj_act)
            total_time_act_render += time.time() - start_time
            n = 20
            start_time = time.time()
            xyz_chunks = [xyzs[i:i+n] for i in range(0, num_sequence, n)]
            act_chunks = [acts[i:i+n] for i in range(0, num_sequence, n)]
            c_acts = []
            for xyz, act in zip(xyz_chunks, act_chunks):
                obj_xyz = torch.as_tensor(np.stack(xyz).astype(np.float32)).to(self._device)
                obj_act = torch.as_tensor(np.stack(act).astype(np.float32)).to(self._device)
                c_act_new = self.obj_act_encoder((obj_xyz, obj_act))
                for chunk_i in range(len(xyz)):
                    c_act = {}
                    for k in c_act_new.keys():
                        c_act[k] = torch.cat([c_act_new[k][chunk_i].unsqueeze(0), c_env[k]], dim=1)
                    c_acts.append(c_act)
            total_time_act_decode += time.time() - start_time
            from src.utils import common_util
            from PIL import Image
            for k,v in c_acts[0].items():
                v_np = v.squeeze().permute(1,2,0).cpu().numpy()
                feature_plane = v_np.reshape([-1, v_np.shape[-1]])
                tsne_result = common_util.embed_tsne(feature_plane)
                colors = common_util.get_color_map(tsne_result)
                colors = colors.reshape((128,128,-1)).astype(np.float32)
                colors = (colors * 255 / np.max(colors)).astype('uint8')
                img = Image.fromarray(colors)
                img.save(f"act_{k}.png")
            import pdb; pdb.set_trace()
            for i in range(num_sequence):
                flow = self.decoder.decode_action(curr_pts[i], c_acts[i], per_features)['flow']
                curr_pts[i] = curr_pts[i] + flow / 10.

            all_traj.append(([p.cpu().numpy().squeeze() for p in curr_pts], xyzs))
        return all_traj
        
    def get_action_encoding_new(self, pts, grasp_loc, target_loc, c_env, vis_idx=None):
        # pts: B*2, N, 3
        import time
        start_time = time.time()
        B,N,_ = pts.shape
        pts = pts.cpu().numpy()

        xyzs, acts = [], []
        # get visable points by rendering pts
        occ_pts = pts[0]
        occ_pts_t = occ_pts * (1200, 1200, 400) / (1.1,1.1,1.1) + (0,0,180)
        if vis_idx is None:
            vis_idx = plushsim_util.render_points(occ_pts_t,
                                            plushsim_util.CAM_EXTR,
                                            plushsim_util.CAM_INTR,
                                            return_index=True)
        obj_xyz = occ_pts[vis_idx]
        #print("time split 1: ", time.time() - start_time)
        start_time = time.time()
        # subsample pts
        indices = np.random.randint(obj_xyz.shape[0], size=5000)
        obj_xyz = obj_xyz[indices]
        # make action feature
        tiled_grasp_loc = np.tile(grasp_loc.cpu().numpy(), (len(obj_xyz), 1)).astype(np.float32)
        tiled_target_loc = np.tile(target_loc.cpu().numpy(), (len(obj_xyz), 1)).astype(np.float32)
        obj_act = np.concatenate([tiled_target_loc, obj_xyz - tiled_grasp_loc], axis=-1)
        return obj_xyz, obj_act

    def get_action_encoding(self, pts, grasp_loc, target_loc, c_env, vis_idx=None):
        # pts: B*2, N, 3
        import time
        start_time = time.time()
        B,N,_ = pts.shape
        pts = pts.cpu().numpy()

        xyzs, acts = [], []
        # get visable points by rendering pts
        occ_pts = pts[0]
        occ_pts_t = occ_pts * (1200, 1200, 400) / (1.1,1.1,1.1) + (0,0,180)
        if vis_idx is None:
            vis_idx = plushsim_util.render_points(occ_pts_t,
                                            plushsim_util.CAM_EXTR,
                                            plushsim_util.CAM_INTR,
                                            return_index=True)
        obj_xyz = occ_pts[vis_idx]
        #print("time split 1: ", time.time() - start_time)
        start_time = time.time()
        # subsample pts
        indices = np.random.randint(obj_xyz.shape[0], size=5000)
        obj_xyz = obj_xyz[indices]
        # make action feature
        tiled_grasp_loc = np.tile(grasp_loc.cpu().numpy(), (len(obj_xyz), 1)).astype(np.float32)
        tiled_target_loc = np.tile(target_loc.cpu().numpy(), (len(obj_xyz), 1)).astype(np.float32)
        obj_act = np.concatenate([tiled_target_loc, obj_xyz - tiled_grasp_loc], axis=-1)
        xyzs.append(obj_xyz)
        acts.append(obj_act)

        obj_xyz = torch.as_tensor(np.stack(xyzs).astype(np.float32)).to(self._device)
        obj_act = torch.as_tensor(np.stack(acts).astype(np.float32)).to(self._device)
        #print("time split 2: ", time.time() - start_time)
        start_time = time.time()
        c_act_new = self.obj_act_encoder((obj_xyz, obj_act))
        #print("time split 3: ", time.time() - start_time)
        start_time = time.time()
        for k in c_act_new.keys():
            c_act_new[k] = torch.cat([c_act_new[k], c_env[k]], dim=1)
        #print("time split 4: ", time.time() - start_time)
        start_time = time.time()
        return c_act_new, obj_xyz

    def encode_perception(self, inputs, merge_env_feature=True):
        obj_pcloud = inputs['obj_obs']
        if len(obj_pcloud.shape) > 3:
            B,_,N,C = obj_pcloud.shape
            obj_pcloud = obj_pcloud.reshape([B*2,N,C])
        obj_xyz, obj_rgb = obj_pcloud[...,:3],obj_pcloud[...,3:6]
        c_per = self.obj_per_encoder((obj_xyz, obj_rgb))

        if self.env_encoder is not None:
            env_pcloud = inputs['env_obs'].cuda()
            if len(env_pcloud.shape) > 3:
                B,_,N,C = env_pcloud.shape
                env_pcloud = env_pcloud.reshape([B*2,N,C])
            env_xyz, env_rgb = env_pcloud[...,:3],env_pcloud[...,3:]
            env_features = self.env_encoder((env_xyz, env_rgb))
            if merge_env_feature:
                for k in c_per.keys():
                    env_f = self.env_upsample(env_features[k])
                    c_per[k] = torch.cat([c_per[k], env_f], dim=1)
            else:
                return c_per, env_features
        return c_per

    def encode_inputs(self, inputs):
        ''' Encodes the input.

        Args:
            input (tensor): the input
        '''

        obj_pcloud = inputs['obj_obs']
        B,_,N,C = obj_pcloud.shape
        obj_pcloud = obj_pcloud.reshape([B*2,N,C])
        obj_xyz, obj_rgb, obj_act = obj_pcloud[...,:3],obj_pcloud[...,3:6],obj_pcloud[...,6:]
        c_per = self.obj_per_encoder((obj_xyz, obj_rgb))
        c_act = self.obj_act_encoder((obj_xyz, obj_act))

        if self.env_encoder is not None:
            env_pcloud = inputs['env_obs'].cuda()
            B,_,N,C = env_pcloud.shape
            env_pcloud = env_pcloud.reshape([B*2,N,C])
            env_xyz, env_rgb = env_pcloud[...,:3],env_pcloud[...,3:]
            env_features = self.env_encoder((env_xyz, env_rgb))
            for k in c_per.keys():
                env_f = self.env_upsample(env_features[k])
                c_per[k] = torch.cat([c_per[k], env_f], dim=1)
                c_act[k] = torch.cat([c_act[k], env_f], dim=1)
        return c_per, c_act
    
    def eval_points(self, pts, c):
        outputs = self.decoder(pts, *c)
        if 'occ' in outputs:
            outputs['occ'] = dist.Bernoulli(logits=outputs['occ'])
        return outputs

    def decode(self, inputs, c1, c2, **kwargs):
        ''' Returns occupancy probabilities for the sampled points.

        Args:
            p (tensor): points
            c (tensor): latent conditioned code c
        '''
        p = inputs['sampled_pts']
        B,_,N,C = p.shape
        p = p.reshape([B*2,N,C])
        outputs = self.decoder(p, c1, c2)
        if 'occ' in outputs:
            outputs['occ'] = dist.Bernoulli(logits=outputs['occ'])
        if 'corr' in outputs:
            _,N,C = outputs['corr'].shape
            corr_f = outputs['corr'].reshape([B,2,N,C])
            if 'skip_indexing' not in kwargs:
                corr_f = torch.transpose(corr_f, 0, 1)
                corr_f = torch.flatten(corr_f, 1, 2)
                inds = inputs['pair_indices']
                corr_f = corr_f[:,inds]
            outputs['corr'] = corr_f
        return outputs

    def to(self, device):
        ''' Puts the model to the device.

        Args:
            device (device): pytorch device
        '''
        model = super().to(device)
        model._device = device
        return model

class ConvOccGeom(nn.Module):
    ''' Occupancy Network class.

    Args:
        decoder (nn.Module): decoder network
        encoder (nn.Module): encoder network
        device (device): torch device
    '''

    def __init__(self, obj_encoder, env_encoder, decoder, device=None, env_scale_factor=2.):
        super().__init__()
        
        self.decoder = decoder.to(device)

        self.obj_encoder = obj_encoder.to(device)
        if env_encoder is None:
            self.env_encoder = env_encoder
        else:
            self.env_encoder = env_encoder.to(device)
            self.env_upsample = torch.nn.UpsamplingBilinear2d(scale_factor=env_scale_factor)
        self._device = device

    def forward(self, inputs, sample=True, **kwargs):
        ''' Performs a forward pass through the network.

        Args:
            p (tensor): sampled points
            inputs (tensor): conditioning input
            sample (bool): whether to sample for z
        '''
        #############
        c = self.encode_inputs(inputs)
        return self.decode(inputs, c, **kwargs)

    def encode_inputs(self, inputs):
        ''' Encodes the input.

        Args:
            input (tensor): the input
        '''

        obj_pcloud = inputs['obj_obs']
        B,_,N,C = obj_pcloud.shape
        obj_pcloud = obj_pcloud.reshape([B*2,N,C])
        obj_xyz, obj_rgb = obj_pcloud[...,:3],obj_pcloud[...,3:]
        obj_features = self.obj_encoder((obj_xyz, obj_rgb))
        if self.env_encoder is None:
            return obj_features
        env_pcloud = inputs['env_obs'].cuda()
        B,_,N,C = env_pcloud.shape
        env_pcloud = env_pcloud.reshape([B*2,N,C])
        env_xyz, env_rgb = env_pcloud[...,:3],env_pcloud[...,3:]
        env_features = self.env_encoder((env_xyz, env_rgb))
        joint_features = {}
        for k in obj_features.keys():
            env_f = self.env_upsample(env_features[k])
            joint_features[k] = torch.cat([obj_features[k], env_f], dim=1)
        return joint_features
    
    def eval_points(self, pts, c):
        outputs = self.decoder(pts, c)
        if 'occ' in outputs:
            outputs['occ'] = dist.Bernoulli(logits=outputs['occ'])
        return outputs

    def decode(self, inputs, c, **kwargs):
        ''' Returns occupancy probabilities for the sampled points.

        Args:
            p (tensor): points
            c (tensor): latent conditioned code c
        '''
        p = inputs['sampled_pts']
        B,_,N,C = p.shape
        p = p.reshape([B*2,N,C])
        outputs = self.decoder(p, c, **kwargs)
        if 'occ' in outputs:
            outputs['occ'] = dist.Bernoulli(logits=outputs['occ'])
        if 'corr' in outputs:
            _,N,C = outputs['corr'].shape
            corr_f = outputs['corr'].reshape([B,2,N,C])
            corr_f = torch.transpose(corr_f, 0, 1)
            corr_f = torch.flatten(corr_f, 1, 2)
            inds = inputs['pair_indices']
            corr_f = corr_f[:,inds]
            outputs['corr'] = corr_f
        return outputs

    def to(self, device):
        ''' Puts the model to the device.

        Args:
            device (device): pytorch device
        '''
        model = super().to(device)
        model._device = device
        return model
