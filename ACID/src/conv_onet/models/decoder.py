import torch
import torch.nn as nn
import torch.nn.functional as F
from src.layers import ResnetBlockFC
from src.common import normalize_coordinate, normalize_3d_coordinate, map2local

class GeomDecoder(nn.Module):
    ''' Decoder.
        Instead of conditioning on global features, on plane/volume local features.

    Args:
        dim (int): input dimension
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        n_blocks (int): number of blocks ResNetBlockFC layers
        leaky (bool): whether to use leaky ReLUs
        sample_mode (str): sampling feature strategy, bilinear|nearest
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
    '''

    def __init__(self, dim=3, c_dim=128, 
                 corr_dim=0, corr_head=True, 
                 hidden_size=256, n_blocks=5, leaky=False, 
                 sample_mode='bilinear', padding=0.1):
        super().__init__()
        self.c_dim = c_dim
        self.n_blocks = n_blocks
        self.corr_dim = corr_dim
        self.corr_head = corr_head

        self.fc_c_occ = nn.ModuleList([
            nn.Linear(c_dim, hidden_size) for i in range(n_blocks)
        ])

        self.fc_p = nn.Linear(dim, hidden_size)

        self.blocks_occ = nn.ModuleList([
            ResnetBlockFC(hidden_size) for i in range(n_blocks)
        ])
        self.fc_occ = nn.Linear(hidden_size, 1)

        if self.corr_dim != 0 and corr_head:
            self.fc_out_corr = nn.Linear(hidden_size, corr_dim)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        self.sample_mode = sample_mode
        self.padding = padding
    

    def sample_plane_feature(self, p, c, plane='xz'):
        xy = normalize_coordinate(p.clone(), plane=plane, padding=self.padding) # normalize to the range of (0, 1)
        xy = xy[:, :, None].float()
        vgrid = 2.0 * xy - 1.0 # normalize to (-1, 1)
        c = F.grid_sample(c, vgrid, padding_mode='border', align_corners=True, mode=self.sample_mode).squeeze(-1)
        return c

    def forward(self, p, c_plane, **kwargs):
        c = 0
        c += self.sample_plane_feature(p, c_plane['xz'], plane='xz')
        c += self.sample_plane_feature(p, c_plane['xy'], plane='xy')
        c += self.sample_plane_feature(p, c_plane['yz'], plane='yz')
        c = c.transpose(1, 2)

        p = p.float()
        x = self.fc_p(p)
        net = x

        for i in range(self.n_blocks):
            net = net + self.fc_c_occ[i](c)
            net = self.blocks_occ[i](net)

        results = {}
        if self.corr_dim != 0 and not self.corr_head:
            results['corr'] = net

        net = self.actvn(net)

        results['occ'] = self.fc_occ(net).squeeze(-1)
        if self.corr_dim != 0 and self.corr_head:
            results['corr'] = self.fc_out_corr(net)

        return results

class CombinedDecoder(nn.Module):
    ''' Decoder.
        Instead of conditioning on global features, on plane/volume local features.

    Args:
        dim (int): input dimension
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        n_blocks (int): number of blocks ResNetBlockFC layers
        leaky (bool): whether to use leaky ReLUs
        sample_mode (str): sampling feature strategy, bilinear|nearest
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
    '''

    def __init__(self, dim=3, c_per_dim=128, c_act_dim=128,
                 corr_dim=0, corr_head=True, 
                 hidden_size=256, n_blocks=5, leaky=False, 
                 sample_mode='bilinear', padding=0.1, fuse=True, detach=False, anneal_gradient=True):
        super().__init__()
        self.c_per_dim = c_per_dim
        self.c_act_dim = c_act_dim
        self.n_blocks = n_blocks
        self.corr_dim = corr_dim
        self.corr_head = corr_head
        self.fuse = fuse
        self.detach = detach
        self.anneal_gradient = anneal_gradient

        self.fc_c_per = nn.ModuleList([
            nn.Linear(c_per_dim, hidden_size) for i in range(n_blocks)
        ])

        self.fc_c_act = nn.ModuleList([
            nn.Linear(c_act_dim, hidden_size) for i in range(n_blocks)
        ])            

        if self.fuse:
            self.fc_c_merge = nn.ModuleList([
                nn.Linear(hidden_size*2, hidden_size) for i in range(n_blocks)
            ])

        self.fc_p_per = nn.Linear(dim, hidden_size)
        self.fc_p_act = nn.Linear(dim, hidden_size)

        self.blocks_per = nn.ModuleList([
            ResnetBlockFC(hidden_size) for i in range(n_blocks)
        ])
        self.blocks_act = nn.ModuleList([
            ResnetBlockFC(hidden_size) for i in range(n_blocks)
        ])

        self.fc_occ = nn.Linear(hidden_size, 1)
        self.fc_flow= nn.Linear(hidden_size, 3)

        if self.corr_dim != 0 and corr_head:
            self.fc_out_corr = nn.Linear(hidden_size, corr_dim)
            if self.fuse:
                self.fc_act_corr_merge = nn.Linear(hidden_size+corr_dim, hidden_size)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        self.sample_mode = sample_mode
        self.padding = padding
    

    def sample_plane_feature(self, p, c, plane='xz'):
        xy = normalize_coordinate(p.clone(), plane=plane, padding=self.padding) # normalize to the range of (0, 1)
        xy = xy[:, :, None].float()
        vgrid = 2.0 * xy - 1.0 # normalize to (-1, 1)
        c = F.grid_sample(c, vgrid, padding_mode='border', align_corners=True, mode=self.sample_mode).squeeze(-1)
        return c

    def decode_perception(self, p, c_per_plane):
        c_per = 0
        c_per += self.sample_plane_feature(p, c_per_plane['xz'], plane='xz')
        c_per += self.sample_plane_feature(p, c_per_plane['xy'], plane='xy')
        c_per += self.sample_plane_feature(p, c_per_plane['yz'], plane='yz')
        c_per = c_per.transpose(1, 2)

        p = p.float()
        net_per = self.fc_p_per(p)
        features = []
        for i in range(self.n_blocks):
            net_per = net_per + self.fc_c_per[i](c_per)
            net_per = self.blocks_per[i](net_per)
            if self.detach:
                features.append(net_per.detach())
            else:
                features.append(net_per)
        net_per = self.actvn(net_per)

        results = {}
        results['occ'] = self.fc_occ(net_per).squeeze(-1)
        if self.corr_dim != 0 and self.corr_head:
            corr = self.fc_out_corr(net_per)
            features.append(corr)
            results['corr'] = corr
        # if self.anneal_gradient:
        #     for i,p in enumerate(features):
        #         features[i] = p * 0.1 + p.detach() * 0.9
        return results, features

    def decode_action(self, p, c_act_plane, per_features):
        c_act = 0
        c_act += self.sample_plane_feature(p, c_act_plane['xz'], plane='xz')
        c_act += self.sample_plane_feature(p, c_act_plane['xy'], plane='xy')
        c_act += self.sample_plane_feature(p, c_act_plane['yz'], plane='yz')
        c_act = c_act.transpose(1, 2)

        p = p.float()
        net_act = self.fc_p_act(p)

        for i in range(self.n_blocks):
            net_act = net_act + self.fc_c_act[i](c_act) 
            if self.fuse:
                net_act = self.blocks_act[i](
                    self.fc_c_merge[i](
                        torch.cat( ( net_act, per_features[i]), dim=-1)))
                        # (net_per.detach()*0.9+net_per * 0.1)), dim=-1)))
            else:
                net_act = self.blocks_act[i](net_act)


        net_act = self.actvn(net_act)

        if self.corr_dim != 0 and self.corr_head:
            if self.fuse:
                net_act = self.fc_act_corr_merge(
                    torch.cat((net_act, per_features[-1].detach()), dim=-1))
        return {'flow':self.fc_flow(net_act)}

    def forward(self, p, c_per_plane, c_act_plane):
        results, per_features = self.decode_perception(p, c_per_plane)
        results['flow'] = self.decode_action(p, c_act_plane, per_features)['flow']
        return results
