import torch
import torch.nn as nn
import torch.nn.functional as F
from src.layers import ResnetBlockFC
from torch_scatter import scatter_mean, scatter_max
from src.common import coordinate2index, normalize_coordinate
from src.encoder.unet import UNet

class GeomEncoder(nn.Module):
    ''' PointNet-based encoder network with ResNet blocks for each point.
        Number of input points are fixed.
    
    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
        scatter_type (str): feature aggregation when doing local pooling
        unet (bool): weather to use U-Net
        unet_kwargs (str): U-Net parameters
        unet3d (bool): weather to use 3D U-Net
        unet3d_kwargs (str): 3D U-Net parameters
        plane_resolution (int): defined resolution for plane feature
        grid_resolution (int): defined resolution for grid feature 
        plane_type (str): feature type, 'xz' - 1-plane, ['xz', 'xy', 'yz'] - 3-plane, ['grid'] - 3D grid volume
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
        n_blocks (int): number of blocks ResNetBlockFC layers
    '''

    def __init__(self, c_dim=128, dim=3, f_dim=9, hidden_dim=128, scatter_type='max', 
                 unet_kwargs=None, plane_resolution=None,  padding=0.1, n_blocks=5):
        super().__init__()
        self.c_dim = c_dim

        self.fc_pos = nn.Linear(dim+f_dim, 2*hidden_dim)
        self.blocks = nn.ModuleList([
            ResnetBlockFC(2*hidden_dim, hidden_dim) for i in range(n_blocks)
        ])
        self.fc_c = nn.Linear(hidden_dim, c_dim)

        self.actvn = nn.ReLU()
        self.hidden_dim = hidden_dim

        self.unet = UNet(c_dim, in_channels=c_dim, **unet_kwargs)

        self.reso_plane = plane_resolution
        self.padding = padding

        if scatter_type == 'max':
            self.scatter = scatter_max
        elif scatter_type == 'mean':
            self.scatter = scatter_mean
        else:
            raise ValueError('incorrect scatter type')


    def generate_plane_features(self, p, c, plane='xz'):
        # acquire indices of features in plane
        xy = normalize_coordinate(p.clone(), plane=plane, padding=self.padding) # normalize to the range of (0, 1)
        index = coordinate2index(xy, self.reso_plane)

        # scatter plane features from points
        fea_plane = c.new_zeros(p.size(0), self.c_dim, self.reso_plane**2)
        c = c.permute(0, 2, 1) # B x 512 x T
        fea_plane = scatter_mean(c, index, out=fea_plane) # B x 512 x reso^2
        fea_plane = fea_plane.reshape(p.size(0), self.c_dim, self.reso_plane, self.reso_plane) # sparce matrix (B x 512 x reso x reso)

        # process the plane features with UNet
        fea_plane = self.unet(fea_plane)

        return fea_plane

    def pool_local(self, xy, index, c):
        bs, fea_dim = c.size(0), c.size(2)
        keys = xy.keys()

        c_out = 0
        for key in keys:
            # scatter plane features from points
            fea = self.scatter(c.permute(0, 2, 1), index[key], dim_size=self.reso_plane**2)
            if self.scatter == scatter_max:
                fea = fea[0]
            # gather feature back to points
            fea = fea.gather(dim=2, index=index[key].expand(-1, fea_dim, -1))
            c_out += fea
        return c_out.permute(0, 2, 1)


    def forward(self, p):
        if type(p) is tuple:
            p, pf = p
        else:
            pf = None
        # acquire the index for each point
        coord = {}
        index = {}
        coord['xz'] = normalize_coordinate(p.clone(), plane='xz', padding=self.padding)
        index['xz'] = coordinate2index(coord['xz'], self.reso_plane)
        coord['xy'] = normalize_coordinate(p.clone(), plane='xy', padding=self.padding)
        index['xy'] = coordinate2index(coord['xy'], self.reso_plane)
        coord['yz'] = normalize_coordinate(p.clone(), plane='yz', padding=self.padding)
        index['yz'] = coordinate2index(coord['yz'], self.reso_plane)

        net = self.fc_pos(torch.cat([p, pf],dim=-1))

        net = self.blocks[0](net)
        for block in self.blocks[1:]:
            pooled = self.pool_local(coord, index, net)
            net = torch.cat([net, pooled], dim=2)
            net = block(net)

        c = self.fc_c(net)

        fea = {}
        fea['xz'] = self.generate_plane_features(p, c, plane='xz')
        fea['xy'] = self.generate_plane_features(p, c, plane='xy')
        fea['yz'] = self.generate_plane_features(p, c, plane='yz')

        return fea

