from torch import nn
import torch.nn.functional as F
import torch
from modules.util import Hourglass, make_coordinate_grid, kp2gaussian

from sync_batchnorm import SynchronizedBatchNorm3d as BatchNorm3d


class DenseMotionNetwork(nn.Module):
    """
    Module that predicting a dense motion from sparse motion representation given by kp_source and kp_driving
    """

    def __init__(self, block_expansion, num_blocks, max_features, num_kp, feature_channel, reshape_depth, compress, num_source,
                 estimate_occlusion_map=False):
        super(DenseMotionNetwork, self).__init__()
        
        # self.hourglass = Hourglass(block_expansion=block_expansion, in_features=(num_kp+1)*(compress+1), max_features=max_features, num_blocks=num_blocks)
        self.hourglass = Hourglass(block_expansion=block_expansion, in_features=(num_kp*num_source+1)*(compress+1), max_features=max_features, num_blocks=num_blocks)
        

        # self.mask = nn.Conv3d(self.hourglass.out_filters, num_kp + 1, kernel_size=7, padding=3)
        self.mask = nn.Conv3d(self.hourglass.out_filters, num_kp*num_source + 1, kernel_size=7, padding=3)

        self.compress = nn.Conv3d(feature_channel, compress, kernel_size=1)
        self.norm = BatchNorm3d(compress, affine=True)

        if estimate_occlusion_map:
            # self.occlusion = nn.Conv2d(reshape_channel*reshape_depth, 1, kernel_size=7, padding=3)
            self.occlusion = nn.Conv2d(self.hourglass.out_filters*reshape_depth, 1, kernel_size=7, padding=3)
        else:
            self.occlusion = None

        self.num_kp = num_kp
        self.num_source = num_source


    def create_sparse_motions(self, feature, kp_driving, kp_source):
        if self.num_source == 1:
            bs, c, d, h, w = feature.shape
            identity_grid = make_coordinate_grid((d, h, w), type=kp_source['value'].type())
        else:
            bs, _, c, d, h, w = feature.shape
            identity_grid = make_coordinate_grid((d, h, w), type=kp_source[0]['value'].type())
        identity_grid = identity_grid.view(1, 1, d, h, w, 3)
        coordinate_grid = identity_grid - kp_driving['value'].view(bs, self.num_kp, 1, 1, 1, 3)

        k = coordinate_grid.shape[1]
        
        # if 'jacobian' in kp_driving:
        if 'jacobian' in kp_driving and kp_driving['jacobian'] is not None:
            jacobian = torch.matmul(kp_source['jacobian'], torch.inverse(kp_driving['jacobian']))
            jacobian = jacobian.unsqueeze(-3).unsqueeze(-3).unsqueeze(-3)
            jacobian = jacobian.repeat(1, 1, d, h, w, 1, 1)
            coordinate_grid = torch.matmul(jacobian, coordinate_grid.unsqueeze(-1))
            coordinate_grid = coordinate_grid.squeeze(-1)
        '''
        if 'rot' in kp_driving:
            rot_s = kp_source['rot']
            rot_d = kp_driving['rot']
            rot = torch.einsum('bij, bjk->bki', rot_s, torch.inverse(rot_d))
            rot = rot.unsqueeze(-3).unsqueeze(-3).unsqueeze(-3).unsqueeze(-3)
            rot = rot.repeat(1, k, d, h, w, 1, 1)
            # print(rot.shape)
            coordinate_grid = torch.matmul(rot, coordinate_grid.unsqueeze(-1))
            coordinate_grid = coordinate_grid.squeeze(-1)
            # print(coordinate_grid.shape)
        '''
        identity_grid = identity_grid.repeat(bs, 1, 1, 1, 1, 1)

        if self.num_source == 1:
            driving_to_source = coordinate_grid + kp_source['value'].view(bs, self.num_kp, 1, 1, 1, 3)    # (bs, num_kp, d, h, w, 3)
        else:
            driving_to_source = []
            for i in range(self.num_source):
                driving_to_source.append(coordinate_grid + kp_source[i]['value'].view(bs, self.num_kp, 1, 1, 1, 3))
            driving_to_source = torch.cat(driving_to_source, dim=1)
        #adding background feature
        sparse_motions = torch.cat([identity_grid, driving_to_source], dim=1)
        
        # sparse_motions = driving_to_source

        return sparse_motions

    def create_deformed_feature(self, feature, sparse_motions):
        if self.num_source == 1:
            bs, c, d, h, w = feature.shape
            feature_repeat = feature.unsqueeze(1).unsqueeze(1).repeat(1, self.num_kp*self.num_source+1, 1, 1, 1, 1, 1) # (bs, num_kp+1, 1, c, d, h, w)
        else:
            bs, _, c, d, h, w = feature.shape
            feature_repeat = []
            for i in range(self.num_source):
                kp = self.num_kp+1 if i == 0 else self.num_kp
                feature_repeat.append(feature[:,i].unsqueeze(1).unsqueeze(1).repeat(1, kp, 1, 1, 1, 1, 1))
            feature_repeat = torch.cat(feature_repeat, dim=1) # (bs, num_kp*num_source+1, 1, c, d, h, w)
        
        feature_repeat = feature_repeat.view(bs * (self.num_kp*self.num_source+1), -1, d, h, w)         # (bs*(num_kp+1), c, d, h, w)
        sparse_motions = sparse_motions.view((bs * (self.num_kp*self.num_source+1), d, h, w, -1))       # (bs*(num_kp+1), d, h, w, 3)
        sparse_deformed = F.grid_sample(feature_repeat, sparse_motions)
        sparse_deformed = sparse_deformed.view((bs, self.num_kp*self.num_source+1, -1, d, h, w))        # (bs, num_kp+1, c, d, h, w)
        return sparse_deformed

    def create_heatmap_representations(self, feature, kp_driving, kp_source):
        spatial_size = feature.shape[3:]
        gaussian_driving = kp2gaussian(kp_driving, spatial_size=spatial_size, kp_variance=0.01)
        if self.num_source == 1:
            gaussian_source = kp2gaussian(kp_source, spatial_size=spatial_size, kp_variance=0.01)
            heatmap = gaussian_driving - gaussian_source
        else:
            heatmap = []
            for i in range(self.num_source):
                gaussian_source = kp2gaussian(kp_source[i], spatial_size=spatial_size, kp_variance=0.01)
                heatmap.append(gaussian_driving - gaussian_source)
            heatmap = torch.cat(heatmap, dim=1)

        # adding background feature
        zeros = torch.zeros(heatmap.shape[0], 1, spatial_size[0], spatial_size[1], spatial_size[2]).type(heatmap.type())
        heatmap = torch.cat([zeros, heatmap], dim=1)
        heatmap = heatmap.unsqueeze(2)         # (bs, num_kp+1, 1, d, h, w)
        return heatmap

    def forward(self, feature, kp_driving, kp_source):
        bs, _, _, _, _ = feature.shape
        feature = self.compress(feature)
        feature = self.norm(feature)
        feature = F.relu(feature)
        _, c, d, h, w = feature.shape

        if self.num_source != 1:
            feature = feature.view(-1, self.num_source, c, d, h, w)
            bs, _, _, d, h, w = feature.shape

        out_dict = dict()
        sparse_motion = self.create_sparse_motions(feature, kp_driving, kp_source) # (bs, num_kp+1, d, h, w, 3)
        deformed_feature = self.create_deformed_feature(feature, sparse_motion) # (bs, num_kp+1, c, d, h, w)

        heatmap = self.create_heatmap_representations(deformed_feature, kp_driving, kp_source) # (bs, num_kp+1, 1, d, h, w)

        input = torch.cat([heatmap, deformed_feature], dim=2)
        input = input.view(bs, -1, d, h, w) # (bs, num_kp+1 * c+1, d, h, w)

        prediction = self.hourglass(input)

        mask = self.mask(prediction)                               # (bs, num_kp+1, d, h, w)
        mask = F.softmax(mask, dim=1)                              
        out_dict['mask'] = mask
        mask = mask.unsqueeze(2)                                   # (bs, num_kp+1, 1, d, h, w)
        sparse_motion = sparse_motion.permute(0, 1, 5, 2, 3, 4)    # (bs, num_kp+1, 3, d, h, w)
        deformation = (sparse_motion * mask).sum(dim=1)            # (bs, 3, d, h, w)
        deformation = deformation.permute(0, 2, 3, 4, 1)           # (bs, d, h, w, 3)

        out_dict['deformation'] = deformation

        if self.occlusion:
            bs, c, d, h, w = prediction.shape
            prediction = prediction.view(bs, -1, h, w)                # (bs, num_kp+1 * d, h, w)
            occlusion_map = torch.sigmoid(self.occlusion(prediction)) # (bs, 1, h, w)
            out_dict['occlusion_map'] = occlusion_map

        return out_dict
