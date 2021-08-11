# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
from torch import nn
import torch.nn.functional as F


class project_ptcld(nn.Module):

    def __init__(self, is_one_way=True):
        super().__init__()
        self.coord = None

    def forward(self, global_p1, R_1_T, t_1, K):

        B, H, W, _, _ = global_p1.shape
        if self.coord is None:
            yy, xx = torch.meshgrid(torch.arange(H).float(), torch.arange(W).float())
            self.coord = torch.ones([1, H, W, 1, 2])
            self.coord[0, ..., 0, 0] = xx
            self.coord[0, ..., 0, 1] = yy

        coord = self.coord.expand([B, H, W, 1, 2])

        p1_camera_1 = torch.matmul(global_p1 - t_1, R_1_T)

        p1_image_1 = torch.matmul(p1_camera_1, K)
        coord_image_sf = (p1_image_1 / (p1_image_1[..., -1:] + 1e-8))[..., : -1]
        displace_field = coord_image_sf - coord
        sf_proj = displace_field.squeeze()
        return sf_proj


# class unproject_ptcld()
class unproject_ptcld(nn.Module):
    # tested
    def __init__(self, is_one_way=True):
        super().__init__()
        self.coord = None

    def forward(self, depth_1, R_1, t_1, K_inv):
        B, _, H, W = depth_1.shape
        if self.coord is None:
            yy, xx = torch.meshgrid(torch.arange(H).float(), torch.arange(W).float())
            self.coord = torch.ones([1, H, W, 1, 3])
            self.coord[0, ..., 0, 0] = xx
            self.coord[0, ..., 0, 1] = yy
            self.coord = self.coord.to(depth_1.device)

        depth_1 = depth_1.view([B, H, W, 1, 1])
        p1_camera_1 = depth_1 * torch.matmul(self.coord, K_inv)
        global_p1 = torch.matmul(p1_camera_1, R_1) + t_1

        return global_p1


class unproject_ptcld_single(nn.Module):
    # tested
    def __init__(self, is_one_way=True):
        super().__init__()
        self.coord = None

    def forward(self, depth, pose, K):
        B, _, H, W = depth.shape
        assert B == 1
        if self.coord is None:
            yy, xx = torch.meshgrid(torch.arange(H).float(), torch.arange(W).float())
            self.coord = torch.ones([H, W, 3])
            self.coord[..., 0] = xx
            self.coord[..., 1] = yy
            self.coord = self.coord.to(depth.device)

        depth = depth.view([H, W, 1])
        p1_camera_1 = depth * torch.matmul(self.coord, torch.inverse(K).transpose(0, 1))
        R = pose[: 3, : 3].T
        t = pose[: 3, 3: 4].T
        global_p1 = torch.matmul(p1_camera_1, R) + t

        return global_p1


class flow_by_depth(nn.Module):
    # tested
    def __init__(self, is_one_way=True):
        super().__init__()
        self.coord = None
        self.sample_grid = None
        self.one_way = is_one_way

    def backward_warp(self, depth_2, flow_1_2):
        # flow[...,0]: dh
        # flow[...,0]: dw
        B, _, H, W = depth_2.shape
        coord = self.coord[..., :2].view(1, H, W, 2).expand([B, H, W, 2])
        sample_grids = coord + flow_1_2
        sample_grids[..., 0] /= (W - 1) / 2
        sample_grids[..., 1] /= (H - 1) / 2
        sample_grids -= 1
        return F.grid_sample(depth_2, sample_grids, align_corners=True, padding_mode='border')

    def forward(self, depth_1, depth_2, flow_1_2, R_1, R_2, R_1_T, R_2_T, t_1, t_2, K, K_inv):
        B, _, H, W = depth_1.shape
        if self.coord is None:
            yy, xx = torch.meshgrid(torch.arange(H).float(), torch.arange(W).float())
            self.coord = torch.ones([1, H, W, 1, 3])
            self.coord[0, ..., 0, 0] = xx
            self.coord[0, ..., 0, 1] = yy
            self.coord = self.coord.to(depth_1.device)

        coord = self.coord.expand([B, H, W, 1, 3])
        depth_1 = depth_1.view([B, H, W, 1, 1])
        depth_2 = depth_2.view([B, H, W, 1, 1])

        p1_camera_1 = depth_1 * torch.matmul(self.coord, K_inv)
        p2_camera_2 = depth_2 * torch.matmul(self.coord, K_inv)

        global_p1 = torch.matmul(p1_camera_1, R_1) + t_1
        global_p2 = torch.matmul(p2_camera_2, R_2) + t_2  # BHW13

        global_p2 = global_p2.squeeze(3).permute([0, 3, 1, 2])  # B3HW
        warped_global_p2 = self.backward_warp(global_p2, flow_1_2)
        warped_global_p2 = warped_global_p2.permute([0, 2, 3, 1])[..., None, :]  # BHW13
        sf_by_depth = warped_global_p2 - global_p1

        p1_camera_2 = torch.matmul(global_p1 - t_2, R_2_T)

        p1_image_2 = torch.matmul(p1_camera_2, K)

        coord_image_2 = (p1_image_2 / (p1_image_2[..., -1:] + 1e-8))[..., :-1]

        idB, idH, idW, idC, idF = torch.where(p1_image_2[..., -1:] < 1e-3)
        tr_coord = coord[..., :-1]
        coord_image_2[idB, idH, idW, idC, idF] = tr_coord[idB, idH, idW, idC, idF]
        coord_image_2[idB, idH, idW, idC, idF + 1] = tr_coord[idB, idH, idW, idC, idF + 1]

        depth_flow_1_2 = (coord_image_2 - coord[..., :-1])[..., 0, :]  # p_{1 -> 2}

        # warp by flow

        return {'dflow_1_2': depth_flow_1_2, 'sf_by_depth': sf_by_depth, 'warped_global_p2': warped_global_p2, 'global_p1': global_p1}


def calc_rigidity_loss(global_p1, sf, depth_1, s=1):
    mp = torch.nn.MaxPool2d(3, stride=1, padding=1, dilation=1)
    p_u = global_p1[:, None, 0:-2, 1:-1, 0, :]
    p_d = global_p1[:, None, 2:, 1:-1, 0, :]
    p_l = global_p1[:, None, 1:-1, 0:-2, 0, :]
    p_r = global_p1[:, None, 1:-1, 2:, 0, :]
    p_c = global_p1[:, None, 1:-1, 1:-1, 0, :]
    p_concat = torch.cat([p_u, p_d, p_c, p_l, p_r], axis=1)
    d_u = depth_1[:, :, 0:-2, 1:-1]
    d_d = depth_1[:, :, 2:, 1:-1]
    d_l = depth_1[:, :, 1:-1, 0:-2]
    d_r = depth_1[:, :, 1:-1, 2:]
    d_c = depth_1[:, :, 1:-1, 1:-1]
    d_concat = torch.cat([d_u, d_d, d_c, d_l, d_r], axis=1)
    s_u = sf[:, None, 0:-2, 1:-1, 0, :]
    s_d = sf[:, None, 2:, 1:-1, 0, :]
    s_l = sf[:, None, 1:-1, 0:-2, 0, :]
    s_r = sf[:, None, 1:-1, 2:, 0, :]
    s_c = sf[:, None, 1:-1, 1:-1, 0, :]
    s_concat = torch.cat([s_u, s_d, s_c, s_l, s_r], axis=1)

    prev_u = p_concat[:, 0, ...] - p_concat[:, 2, ...]
    prev_d = p_concat[:, 1, ...] - p_concat[:, 2, ...]
    prev_l = p_concat[:, 3, ...] - p_concat[:, 2, ...]
    prev_r = p_concat[:, 4, ...] - p_concat[:, 2, ...]
    after_u = s_concat[:, 0, ...] - s_concat[:, 2, ...]
    after_d = s_concat[:, 1, ...] - s_concat[:, 2, ...]
    after_l = s_concat[:, 3, ...] - s_concat[:, 2, ...]
    after_r = s_concat[:, 4, ...] - s_concat[:, 2, ...]
    gradd_u = d_concat[:, 0, ...] - d_concat[:, 2, ...]
    gradd_d = d_concat[:, 1, ...] - d_concat[:, 2, ...]
    gradd_l = d_concat[:, 3, ...] - d_concat[:, 2, ...]
    gradd_r = d_concat[:, 4, ...] - d_concat[:, 2, ...]

    lu = torch.abs(torch.norm(prev_u, dim=-1) - torch.norm(after_u, dim=-1))
    ld = torch.abs(torch.norm(prev_d, dim=-1) - torch.norm(after_d, dim=-1))
    lr = torch.abs(torch.norm(prev_r, dim=-1) - torch.norm(after_r, dim=-1))
    ll = torch.abs(torch.norm(prev_l, dim=-1) - torch.norm(after_l, dim=-1))

    weight_u = torch.exp(-s * mp(torch.abs(gradd_u)))
    weight_d = torch.exp(-s * mp(torch.abs(gradd_d)))
    weight_l = torch.exp(-s * mp(torch.abs(gradd_l)))
    weight_r = torch.exp(-s * mp(torch.abs(gradd_r)))
    total_loss = weight_u * lu + weight_r * lr + weight_d * ld + weight_l * ll
    loss_items = {'lu': lu, 'lr': lr, 'ld': ld, 'll': ll, 'weight_u': weight_u, 'weight_d': weight_d, 'weight_r': weight_r, 'weight_l': weight_l}
    return total_loss, loss_items


class scene_flow_projection_slack(nn.Module):
    # tested
    def __init__(self, is_one_way=False):
        super().__init__()
        self.coord = None
        self.sample_grid = None
        self.is_one_way = is_one_way

    def backward_warp(self, depth_2, flow_1_2):

        B, _, H, W = depth_2.shape
        coord = self.coord[..., :2].view(1, H, W, 2).expand([B, H, W, 2])
        sample_grids = coord + flow_1_2
        sample_grids[..., 0] /= (W - 1) / 2
        sample_grids[..., 1] /= (H - 1) / 2
        sample_grids -= 1
        return F.grid_sample(depth_2, sample_grids, align_corners=True, padding_mode='border')

    def forward(self, depth_1, depth_2, flow_1_2, flow_2_1, R_1, R_2, R_1_T, R_2_T, t_1, t_2, K, K_inv, sflow_1_2, sflow_2_1):
        B, _, H, W = depth_1.shape
        if self.coord is None:
            yy, xx = torch.meshgrid(torch.arange(H).float(), torch.arange(W).float())
            self.coord = torch.ones([1, H, W, 1, 3])
            self.coord[0, ..., 0, 0] = xx
            self.coord[0, ..., 0, 1] = yy
            self.coord = self.coord.to(depth_1.device)

        coord = self.coord.expand([B, H, W, 1, 3])
        depth_1 = depth_1.view([B, H, W, 1, 1])
        depth_2 = depth_2.view([B, H, W, 1, 1])

        p1_camera_1 = depth_1 * torch.matmul(self.coord, K_inv)
        p2_camera_2 = depth_2 * torch.matmul(self.coord, K_inv)
        global_p1 = torch.matmul(p1_camera_1, R_1) + t_1
        global_p2 = torch.matmul(p2_camera_2, R_2) + t_2

        p2_camera_2_w = p2_camera_2.squeeze(3).permute([0, 3, 1, 2])  # B3HW
        warped_p2_camera_2 = self.backward_warp(p2_camera_2_w, flow_1_2)
        warped_p2_camera_2 = warped_p2_camera_2.permute([0, 2, 3, 1])[..., None, :]  # BHW13

        p1_camera_2 = torch.matmul(global_p1 + sflow_1_2 - t_2, R_2_T)
        p1_camera_2_static = torch.matmul(global_p1 - t_2, R_2_T)
        p2_camera_1 = torch.matmul(global_p2 + sflow_2_1 - t_1, R_1_T)
        p1_image_2 = torch.matmul(p1_camera_2, K)
        p2_image_1 = torch.matmul(p2_camera_1, K)
        p1_image_2_static = torch.matmul(p1_camera_2_static, K)
        coord_image_2_static = (p1_image_2_static / (p1_image_2_static[..., -1:] + 1e-8))[..., : -1]
        coord_image_2 = (p1_image_2 / (p1_image_2[..., -1:] + 1e-8))[..., : -1]
        coord_image_1 = (p2_image_1 / (p2_image_1[..., -1:] + 1e-8))[..., : -1]
        idB, idH, idW, idC, idF = torch.where(p1_image_2[..., -1:] < 1e-3)
        tr_coord = coord[..., :-1]
        coord_image_2[idB, idH, idW, idC, idF] = tr_coord[idB, idH, idW, idC, idF]
        coord_image_2[idB, idH, idW, idC, idF + 1] = tr_coord[idB, idH, idW, idC, idF + 1]
        idB, idH, idW, idC, idF = torch.where(p2_image_1[..., -1:] < 1e-3)
        coord_image_1[idB, idH, idW, idC, idF] = tr_coord[idB, idH, idW, idC, idF]
        coord_image_1[idB, idH, idW, idC, idF + 1] = tr_coord[idB, idH, idW, idC, idF + 1]

        idB, idH, idW, idC, idF = torch.where(p1_image_2_static[..., -1:] < 1e-3)
        coord_image_2_static[idB, idH, idW, idC, idF] = tr_coord[idB, idH, idW, idC, idF]
        coord_image_2_static[idB, idH, idW, idC, idF + 1] = tr_coord[idB, idH, idW, idC, idF + 1]

        depth_flow_1_2 = (coord_image_2 - coord[..., :-1])[..., 0, :]  # p_{1 -> 2}

        depth_flow_1_2_static = (coord_image_2_static - coord[..., :-1])[..., 0, :]
        depth_image_1_2 = p1_image_2[..., -1].permute(0, 3, 1, 2)  # z_{1 -> 2}

        # forward warping depth
        depth_1 = depth_1.view(B, 1, H, W)
        depth_2 = depth_2.view(B, 1, H, W)

        depth_warp_1_2 = self.backward_warp(depth_2, flow_1_2)

        depth_warp_1_2 = depth_warp_1_2.view([B, 1, H, W])

        return {'dflow_1_2': depth_flow_1_2, 'depth_image_1_2': depth_image_1_2, 'depth_warp_1_2': depth_warp_1_2, 'depth_1': depth_1, 'depth_2': depth_2, 'scenef_1_2': sflow_1_2, 'global_p1': global_p1, 'staticflow_1_2': depth_flow_1_2_static, 'p1_camera_2': p1_camera_2, 'warped_p2_camera_2': warped_p2_camera_2}


class BackwardWarp(nn.Module):

    def __init__(self, is_one_way=False):
        super().__init__()
        self.coord = None
        self.sample_grid = None
        self.is_one_way = is_one_way

    def backward_warp(self, buffer, flow_1_2):

        B, _, H, W = buffer.shape
        coord = self.coord[..., :2].view(1, H, W, 2).expand([B, H, W, 2])
        sample_grids = coord + flow_1_2
        sample_grids[..., 0] /= (W - 1) / 2
        sample_grids[..., 1] /= (H - 1) / 2
        sample_grids -= 1
        return F.grid_sample(buffer, sample_grids, align_corners=True, padding_mode='border')

    def forward(self, buffer, flow_1_2):
        B, _, H, W = buffer.shape
        if self.coord is None:
            yy, xx = torch.meshgrid(torch.arange(H).float(), torch.arange(W).float())
            self.coord = torch.ones([1, H, W, 1, 3])
            self.coord[0, ..., 0, 0] = xx
            self.coord[0, ..., 0, 1] = yy
            self.coord = self.coord.to(buffer.device)
        return self.backward_warp(buffer, flow_1_2)
