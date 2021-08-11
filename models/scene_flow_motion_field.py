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
from models.video_base import VideoBaseModel
from third_party.hourglass import HourglassModel_Embed
from third_party.MiDaS import MidasNet
from visualize.html_visualizer import HTMLVisualizer as Visualizer
import torch.nn.functional as F
import inspect
from functools import partial
from configs import depth_pretrain_path, midas_pretrain_path
from losses.scene_flow_projection import scene_flow_projection_slack, flow_by_depth, BackwardWarp, unproject_ptcld
from networks.FCNUnet import FCNUnet
from networks.sceneflow_field import SceneFlowFieldNet
import numpy as np
from os.path import join
from os import makedirs


class Model(VideoBaseModel):
    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument('--l1_mul', type=float, default=1e-4, help='L1 multiplier')
        parser.add_argument('--disp_mul', type=float, default=10, help='disparity multiplier')
        parser.add_argument('--one_way', action='store_true', help='use only losses on 1 to 2')
        parser.add_argument('--loss_type', type=str, default='l2', help='use l2 on sceneflow')
        parser.add_argument('--scene_lr_mul', type=float, default=1, help='lr multiplier for scene flow network')
        parser.add_argument('--n_down', type=int, default=3, help='sf net size')
        parser.add_argument('--weight_steps', action='store_true', help='weight steps by baselines')
        parser.add_argument('--sf_min_mul', type=float, default=0, help='minimize sf')
        parser.add_argument('--sf_quantile', type=float, default=0.5, help='minimize sf for 50% pixels')
        parser.add_argument('--static', action='store_true', help='optimize static regions with skip frames')
        parser.add_argument('--static_mul', type=float, default=1, help='multiplier for static large baseline losses')
        parser.add_argument('--flow_mul', type=float, default=10, help='multiplier for flow losses')
        parser.add_argument('--acc_mul', type=float, default=100, help='multiplier for acceleration regularization losses')
        parser.add_argument('--si_mul', type=float, default=0, help='multiplier for scale invariant losses')
        parser.add_argument('--cos_mul', type=float, default=0, help='multiplier for cosine angle losses for optical flow')
        parser.add_argument('--motion_seg_hard', action='store_true', help='flag for using hard motion segmentations')
        parser.add_argument('--warm_mul', type=float, default=1, help='multiplier for warm up state training')
        parser.add_argument('--interp_steps', type=int, default=5, help='steps for interpolation')
        parser.add_argument('--warm_static', action='store_true', help='only use static loss for warm up')
        parser.add_argument('--use_disp', action='store_true', help='flag for using disp losses')
        parser.add_argument('--use_disp_ratio', action='store_true', help='use  disp ratio losses')
        parser.add_argument('--time_dependent', action='store_true', help='flag for time dependent scene flow model')
        parser.add_argument('--use_cnn', action='store_true', help='flag for using CNN for scene flow model')
        parser.add_argument('--use_embedding', action='store_true', help='flag for using optimizable embedding for each frame')
        parser.add_argument('--use_motion_seg', action='store_true', help='flag for using motion seg')
        parser.add_argument('--warm_reg', action='store_true', help='use reg for warm up as well')
        parser.add_argument('--warm_sf', type=int, default=0, help='warm up flow network for k epochs')
        parser.add_argument('--n_freq_xyz', type=int, default=16, help='xyz_embeddings')
        parser.add_argument('--n_freq_t', type=int, default=16, help='time embeddings')
        parser.add_argument('--sf_mag_div', type=float, default=100, help='divident for sceneflow network output, making it easier to optimize')
        parser.add_argument('--midas', action='store_true', help='use midas for depth prediction')

        return parser, set()

    def weighted_mean_loss(x, weights, eps=1e-6):
        assert x.ndimension() == weights.ndimension() and x.shape[0] == weights.shape[0]
        B = weights.shape[0]
        weights_sum = torch.sum(weights.view(B, -1), dim=-1).view(B, 1, 1, 1)
        weights_sum = torch.clamp(weights_sum, min=eps)
        weights_n = weights / weights_sum

        return torch.sum((weights_n * x).reshape(B, -1), dim=1)

    def __init__(self, opt, loggers):
        super().__init__(opt, loggers)
        self.input_names = ['img', 'img_1', 'img_2', 'pose', 'intrinsic', 'mask_1', 'mask_2', 'R_1', 'R_1_T',
                            'R_2', 'R_2_T', 't_1', 't_2', 'flow_1_2', 'flow_2_1', 'K', 'K_inv', 'motion_seg_1', 'time_stamp_1', 'time_stamp_2', 'frame_id_1', 'frame_id_2', 'time_step']
        self.gt_names = []
        self.requires = list(set().union(self.input_names, self.gt_names))
        if self.opt.midas:
            resize = None
            if 'real_video' in self.opt.dataset:
                resize = [224, 384]
            if 'korean' in self.opt.dataset:
                resize = [224, 384]
            if 'mctest' in self.opt.dataset:
                resize = [224, 384]
            if 'cube' in self.opt.dataset:
                resize = [224, 384]
            self.net_depth = MidasNet(path=midas_pretrain_path, non_negative=True, normalize_input=True, resize=resize)

        else:
            self.net_depth = HourglassModel_Embed(noexp=False, use_embedding=opt.use_embedding)

        assert hasattr(self._input, 'depth_1') is False
        assert hasattr(self._input, 'depth_2') is False

        conv_setup = {'norm': 'none', 'activation': 'lrelu', 'pad_type': 'reflect', 'stride': 1}
        if opt.use_cnn:
            in_channel = 4 if opt.time_dependent else 3
            self.net_sceneflow = FCNUnet(conv_setup, n_down=self.opt.n_down, feat=32, block_type='double_conv', in_channel=in_channel, out_channel=3)
        else:
            self.net_sceneflow = SceneFlowFieldNet(net_width=256, n_layers=4, time_dependent=opt.time_dependent, N_freq_xyz=opt.n_freq_xyz, N_freq_t=opt.n_freq_t)
        self.bkwarp = BackwardWarp()
        self.unproject_points = unproject_ptcld()

        self.global_rank = opt.global_rank
        self._nets = [self.net_depth, self.net_sceneflow]
        self.optimizer_depth = self.optim(self.net_depth.parameters(), lr=opt.lr, **self.optim_params)
        self.optimizer_scene = self.optim(self.net_sceneflow.parameters(), lr=opt.lr * opt.scene_lr_mul, **self.optim_params)
        self._optimizers = [self.optimizer_depth, self.optimizer_scene]
        self._metrics = ['flow_loss_1_2', 'loss', 'disp_loss_1_2', 'data_time', 'acc_reg', 'sf_loss', ]
        self.init_vars(add_path=False)
        if opt.midas:
            pass
        else:
            self.net_depth.net_depth.load_state_dict(torch.load(depth_pretrain_path))

        self.init_weight(self.net_sceneflow, 'kaiming', 0.01, a=0.2)
        self.visualizer = Visualizer(loggers.get_html_logger(), n_workers=4)
        self.L1_crit = partial(F.l1_loss, reduction='none')
        self.L2_crit = partial(F.mse_loss, reduction='none')
        self.warp = scene_flow_projection_slack()
        warp_arg_list = inspect.getargspec(self.warp.forward).args[1:]
        self.warp_args = []
        for x in warp_arg_list:
            if hasattr(self._input, x):
                self.warp_args.append(x)
        self.depth_flow = flow_by_depth()
        flow_arg_list = inspect.getargspec(self.depth_flow.forward).args[1:]
        self.flow_args = []
        for x in flow_arg_list:
            if hasattr(self._input, x):
                self.flow_args.append(x)

    def disp_loss(self, d1, d2):
        if self.opt.use_disp:
            t1 = torch.clamp(d1, min=1e-3)
            t2 = torch.clamp(d2, min=1e-3)
            return 100 * torch.abs((1 / t1) - (1 / t2))
        if self.opt.use_disp_ratio:
            t1 = torch.clamp(d1, min=1e-3)
            t2 = torch.clamp(d2, min=1e-3)
            return torch.max(t1, t2) / torch.min(t1, t2) - 1
        else:
            return torch.abs(d1 - d2)

    def _train_on_batch(self, epoch, batch_ind, batch):

        if epoch <= self.opt.warm_sf:
            # raise NotImplementedError
            if self.opt.midas:
                self.net_depth.eval()
                for param in self.net_depth.parameters():
                    param.requires_grad = False
            else:

                self.net_depth.freeze()    # freeze bn

            self.warm = True
        else:
            self.warm = False
            if self.opt.midas:
                self.net_depth.eval()
                for param in self.net_depth.parameters():
                    param.requires_grad = True
            else:
                self.net_depth.defrost()

        for n in self._nets:
            n.zero_grad()

        for k, v in batch.items():
            if type(v) != list:
                batch[k] = v.squeeze(0)

        self.load_batch(batch)

        pred = {}
        loss = 0
        loss_data = {}

        pred = self._predict_on_batch()
        loss, loss_data = self._calc_loss(pred)
        if self.opt.weight_steps:
            loss = loss * self.steps

        if self.opt.interp_steps > 0 and (not self.warm or self.opt.warm_reg) and self.opt.acc_mul > 0:
            loss.backward(retain_graph=True)
            reg_loss = self._opt_reg(pred, steps=self.opt.interp_steps)
            loss_data['acc_reg'] = reg_loss

        else:
            loss.backward()
            loss_data['acc_reg'] = 0

        for k, v in pred.items():
            pred[k] = v.data.cpu().numpy()

        pred_static = {}
        loss_data_static = {}
        for k, v in pred_static.items():
            pred[k + '_static'] = v.data.cpu().numpy()

        for k, v in loss_data_static.items():
            loss_data[k] = v

        for optimizer in self._optimizers:
            optimizer.step()

        if np.mod(epoch, self.opt.vis_every_train) == 0:
            indx = batch_ind if self.opt.vis_at_start else self.opt.epoch_batches - batch_ind
            if indx <= self.opt.vis_batches_train:

                outdir = join(self.full_logdir, 'visualize', 'epoch%04d_train' % epoch)
                makedirs(outdir, exist_ok=True)
                output = self.pack_output(pred, batch)
                if self.global_rank == 0:
                    if self.visualizer is not None:
                        self.visualizer.visualize(output, indx + (1000 * epoch), outdir)
                np.savez(join(outdir, 'rank%04d_batch%04d' % (self.global_rank, batch_ind)), **output)
        batch_log = {'size': self.opt.batch_size, 'loss': loss.item(), **loss_data}
        return batch_log

    def _predict_on_batch(self, is_train=True):

        if is_train:
            if self.opt.midas:
                depth_1 = self.net_depth(self._input.img_1)
                depth_2 = self.net_depth(self._input.img_2)

            else:
                depth_1 = self.net_depth(self._input.img_1, self._input.frame_id_1.long())
                depth_2 = self.net_depth(self._input.img_2, self._input.frame_id_2.long())

            flow_data_input = {'depth_1': depth_1, 'depth_2': depth_2}
            for k in self.flow_args:
                flow_data_input[k] = getattr(self._input, k)
            dflow = self.depth_flow(**flow_data_input)

            global_p1 = dflow['global_p1'].squeeze(3).permute(0, 3, 1, 2)  # .detach()  # B3HW

            time_step = self._input.time_step.squeeze().item()
            time_gap = torch.mean(self._input.time_stamp_2 - self._input.time_stamp_1)
            steps = (time_gap / time_step).round().long().item()
            self.steps = steps

            sf_1_2 = self.forward_sf_net_multi_step(global_p1, self._input.time_stamp_1, time_step=time_step, steps=steps)
            if self.opt.use_motion_seg:
                sf_1_2 *= self._input.motion_seg_1.squeeze(3).permute(0, 3, 1, 2)

            flow_data_input['sflow_1_2'] = sf_1_2.permute(0, 2, 3, 1)[..., None, :]  # .fill_(0)
            flow_data_input['sflow_2_1'] = sf_1_2.permute(0, 2, 3, 1)[..., None, :]
            flow_data_input['flow_1_2'] = self._input.flow_1_2
            flow_data_input['flow_2_1'] = self._input.flow_2_1
            result = self.warp(**flow_data_input)
            result['sf_1_2'] = sf_1_2
            result['global_p1'] = global_p1

            result['sf_by_dep_1_2'] = dflow['sf_by_depth']
        else:
            if self.opt.midas:
                depth = self.net_depth(self._input.img)
            else:
                depth = self.net_depth(self._input.img, self._input.frame_id_1.long())

            global_p1 = self.unproject_points(depth, self._input.R_1, self._input.t_1, self._input.K_inv)
            global_p1 = global_p1.squeeze(3).permute(0, 3, 1, 2)

            sf_1_2 = self.forward_sf_net_multi_step(global_p1, self._input.time_stamp_1, time_step=self._input.time_step, steps=1)
            result = {'depth': depth, 'sf_1_2': sf_1_2}
        return result

    def flow_cos_norm(self, flow_1, flow_2):
        flow_mag_1 = torch.norm(flow_1, dim=-1, keepdim=True)
        flow_mag_2 = torch.norm(flow_1, dim=-1, keepdim=True)
        flow_cos = torch.sum(flow_1 * flow_2, dim=-1, keepdim=True)
        flow_cos_norm = flow_cos / (flow_mag_1 * flow_mag_2 + 1e-8)
        return flow_cos_norm

    def _calc_loss(self, pred):
        mask = self._input.mask_2
        if self.opt.midas:
            mask = (pred['depth_1'] < 100).float().squeeze(1)[..., None, None] * mask
            mask = (pred['warped_p2_camera_2'][..., 2] < 100).float().squeeze(3)[..., None, None] * mask

        crit = self.L2_crit if self.warm else self.L1_crit
        scene_flow_loss_1_2 = crit(pred['dflow_1_2'], self._input.flow_1_2)
        flow_cos_norm = self.flow_cos_norm(pred['dflow_1_2'], self._input.flow_1_2)
        scene_flow_loss_cos = crit(flow_cos_norm, torch.ones_like(flow_cos_norm))

        occ_mask = mask[:, None, ..., 0, 0].permute([0, 2, 3, 1])
        scene_flow_loss_cos = torch.sum(occ_mask * scene_flow_loss_cos) / (torch.sum(occ_mask) + 1e-8)

        flow_loss_1_2 = torch.sum(occ_mask * scene_flow_loss_1_2.squeeze(3)) / (torch.sum(occ_mask) + 1e-8)

        disp_loss_1_2 = self.disp_loss(pred['p1_camera_2'][..., -1], pred['warped_p2_camera_2'][..., -1]).permute([0, 3, 1, 2])

        disp_loss_1_2 = torch.sum(occ_mask[:, None, ..., 0] * disp_loss_1_2[:, 0:1, ...]) / (torch.sum(occ_mask) + 1e-8)

        sf_loss_pp = torch.abs(pred['sf_by_dep_1_2'].squeeze(3).permute(0, 3, 1, 2) - pred['sf_1_2'])
        sf_loss = torch.sum(occ_mask[:, None, ..., 0] * sf_loss_pp[:, ...]) / (torch.sum(occ_mask) + 1e-8)

        pred['sf_loss_pp'] = sf_loss_pp.sum(1).detach()

        if not self.opt.use_disp:
            if self.warm:
                loss = flow_loss_1_2 * self.opt.flow_mul + sf_loss * self.opt.disp_mul
            else:
                loss = flow_loss_1_2 * self.opt.flow_mul + sf_loss * self.opt.disp_mul
        else:
            if self.warm:
                loss = flow_loss_1_2 * self.opt.flow_mul + disp_loss_1_2 * self.opt.disp_mul
            else:
                loss = flow_loss_1_2 * self.opt.flow_mul + disp_loss_1_2 * self.opt.disp_mul

        loss_data = {'total_loss': loss.item(), 'loss': loss.item(),
                     'flow_loss_1_2': flow_loss_1_2.item(),
                     'disp_loss_1_2': disp_loss_1_2.item(), 'sf_loss': sf_loss.item()}
        return loss, loss_data

    def _opt_reg(self, pred, steps=2):
        global_p1 = pred['global_p1']

        sf_1_2 = self.forward_sf_net(global_p1, self._input.time_stamp_1)

        mseg = torch.ones_like(sf_1_2)

        time_step = self._input.time_step.squeeze().item()

        time_stamp = (self._input.time_stamp_1 + time_step)
        global_p1_interp = (global_p1 + sf_1_2)

        sf_1_2_t1 = self.forward_sf_net(global_p1_interp, time_stamp)

        acc_loss = (mseg * torch.abs(sf_1_2_t1 - sf_1_2)).sum() / (mseg.sum() + 1e-6)

        loss = acc_loss * self.opt.acc_mul
        loss.backward()
        return loss.item()

    def forward_sf_net(self, global_p1, ts):
        if self.opt.use_cnn:
            if self.opt.time_dependent:
                sf_1_2 = self.net_sceneflow(torch.cat([global_p1, ts], 1))
            else:
                sf_1_2 = self.net_sceneflow(global_p1)
        elif self.opt.time_dependent:
            sf_1_2 = self.net_sceneflow(global_p1, ts)
        else:
            sf_1_2 = self.net_sceneflow(global_p1)

        sf_1_2 /= self.opt.sf_mag_div  # s1000
        return sf_1_2

    def forward_sf_net_multi_step(self, global_p1, time_stamp, time_step, steps):
        sf_acc = 0
        for i in range(steps):
            sf_1_2 = self.forward_sf_net(global_p1, time_stamp)
            sf_acc = sf_acc + sf_1_2
            global_p1 = global_p1 + sf_1_2
            time_stamp = time_stamp + time_step
        return sf_acc
