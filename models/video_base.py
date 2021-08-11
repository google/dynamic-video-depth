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

from os.path import join, dirname
import numpy as np
import torch
from models.netinterface import NetInterface
from os import makedirs
import matplotlib.pyplot as plt
import matplotlib.cm
from matplotlib.colors import ListedColormap
from third_party.util_colormap import turbo_colormap_data
# matplotlib.cm.register_cmap('turbo', cmap=ListedColormap(turbo_colormap_data))
import matplotlib
import shutil


class VideoBaseModel(NetInterface):
    def disp_loss(self, d1, d2):
        if self.opt.use_disp:
            t1 = torch.clamp(d1, min=1e-3)
            t2 = torch.clamp(d2, min=1e-3)
            return 300 * torch.abs((1 / t1) - (1 / t2))
        else:
            return torch.abs(d1 - d2)

    def _train_on_batch(self, epoch, batch_ind, batch):
        for n in self._nets:
            n.zero_grad()
        # self.net_depth.eval()  # freeze bn to check

        self.load_batch(batch)
        batch_size = batch['img_1'].shape[0]
        pred = self._predict_on_batch()
        loss, loss_data = self._calc_loss(pred)
        loss.backward()
        for optimizer in self._optimizers:
            optimizer.step()

        if np.mod(epoch, self.opt.vis_every_train) == 0:
            indx = batch_ind if self.opt.vis_at_start else self.opt.epoch_batches - batch_ind
            if indx <= self.opt.vis_batches_train:
                for k, v in pred.items():
                    pred[k] = v.data.cpu().numpy()
                outdir = join(self.full_logdir, 'visualize', 'epoch%04d_train' % epoch)
                makedirs(outdir, exist_ok=True)
                output = self.pack_output(pred, batch)
                if self.global_rank == 0:
                    if self.visualizer is not None:
                        self.visualizer.visualize(output, indx + (1000 * epoch), outdir)
                np.savez(join(outdir, 'rank%04d_batch%04d' % (self.global_rank, batch_ind)), **output)
        batch_log = {'size': batch_size, 'loss': loss.item(), **loss_data}
        return batch_log

    @staticmethod
    def depth2disp(depth):
        valid = depth > 1e-2
        valid = valid.float()
        return (1 / (depth + (1 - valid) * 1e-8)) * valid

    def disp_vali(self, d1, d2):
        vali = d2 > 1e-2
        return torch.nn.functional.mse_loss(self.depth2disp(d1) * vali, self.depth2disp(d2) * vali)

    def _vali_on_batch(self, epoch, batch_idx, batch):
        for n in self._nets:
            n.eval()
        self.load_batch(batch)
        with torch.no_grad():
            pred = self._predict_on_batch(is_train=False)
        gt_depth = batch['depth_mvs'].to(pred['depth'].device)
        # try:
        loss = self.disp_vali(pred['depth'], gt_depth).item()
        # except:
        #    print('error when eval losses, might be in test mode')
        #    pass

        if np.mod(epoch, self.opt.vis_every_vali) == 0:
            if batch_idx < self.opt.vis_batches_vali:
                for k, v in pred.items():
                    pred[k] = v.cpu().numpy()
                outdir = join(self.full_logdir, 'visualize', 'epoch%04d_vali' % epoch)
                makedirs(outdir, exist_ok=True)
                output = self.pack_output(pred, batch)
                if self.global_rank == 0:
                    if self.visualizer is not None:
                        self.visualizer.visualize(output, batch_idx + (1000 * epoch), outdir)
                np.savez(join(outdir, 'rank%04d_batch%04d' % (self.global_rank, batch_idx)), **output)
        batch_size = batch['img'].shape[0]

        batch_log = {'size': batch_size, 'loss': loss}
        return batch_log

    def pack_output(self, pred_all, batch):
        batch_size = len(batch['pair_path'])
        if 'img' not in batch.keys():
            img_1 = batch['img_1'].cpu().numpy()
            img_2 = batch['img_2'].cpu().numpy()
        else:
            img_1 = batch['img']
            img_2 = batch['img']
        output = {'batch_size': batch_size, 'img_1': img_1, 'img_2': img_2, **pred_all}

        if 'img' not in batch.keys():
            output['flow_1_2'] = self._input.flow_1_2.cpu().numpy()
            output['flow_2_1'] = self._input.flow_2_1.cpu().numpy()
            output['depth_nn_1'] = batch['depth_pred_1'].cpu().numpy()

        else:
            output['depth_nn'] = batch['depth_pred'].cpu().numpy()
            output['depth_gt'] = batch['depth_mvs'].cpu().numpy()
            output['cam_c2w'] = batch['cam_c2w'].cpu().numpy()
            output['K'] = batch['K'].cpu().numpy()
        output['pair_path'] = batch['pair_path']
        return output

    def test_on_batch(self, batch_idx, batch):
        if not hasattr(self, 'test_cache'):
            self.test_cache = []
        for n in self._nets:
            n.eval()
        self.load_batch(batch)
        with torch.no_grad():
            pred = self._predict_on_batch(is_train=False)

        if not hasattr(self, 'test_loss'):
            self.test_loss = 0

        for k, v in pred.items():
            pred[k] = v.cpu().numpy()
        epoch_string = 'best' if self.opt.epoch < 0 else '%04d' % self.opt.epoch
        outdir = join(self.opt.output_dir, 'epoch%s_test' % epoch_string)
        if not hasattr(self, 'outdir'):
            self.outdir = outdir
        makedirs(outdir, exist_ok=True)
        output = self.pack_output(pred, batch)
        if batch_idx == 223:
            output['depth'][0, 0, 0, :] = output['depth'][0, 0, 2, :]
            output['depth'][0, 0, 1, :] = output['depth'][0, 0, 2, :]
        self.test_cache.append(output.copy())
        if self.global_rank == 0:
            if self.visualizer is not None:
                self.visualizer.visualize(output, batch_idx, outdir)
        np.savez(join(outdir, 'batch%04d' % (batch_idx)), **output)

    def on_test_end(self):

        # make test video:
        from subprocess import call
        from util.util_html import Webpage
        from tqdm import tqdm
        depth_pred = []
        depth_nn = []
        depth_gt = []
        imgs = []
        c2ws = []
        Ks = []
        for pack in self.test_cache:
            depth_pred.append(pack['depth'])
            depth_nn.append(pack['depth_nn'])
            imgs.append(pack['img_1'])
            c2ws.append(pack['cam_c2w'])
            Ks.append(pack['K'])
            depth_gt.append(pack['depth_gt'])

        depth_pred = np.concatenate(depth_pred, axis=0)
        depth_nn = np.concatenate(depth_nn, axis=0)
        imgs = np.concatenate(imgs, axis=0)
        c2ws = np.concatenate(c2ws, axis=0)
        Ks = np.concatenate(Ks, axis=0)
        depth_gt = np.concatenate(depth_gt, axis=0)

        pred_max = depth_pred.max()

        pred_min = depth_pred.min()

        print(pred_max, pred_min)
        depth_cmap = 'turbo'
        mask_valid = np.where(depth_gt > 1e-8, 1, 0)

        for i in tqdm(range(depth_pred.shape[0])):
            plt.figure(figsize=[60, 20], dpi=40, facecolor='black')

            plt.subplot(1, 3, 1)
            plt.title('Refined', fontsize=100, color='w')
            plt.imshow(1 / depth_pred[i, 0, ...], cmap=depth_cmap, vmax=1 / pred_min, vmin=1 / pred_max)
            cbar = plt.colorbar(fraction=0.048 * 0.5, pad=0.01)
            plt.axis('off')
            cbar.ax.yaxis.set_tick_params(color='w', labelsize=40)
            plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='w')

            plt.subplot(1, 3, 2)
            plt.title('Initial', fontsize=100, color='w')
            plt.imshow(1 / depth_nn[i, 0, ...], cmap=depth_cmap, vmax=1 / pred_min, vmin=1 / pred_max)
            plt.axis('off')
            cbar = plt.colorbar(fraction=0.048 * 0.5, pad=0.01)
            cbar.ax.yaxis.set_tick_params(color='w', labelsize=40)
            plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='w')

            plt.subplot(1, 3, 3)
            plt.title('GT', fontsize=100, color='w')

            plt.imshow(mask_valid[i, 0, ...] / (depth_gt[i, 0, ...] + 1e-8), cmap=depth_cmap, vmax=1 / pred_min, vmin=1 / pred_max)
            plt.axis('off')
            cbar = plt.colorbar(fraction=0.048 * 0.5, pad=0.01)
            cbar.ax.yaxis.set_tick_params(color='w', labelsize=40)
            plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='w')
            plt.savefig(join(self.outdir, 'compare_%04d.png' % i), bbox_inches='tight', facecolor='black', dpi='figure')
            plt.close()

            plt.imshow(imgs[i, ...].transpose(1, 2, 0))
            plt.axis('off')
            plt.savefig(join(self.outdir, 'rgb_%04d.png' % i), bbox_inches='tight', facecolor='black', dpi='figure')
            plt.close()

        epoch_string = self.outdir.split('/')[-1]

        gen_vid_command = 'ffmpeg -y -r 30 -i {img_template} -vcodec libx264 -crf 25 -pix_fmt yuv420p -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" {video_path} > /dev/null'
        gen_vid_command_slow = 'ffmpeg -y -r 2 -i {img_template} -vcodec libx264 -crf 25 -pix_fmt yuv420p -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" {video_path} > /dev/null'
        for r_number in range(120, 140):
            plt.figure(figsize=[60, 20], dpi=20, facecolor='black')

            plt.subplot(1, 2, 1)
            plt.title('Refined', fontsize=100, color='w')
            plt.imshow(1 / depth_pred[:, 0, r_number, :], cmap=depth_cmap)
            cbar = plt.colorbar(fraction=0.048 * 0.5, pad=0.01)
            plt.axis('off')
            cbar.ax.yaxis.set_tick_params(color='w', labelsize=40)
            plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='w')

            plt.subplot(1, 2, 2)
            plt.title('Initial', fontsize=100, color='w')
            plt.imshow(1 / depth_nn[:, 0, r_number, :], cmap=depth_cmap)

            plt.axis('off')
            cbar = plt.colorbar(fraction=0.048 * 0.5, pad=0.01)
            cbar.ax.yaxis.set_tick_params(color='w', labelsize=40)
            plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='w')
            plt.savefig(join(self.outdir, 'temporal_slice_%04d.png' % (r_number - 120)), bbox_inches='tight', facecolor='black', dpi='figure')
            plt.close()

        img_template = join(self.outdir, 'compare_%04d.png')

        img_template_t = join(self.outdir, 'temporal_slice_%04d.png')

        video_path = join(dirname(self.outdir), epoch_string + '.mp4')

        video_path_t = join(dirname(self.outdir), epoch_string + '_temporal.mp4')

        gen_vid_command_c = gen_vid_command.format(img_template=img_template, video_path=video_path)
        call(gen_vid_command_c, shell=True)
        gen_vid_command_t = gen_vid_command_slow.format(img_template=img_template_t, video_path=video_path_t)

        call(gen_vid_command_t, shell=True)

        web = Webpage()

        web.add_video(epoch_string + '_rgb.mp4', title='original video')
        web.add_video(epoch_string + '.mp4', title=f'Disparity loss {self.test_loss}')

        web.save(join(dirname(self.outdir), epoch_string + '.html'))

    @staticmethod
    def copy_and_make_dir(src, target):
        fname = dirname(target)
        makedirs(fname, exist_ok=True)
        shutil.copy(src, target)

    @staticmethod
    def scale_tesnor(t):
        t = (t - t.min()) / (t.max() - t.min() + 1e-9)
        return t


# %%
