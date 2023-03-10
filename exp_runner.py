import os
import time
import logging
import argparse
import ipdb
import numpy as np
import cv2 as cv
import trimesh
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from models.dataset_patch import Dataset_patch
from models.dataset import Dataset
from shutil import copyfile
from icecream import ic
from tqdm import tqdm
from pyhocon import ConfigFactory
from models.fields import SDFNetwork, SingleVarianceNetwork, COLORNetwork
from models.renderer import Renderer
import matplotlib.pyplot as plt
import random
import time
seed = 841232111
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
def standard_metrics(input_gt_depth_image,pred_depth_image, verbose=True):
    input_gt_depth = input_gt_depth_image.copy()
    pred_depth = pred_depth_image.copy()
    
    input_gt_depth[input_gt_depth>10] = 0

    n = np.sum((input_gt_depth > 1e-3)) ####valid samples
                        
    ###invalid samples - no measures
    idxs = ( (input_gt_depth <= 1e-3) )
    pred_depth[idxs] = 1
    input_gt_depth[idxs] = 1

    if(verbose):
        print('valid samples:',n,'masked samples:', np.sum(idxs))

    ####STEP 1: compute delta################################################################
    #######prepare mask
    pred_d_gt = pred_depth / input_gt_depth
    pred_d_gt[idxs] = 100
    gt_d_pred = input_gt_depth / pred_depth
    gt_d_pred[idxs] = 100

    Threshold_1_25 = np.sum(np.maximum(pred_d_gt, gt_d_pred) < 1.25) / n
    Threshold_1_25_2 = np.sum(np.maximum(pred_d_gt, gt_d_pred) < 1.25 * 1.25) / n
    Threshold_1_25_3 = np.sum(np.maximum(pred_d_gt, gt_d_pred) < 1.25 * 1.25 * 1.25) / n
    ########################################################################################        

    #####STEP 2: compute mean error##########################################################
    input_gt_depth_norm = input_gt_depth / np.max(input_gt_depth)
    pred_depth_norm = pred_depth / np.max(pred_depth)
    if(verbose):
        print(np.max(input_gt_depth),np.max(pred_depth))
    log_pred = np.log(pred_depth_norm)
    log_gt = np.log(input_gt_depth_norm)
               
    ###OmniDepth: 
    RMSE_linear = ((pred_depth - input_gt_depth) ** 2).mean()
    RMSE_log = np.sqrt(((log_pred - log_gt) ** 2).mean())
    ARD = (np.abs((pred_depth_norm - input_gt_depth_norm)) / input_gt_depth_norm).mean()
    SRD = (((pred_depth_norm - input_gt_depth_norm)** 2) / input_gt_depth_norm).mean()
    MAE = np.abs((pred_depth - input_gt_depth)).mean()
    REL = (np.abs((pred_depth - input_gt_depth)) / input_gt_depth).mean()
    if(verbose):
        print('MAE\tREL\tLog\tSRD\tARD\tRMSE\tTh1\tTh2\tTh3')
        print('%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f'%(MAE,REL,RMSE_log,SRD,ARD,RMSE_linear,Threshold_1_25,Threshold_1_25_2,Threshold_1_25_3))
        
    return [MAE,REL,RMSE_linear,Threshold_1_25,Threshold_1_25_2,Threshold_1_25_3]



class Runner:
    def __init__(self, conf_path, base_exp_dir, case='CASE_NAME', is_continue=False,d = 0,numbers=0, random_epoch = None, ps = None):
        self.device = torch.device('cuda')

        # Configuration
        self.conf_path = conf_path
        f = open(self.conf_path)
        conf_text = f.read()
        conf_text = conf_text.replace('CASE_NAME', case)
        f.close()
        self.bs = 32
        self.random_epoch = random_epoch
        self.conf = ConfigFactory.parse_string(conf_text)
        #ipdb.set_trace()
        self.conf['dataset.data_dir'] = self.conf['dataset.data_dir'].replace('CASE_NAME', case)
        self.base_exp_dir = base_exp_dir+ '/%d_images/'%numbers+ case
        os.makedirs(self.base_exp_dir, exist_ok=True)
        self.dataset = Dataset(self.conf['dataset'],numbers,patch = 4,degree=args.degree)
        self.dataset_patch = Dataset_patch(self.conf['dataset'],numbers,patch = ps,degree=args.degree)
        print(self.dataset.n_images)
        print(self.dataset_patch.n_images)
        print(self.dataset_patch[0].shape)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.bs,
            shuffle=False,
            num_workers=8,
            drop_last=False,
            pin_memory=True
        )
        self.dataloader_patch = torch.utils.data.DataLoader(
            self.dataset_patch,
            batch_size=self.bs,
            shuffle=False,
            num_workers=8,
            drop_last=False,
            pin_memory=True
        )
        self.iter_step = 0

        # Training parameters
        self.end_iter = self.conf.get_int('train.end_iter')
        self.save_freq = self.conf.get_int('train.save_freq')
        self.report_freq = self.conf.get_int('train.report_freq')
        self.val_freq = self.conf.get_int('train.val_freq')
        self.val_mesh_freq = self.conf.get_int('train.val_mesh_freq')
        self.batch_size = self.conf.get_int('train.batch_size')
        self.validate_resolution_level = self.conf.get_int('train.validate_resolution_level')
        self.learning_rate = self.conf.get_float('train.learning_rate')
        self.learning_rate_alpha = self.conf.get_float('train.learning_rate_alpha')
        self.use_white_bkgd = self.conf.get_bool('train.use_white_bkgd')
        self.warm_up_end = self.conf.get_float('train.warm_up_end', default=0.0)
        self.anneal_end = self.conf.get_float('train.anneal_end', default=0.0)

        # Weights
        self.igr_weight = self.conf.get_float('train.igr_weight')
        self.mask_weight = self.conf.get_float('train.mask_weight')
        self.is_continue = is_continue
        self.model_list = []
        self.writer = None

        # Networks
        params_to_train = []
        self.color_network = COLORNetwork(**self.conf['model.color_network']).to(self.device)
        self.sdf_network = SDFNetwork(distance = d,**self.conf['model.sdf_network']).to(self.device)
        self.deviation_network = SingleVarianceNetwork(**self.conf['model.variance_network']).to(self.device)
        params_to_train += list(self.color_network.parameters())
        params_to_train += list(self.sdf_network.parameters())
        params_to_train += list(self.deviation_network.parameters())

        self.optimizer = torch.optim.Adam(params_to_train, lr=self.learning_rate)

        self.renderer = Renderer(self.color_network,
                                     self.sdf_network,
                                     self.deviation_network,
                                     **self.conf['model.renderer'])
        #ipdb.set_trace()
        # Load checkpoint
        latest_model_name = None
        if is_continue is not None:
            latest_model_name = 'exp/%.1f_%s/%d_images/%s/checkpoints/ckpt_%06d.pth'%(d,is_continue,numbers,case,80)

        if latest_model_name is not None:
            print('Find checkpoint: {}'.format(latest_model_name))
            self.load_checkpoint(latest_model_name)

        # Backup codes and configs for debug
        self.file_backup()

    def train(self):
        self.writer = SummaryWriter(log_dir=os.path.join(self.base_exp_dir, 'logs'))
        self.update_learning_rate()
        res_step = self.end_iter - self.iter_step
        _,_ = self.validate_image(idx=2)      
        near = 0
        out_average = []
        #ipdb.set_trace()
        if self.iter_step < self.random_epoch:
            for epoch in tqdm(range(self.random_epoch - self.iter_step)):
                for i, data in enumerate(self.dataloader):
                    rays_o, rays_d, true_rgb, depth_gt, far  = data[..., :3], data[..., 3: 6], data[..., 6: 9], data[..., 9], data[..., 10]
                    rays_o = rays_o.reshape(-1,3).cuda()
                    rays_d = rays_d.reshape(-1,3).cuda()
                    true_rgb = true_rgb.reshape(-1,3).cuda()
                    depth_gt = depth_gt.reshape(-1).cuda()
                    far = far.reshape(-1).cuda()
                    background_rgb = None
 
                    mask = torch.ones_like(rays_o[:,:1])

                    mask_sum = mask.sum() + 1e-5
                    render_out = self.renderer.render(rays_o, rays_d, near, far,
                                                    background_rgb=background_rgb,
                                                    cos_anneal_ratio=self.get_cos_anneal_ratio())

                    #ipdb.set_trace()
                    color_fine = render_out['color_fine']
                    depth_mask = depth_gt < 10.0
                    depth_l1 = torch.abs((render_out['depth_fine'] - depth_gt)*depth_mask).mean()
                    # Loss
                    color_error_fine = (color_fine - true_rgb) * mask
                    color_fine_loss = F.l1_loss(color_error_fine, torch.zeros_like(color_error_fine), reduction='sum') / mask_sum
                    psnr = 20.0 * torch.log10(1.0 / (((color_fine - true_rgb)**2 * mask).sum() / (mask_sum * 3.0)).sqrt())
                    eikonal_loss = render_out['gradient_error']

                    loss = color_fine_loss + eikonal_loss * self.igr_weight #+ color_coarse_loss

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                self.iter_step += 1
                #ipdb.set_trace()

                if self.iter_step % self.report_freq == 0:

                    if self.iter_step > 0 :
                        self.writer.add_scalar('Loss/loss', loss, self.iter_step)
                        self.writer.add_scalar('Loss/color_fine_loss', color_fine_loss, self.iter_step)
                        self.writer.add_scalar('Statistics/psnr', psnr, self.iter_step)
                        self.writer.add_scalar('Statistics/depth', depth_l1, self.iter_step)
                        print('iter:{:8>d} loss = {} lr = {}'.format(self.iter_step, loss,self.optimizer.param_groups[0]['lr']))
                if self.iter_step % self.val_freq == 0:
                    errors_weight, weight_sdf = self.validate_image(idx=2)
                    self.writer.add_scalar('Weight/MRE', errors_weight[1], self.iter_step)
                    self.writer.add_scalar('Weight/RMSE', errors_weight[2], self.iter_step)
                    self.writer.add_scalar('Weight/MAE', errors_weight[0], self.iter_step)
                if self.iter_step % self.save_freq == 0:
                    self.save_checkpoint()
                self.update_learning_rate()
        res_epochs = self.end_iter - self.iter_step
        for epoch in tqdm(range(res_epochs)):
            for i, data in enumerate(self.dataloader_patch):
                rays_o, rays_d, true_rgb, depth_gt, far  = data[..., :3], data[..., 3: 6], data[..., 6: 9], data[..., 9], data[..., 10]
                batch = rays_o.shape[0]
                rays_o = rays_o.reshape(-1,3).cuda()
                rays_d = rays_d.reshape(-1,3).cuda()
                true_rgb = true_rgb.reshape(-1,3).cuda()
                depth_gt = depth_gt.reshape(-1).cuda()
                far = far.reshape(-1).cuda()
                background_rgb = None
                if self.use_white_bkgd:
                    background_rgb = torch.ones([1, 3])

                mask = torch.ones_like(rays_o[:,:1])

                mask_sum = mask.sum() + 1e-5
                render_out = self.renderer.render(rays_o, rays_d, near, far,
                                                background_rgb=background_rgb,
                                                cos_anneal_ratio=self.get_cos_anneal_ratio())
                normals = render_out['gradients_out'] * render_out['weights'][..., None]
                #ipdb.set_trace()
                normals = -F.normalize(normals.sum(dim=1),p=2,dim=-1).reshape(batch,-1,3).mean(1)
                matrix_a = render_out['pts'].reshape(batch,-1,3)
                matrix_a_trans = matrix_a.permute(0,2,1)
                matrix_b = torch.ones(matrix_a.shape[:2]).unsqueeze(-1)
                point_multi = torch.matmul(matrix_a_trans,matrix_a)
                point_multi_inverse = torch.inverse(point_multi)
                normals_from_points = torch.matmul(torch.matmul(point_multi_inverse,matrix_a_trans),matrix_b).squeeze(-1)
                normals_from_points = F.normalize(normals_from_points,p=2,dim=1)
                normal_error_fine = (1 - torch.sum(normals_from_points * normals,-1)).mean()
                color_fine = render_out['color_fine']
                depth_mask = depth_gt < 10.0
                depth_l1 = torch.abs((render_out['depth_fine'] - depth_gt)*depth_mask).mean()
                # Loss
                color_error_fine = (color_fine - true_rgb) * mask
                color_fine_loss = F.l1_loss(color_error_fine, torch.zeros_like(color_error_fine), reduction='sum') / mask_sum
                psnr = 20.0 * torch.log10(1.0 / (((color_fine - true_rgb)**2 * mask).sum() / (mask_sum * 3.0)).sqrt())
                eikonal_loss = render_out['gradient_error']

                loss = color_fine_loss + eikonal_loss * self.igr_weight + 0.01*normal_error_fine#+ color_coarse_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            self.iter_step += 1

            if self.iter_step % self.report_freq == 0:
                    #print(self.base_exp_dir)
                if self.iter_step > 0 :
                    self.writer.add_scalar('Loss/loss', loss, self.iter_step)
                    self.writer.add_scalar('Loss/color_fine_loss', color_fine_loss, self.iter_step)
                    self.writer.add_scalar('Statistics/psnr', psnr, self.iter_step)
                    self.writer.add_scalar('Statistics/depth', depth_l1, self.iter_step)
                    print('iter:{:8>d} loss = {} normal_loss = {} lr = {}'.format(self.iter_step, loss, normal_error_fine,self.optimizer.param_groups[0]['lr']))
            if self.iter_step % self.val_freq == 0:
                errors_weight, weight_sdf = self.validate_image(idx=2)
                self.writer.add_scalar('Weight/MRE', errors_weight[1], self.iter_step)
                self.writer.add_scalar('Weight/RMSE', errors_weight[2], self.iter_step)
                self.writer.add_scalar('Weight/MAE', errors_weight[0], self.iter_step)
            if self.iter_step % self.save_freq == 0:
                self.save_checkpoint()
            self.update_learning_rate()
        #np.save(os.path.join(self.base_exp_dir, 'average'),np.array(out_average))

    def get_image_perm(self):
        return torch.randperm(self.dataset.n_images)

    def get_cos_anneal_ratio(self):
        if self.anneal_end == 0.0:
            return 1.0
        else:
            return np.min([1.0, self.iter_step / self.anneal_end])

    def update_learning_rate(self):
        if self.iter_step < self.warm_up_end:
            learning_factor = self.iter_step / self.warm_up_end
        else:
            alpha = self.learning_rate_alpha
            progress = (self.iter_step - self.warm_up_end) / (self.end_iter - self.warm_up_end)
            learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha

        for g in self.optimizer.param_groups:
            g['lr'] = self.learning_rate * learning_factor

    def file_backup(self):
        dir_lis = self.conf['general.recording']
        os.makedirs(os.path.join(self.base_exp_dir, 'recording'), exist_ok=True)
        for dir_name in dir_lis:
            cur_dir = os.path.join(self.base_exp_dir, 'recording', dir_name)
            os.makedirs(cur_dir, exist_ok=True)
            files = os.listdir(dir_name)
            for f_name in files:
                if f_name[-3:] == '.py':
                    copyfile(os.path.join(dir_name, f_name), os.path.join(cur_dir, f_name))

        copyfile(self.conf_path, os.path.join(self.base_exp_dir, 'recording', 'config.conf'))

    def load_checkpoint(self, checkpoint_name):
        #ipdb.set_trace()
        checkpoint = torch.load(os.path.join(checkpoint_name), map_location=self.device)
        self.color_network.load_state_dict(checkpoint['color_network'])
        self.sdf_network.load_state_dict(checkpoint['sdf_network'])
        self.deviation_network.load_state_dict(checkpoint['variance_network_fine'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.iter_step = checkpoint['iter_step']
        #ipdb.set_trace()
        torch.cuda.set_rng_state(checkpoint['rng'].cpu())
        np.random.set_state(checkpoint['rng_np'])

        logging.info('End')

    def save_checkpoint(self):
        checkpoint = {
            'color_network': self.color_network.state_dict(),
            'sdf_network': self.sdf_network.state_dict(),
            'variance_network_fine': self.deviation_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'iter_step': self.iter_step,
            'rng':torch.cuda.get_rng_state(),
            'rng_np':np.random.get_state(),
        }

        os.makedirs(os.path.join(self.base_exp_dir, 'checkpoints'), exist_ok=True)
        torch.save(checkpoint, os.path.join(self.base_exp_dir, 'checkpoints', 'ckpt_{:0>6d}.pth'.format(self.iter_step)))

    def validate_image(self, idx=-1, resolution_level=-1, verbose = True):
        if idx < 0:
            idx = np.random.randint(self.dataset.n_images)
        if(verbose):
            print('Validate: iter: {}, camera: {}'.format(self.iter_step, idx))

        if resolution_level < 0:
            resolution_level = self.validate_resolution_level
        rays_o, rays_d, gt_depth, far = self.dataset.gen_rays_at(idx, resolution_level=resolution_level)
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)
        far = far.reshape(-1).split(self.batch_size)
        out_rgb_fine = []
        out_normal_fine = []
        out_depth = []
        #out_sdf = []
        out_val_z = []
        out_weights = []
        out_normal = []
        out_pts = []
        near = 0
        for rays_o_batch, rays_d_batch, far_batch in zip(rays_o, rays_d,far):
            near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
            background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None
            #ipdb.set_trace()
            render_out = self.renderer.render(rays_o_batch,
                                              rays_d_batch,
                                              near,
                                              far_batch,
                                              perturb_overwrite = 0,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                              background_rgb=background_rgb)

            def feasible(key): return (key in render_out) and (render_out[key] is not None)

            if feasible('color_fine'):
                out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())
                out_depth.append(render_out['depth_fine'].detach().cpu().numpy())
                #out_sdf.append(render_out['sdf'].detach().cpu().numpy())
                out_val_z.append(render_out['val_z'].detach().cpu().numpy())
                out_weights.append(render_out['weights'].detach().cpu().numpy())
                #n_samples = self.renderer.n_samples + self.renderer.n_importance
                normals = render_out['gradients_out'] * render_out['weights'][..., None]
                normals = F.normalize(normals.sum(dim=1),p=2,dim=-1).detach().cpu().numpy()
                out_normal.append(normals)
                out_pts.append(render_out['pts'].detach().cpu().numpy())

            if feasible('gradients') and feasible('weights'):
                n_samples = self.renderer.n_samples + self.renderer.n_importance
                normals = render_out['gradients'] * render_out['weights'][:, :n_samples, None]
                if feasible('inside_sphere'):
                    normals = normals * render_out['inside_sphere'][..., None]
                normals = normals.sum(dim=1).detach().cpu().numpy()
                out_normal_fine.append(normals)
            del render_out

        img_fine = None
        if len(out_rgb_fine) > 0:
            img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3, -1]) * 256).clip(0, 255)
            depth_fine = np.concatenate(out_depth, axis=0).reshape([H, W, 1, -1])
            val_z = np.concatenate(out_val_z, axis=0).reshape([H, W, -1])
            weights = np.concatenate(out_weights, axis=0).reshape([H, W, -1])
            out_normal = np.concatenate(out_normal, axis=0).reshape([H, W, -1])
            #sdf = np.concatenate(out_sdf, axis=0).reshape([H, W, -1])
            #temp = sdf < 0
            #idx =  np.expand_dims(np.argmax(temp, axis=2), axis=2)
            #depth_sdf = np.take_along_axis(val_z,idx,axis=-1)[:,:,0]

        normal_img = None
        if len(out_normal_fine) > 0:
            normal_img = np.concatenate(out_normal_fine, axis=0)
            rot = np.linalg.inv(self.dataset.pose_all[idx, :3, :3].detach().cpu().numpy())
            normal_img = (np.matmul(rot[None, :, :], normal_img[:, :, None])
                          .reshape([H, W, 3, -1]) * 128 + 128).clip(0, 255)

        os.makedirs(os.path.join(self.base_exp_dir, 'validations_fine'), exist_ok=True)
        os.makedirs(os.path.join(self.base_exp_dir, 'validations_fine','{:0>8d}'.format(self.iter_step)), exist_ok=True)
        os.makedirs(os.path.join(self.base_exp_dir, 'normals'), exist_ok=True)
        
        for i in range(img_fine.shape[-1]):
            if len(out_rgb_fine) > 0:
                cv.imwrite(os.path.join(self.base_exp_dir,
                                        'validations_fine',
                                        '{:0>8d}'.format(self.iter_step),'rgb.png'),
                           img_fine[..., i])
                if(verbose):
                    print('==========depth from weight==========')
                errors_weight = standard_metrics(gt_depth.detach().cpu().numpy(),depth_fine[:,:,0,i],verbose=verbose)
                # print('==========depth from SDF==========')
                # errors_sdf = standard_metrics(gt_depth.detach().cpu().numpy(),depth_sdf)
                _ = plt.imshow(depth_fine[:,:,0,i])
                plt.tight_layout()
                plt.savefig(os.path.join(self.base_exp_dir,
                                        'validations_fine',
                                        '{:0>8d}'.format(self.iter_step),'depth.png'))
                plt.close()

                #ipdb.set_trace()
                _ = plt.imshow((out_normal + 1)/2)
                plt.tight_layout()
                plt.savefig(os.path.join(self.base_exp_dir,
                                        'validations_fine',
                                        '{:0>8d}'.format(self.iter_step),'normal.png'))
                plt.close()
                np.save(os.path.join(self.base_exp_dir,
                                        'validations_fine',
                                        '{:0>8d}'.format(self.iter_step),'depth'), depth_fine[:,:,0,i])
        return errors_weight, None





if __name__ == '__main__':

    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/base.conf')
    parser.add_argument('--mcube_threshold', type=float, default=0.0)
    parser.add_argument('--is_continue', type=str, default=None)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--case', type=str, default='')
    parser.add_argument('--d', type=float, default=1.5)
    parser.add_argument('--n', type=int, default=0)
    parser.add_argument('--random', type=int, default=80)
    parser.add_argument('--degree', type=int, default=0)
    parser.add_argument('--ps', type=int, default=4)
    parser.add_argument('--dir', type=str, default='./exp')
    #ipdb.set_trace()
    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    runner = Runner(args.conf, args.dir, args.case, args.is_continue,args.d,args.n,args.random,args.ps)
    runner.train()
    temp = []
    for i in range(args.n+3):
        errors_weight,_ = runner.validate_image(idx=i)   
        temp.append(errors_weight)
    temp = np.array(temp)
    average = np.mean(temp, axis=0)
    np.save(os.path.join(runner.base_exp_dir, 'average'),np.array(average))
