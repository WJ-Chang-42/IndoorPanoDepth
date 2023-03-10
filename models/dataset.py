import torch
import torch.nn.functional as F
import cv2 as cv
import numpy as np
import os
from glob import glob
from icecream import ic
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp
import ipdb
import cv2

# This function is borrowed from IDR: https://github.com/lioryariv/idr
def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose


class Dataset: 
    def __init__(self, conf,numbers,patch=4,degree=0):
        super(Dataset, self).__init__()
        print('Load data: Begin:Random_sampling')
        
        self.device = torch.device('cuda')
        self.conf = conf
        self.data_dir = conf.get_string('data_dir')
        if numbers>0:
            self.position = np.load(self.data_dir+'_position.npy')
        
        images = []
        depths = []
        #ipdb.set_trace()
        img = np.array(cv2.imread(self.data_dir+'_color_0_Left_Down_0.0.png', cv2.IMREAD_ANYCOLOR)) / 255.
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.
        depth = np.array(cv2.imread(self.data_dir+'_depth_0_Left_Down_0.0.exr',cv2.IMREAD_UNCHANGED)[:,:,0])
        images.append(img)
        depths.append(depth)
        img = np.array(cv2.imread(self.data_dir+'_color_0_Right_0.0.png', cv2.IMREAD_ANYCOLOR)) / 255.
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.
        depth = np.array(cv2.imread(self.data_dir+'_depth_0_Right_0.0.exr',cv2.IMREAD_UNCHANGED)[:,:,0])
        images.append(img)
        depths.append(depth)
        img = np.array(cv2.imread(self.data_dir+'_color_0_Up_0.0.png', cv2.IMREAD_ANYCOLOR)) / 255.
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.
        depth = np.array(cv2.imread(self.data_dir+'_depth_0_Up_0.0.exr',cv2.IMREAD_UNCHANGED)[:,:,0])
        images.append(img)
        depths.append(depth)
        for i in range(numbers):
           img = np.array(cv2.imread(self.data_dir+'_color_0_%02d_0.0.png'%(i+1), cv2.IMREAD_ANYCOLOR)) / 255.
           #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.
           depth = np.array(cv2.imread(self.data_dir+'_depth_0_%02d_0.0.exr'%(i+1),cv2.IMREAD_UNCHANGED)[:,:,0])
           images.append(img)
           depths.append(depth)
        
        self.n_images = len(images)
            
        self.images_np = np.stack(images, axis=0)
        self.depths_np = np.stack(depths, axis=0)

        self.images = torch.from_numpy(self.images_np.astype(np.float32))  # [n_images, H, W, 3]
        self.depths = torch.from_numpy(self.depths_np.astype(np.float32)) # [n_images, H, W]
        self.H, self.W = self.images.shape[1], self.images.shape[2]
        self.image_pixels = self.H * self.W
        self.patch_size = patch
        self.n_img_patches = self.image_pixels
        self.n_patches = self.n_img_patches * (self.n_images)
        self.patch_row = torch.arange(patch, device='cpu').repeat(patch,1).reshape(-1)
        self.patch_col = torch.arange(patch, device='cpu').repeat(patch,1).permute(1,0).reshape(-1)
        #ipdb.set_trace()
        cen_x = (self.W - 1) / 2.0
        cen_y = (self.H - 1) / 2.0
        theta = (2 * (np.arange(self.W) - cen_x) / self.W) * np.pi
        phi = (2 * (np.arange(self.H) - cen_y) / self.H) * (np.pi / 2)
        theta = np.tile(theta[None, :], [self.H, 1])
        phi = np.tile(phi[None, :], [self.W, 1]).T

        x = (np.cos(phi) * np.sin(theta)).reshape([self.H, self.W, 1])
        y = (np.sin(phi)).reshape([self.H, self.W, 1])
        z = (np.cos(phi) * np.cos(theta)).reshape([self.H, self.W, 1])
        directions = np.concatenate([x, y, z], axis=-1).reshape(-1,3)
        ellipsoid = np.concatenate([16*x, 16*y, 16*z], axis=-1)
        radius = np.linalg.norm(ellipsoid,ord=2,axis=-1)
        #phi_ = torch.arcsin(y)
        #theta_ =  torch.arctan(x/z)[...,0]
        #theta_[z[...,0]<0] += torch.pi
        #theta_[theta_>np.pi] -= 2*np.pi
        ro = degree/180*np.pi
        rotation = np.array([[np.cos(ro),np.sin(ro),0],[-np.sin(ro),np.cos(ro),0],[0,0,1]])
        directions = (rotation @ directions.T ).T
        directions = directions.reshape([self.H,self.W,-1])[None,:,:,:]
        
        directions = np.concatenate([directions, directions, directions,directions,directions,directions,directions,directions,directions,directions,directions,directions,directions], axis=0)
        #directions = np.concatenate([directions, directions], axis=0)
        origins = np.zeros(directions.shape, dtype = directions.dtype)
        origins[1] = origins[1] + (rotation @ np.array([0.26,0,0]).T)[None,None,:]
        origins[2] = origins[2] + (rotation @ np.array([0,-0.26,0]).T)[None,None,:]
        for i in range(numbers):
            origins[i+3] = origins[i+3] + np.array([self.position[i][0],-self.position[i][2],self.position[i][1]])[None,None,:]
        #ipdb.set_trace()
        self.rays_v = torch.from_numpy(directions.astype(np.float32))
        self.rays_o = torch.from_numpy(origins.astype(np.float32))
        self.radius = torch.from_numpy(radius.astype(np.float32))
        self.all_radius = torch.from_numpy(radius[None,:].astype(np.float32).repeat(self.n_images,axis=0))
        self.all_images = self.images.reshape(-1,3)
        self.all_depths = self.depths.reshape(-1)
        self.all_rays_v = self.rays_v.reshape(-1,3)
        self.all_rays_o = self.rays_o.reshape(-1,3)
        self.all_radius = self.all_radius.reshape(-1)
        

        print('Load data: End')

    def gen_rays_at(self, img_idx, resolution_level=1):
        """
        Generate rays at world space from one camera.
        """
        l = resolution_level


        return self.rays_o[img_idx].cuda(), self.rays_v[img_idx].cuda(), self.depths[img_idx].cuda(), self.radius.cuda()


    def __len__(self):
        return 512*256*self.n_images // self.patch_size**2


    def gen_random_rays_at(self, img_idx, batch_size):
        """
        Generate random rays at world space from one camera.
        """
        pixels_x = torch.randint(low=0, high=self.W, size=[batch_size],device='cpu')
        pixels_y = torch.randint(low=0, high=self.H, size=[batch_size],device='cpu')
        color = self.images[img_idx][(pixels_y, pixels_x)]    # batch_size, 3
        rays_v = self.rays_v[img_idx][(pixels_y, pixels_x)]   # batch_size, 3
        rays_o = self.rays_o[img_idx][(pixels_y, pixels_x)]  # batch_size, 3
        depth = self.depths[img_idx][(pixels_y, pixels_x)] 
        #ipdb.set_trace()
        far = self.radius[(pixels_y, pixels_x)] 
        return torch.cat([rays_o, rays_v, color, depth.unsqueeze(-1),far.unsqueeze(-1)], dim=-1)    # batch_size, 9

    def __getitem__(self, index):
        #ipdb.set_trace()
        pixels_x = torch.randint(low=0, high=self.W, size=[self.patch_size*self.patch_size],device='cpu')#[0].item()
        pixels_y = torch.randint(low=0, high=self.H, size=[self.patch_size*self.patch_size],device='cpu')#[0].item()
        #pixels_x = x + self.patch_row
        #pixels_x[pixels_x >= self.W] = pixels_x[pixels_x >= self.W] - self.W
        #pixels_y = y + self.patch_col
        #pixels_y[pixels_y >= self.H] = pixels_y[pixels_y >= self.H] - self.H
        img_idx = torch.randint(low=0, high=self.n_images, size=[1],device='cpu')[0].item()

        rays_o = self.rays_o[img_idx][(pixels_y, pixels_x)] 
        rays_v = self.rays_v[img_idx][(pixels_y, pixels_x)] 
        color = self.images[img_idx][(pixels_y, pixels_x)] 
        depth = self.depths[img_idx][(pixels_y, pixels_x)] 
        far = self.radius[(pixels_y, pixels_x)] 
        return torch.cat([rays_o, rays_v, color, depth.unsqueeze(-1),far.unsqueeze(-1)], dim=-1)
    
    # def __getitem__(self, index):
    #     i_patch = torch.randint(high=self.n_patches, size=(1,),device='cpu')[0].item()
    #     i_img, i_patch = i_patch // self.n_img_patches, i_patch % self.n_img_patches
    #     row, col = i_patch // (self.W - self.patch_size + 1), i_patch % (self.W - self.patch_size + 1)
    #     start_idx = i_img * self.W * self.H + row * self.W + col
    #     idxs = start_idx + torch.cat([torch.arange(self.patch_size,device='cpu') + i * self.W for i in range(self.patch_size)])

    #     rays_o = self.all_rays_o[idxs]
    #     rays_v = self.all_rays_v[idxs]
    #     color = self.all_images[idxs]
    #     depth = self.all_depths[idxs]
    #     far = self.all_radius[idxs]
    #     return torch.cat([rays_o, rays_v, color, depth.unsqueeze(-1),far.unsqueeze(-1)], dim=-1)
    
    def gen_rays_between(self, idx_0, idx_1, ratio, resolution_level=1):
        """
        Interpolate pose between two cameras.
        """
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1)  # W, H, 3
        p = torch.matmul(self.intrinsics_all_inv[0, None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        trans = self.pose_all[idx_0, :3, 3] * (1.0 - ratio) + self.pose_all[idx_1, :3, 3] * ratio
        pose_0 = self.pose_all[idx_0].detach().cpu().numpy()
        pose_1 = self.pose_all[idx_1].detach().cpu().numpy()
        pose_0 = np.linalg.inv(pose_0)
        pose_1 = np.linalg.inv(pose_1)
        rot_0 = pose_0[:3, :3]
        rot_1 = pose_1[:3, :3]
        rots = Rot.from_matrix(np.stack([rot_0, rot_1]))
        key_times = [0, 1]
        slerp = Slerp(key_times, rots)
        rot = slerp(ratio)
        pose = np.diag([1.0, 1.0, 1.0, 1.0])
        pose = pose.astype(np.float32)
        pose[:3, :3] = rot.as_matrix()
        pose[:3, 3] = ((1.0 - ratio) * pose_0 + ratio * pose_1)[:3, 3]
        pose = np.linalg.inv(pose)
        rot = torch.from_numpy(pose[:3, :3]).cuda()
        trans = torch.from_numpy(pose[:3, 3]).cuda()
        rays_v = torch.matmul(rot[None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        rays_o = trans[None, None, :3].expand(rays_v.shape)  # W, H, 3
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)

    def near_far_from_sphere(self, rays_o, rays_d):
        #a = torch.sum(rays_d**2, dim=-1, keepdim=True)
        #b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
        #mid = 0.5 * (-b) / a
        #near = mid - 1.0
        #far = mid + 1.0
        near = 0
        far = 16
        return near, far

    def image_at(self, idx, resolution_level):
        img = cv.imread(self.images_lis[idx])
        return (cv.resize(img, (self.W // resolution_level, self.H // resolution_level))).clip(0, 255)

