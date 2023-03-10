import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import mcubes
from icecream import ic
import ipdb


def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    # Get pdf
    weights = weights + 1e-5 # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples


def sample_pdf(bins, weights, n_samples, det=False):
    #ipdb.set_trace()
    # This implementation is from NeRF
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    # Take uniform samples
    if det:
        u = torch.linspace(0. + 0.5 / n_samples, 1. - 0.5 / n_samples, steps=n_samples)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples])

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


class Renderer:
    def __init__(self,
                 color_network,
                 sdf_network,
                 deviation_network,
                 n_samples,
                 n_importance,
                 perturb):
        self.color_network = color_network
        self.sdf_network = sdf_network
        self.deviation_network = deviation_network
        self.n_samples = n_samples
        self.n_importance = n_importance
        self.perturb = perturb

    def render_inference(self, rays_o, rays_d, z_vals, sample_dist, color_network,sdf,deviation_network,cos_anneal_ratio, background_rgb=None):
        """
        Render background
        """
        batch_size, n_samples = z_vals.shape

        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, sample_dist], -1)
        mid_z_vals = z_vals + dists * 0.5

        pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]  # batch_size, n_samples, 3

        dirs = rays_d[:, None, :].expand(batch_size, n_samples, 3)

        #pts_color = pts_color.reshape(-1, 3 + int(self.n_outside > 0))
        pts = pts.reshape(-1, 3)
        dirs = dirs.reshape(-1, 3)

        gradient, sdf_nn_output = sdf.gradient_inference(pts)
        gradients = gradient.squeeze()
        with torch.no_grad():
            sdf = sdf_nn_output[:, :1]
            inv_s = deviation_network(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)           # Single parameter
            inv_s = inv_s.expand(batch_size * n_samples, 1)

            true_cos = (dirs * gradients).sum(-1, keepdim=True)

            # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
            # the cos value "not dead" at the beginning training iterations, for better convergence.
            iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) +
                        F.relu(-true_cos) * cos_anneal_ratio)  # always non-positive

            # Estimate signed distances at section points
            estimated_next_sdf = sdf + iter_cos * dists.reshape(-1, 1) * 0.5
            estimated_prev_sdf = sdf - iter_cos * dists.reshape(-1, 1) * 0.5

            prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
            next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

            p = prev_cdf - next_cdf
            c = prev_cdf

            alpha = ((p + 1e-5) / (c + 1e-5)).reshape(batch_size, n_samples).clip(0.0, 1.0)
            weights = alpha * torch.cumprod(torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]

        return {
            'alpha': alpha,
            'weights': weights,
            'mid_z':mid_z_vals
        }

    def render_gradient(self, rays_o, rays_d, z_vals, sample_dist, color_network,sdf,deviation_network,cos_anneal_ratio, background_rgb=None):
        """
        Render background
        """
        batch_size, n_samples = z_vals.shape

        # Section length
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, sample_dist], -1)
        mid_z_vals = z_vals + dists * 0.5
        #ipdb.set_trace()
        # Section midpoints
        pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]  # batch_size, n_samples, 3

        dis_to_center = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=True).clip(10e-4, 1e10)
        pts_color = torch.cat([pts , 1/dis_to_center], dim=-1)       # batch_size, n_samples, 4

        dirs = rays_d[:, None, :].expand(batch_size, n_samples, 3)

        pts_color = pts_color.reshape(-1, 4)
        pts = pts.reshape(-1, 3)
        dirs = dirs.reshape(-1, 3)

        gradient, sdf_nn_output = sdf.gradient(pts)
        gradients = gradient.squeeze()
        sdf = sdf_nn_output[:, :1]
        #sdf = torch.clamp(sdf, min=-16, max=16)
        feature_vector = sdf_nn_output[:, 1:]
        sampled_color = color_network(pts, dirs, feature_vector)
        #################SDF rendering equation####################
        inv_s = deviation_network(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)           # Single parameter
        inv_s = inv_s.expand(batch_size * n_samples, 1)

        true_cos = (dirs * gradients).sum(-1, keepdim=True)

        # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
        # the cos value "not dead" at the beginning training iterations, for better convergence.
        iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) +
                     F.relu(-true_cos) * cos_anneal_ratio)  # always non-positive

        # Estimate signed distances at section points
        estimated_next_sdf = sdf + iter_cos * dists.reshape(-1, 1) * 0.5
        estimated_prev_sdf = sdf - iter_cos * dists.reshape(-1, 1) * 0.5

        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

        p = prev_cdf - next_cdf
        c = prev_cdf

        alpha = ((p + 1e-5) / (c + 1e-5)).reshape(batch_size, n_samples).clip(0.0, 1.0)
        #############classic NeRF rendering equation###############
        # alpha = 1.0 - torch.exp(-F.softplus(sdf.reshape(batch_size, n_samples)) * dists)
        # alpha = alpha.reshape(batch_size, n_samples)
        ###########################################################
        weights = alpha * torch.cumprod(torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]
        sampled_color = sampled_color.reshape(batch_size, n_samples, 3)
        color = (weights[:, :, None] * sampled_color).sum(dim=1)
        if background_rgb is not None:
            color = color + background_rgb * (1.0 - weights.sum(dim=-1, keepdim=True))

        pts_norm = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=True).reshape(batch_size, n_samples)
        relax_inside_sphere = (pts_norm < 10).float().detach()
        
        gradient_error = (torch.linalg.norm(gradients.reshape(batch_size, n_samples, 3), ord=2,
                                            dim=-1) - 1.0) ** 2
        gradient_error = (relax_inside_sphere * gradient_error).sum() / (relax_inside_sphere.sum() + 1e-5)

        return {
            'color': color,
            'sampled_color': sampled_color,
            'gradients': gradients.reshape(batch_size, n_samples, 3),
            'alpha': alpha,
            'weights': weights,
            'mid_z':mid_z_vals,
            'sdf':sdf,
            'gradient_error':gradient_error
        }


    def cat_z_vals(self, rays_o, rays_d, z_vals, new_z_vals, sdf, last=False):
        batch_size, n_samples = z_vals.shape
        _, n_importance = new_z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * new_z_vals[..., :, None]
        z_vals = torch.cat([z_vals, new_z_vals], dim=-1)
        z_vals, index = torch.sort(z_vals, dim=-1)

        if not last:
            new_sdf = self.sdf_network.sdf(pts.reshape(-1, 3)).reshape(batch_size, n_importance)
            sdf = torch.cat([sdf, new_sdf], dim=-1)
            xx = torch.arange(batch_size)[:, None].expand(batch_size, n_samples + n_importance).reshape(-1)
            index = index.reshape(-1)
            sdf = sdf[(xx, index)].reshape(batch_size, n_samples + n_importance)

        return z_vals, sdf

    def render(self, rays_o, rays_d, near, far, perturb_overwrite=-1, background_rgb=None, cos_anneal_ratio=0.0):
        #ipdb.set_trace()
        batch_size = len(rays_o)
        sample_dist = (far - near)[...,None] / self.n_samples   # Assuming the region of interest is a unit sphere
        z_vals = torch.linspace(0.0, 1.0, self.n_samples)
        z_vals = near + (far - near)[...,None] * z_vals[None, :]

        perturb = self.perturb

        if perturb_overwrite >= 0:
            perturb = perturb_overwrite
        if perturb > 0:
            t_rand = (torch.rand([batch_size, 1]) - 0.5)
            z_vals = z_vals + t_rand * (far - near)[...,None] / self.n_samples
        else:
            t_rand = torch.zeros([batch_size, 1])
            z_vals = z_vals + t_rand * (far - near)[...,None] / self.n_samples


        ret = self.render_inference(rays_o, rays_d, z_vals, sample_dist, self.color_network, self.sdf_network, self.deviation_network, cos_anneal_ratio)
        z_samples = sample_pdf(0.5*(ret['mid_z'][..., 1:] + ret['mid_z'][..., :-1]), ret['weights'][...,1:-1], self.n_importance, det=(perturb == 0.))
        z_vals_feed = torch.cat([z_vals, z_samples], dim=-1)
        z_vals_feed, _ = torch.sort(z_vals_feed, dim=-1)

        ret_outside = self.render_gradient(rays_o, rays_d, z_vals_feed, sample_dist, self.color_network, self.sdf_network, self.deviation_network,cos_anneal_ratio)

        depth_fine = torch.sum(ret_outside['mid_z']*ret_outside['weights'], -1)

        pts = rays_o + rays_d * depth_fine[...,None]
        return {
            #'color_coarse': ret['color'],
            'color_fine': ret_outside['color'],
            'depth_fine': depth_fine,
            'val_z':ret_outside['mid_z'],
            'weights':ret_outside['weights'],
            'sdf':ret_outside['sdf'],
            'pts':pts,
            'gradients_out':ret_outside['gradients'],
            'gradient_error':ret_outside['gradient_error']
        }

