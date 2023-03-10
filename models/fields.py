import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.embedder import get_embedder
import ipdb

# This implementation is borrowed from IDR: https://github.com/lioryariv/idr
class SDFNetwork(nn.Module):
    def __init__(self,
                 distance,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 skip_in=(4,),
                 multires=0,
                 bias=0.5,
                 scale=1,
                 geometric_init=True,
                 weight_norm=True,
                 inside_outside=False):
        super(SDFNetwork, self).__init__()
        #ipdb.set_trace()
        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embed_fn_fine = None

        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=3)
            self.embed_fn_fine = embed_fn
            dims[0] = input_ch 

        self.num_layers = len(dims)
        self.skip_in = skip_in
        self.scale = scale
        dim_ = 3
        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)
            if distance > 0:
                if l == self.num_layers - 2:
                    if not inside_outside:
                        torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, -distance)
                    else:
                        torch.nn.init.normal_(lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, distance)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight, 0.0)
                    #torch.nn.init.normal_(lin.weight[:, 0], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.normal_(lin.weight[:, 1], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    #torch.nn.init.normal_(lin.weight[:, 2], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    #ipdb.set_trace()
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - dim_):], 0.0)
                    #torch.nn.init.constant_(lin.weight[:, -(dims[0] - 1)], 0.0)
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 0)], 0.0)
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 2)], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
            # else:
            #    if l == self.num_layers - 2:
            #        if not inside_outside:
            #            torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
            #            torch.nn.init.constant_(lin.bias, -distance)
            #        else:
            #            torch.nn.init.normal_(lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
            #            torch.nn.init.constant_(lin.bias, distance)
            #    elif multires > 0 and l == 0:
            #        torch.nn.init.constant_(lin.bias, 0.0)
            #        torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
            #        torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
            #    elif multires > 0 and l in self.skip_in:
            #        torch.nn.init.constant_(lin.bias, 0.0)
            #        torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
            #        torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
            #    else:
            #        torch.nn.init.constant_(lin.bias, 0.0)
            #        torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.activation = nn.Softplus(beta=100)

    def forward(self, inputs):
        #ipdb.set_trace()
        dis_to_center = torch.linalg.norm(inputs, ord=2, dim=-1)
        # phi_ = torch.arctan(inputs[...,1]/(torch.sqrt(inputs[...,0]**2 + inputs[...,2]**2)+10e-10))
        # theta_ =  torch.arctan(inputs[...,0]/((inputs[...,2]+10e-10)))
        # theta_[inputs[...,2]<0] += np.pi
        # theta_[theta_>np.pi] -= 2*np.pi
        # sphere = torch.cat([theta_.unsqueeze(-1),phi_.unsqueeze(-1),1/(dis_to_center.unsqueeze(-1)+ 10e-10)],-1)
        if self.embed_fn_fine is not None:
            #inputs = torch.cat([inputs,self.embed_fn_fine(torch.cat([inputs /(dis_to_center.unsqueeze(-1)+ 10e-10),1 /(dis_to_center.unsqueeze(-1)+ 10e-10)], dim=-1))],-1)
            inputs = self.embed_fn_fine(inputs)

        x = inputs
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, inputs], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.activation(x)
        return torch.cat([x[:, :1] / self.scale, x[:, 1:]], dim=-1)

    def sdf(self, x):
        return self.forward(x)[:, :1]

    def sdf_hidden_appearance(self, x):
        return self.forward(x)

    def gradient(self, x):
        x.requires_grad_(True)
        result = self.sdf_hidden_appearance(x)
        y = result[:, :1]
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients.unsqueeze(1), result

    def gradient_inference(self, x):
        x.requires_grad_(True)
        result = self.sdf_hidden_appearance(x)
        y = result[:, :1]
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            only_inputs=True)[0]
        return gradients.unsqueeze(1), result

class COLORNetwork(nn.Module):
    def __init__(self,
                 D=8,
                 W=256,
                 d_in=3,
                 d_in_view=3,
                 multires=0,
                 multires_view=0,
                 output_ch=4,
                 skips=[4],
                 use_viewdirs=False):
        super(COLORNetwork, self).__init__()
        self.D = D
        self.W = W
        self.d_in = d_in
        self.d_in_view = d_in_view
        self.input_ch = 3
        self.input_ch_view = 3
        self.embed_fn = None
        self.embed_fn_view = None
        #ipdb.set_trace()
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=6)
            self.embed_fn = embed_fn
            self.input_ch = input_ch

        if multires_view > 0:
            embed_fn_view, input_ch_view = get_embedder(multires_view, input_dims=3)
            self.embed_fn_view = embed_fn_view
            self.input_ch_view = input_ch_view

        self.skips = skips
        self.use_viewdirs = use_viewdirs

        self.pts_linears = nn.ModuleList(
            [nn.Linear(self.input_ch+256, W)] +
            [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_ch, W) for i in range(D - 1)])

        ### Implementation according to the official code release
        ### (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(self.input_ch_view + W, W // 2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            #self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W // 2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, input_pts, input_views, feature_vectors):
        #ipdb.set_trace()
        with torch.no_grad():
            dis_to_center = torch.linalg.norm(input_pts, ord=2, dim=-1).clip(10e-4,100)
            sin_phi = input_pts[...,1]/dis_to_center
            cos_theta_ = input_pts[...,0]/dis_to_center#/torch.sqrt(1 - sin_phi**2 + 10e-6)
            cos_theta = cos_theta_ /torch.sqrt(1 - sin_phi**2 + 10e-6)
            sin_theta_ = input_pts[...,2]/dis_to_center#/torch.sqrt(1 - sin_phi**2 + 10e-6)
            sin_theta = sin_theta_ /torch.sqrt(1 - sin_phi**2 + 10e-6)


            if self.embed_fn is not None:
                #input_pts = self.embed_fn(input_pts/16)
                #input_pts = self.embed_fn(sphere)
                #input_pts = self.embed_fn(torch.cat([input_pts/(dis_to_center.unsqueeze(-1)+ 10e-10),1/(dis_to_center.unsqueeze(-1)+ 10e-10)],-1))
                input_pts = self.embed_fn(torch.cat([sin_theta_.unsqueeze(-1),cos_theta_.unsqueeze(-1),sin_phi.unsqueeze(-1),1/(dis_to_center.unsqueeze(-1)),sin_theta.unsqueeze(-1),cos_theta.unsqueeze(-1),],-1))
            if self.embed_fn_view is not None:
                input_views = self.embed_fn_view(input_views)
            
        h = torch.cat([input_pts, feature_vectors], dim=-1)
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)


        feature = self.feature_linear(h)
        h = torch.cat([feature, input_views], -1)

        for i, l in enumerate(self.views_linears):
            h = self.views_linears[i](h)
            h = F.relu(h)

        rgb = self.rgb_linear(h)
        return rgb

class SingleVarianceNetwork(nn.Module):
    def __init__(self, init_val):
        super(SingleVarianceNetwork, self).__init__()
        self.register_parameter('variance', nn.Parameter(torch.tensor(init_val)))

    def forward(self, x):
        return torch.ones([len(x), 1]) * torch.exp(self.variance * 10.0)
