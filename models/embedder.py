import torch
import torch.nn as nn
import numpy as np
import ipdb

# Positional encoding embedding. Code was taken from https://github.com/bmild/nerf.
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        #d = 4
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        
        if False:
            dis_to_center = torch.linalg.norm(inputs, ord=2, dim=-1)
            #phi_ = torch.arcsin((inputs[...,1]+10e-8)/(dis_to_center+10e-8))
            phi_ = torch.arctan(inputs[...,1]/(torch.sqrt(inputs[...,0]**2 + inputs[...,2]**2)+10e-10))
            theta_ =  torch.arctan(inputs[...,0]/((inputs[...,2]+10e-10)))
            theta_[inputs[...,2]<0] += np.pi
            theta_[theta_>np.pi] -= 2*np.pi
            sphere = torch.cat([theta_.unsqueeze(-1),phi_.unsqueeze(-1),1/(dis_to_center.unsqueeze(-1)+ 10e-10)],-1)
            #print(torch.abs(dis_to_center*torch.cos(phi_)*torch.cos(theta_) - inputs[...,2]).max(), theta_.max(),phi_.max())
            results = torch.cat([inputs]+[fn(sphere) for fn in self.embed_fns], -1)
            # if torch.sum(torch.isnan(results)) > 0:
            #     torch.sum(torch.isnan(phi_))
            #     ipdb.set_trace()
        if False:
            #ipdb.set_trace()
            dis_to_center = torch.linalg.norm(inputs, ord=2, dim=-1, keepdim=True)
            pts_color = torch.cat([inputs /dis_to_center, 1/dis_to_center], dim=-1) 
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, input_dims=3):
    embed_kwargs = {
        'include_input': True,
        'input_dims': input_dims,
        'max_freq_log2': multires-1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    def embed(x, eo=embedder_obj): return eo.embed(x)
    return embed, embedder_obj.out_dim
