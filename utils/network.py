import numpy as np
import pickle
import torch
import torch.nn as nn
import argparse

from torch import DeviceObjType, nn, optim
from torch.nn import parameter
import torch.nn.functional as F
import open3d as o3d
import time
import os
from torch.nn.modules.linear import Linear
import tqdm
import MinkowskiEngine as ME

from torch.autograd import Function
from gdn_3d import GDN3d, IGDN3d

SEED2 = np.load('SEED3.npy')
SEED4_Gauss = np.load('SEED4_Gaussian.npy')
seed_ptr = 0

# differentiable rounding function
class BypassRound(Function):
    @staticmethod
    def forward(ctx, inputs):
        return torch.round(inputs)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class BypassRound32(Function):
    @staticmethod
    def forward(ctx, inputs):
        return torch.round(inputs*32)/32

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class BypassRound16(Function):
    @staticmethod
    def forward(ctx, inputs):
        return torch.round(inputs*16)/16

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

bypass_round = BypassRound.apply
bypass_round32 = BypassRound32.apply
bypass_round16 = BypassRound16.apply

class LowerBound(Function):
    """
    Low_bound make the numerical calculation close to the bound
    """
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y*torch.ones_like(x))
        x = torch.clamp(x, min=y)
        return x

    @staticmethod
    def backward(ctx, g):
        x, y = ctx.saved_tensors
        grad1 = g.clone()
        pass_through_if = torch.logical_or(x >= y, g < 0)
        t = pass_through_if
        return grad1*t, None

class UpperBound(Function):
    """
    Low_bound make the numerical calculation close to the bound
    """
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y*torch.ones_like(x))
        x = torch.clamp(x, max=y)
        return x

    @staticmethod
    def backward(ctx, g):
        x, y = ctx.saved_tensors
        grad1 = g.clone()
        pass_through_if = torch.logical_or(x <= y, g > 0)
        t = pass_through_if
        return grad1*t, None

lower_bound = LowerBound.apply
upper_bound = UpperBound.apply

class FCLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(FCLayer, self).__init__()
        self.transform = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU()
        )
    def forward(self, x):
        return self.transform(x)    

class CDFModel(nn.Module):
    def __init__(self, init_scale=10):
        super(CDFModel, self).__init__()
        self.init_scale = init_scale
        self.filters = [1,3,3,3,1]
        self.matrices = []
        self.biases = []
        self.factors = []
        self.softplus = nn.Softplus()
        scale = self.init_scale ** (1 / (len(self.filters) + 1))
        for i in range(len(self.filters) - 1):
            init = np.log(np.expm1(1 / scale / self.filters[i + 1]))
            self.matrices.append(nn.parameter.Parameter(torch.ones((self.filters[i], self.filters[i+1]), dtype=torch.float32, requires_grad=True)*scale))
            self.biases.append(nn.parameter.Parameter(torch.rand((self.filters[i+1]), dtype=torch.float32, requires_grad=True)-0.5))

            self.register_parameter(f'matrix_{i+1}', self.matrices[-1])
            self.register_parameter(f'bias_{i+1}', self.biases[-1])

            if i < len(self.filters) - 2:
                self.factors.append(nn.parameter.Parameter(torch.zeros(self.filters[i+1], dtype=torch.float32, requires_grad=True)))
                self.register_parameter(f'factor_{i+1}', self.factors[-1])

    def forward(self, x):
        # x of shape [Nxhxw, 1]
        y = x
        for i in range(len(self.filters) - 1):
            mat = self.softplus(self.matrices[i])
            y = torch.matmul(y, mat)
            y = y + self.biases[i]
            if i < len(self.filters) - 2:
                factor = torch.tanh(self.factors[i])
                y += factor * torch.tanh(y)
        return y

class GaussianModel(nn.Module):
  def __init__(self, qp=1):
    super(GaussianModel, self).__init__()
    self.m_normal_dist = torch.distributions.normal.Normal(0., 1.)
    self.qp = qp

  def _cumulative(self, inputs, stds, mu):
    half = 0.5 * self.qp
    eps = 1e-6
    upper = (inputs - mu + half) / (stds)
    lower = (inputs - mu - half) / (stds)
    cdf_upper = self.m_normal_dist.cdf(upper)
    cdf_lower = self.m_normal_dist.cdf(lower)
    res = cdf_upper - cdf_lower
    return res
  
  def forward(self, inputs, hyper_sigma, hyper_mu):
    likelihood = self._cumulative(inputs, hyper_sigma, hyper_mu)
    likelihood_bound = 1e-8
    # likelihood = torch.clamp(likelihood, min=likelihood_bound)
    likelihood = lower_bound(likelihood, likelihood_bound)
    rates = -1 * torch.log(likelihood) / np.log(2)
    return rates.sum()

  def evaluate(self, inputs, hyper_sigma, hyper_mu):
    likelihood = self._cumulative(inputs, hyper_sigma, hyper_mu)
    likelihood_bound = 1e-8
    # likelihood = torch.clamp(likelihood, min=likelihood_bound)
    likelihood = lower_bound(likelihood, likelihood_bound)
    rates = -1 * torch.log(likelihood) / np.log(2)
    return rates.sum()

class LaplaceModel(nn.Module):
  def __init__(self, qp=1):
    super(LaplaceModel, self).__init__()
    self.m_normal_dist = torch.distributions.laplace.Laplace(0., 1.)
    self.qp = qp

  def _cumulative(self, inputs, stds, mu):
    half = 0.5 * self.qp
    eps = 1e-6
    upper = (inputs - mu + half) / (stds)
    lower = (inputs - mu - half) / (stds)
    cdf_upper = self.m_normal_dist.cdf(upper)
    cdf_lower = self.m_normal_dist.cdf(lower)
    res = cdf_upper - cdf_lower
    return res
  
  def forward(self, inputs, hyper_sigma, hyper_mu):
    likelihood = self._cumulative(inputs, hyper_sigma, hyper_mu)
    likelihood_bound = 1e-8
    # likelihood = torch.clamp(likelihood, min=likelihood_bound)
    likelihood = lower_bound(likelihood, likelihood_bound)
    rates = -1 * torch.log(likelihood) / np.log(2)
    return rates.sum()

  def evaluate(self, inputs, hyper_sigma, hyper_mu):
    likelihood = self._cumulative(inputs, hyper_sigma, hyper_mu)
    likelihood_bound = 1e-8
    # likelihood = torch.clamp(likelihood, min=likelihood_bound)
    likelihood = lower_bound(likelihood, likelihood_bound)
    rates = -1 * torch.log(likelihood) / np.log(2)
    return rates.sum()

class LikelihoodModel(nn.Module):
    def __init__(self, step_size=1):
        super(LikelihoodModel, self).__init__()
        self.cdf = CDFModel()
        self.step_size = step_size
        self.half_step_size = step_size / 2
    
    def forward(self, x):
        batch_x = torch.reshape(x, (-1, 1))
        N = batch_x.shape[0]
        lower = batch_x - self.half_step_size
        upper = batch_x + self.half_step_size
        logit_cdf_lower = self.cdf(lower)
        logit_cdf_upper = self.cdf(upper)
        sign = -torch.sign(logit_cdf_lower + logit_cdf_upper)
        sign = sign.detach()
        likelihood = torch.abs(torch.sigmoid(sign*logit_cdf_upper)-torch.sigmoid(sign*logit_cdf_lower))
        likelihood = lower_bound(likelihood, 1e-9)
        rates = -1 * torch.log(likelihood) / np.log(2)
        return rates.sum()
    
    def verbose_forward(self, x):
        batch_x = torch.reshape(x, (-1, 1))
        N = batch_x.shape[0]
        lower = batch_x - self.half_step_size
        upper = batch_x + self.half_step_size
        logit_cdf_lower = self.cdf(lower)
        logit_cdf_upper = self.cdf(upper)
        sign = -torch.sign(logit_cdf_lower + logit_cdf_upper)
        sign = sign.detach()
        likelihood = torch.abs(torch.sigmoid(sign*logit_cdf_upper)-torch.sigmoid(sign*logit_cdf_lower))
        likelihood = lower_bound(likelihood, 1e-9)
        rates = -1 * torch.log(likelihood) / np.log(2)
        return likelihood, logit_cdf_lower, logit_cdf_upper

class GMM2(nn.Module):
    def __init__(self, step_size=1):
        super(GMM2, self).__init__()
        self.m_normal_dist = torch.distributions.normal.Normal(0., 1.)
        self.step_size = step_size

    def _gauss_cumulative(self, inputs, stds, mu):
        half = 0.5 * self.step_size
        eps = 1e-6
        upper = (inputs - mu + half) / (stds)
        lower = (inputs - mu - half) / (stds)
        cdf_upper = self.m_normal_dist.cdf(upper)
        cdf_lower = self.m_normal_dist.cdf(lower)
        res = cdf_upper - cdf_lower
        return res
    
    def forward(self, inputs, hyper_sigmas, hyper_mus, p):
        likelihood0 = self._gauss_cumulative(inputs, hyper_sigmas[0], hyper_mus[0])
        likelihood1 = self._gauss_cumulative(inputs, hyper_sigmas[1], hyper_mus[1])
        likelihood = p * likelihood0 + (1-p) * likelihood1
        likelihood_bound = 1e-8
        likelihood = lower_bound(likelihood, likelihood_bound)
        rates = -1 * torch.log(likelihood) / np.log(2)
        return rates, likelihood

class GMMLikelihoodModel(nn.Module):
    def __init__(self, step_size=1):
        super(GMMLikelihoodModel, self).__init__()
        self.gmm = GMM2(step_size)
        sigma = torch.nn.Parameter(torch.ones(
            (2), dtype=torch.float32, requires_grad=True))
        self.register_parameter('sigma', sigma)

        mu = torch.nn.Parameter(torch.from_numpy(np.random.rand(2)-0.5).float(), requires_grad=True)
        self.register_parameter('mu', mu)
        
        p = torch.nn.Parameter(torch.zeros(
            (1), dtype=torch.float32, requires_grad=True))
        self.register_parameter('p', p)

        self.step_size = step_size
        self.half_step_size = step_size / 2
    
    def forward(self, x):
        batch_x = torch.reshape(x, (-1, 1))
        N = batch_x.shape[0]
        bits, lklh = self.gmm(batch_x, torch.abs(self.sigma), self.mu, F.sigmoid(self.p))
        return bits.sum()

class GaussianLikelihoodModel(nn.Module):
    def __init__(self, step_size=1):
        super(GaussianLikelihoodModel, self).__init__()
        self.gaussian_model = GaussianModel(step_size)
        sigma = torch.nn.Parameter(torch.ones(
            (1), dtype=torch.float32, requires_grad=True))
        self.register_parameter('sigma', sigma)

        mu = torch.nn.Parameter(torch.from_numpy(np.zeros(1)).float(), requires_grad=True)
        self.register_parameter('mu', mu)

        self.step_size = step_size
        self.half_step_size = step_size / 2
    
    def forward(self, x):
        batch_x = torch.reshape(x, (-1, 1))
        N = batch_x.shape[0]
        bits = self.gaussian_model(batch_x, torch.abs(self.sigma), self.mu)
        return bits.sum()

class StatGaussianLikelihoodModel(nn.Module):
    def __init__(self, step_size=1):
        super(StatGaussianLikelihoodModel, self).__init__()
        self.gaussian_model = GaussianModel(step_size)
        sigma = torch.nn.Parameter(torch.ones(
            (1), dtype=torch.float32, requires_grad=True))
        self.register_parameter('sigma', sigma)

        mu = torch.nn.Parameter(torch.from_numpy(np.zeros(1)).float(), requires_grad=True)
        self.register_parameter('mu', mu)

        self.step_size = step_size
        self.half_step_size = step_size / 2
    
    def forward(self, x):
        batch_x = torch.reshape(x, (-1, 1))
        N = batch_x.shape[0]
        with torch.no_grad():
            scale, mu = torch.std_mean(batch_x)
            self.sigma[...] = scale + 1e-6
            self.mu[...] = mu
        bits = self.gaussian_model(batch_x, torch.abs(self.sigma.detach()), self.mu.detach())
        return bits.sum()

class LaplaceLikelihoodModel(nn.Module):
    def __init__(self, step_size=1):
        super(LaplaceLikelihoodModel, self).__init__()
        self.model = LaplaceModel(step_size)
        sigma = torch.nn.Parameter(torch.ones(
            (1), dtype=torch.float32, requires_grad=True))
        self.register_parameter('sigma', sigma)

        mu = torch.nn.Parameter(torch.from_numpy(np.zeros(1)).float(), requires_grad=True)
        self.register_parameter('mu', mu)

        self.step_size = step_size
        self.half_step_size = step_size / 2
    
    def forward(self, x):
        batch_x = torch.reshape(x, (-1, 1))
        N = batch_x.shape[0]
        bits = self.model(batch_x, torch.abs(self.sigma), self.mu)
        return bits.sum()

class StatLaplaceLikelihoodModel(nn.Module):
    def __init__(self, step_size=1):
        super(StatLaplaceLikelihoodModel, self).__init__()
        self.model = LaplaceModel(step_size)
        sigma = torch.nn.Parameter(torch.ones(
            (1), dtype=torch.float32, requires_grad=True))
        self.register_parameter('sigma', sigma)

        mu = torch.nn.Parameter(torch.from_numpy(np.zeros(1)).float(), requires_grad=True)
        self.register_parameter('mu', mu)

        self.step_size = step_size
        self.half_step_size = step_size / 2
    
    def forward(self, x):
        batch_x = torch.reshape(x, (-1, 1))
        N = batch_x.shape[0]
        with torch.no_grad():
            scale, mu = torch.std_mean(batch_x)
            self.sigma[...] = scale + 1e-6
            self.mu[...] = mu
        bits = self.model(batch_x, torch.abs(self.sigma.detach()), self.mu.detach())
        return bits.sum()

BOUND = 1/4

def _calculate_fan_in_and_fan_out(tensor):
    dimensions = tensor.dim()
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")

    num_input_fmaps = tensor.size(1)
    num_output_fmaps = tensor.size(0)
    receptive_field_size = 1
    if tensor.dim() > 2:
        receptive_field_size = tensor[0][0].numel()
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out

def get_kaiming_init_from_seed(w, seed, nonlinearity='relu'):
    fan_in, _ = _calculate_fan_in_and_fan_out(w)
    if nonlinearity == 'relu':
        gain = np.sqrt(2.0)
    else:
        raise NotImplementedError
    std = gain / np.sqrt(fan_in)
    bound = np.sqrt(3.0) * std
    return (seed - 0.5) * 2 * bound

class QFCLayer(nn.Module):
    def __init__(self, in_dim, out_dim, relu=True, iQ=16, SEED=None):
        super(QFCLayer, self).__init__()
        w = nn.parameter.Parameter(torch.zeros((in_dim, out_dim)).float(), requires_grad=True)
        b = nn.parameter.Parameter(torch.zeros(out_dim).float(), requires_grad=True)

        off1 = in_dim * out_dim
        w_seed = torch.from_numpy(SEED[:off1].reshape(w.shape)).float()
        w_init = get_kaiming_init_from_seed(w, w_seed)
        self.register_buffer('w_init', w_init)

        off2 = out_dim + off1
        b_seed = torch.from_numpy(SEED[off1:off2].reshape(b.shape)).float()
        init_bound = 1/np.sqrt(in_dim)
        b_init = (b_seed - 0.5) * 2 * init_bound
        self.register_buffer('b_init', b_init)

        self.register_parameter('w', w)
        self.register_parameter('b', b)

        if relu:
            self.relu = nn.ReLU()
        else:
            self.relu = torch.nn.Identity()
        self.Q = 1/iQ
        self.offset = off2
        
    def forward(self, x, q):
        # k_rounded = bypass_round32(self.kernel)
        # w = self.w + self.w_init
        b = self.b + self.b_init
        w = self.w
        if q == 1:
            k_rounded = w + (torch.rand_like(w)-0.5)*self.Q
        elif q == 2:
            # k_rounded = torch.round(w * 16) / 16
            k_rounded = bypass_round16(w)
        else:
            k_rounded = w
        # k_rounded = lower_bound(k_rounded, -BOUND)
        # k_rounded = upper_bound(k_rounded, BOUND)
        k_rounded = k_rounded + self.w_init
        y = torch.matmul(x, k_rounded)
        y = y + b
        y = self.relu(y)
        return y

class FCLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(FCLayer, self).__init__()
        self.transform = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU()
        )
    def forward(self, x):
        return self.transform(x)

class NeRFPosEmbLinear(nn.Module):

    def __init__(self, in_dim, out_dim, angular=False, no_linear=False, cat_input=False, use_seed=False):
        super().__init__()
        assert out_dim % (2 * in_dim) == 0, "dimension must be dividable"
        L = out_dim // 2 // in_dim
        if use_seed:
            seed = np.load('SEED.npy')
            emb = torch.tensor(seed[:L]).float()
        else:
            emb = torch.exp((torch.arange(L, dtype=torch.float)+32) * np.log(2.))
        if not angular:
            emb = emb * np.pi

        self.emb = nn.Parameter(emb, requires_grad=False)
        self.angular = angular
        self.linear = nn.Linear(out_dim, out_dim) if not no_linear else None
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.cat_input = cat_input

    def forward(self, x):
        assert x.size(-1) == self.in_dim, "size must match"
        sizes = x.size() 
        inputs = x.clone()

        if self.angular:
            x = torch.acos(x.clamp(-1 + 1e-6, 1 - 1e-6))
        x = x.unsqueeze(-1) @ self.emb.unsqueeze(0)
        x = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
        x = x.view(*sizes[:-1], self.out_dim)
        if self.linear is not None:
            x = self.linear(x)
        if self.cat_input:
            x = torch.cat([x, inputs], -1)
        return x

    def extra_repr(self) -> str:
        outstr = 'Sinusoidal (in={}, out={}, angular={})'.format(
            self.in_dim, self.out_dim, self.angular)
        if self.cat_input:
            outstr = 'Cat({}, {})'.format(outstr, self.in_dim)
        return outstr

def make_layer(block, block_layers, channels):
    """make stacked InceptionResNet layers.
    """
    layers = []
    for i in range(block_layers):
        layers.append(block(channels=channels))
        
    return torch.nn.Sequential(*layers)


class InceptionResNetDense(torch.nn.Module):
    """Inception Residual Network
    """
    
    def __init__(self, channels):
        super().__init__()
        self.conv0_0 = nn.Conv3d(
            in_channels=channels,
            out_channels=channels//4,
            kernel_size= 3,
            stride=1,
            bias=True,
            padding=1)
        self.conv0_1 = nn.Conv3d(
            in_channels=channels//4,
            out_channels=channels//2,
            kernel_size= 3,
            stride=1,
            bias=True,
            padding=1)
        self.conv1_0 = nn.Conv3d(
            in_channels=channels,
            out_channels=channels//4,
            kernel_size= 1,
            stride=1,
            bias=True,
            padding=0)
        self.conv1_1 = nn.Conv3d(
            in_channels=channels//4,
            out_channels=channels//4,
            kernel_size= 3,
            stride=1,
            bias=True,
            padding=1)
        self.conv1_2 = nn.Conv3d(
            in_channels=channels//4,
            out_channels=channels//2,
            kernel_size= 1,
            stride=1,
            bias=True,
            padding=0)

        self.relu = nn.ReLU()
        
    def forward(self, x):
        out0 = self.conv0_1(self.relu(self.conv0_0(x)))
        out1 = self.conv1_2(self.relu(self.conv1_1(self.relu(self.conv1_0(x)))))
        out = torch.cat([out0, out1], 1) + x

        return out

class QConvTranspose3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, bias, padding=0, output_padding=0, iQ=16, SEED=None, groups=1, zero_bias=False):
        super(QConvTranspose3d, self).__init__()
        assert in_channels % groups == 0
        assert out_channels % groups == 0
        self.padding = padding
        self.output_padding = output_padding
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.bias = bias
        eff_out_ch = out_channels // groups
        self.groups = groups
        
        kernel = nn.parameter.Parameter(torch.zeros(
                (in_channels, eff_out_ch,
                kernel_size, kernel_size, kernel_size)
            ),
            requires_grad=True
        )
        b = nn.parameter.Parameter(torch.zeros((out_channels)).float(), requires_grad=True)

        off1 = in_channels * eff_out_ch * (kernel_size**3)
        kernel_seed = torch.from_numpy(SEED[:off1].reshape(kernel.shape)).float()
        kernel_init = get_kaiming_init_from_seed(kernel, kernel_seed)
        self.register_buffer('kernel_init', kernel_init)

        off2 = out_channels + off1
        b_seed = torch.from_numpy(SEED[off1:off2].reshape(b.shape)).float()
        init_bound = 1/np.sqrt(in_channels)
        if zero_bias:
            b_init = torch.zeros_like(b_seed)
        else:
            b_init = (b_seed - 0.5) * 2 * init_bound
        self.register_buffer('b_init', b_init)

        self.register_parameter('kernel', kernel)
        self.register_parameter('b', b)
        self.Q = 1/iQ
        self.offset = off2

    def forward(self, x, q):
        # k_rounded = bypass_round32(self.kernel)
        # kernel = self.kernel + self.kernel_init
        kernel = self.kernel
        b = self.b + self.b_init
        if q == 1:
            k_rounded = kernel + (torch.rand_like(kernel)-0.5)*self.Q
        elif q == 2:
            # k_rounded = torch.round(kernel * 16) / 16
            k_rounded = bypass_round16(kernel)
        else:
            k_rounded = kernel
        # k_rounded = lower_bound(k_rounded, -BOUND)
        # k_rounded = upper_bound(k_rounded, BOUND)
        k_rounded = k_rounded + self.kernel_init
        out = F.conv_transpose3d(x, k_rounded, b, self.stride, self.padding, self.output_padding, groups=self.groups)
        return out

class QConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, bias, padding=0, iQ=16, SEED=None, groups=1, zero_bias=False):
        super(QConv3d, self).__init__()
        assert in_channels % groups == 0
        self.padding = padding
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.bias = bias
        self.groups = groups
        eff_in_ch = in_channels // groups
        
        kernel = nn.parameter.Parameter(torch.zeros(
                (out_channels, eff_in_ch,
                kernel_size, kernel_size, kernel_size)
            ),
            requires_grad=True
        )
        b = nn.parameter.Parameter(torch.zeros((out_channels)).float(), requires_grad=True)

        off1 = eff_in_ch * out_channels * (kernel_size**3)
        kernel_seed = torch.from_numpy(SEED[:off1].reshape(kernel.shape)).float()
        kernel_init = get_kaiming_init_from_seed(kernel, kernel_seed)
        self.register_buffer('kernel_init', kernel_init)
        self.register_parameter('kernel', kernel)

        if self.bias:
            off2 = out_channels + off1
            b_seed = torch.from_numpy(SEED[off1:off2].reshape(b.shape)).float()
            init_bound = 1/np.sqrt(eff_in_ch)

            if zero_bias:
                b_init = torch.zeros_like(b_seed)
            else:
                b_init = (b_seed - 0.5) * 2 * init_bound

            self.register_buffer('b_init', b_init)
            self.register_parameter('b', b)
        else:
            off2 = off1
        
        self.Q = 1/iQ
        self.offset = off2

    def forward(self, x, q):
        # k_rounded = bypass_round32(self.kernel)
        # kernel = self.kernel + self.kernel_init
        kernel = self.kernel
        if self.bias:
            b = self.b + self.b_init
        else:
            b = None
        if q == 1:
            k_rounded = kernel + (torch.rand_like(kernel)-0.5)*self.Q
        elif q == 2:
            # k_rounded = torch.round(kernel * 16) / 16
            k_rounded = bypass_round16(kernel)
        else:
            k_rounded = kernel
        # k_rounded = lower_bound(k_rounded, -BOUND)
        # k_rounded = upper_bound(k_rounded, BOUND)
        k_rounded = k_rounded + self.kernel_init
        out = F.conv3d(x, k_rounded, b, self.stride, self.padding, groups=self.groups)
        return out

class IConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, bias, padding=0, SEED=None, groups=1, zero_bias=False):
        super(IConv3d, self).__init__()
        assert in_channels % groups == 0
        self.padding = padding
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.bias = bias
        self.groups = groups
        eff_in_ch = in_channels // groups
        
        kernel = nn.parameter.Parameter(torch.zeros(
                (out_channels, eff_in_ch,
                kernel_size, kernel_size, kernel_size)
            ),
            requires_grad=True
        )
        b = nn.parameter.Parameter(torch.zeros((out_channels)).float(), requires_grad=True)

        off1 = eff_in_ch * out_channels * (kernel_size**3)
        kernel_seed = torch.from_numpy(SEED[:off1].reshape(kernel.shape)).float()
        kernel_init = get_kaiming_init_from_seed(kernel, kernel_seed)
        self.register_buffer('kernel_init', kernel_init)
        self.register_parameter('kernel', kernel)

        if self.bias:
            off2 = out_channels + off1
            b_seed = torch.from_numpy(SEED[off1:off2].reshape(b.shape)).float()
            init_bound = 1/np.sqrt(eff_in_ch)
            b_init = (b_seed - 0.5) * 2 * init_bound
            
            if zero_bias:
                b_init = torch.zeros_like(b_seed)
            else:
                b_init = (b_seed - 0.5) * 2 * init_bound
            
            self.register_buffer('b_init', b_init)
            self.register_parameter('b', b)
        else:
            off2 = off1
        
        self.offset = off2

    def forward(self, x):
        if self.bias:
            b = self.b + self.b_init
        else:
            b = None
        kernel = self.kernel + self.kernel_init
        out = F.conv3d(x, kernel, b, self.stride, self.padding, groups=self.groups)
        return out

class QDSepConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, bias, activation, padding=0, iQ=16, SEED=None, inner_channels=None):
        super(QDSepConv3d, self).__init__()
        if inner_channels is None:
            inner_channels = in_channels
        self.filter = QConv3d(in_channels, inner_channels, kernel_size, stride, bias, padding, iQ, SEED, groups=in_channels)
        off1 = self.filter.offset
        self.mixer = QConv3d(inner_channels, out_channels, 1, 1, True, 0, iQ, SEED[off1:], groups=1)
        self.offset = off1 + self.mixer.offset
        self.activation = activation
    def forward(self, x, q):
        out = self.activation(self.filter(x, q))
        out = self.mixer(out, q)
        return out
    def get_q_params(self):
        return [self.filter.kernel, self.mixer.kernel]
        
class QDSepConvTransposed3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, bias, activation, padding=0, output_padding=0, iQ=16, SEED=None, inner_channels=None):
        super(QDSepConvTransposed3d, self).__init__()
        if inner_channels is None:
            inner_channels = in_channels
        self.filter = QConvTranspose3d(in_channels, inner_channels, kernel_size, stride, bias, padding, output_padding, iQ, SEED, groups=in_channels)
        off1 = self.filter.offset
        self.mixer = QConv3d(inner_channels, out_channels, 1, 1, True, 0, iQ, SEED[off1:], groups=1)
        self.offset = off1 + self.mixer.offset
        self.activation = activation
    def forward(self, x, q):
        out = self.activation(self.filter(x, q))
        out = self.mixer(out, q)
        return out
    def get_q_params(self):
        return [self.filter.kernel, self.mixer.kernel]

class CompShapeEmb(nn.Module):
    def __init__(self, nLeaves, out_dim, prob_model='Gaussian', iQ=1):
        super(CompShapeEmb, self).__init__()
        global SEED2
        global seed_ptr

        emb = nn.parameter.Parameter(torch.ones((nLeaves,16,2,2,2), dtype=torch.float32), requires_grad=True)
        print('Embedding shape: ', emb.shape)
        self.register_parameter('emb', emb)
        ch = 16 # 1a_2 1a_5
        print('Bottleneck channel: ', ch)
        # ch = 2 # 1a_3 test
        
        self.h_analysis_2 = nn.Conv3d(
            in_channels=16,
            out_channels=ch,
            kernel_size=3,
            stride=1,
            bias=True,
            padding=1
        )
        self.gdn_2 = GDN3d(ch)
        print('Bottleneck shape: ', (ch, 2, 2, 2))

        h1_sigma = torch.nn.Parameter(torch.ones(
            (1, ch, 2, 2, 2), dtype=torch.float32, requires_grad=True))
        self.register_parameter('h1_sigma', h1_sigma)

        h1_mu = torch.nn.Parameter(torch.zeros(
            (1, ch, 2, 2, 2), dtype=torch.float32, requires_grad=True))
        self.register_buffer('h1_mu', h1_mu)
        if prob_model == 'Gaussian':
            self.gauss_model = GaussianModel(iQ)
        elif prob_model == 'Laplacian':
            self.gauss_model = LaplaceModel(iQ)
        self.h_synthesis_1 = QConvTranspose3d(ch, 8, 5, 2, True, padding=2, output_padding=1, SEED=SEED2[seed_ptr:])
        seed_ptr += self.h_synthesis_1.offset
        self.igdn_1 = IGDN3d(8)

        self.h_synthesis_2 = QConvTranspose3d(8, out_dim, 5, 2, True, padding=2, output_padding=1, SEED=SEED2[seed_ptr:])
        seed_ptr += self.h_synthesis_2.offset
        # self.igdn_2 = IGDN3d(out_dim)
        self.igdn_2 = nn.ReLU()
        if iQ == 1:
            self.bypass_round = bypass_round
        elif iQ == 16:
            self.bypass_round = bypass_round16
        self.Q = 1 / iQ
    
    def forward(self, idx, q):
        x = self.emb[idx]

        # h_x = self.gdn_1(self.h_analysis_1(x))
        # print(h_x.shape)
        h_x = self.gdn_2(self.h_analysis_2(x))
        # print(h_x.shape)
        
        noise = torch.rand_like(h_x) - 0.5
        h_x_noisy = h_x + noise * self.Q
        h_x_rounded = self.bypass_round(h_x)

        if q == 1:
            h_x_form = h_x_noisy
        elif q == 2:
            h_x_form = h_x_rounded
        elif q == 0:
            h_x_form = h_x_noisy
        else:
            raise NotImplementedError
        
        h_likelihood = self.gauss_model(h_x_form, torch.abs(self.h1_sigma), self.h1_mu)

        h_y = self.igdn_2(self.h_synthesis_2(self.igdn_1(self.h_synthesis_1(h_x_rounded, q)), q))

        bits_list = [h_likelihood, torch.tensor(0).float()]

        if torch.any(torch.isnan(h_likelihood)):
            print('problems with bits')
            import IPython
            IPython.embed()
            raise ValueError()

        if torch.any(torch.isnan(x)):
            print('problems with x_form')
            import IPython
            IPython.embed()
            raise ValueError()

        return h_y, bits_list
    
    def eval(self, idx, q):
        x = self.emb[idx]

        # h_x = self.gdn_1(self.h_analysis_1(x))
        # print(h_x.shape)
        h_x = self.gdn_2(self.h_analysis_2(x))
        # print(h_x.shape)
        
        noise = torch.rand_like(h_x) - 0.5
        h_x_noisy = h_x + noise
        h_x_rounded = bypass_round(h_x)

        if q == 1:
            h_x_form = h_x_noisy
        elif q == 2:
            h_x_form = h_x_rounded
        elif q == 0:
            h_x_form = h_x_noisy
        else:
            raise NotImplementedError
        
        h_likelihood = self.gauss_model(h_x_form, torch.abs(self.h1_sigma), self.h1_mu)

        h_y = self.igdn_2(self.h_synthesis_2(self.igdn_1(self.h_synthesis_1(h_x_rounded, q)), q))

        bits_list = [h_likelihood, torch.tensor(0).float()]

        if torch.any(torch.isnan(h_likelihood)):
            print('problems with bits')
            import IPython
            IPython.embed()
            raise ValueError()

        if torch.any(torch.isnan(x)):
            print('problems with x_form')
            import IPython
            IPython.embed()
            raise ValueError()

        return h_y, bits_list, h_x_rounded
    
    def get_q_params(self):
        params = []
        params.append(self.h_synthesis_1.kernel) 
        params.append(self.h_synthesis_2.kernel)
        return params
    
    def get_all_latent(self):
        h_x = self.gdn_2(self.h_analysis_2(self.emb))
        # h_x_rounded = torch.round(h_x)
        return h_x

class RootlessShapeEmb(nn.Module):
    def __init__(self, out_dim, prob_model='Gaussian', iQ=1):
        super(RootlessShapeEmb, self).__init__()
        global SEED2
        global seed_ptr

        ch = 8 # 1a_2 1a_5
        print('Bottleneck channel: ', ch)
        # ch = 2 # 1a_3 test
        
        self.h_analysis_2 = IConv3d(
            in_channels=8,
            out_channels=ch,
            kernel_size=3,
            stride=1,
            bias=True,
            padding=1,
            SEED=SEED2[seed_ptr:]
        )
        seed_ptr += self.h_analysis_2.offset
        self.gdn_2 = GDN3d(ch)
        print('Bottleneck shape: ', (ch, 2, 2, 2))

        h1_sigma = torch.nn.Parameter(torch.ones(
            (1, ch, 2, 2, 2), dtype=torch.float32, requires_grad=True))
        self.register_parameter('h1_sigma', h1_sigma)

        h1_mu = torch.nn.Parameter(torch.zeros(
            (1, ch, 2, 2, 2), dtype=torch.float32, requires_grad=True))
        self.register_buffer('h1_mu', h1_mu)
        if prob_model == 'Gaussian':
            self.gauss_model = GaussianModel(iQ)
        elif prob_model == 'Laplacian':
            self.gauss_model = LaplaceModel(iQ)
        self.h_synthesis_1 = QConvTranspose3d(ch, 8, 5, 2, True, padding=2, output_padding=1, SEED=SEED2[seed_ptr:])
        seed_ptr += self.h_synthesis_1.offset
        self.igdn_1 = IGDN3d(8)

        self.h_synthesis_2 = QConvTranspose3d(8, out_dim, 5, 2, True, padding=2, output_padding=1, SEED=SEED2[seed_ptr:])
        seed_ptr += self.h_synthesis_2.offset
        # self.igdn_2 = IGDN3d(out_dim)
        self.igdn_2 = nn.ReLU()
        if iQ == 1:
            self.bypass_round = bypass_round
        elif iQ == 16:
            self.bypass_round = bypass_round16
        self.Q = 1 / iQ
    
    def forward(self, x, q):
        h_x = self.gdn_2(self.h_analysis_2(x))
        
        noise = torch.rand_like(h_x) - 0.5
        h_x_noisy = h_x + noise * self.Q
        h_x_rounded = self.bypass_round(h_x)

        if q == 1:
            h_x_form = h_x_noisy
        elif q == 2:
            h_x_form = h_x_rounded
        elif q == 0:
            h_x_form = h_x_noisy
        else:
            raise NotImplementedError
        
        h_likelihood = self.gauss_model(h_x_form, torch.abs(self.h1_sigma), self.h1_mu)

        h_y = self.igdn_2(self.h_synthesis_2(self.igdn_1(self.h_synthesis_1(h_x_rounded, q)), q))

        bits_list = [h_likelihood, torch.tensor(0).float()]

        if torch.any(torch.isnan(h_likelihood)):
            print('problems with bits')
            import IPython
            IPython.embed()
            raise ValueError()

        if torch.any(torch.isnan(x)):
            print('problems with x_form')
            import IPython
            IPython.embed()
            raise ValueError()

        return h_y, bits_list

    def get_q_params(self):
        params = []
        params.append(self.h_synthesis_1.kernel) 
        params.append(self.h_synthesis_2.kernel)
        return params

    def get_all_latent_and_bits(self, x):
        h_x = self.gdn_2(self.h_analysis_2(x))
        h_x_rounded = self.bypass_round(h_x)
        h_likelihood = self.gauss_model(h_x_rounded, torch.abs(self.h1_sigma), self.h1_mu)
        bits_list = [h_likelihood, torch.tensor(0).float()]
        return h_x_rounded, bits_list

class GridRootlessShapeEmb(nn.Module):
    def __init__(self, out_dim, prob_model='Gaussian', iQ=1):
        super(GridRootlessShapeEmb, self).__init__()
        global SEED2
        global seed_ptr

        ch = 8 # 1a_2 1a_5
        print('Bottleneck channel: ', ch)
        # ch = 2 # 1a_3 test
        
        self.h_analysis_2 = IConv3d(
            in_channels=8,
            out_channels=ch,
            kernel_size=1,
            stride=1,
            bias=True,
            padding=0,
            SEED=SEED2[seed_ptr:]
        )
        seed_ptr += self.h_analysis_2.offset
        self.gdn_2 = GDN3d(ch)
        print('Bottleneck shape: ', (ch, 2, 2, 2))

        h1_sigma = torch.nn.Parameter(torch.ones(
            (1, ch, 1, 1, 1), dtype=torch.float32, requires_grad=True))
        self.register_parameter('h1_sigma', h1_sigma)

        h1_mu = torch.nn.Parameter(torch.zeros(
            (1, ch, 1, 1, 1), dtype=torch.float32, requires_grad=True))
        self.register_buffer('h1_mu', h1_mu)
        if prob_model == 'Gaussian':
            self.gauss_model = GaussianModel(iQ)
        elif prob_model == 'Laplacian':
            self.gauss_model = LaplaceModel(iQ)
        self.h_synthesis_1 = QConvTranspose3d(ch, 8, 5, 2, True, padding=2, output_padding=1, SEED=SEED2[seed_ptr:])
        seed_ptr += self.h_synthesis_1.offset
        self.igdn_1 = IGDN3d(8)

        self.h_synthesis_2 = QConvTranspose3d(8, out_dim, 5, 2, True, padding=2, output_padding=1, SEED=SEED2[seed_ptr:])
        seed_ptr += self.h_synthesis_2.offset
        # self.igdn_2 = IGDN3d(out_dim)
        self.igdn_2 = nn.ReLU()
        if iQ == 1:
            self.bypass_round = bypass_round
        elif iQ == 16:
            self.bypass_round = bypass_round16
        self.Q = 1 / iQ
    
    def forward(self, x, q):
        h_x = self.gdn_2(self.h_analysis_2(x))
        
        noise = torch.rand_like(h_x) - 0.5
        h_x_noisy = h_x + noise * self.Q
        h_x_rounded = self.bypass_round(h_x)

        if q == 1:
            h_x_form = h_x_noisy
        elif q == 2:
            h_x_form = h_x_rounded
        elif q == 0:
            h_x_form = h_x_noisy
        else:
            raise NotImplementedError
        
        h_likelihood = self.gauss_model(h_x_form, torch.abs(self.h1_sigma), self.h1_mu)

        h_y = self.igdn_2(self.h_synthesis_2(self.igdn_1(self.h_synthesis_1(h_x_rounded, q)), q))

        bits_list = [h_likelihood, torch.tensor(0).float()]

        if torch.any(torch.isnan(h_likelihood)):
            print('problems with bits')
            import IPython
            IPython.embed()
            raise ValueError()

        if torch.any(torch.isnan(x)):
            print('problems with x_form')
            import IPython
            IPython.embed()
            raise ValueError()

        return h_y, bits_list

    def get_q_params(self):
        params = []
        params.append(self.h_synthesis_1.kernel) 
        params.append(self.h_synthesis_2.kernel)
        return params

    def get_all_latent_and_bits(self, x):
        h_x = self.gdn_2(self.h_analysis_2(x))
        h_x_rounded = self.bypass_round(h_x)
        h_likelihood = self.gauss_model(h_x_rounded, torch.abs(self.h1_sigma), self.h1_mu)
        bits_list = [h_likelihood, torch.tensor(0).float()]
        return h_x_rounded, bits_list

class MixtureCompShapeEmb(nn.Module):
    def __init__(self, nLeaves, out_dim, prob_model='Gaussian'):
        super(MixtureCompShapeEmb, self).__init__()
        global SEED2
        global seed_ptr

        ch = 4 # 1a_2 1a_5
        print('Bottleneck channel: ', ch)
        
        emb = nn.parameter.Parameter(torch.ones((nLeaves,16,2,2,2), dtype=torch.float32), requires_grad=True)

        emb_off = nLeaves * ch * 2 * 2 * 2
        emb_init = torch.from_numpy(SEED2[seed_ptr:seed_ptr+emb_off]).float().reshape((nLeaves,ch,2,2,2))
        seed_ptr = seed_ptr + emb_off
        self.register_buffer('emb_init', emb_init)
        
        print('Embedding shape: ', emb.shape)
        self.register_parameter('emb', emb)
        # ch = 2 # 1a_3 test
        
        self.h_analysis_2 = nn.Conv3d(
            in_channels=16,
            out_channels=ch,
            kernel_size=3,
            stride=1,
            bias=True,
            padding=1
        )
        self.gdn_2 = GDN3d(ch)
        print('Bottleneck shape: ', (ch, 2, 2, 2))

        h1_sigma = torch.nn.Parameter(torch.ones(
            (1, ch, 2, 2, 2), dtype=torch.float32, requires_grad=True))
        self.register_parameter('h1_sigma', h1_sigma)

        h1_mu = torch.nn.Parameter(torch.zeros(
            (1, ch, 2, 2, 2), dtype=torch.float32, requires_grad=True))
        self.register_buffer('h1_mu', h1_mu)
        if prob_model == 'Gaussian':
            self.gauss_model = GaussianModel()
        elif prob_model == 'Laplacian':
            self.gauss_model = LaplaceModel()
        self.h_synthesis_1 = QConvTranspose3d(ch, 8, 5, 2, True, padding=2, output_padding=1, SEED=SEED2[seed_ptr:])
        seed_ptr += self.h_synthesis_1.offset
        self.igdn_1 = IGDN3d(8)

        self.h_synthesis_2 = QConvTranspose3d(8, out_dim, 5, 2, True, padding=2, output_padding=1, SEED=SEED2[seed_ptr:])
        seed_ptr += self.h_synthesis_2.offset
        # self.igdn_2 = IGDN3d(out_dim)
        self.igdn_2 = nn.ReLU()
    
    def forward(self, idx, q):
        x = self.emb[idx]

        # h_x = self.gdn_1(self.h_analysis_1(x))
        # print(h_x.shape)
        h_x = self.gdn_2(self.h_analysis_2(x))
        # print(h_x.shape)
        
        noise = torch.rand_like(h_x) - 0.5
        h_x_noisy = h_x + noise
        h_x_rounded = bypass_round(h_x)

        if q == 1:
            h_x_form = h_x_noisy
        elif q == 2:
            h_x_form = h_x_rounded
        elif q == 0:
            h_x_form = h_x_noisy
        else:
            raise NotImplementedError
        
        h_likelihood = self.gauss_model(h_x_form, torch.abs(self.h1_sigma), self.h1_mu)

        h_y = self.igdn_2(self.h_synthesis_2(self.igdn_1(self.h_synthesis_1(h_x_rounded / 16 + self.emb_init[idx], q)), q))

        bits_list = [h_likelihood, torch.tensor(0).float()]

        if torch.any(torch.isnan(h_likelihood)):
            print('problems with bits')
            import IPython
            IPython.embed()
            raise ValueError()

        if torch.any(torch.isnan(x)):
            print('problems with x_form')
            import IPython
            IPython.embed()
            raise ValueError()

        return h_y, bits_list
    
    def eval(self, idx, q):
        x = self.emb[idx]

        # h_x = self.gdn_1(self.h_analysis_1(x))
        # print(h_x.shape)
        h_x = self.gdn_2(self.h_analysis_2(x))
        # print(h_x.shape)
        
        noise = torch.rand_like(h_x) - 0.5
        h_x_noisy = h_x + noise
        h_x_rounded = bypass_round(h_x)

        if q == 1:
            h_x_form = h_x_noisy
        elif q == 2:
            h_x_form = h_x_rounded
        elif q == 0:
            h_x_form = h_x_noisy
        else:
            raise NotImplementedError
        
        h_likelihood = self.gauss_model(h_x_form, torch.abs(self.h1_sigma), self.h1_mu)

        h_y = self.igdn_2(self.h_synthesis_2(self.igdn_1(self.h_synthesis_1(h_x_rounded, q)), q))

        bits_list = [h_likelihood, torch.tensor(0).float()]

        if torch.any(torch.isnan(h_likelihood)):
            print('problems with bits')
            import IPython
            IPython.embed()
            raise ValueError()

        if torch.any(torch.isnan(x)):
            print('problems with x_form')
            import IPython
            IPython.embed()
            raise ValueError()

        return h_y, bits_list, h_x_rounded
    
    def get_q_params(self):
        params = []
        params.append(self.h_synthesis_1.kernel) 
        params.append(self.h_synthesis_2.kernel)
        return params

class StatCompShapeEmb(nn.Module):
    def __init__(self, nLeaves, out_dim, prob_model='Gaussian', iQ=16):
        super(StatCompShapeEmb, self).__init__()
        global SEED2
        global seed_ptr

        emb = nn.parameter.Parameter(torch.ones((nLeaves,16,2,2,2), dtype=torch.float32), requires_grad=True)
        print('Embedding shape: ', emb.shape)
        self.register_parameter('emb', emb)
        ch = 4
        print('Bottleneck channel: ', ch)
        
        self.h_analysis_2 = nn.Conv3d(
            in_channels=16,
            out_channels=ch,
            kernel_size=3,
            stride=1,
            bias=True,
            padding=1
        )
        self.act_ana_2 = NotActivation()
        print('Bottleneck shape: ', (ch, 2, 2, 2))

        h1_sigma = torch.nn.Parameter(torch.ones(
            (1, ch, 2, 2, 2), dtype=torch.float32, requires_grad=True))
        self.register_parameter('h1_sigma', h1_sigma)

        h1_mu = torch.nn.Parameter(torch.zeros(
            (1, ch, 2, 2, 2), dtype=torch.float32, requires_grad=True))
        self.register_parameter('h1_mu', h1_mu)

        if prob_model == 'Gaussian':
            self.gauss_model = GaussianModel(qp=1/16)
        elif prob_model == 'Laplacian':
            self.gauss_model = LaplaceModel(qp=1/16)

        self.h_synthesis_1 = QConvTranspose3d(ch, 8, 5, 2, True, padding=2, output_padding=1, SEED=SEED2[seed_ptr:])
        seed_ptr += self.h_synthesis_1.offset
        # self.act_syn_1 = IGDN3d(8)
        self.act_syn_1 = nn.GELU()

        self.h_synthesis_2 = QConvTranspose3d(8, out_dim, 5, 2, True, padding=2, output_padding=1, SEED=SEED2[seed_ptr:])
        seed_ptr += self.h_synthesis_2.offset
        # self.igdn_2 = IGDN3d(out_dim)
        self.act_syn_2 = nn.GELU()
        self.Q = 1 / iQ
        self.iQ = iQ
        if iQ == 16:
            self.bpround = bypass_round16
        elif iQ == 1:
            self.bpround = bypass_round
    
    def forward(self, idx, q):
        x = self.emb[idx]

        h_x = self.act_ana_2(self.h_analysis_2(x))
        
        noise = (torch.rand_like(h_x) - 0.5) * self.Q
        h_x_noisy = h_x + noise
        h_x_rounded = self.bpround(h_x)

        if q == 1:
            h_x_form = h_x_noisy
        elif q == 2:
            h_x_form = h_x_rounded
        elif q == 0:
            h_x_form = h_x_noisy
        else:
            raise NotImplementedError
        
        h_likelihood = self.gauss_model(h_x_form, torch.abs(self.h1_sigma.detach()), self.h1_mu.detach())

        h_y = self.act_syn_2(self.h_synthesis_2(self.act_syn_1(self.h_synthesis_1(h_x_rounded, q)), q))

        bits_list = [h_likelihood, torch.tensor(0).float()]

        if torch.any(torch.isnan(h_likelihood)):
            print('problems with bits')
            import IPython
            IPython.embed()
            raise ValueError()

        if torch.any(torch.isnan(x)):
            print('problems with x_form')
            import IPython
            IPython.embed()
            raise ValueError()

        return h_y, bits_list
    
    def eval(self, idx, q):
        x = self.emb[idx]

        h_x = self.act_ana_2(self.h_analysis_2(x))
        
        noise = torch.rand_like(h_x) - 0.5
        h_x_noisy = h_x + noise
        h_x_rounded = bypass_round(h_x)

        if q == 1:
            h_x_form = h_x_noisy
        elif q == 2:
            h_x_form = h_x_rounded
        elif q == 0:
            h_x_form = h_x_noisy
        else:
            raise NotImplementedError
        
        h_likelihood = self.gauss_model(h_x_form, torch.abs(self.h1_sigma.detach()), self.h1_mu.detach())

        h_y = self.act_syn_2(self.h_synthesis_2(self.act_syn_1(self.h_synthesis_1(h_x_rounded, q)), q))

        bits_list = [h_likelihood, torch.tensor(0).float()]

        if torch.any(torch.isnan(h_likelihood)):
            print('problems with bits')
            import IPython
            IPython.embed()
            raise ValueError()

        if torch.any(torch.isnan(x)):
            print('problems with x_form')
            import IPython
            IPython.embed()
            raise ValueError()

        return h_y, bits_list, h_x_rounded
    
    def get_q_params(self):
        params = []
        params.append(self.h_synthesis_1.kernel) 
        params.append(self.h_synthesis_2.kernel)
        return params

    def update_stats(self):
        with torch.no_grad():
            feat = self.h_analysis_2(self.emb)
            qfeat = torch.round(feat)
            scale, mu = torch.std_mean(qfeat, 0)
            self.h1_sigma[...] = scale + 1e-6
            self.h1_mu[...] = mu

class RandStatCompShapeEmb(nn.Module):
    def __init__(self, nLeaves, out_dim, prob_model='Gaussian', iQ=16):
        super(RandStatCompShapeEmb, self).__init__()
        global SEED2
        global seed_ptr

        emb = nn.parameter.Parameter(torch.zeros((nLeaves,16,2,2,2), dtype=torch.float32), requires_grad=True)
        emb_off = nLeaves * 16 * 2 * 2 * 2
        emb_init = torch.from_numpy(SEED2[seed_ptr:seed_ptr+emb_off]).float().reshape(emb.shape)
        seed_ptr = seed_ptr + emb_off
        self.register_buffer('emb_init', emb_init)
        print('Embedding shape: ', emb.shape)
        self.register_parameter('emb', emb)
        ch = 4 
        print('Bottleneck channel: ', ch)
        
        self.h_analysis_2 = nn.Conv3d(
            in_channels=16,
            out_channels=ch,
            kernel_size=3,
            stride=1,
            bias=True,
            padding=1
        )
        self.act_ana_2 = NotActivation()
        print('Bottleneck shape: ', (ch, 2, 2, 2))

        h1_sigma = torch.nn.Parameter(torch.ones(
            (1, ch, 2, 2, 2), dtype=torch.float32, requires_grad=True))
        self.register_parameter('h1_sigma', h1_sigma)

        h1_mu = torch.nn.Parameter(torch.zeros(
            (1, ch, 2, 2, 2), dtype=torch.float32, requires_grad=True))
        self.register_parameter('h1_mu', h1_mu)

        if prob_model == 'Gaussian':
            self.gauss_model = GaussianModel(qp=1/16)
        elif prob_model == 'Laplacian':
            self.gauss_model = LaplaceModel(qp=1/16)

        self.h_synthesis_1 = QConvTranspose3d(ch, 8, 5, 2, True, padding=2, output_padding=1, SEED=SEED2[seed_ptr:])
        seed_ptr += self.h_synthesis_1.offset
        # self.act_syn_1 = IGDN3d(8)
        self.act_syn_1 = nn.GELU()

        self.h_synthesis_2 = QConvTranspose3d(8, out_dim, 5, 2, True, padding=2, output_padding=1, SEED=SEED2[seed_ptr:])
        seed_ptr += self.h_synthesis_2.offset
        # self.igdn_2 = IGDN3d(out_dim)
        self.act_syn_2 = nn.GELU()
        self.Q = 1 / iQ
        self.iQ = iQ
        if iQ == 16:
            self.bpround = bypass_round16
        elif iQ == 1:
            self.bpround = bypass_round
    
    def forward(self, idx, q):
        x = self.emb[idx] + self.emb_init[idx]

        h_x = self.act_ana_2(self.h_analysis_2(x))
        
        noise = (torch.rand_like(h_x) - 0.5) * self.Q
        h_x_noisy = h_x + noise
        h_x_rounded = self.bpround(h_x)

        if q == 1:
            h_x_form = h_x_noisy
        elif q == 2:
            h_x_form = h_x_rounded
        elif q == 0:
            h_x_form = h_x_noisy
        else:
            raise NotImplementedError
        
        h_likelihood = self.gauss_model(h_x_form, torch.abs(self.h1_sigma.detach()), self.h1_mu.detach())

        h_y = self.act_syn_2(self.h_synthesis_2(self.act_syn_1(self.h_synthesis_1(h_x_rounded, q)), q))

        bits_list = [h_likelihood, torch.tensor(0).float()]

        if torch.any(torch.isnan(h_likelihood)):
            print('problems with bits')
            import IPython
            IPython.embed()
            raise ValueError()

        if torch.any(torch.isnan(x)):
            print('problems with x_form')
            import IPython
            IPython.embed()
            raise ValueError()

        return h_y, bits_list
    

    def eval(self, idx, q):
        x = self.emb[idx]

        h_x = self.act_ana_2(self.h_analysis_2(x))
        
        noise = torch.rand_like(h_x) - 0.5
        h_x_noisy = h_x + noise
        h_x_rounded = bypass_round(h_x)

        if q == 1:
            h_x_form = h_x_noisy
        elif q == 2:
            h_x_form = h_x_rounded
        elif q == 0:
            h_x_form = h_x_noisy
        else:
            raise NotImplementedError
        
        h_likelihood = self.gauss_model(h_x_form, torch.abs(self.h1_sigma.detach()), self.h1_mu.detach())

        h_y = self.act_syn_2(self.h_synthesis_2(self.act_syn_1(self.h_synthesis_1(h_x_rounded, q)), q))

        bits_list = [h_likelihood, torch.tensor(0).float()]

        if torch.any(torch.isnan(h_likelihood)):
            print('problems with bits')
            import IPython
            IPython.embed()
            raise ValueError()

        if torch.any(torch.isnan(x)):
            print('problems with x_form')
            import IPython
            IPython.embed()
            raise ValueError()

        return h_y, bits_list, h_x_rounded
    
    def get_q_params(self):
        params = []
        params.append(self.h_synthesis_1.kernel) 
        params.append(self.h_synthesis_2.kernel)
        return params

    def update_stats(self):
        with torch.no_grad():
            feat = self.h_analysis_2(self.emb)
            qfeat = torch.round(feat)
            scale, mu = torch.std_mean(qfeat, 0)
            self.h1_sigma[...] = scale + 1e-6
            self.h1_mu[...] = mu

class DirectRandStatCompShapeEmb(nn.Module):
    def __init__(self, nLeaves, out_dim, prob_model='Gaussian', iQ=16):
        super(DirectRandStatCompShapeEmb, self).__init__()
        global SEED2
        global seed_ptr
        ch = 4 
        print('Bottleneck channel: ', ch)
        print('Bottleneck shape: ', (ch, 2, 2, 2))

        emb = nn.parameter.Parameter(torch.zeros((nLeaves,ch,2,2,2), dtype=torch.float32), requires_grad=True)
        emb_off = nLeaves * ch * 2 * 2 * 2
        emb_init = torch.from_numpy(SEED2[seed_ptr:seed_ptr+emb_off]).float().reshape(emb.shape)
        seed_ptr = seed_ptr + emb_off
        self.register_buffer('emb_init', emb_init)
        print('Embedding shape: ', emb.shape)
        self.register_parameter('emb', emb)
        
        h1_sigma = torch.nn.Parameter(torch.ones(
            (1, ch, 2, 2, 2), dtype=torch.float32, requires_grad=True))
        self.register_parameter('h1_sigma', h1_sigma)

        h1_mu = torch.nn.Parameter(torch.zeros(
            (1, ch, 2, 2, 2), dtype=torch.float32, requires_grad=True))
        self.register_parameter('h1_mu', h1_mu)

        if prob_model == 'Gaussian':
            self.gauss_model = GaussianModel(qp=1/16)
        elif prob_model == 'Laplacian':
            self.gauss_model = LaplaceModel(qp=1/16)

        self.h_synthesis_1 = QConvTranspose3d(ch, 8, 5, 2, True, padding=2, output_padding=1, SEED=SEED2[seed_ptr:])
        seed_ptr += self.h_synthesis_1.offset
        # self.act_syn_1 = IGDN3d(8)
        self.act_syn_1 = nn.GELU()

        self.h_synthesis_2 = QConvTranspose3d(8, out_dim, 5, 2, True, padding=2, output_padding=1, SEED=SEED2[seed_ptr:])
        seed_ptr += self.h_synthesis_2.offset
        # self.igdn_2 = IGDN3d(out_dim)
        self.act_syn_2 = nn.GELU()
        self.Q = 1 / iQ
        self.iQ = iQ
        if iQ == 16:
            self.bpround = bypass_round16
        elif iQ == 1:
            self.bpround = bypass_round
    
    def forward(self, idx, q):
        x = self.emb[idx]

        h_x = x
        
        noise = (torch.rand_like(h_x) - 0.5) * self.Q
        h_x_noisy = h_x + noise
        h_x_rounded = self.bpround(h_x)

        if q == 1:
            h_x_form = h_x_noisy
        elif q == 2:
            h_x_form = h_x_rounded
        elif q == 0:
            h_x_form = h_x_noisy
        else:
            raise NotImplementedError
        
        h_likelihood = self.gauss_model(h_x_form, torch.abs(self.h1_sigma.detach()), self.h1_mu.detach())

        h_y = self.act_syn_2(self.h_synthesis_2(self.act_syn_1(self.h_synthesis_1(h_x_rounded + self.emb_init[idx], q)), q))

        bits_list = [h_likelihood, torch.tensor(0).float()]

        if torch.any(torch.isnan(h_likelihood)):
            print('problems with bits')
            import IPython
            IPython.embed()
            raise ValueError()

        if torch.any(torch.isnan(x)):
            print('problems with x_form')
            import IPython
            IPython.embed()
            raise ValueError()

        return h_y, bits_list
    

    def eval(self, idx, q):
        x = self.emb[idx]

        h_x = self.act_ana_2(self.h_analysis_2(x))
        
        noise = torch.rand_like(h_x) - 0.5
        h_x_noisy = h_x + noise
        h_x_rounded = bypass_round(h_x)

        if q == 1:
            h_x_form = h_x_noisy
        elif q == 2:
            h_x_form = h_x_rounded
        elif q == 0:
            h_x_form = h_x_noisy
        else:
            raise NotImplementedError
        
        h_likelihood = self.gauss_model(h_x_form, torch.abs(self.h1_sigma.detach()), self.h1_mu.detach())

        h_y = self.act_syn_2(self.h_synthesis_2(self.act_syn_1(self.h_synthesis_1(h_x_rounded, q)), q))

        bits_list = [h_likelihood, torch.tensor(0).float()]

        if torch.any(torch.isnan(h_likelihood)):
            print('problems with bits')
            import IPython
            IPython.embed()
            raise ValueError()

        if torch.any(torch.isnan(x)):
            print('problems with x_form')
            import IPython
            IPython.embed()
            raise ValueError()

        return h_y, bits_list, h_x_rounded
    
    def get_q_params(self):
        params = []
        params.append(self.h_synthesis_1.kernel) 
        params.append(self.h_synthesis_2.kernel)
        return params

    def update_stats(self):
        with torch.no_grad():
            # feat = self.h_analysis_2(self.emb)
            feat = self.emb
            qfeat = torch.round(feat)
            scale, mu = torch.std_mean(qfeat, 0)
            self.h1_sigma[...] = scale + 1e-6
            self.h1_mu[...] = mu

class RootlessRandInitShapeEmb(nn.Module):
    def __init__(self, out_dim, prob_model='Gaussian', iQ=1):
        super(RootlessRandInitShapeEmb, self).__init__()
        global SEED2
        global seed_ptr

        ch = 16 # 1a_2 1a_5
        print('Bottleneck channel: ', ch)
        # ch = 2 # 1a_3 test
        
        self.h_analysis_2 = IConv3d(
            in_channels=16,
            out_channels=ch,
            kernel_size=3,
            stride=1,
            bias=True,
            padding=1,
            SEED=SEED2[seed_ptr:]
        )
        seed_ptr += self.h_analysis_2.offset
        self.gdn_2 = GDN3d(ch)
        print('Bottleneck shape: ', (ch, 2, 2, 2))

        h1_sigma = torch.nn.Parameter(torch.ones(
            (1, ch, 2, 2, 2), dtype=torch.float32, requires_grad=True))
        self.register_parameter('h1_sigma', h1_sigma)

        h1_mu = torch.nn.Parameter(torch.zeros(
            (1, ch, 2, 2, 2), dtype=torch.float32, requires_grad=True))
        self.register_buffer('h1_mu', h1_mu)
        if prob_model == 'Gaussian':
            self.gauss_model = GaussianModel(iQ)
        elif prob_model == 'Laplacian':
            self.gauss_model = LaplaceModel(iQ)
        self.h_synthesis_1 = QConvTranspose3d(ch, 8, 5, 2, True, padding=2, output_padding=1, SEED=SEED2[seed_ptr:])
        seed_ptr += self.h_synthesis_1.offset
        self.igdn_1 = IGDN3d(8)

        self.h_synthesis_2 = QConvTranspose3d(8, out_dim, 5, 2, True, padding=2, output_padding=1, SEED=SEED2[seed_ptr:])
        seed_ptr += self.h_synthesis_2.offset
        # self.igdn_2 = IGDN3d(out_dim)
        self.igdn_2 = nn.ReLU()
        if iQ == 1:
            self.bypass_round = bypass_round
        elif iQ == 16:
            self.bypass_round = bypass_round16
        self.Q = 1 / iQ
    
    def forward(self, x, q, x_emb_init):
        h_x = self.gdn_2(self.h_analysis_2(x))
        
        noise = torch.rand_like(h_x) - 0.5
        h_x_noisy = h_x + noise * self.Q
        h_x_rounded = self.bypass_round(h_x)

        if q == 1:
            h_x_form = h_x_noisy
        elif q == 2:
            h_x_form = h_x_rounded
        elif q == 0:
            h_x_form = h_x_noisy
        else:
            raise NotImplementedError
        
        h_likelihood = self.gauss_model(h_x_form, torch.abs(self.h1_sigma), self.h1_mu)

        h_y = self.igdn_2(self.h_synthesis_2(self.igdn_1(self.h_synthesis_1(h_x_rounded+x_emb_init, q)), q))

        bits_list = [h_likelihood, torch.tensor(0).float()]

        if torch.any(torch.isnan(h_likelihood)):
            print('problems with bits')
            import IPython
            IPython.embed()
            raise ValueError()

        if torch.any(torch.isnan(x)):
            print('problems with x_form')
            import IPython
            IPython.embed()
            raise ValueError()

        return h_y, bits_list
    
    def eval(self, x, q, x_emb_init):
        h_x = self.gdn_2(self.h_analysis_2(x))
        
        noise = torch.rand_like(h_x) - 0.5
        h_x_noisy = h_x + noise * self.Q
        h_x_rounded = self.bypass_round(h_x)

        if q == 1:
            h_x_form = h_x_noisy
        elif q == 2:
            h_x_form = h_x_rounded
        elif q == 0:
            h_x_form = h_x_noisy
        else:
            raise NotImplementedError
        
        h_likelihood = self.gauss_model(h_x_form, torch.abs(self.h1_sigma), self.h1_mu)

        h_y = self.igdn_2(self.h_synthesis_2(self.igdn_1(self.h_synthesis_1(h_x_rounded+x_emb_init, q)), q))

        bits_list = [h_likelihood, torch.tensor(0).float()]

        if torch.any(torch.isnan(h_likelihood)):
            print('problems with bits')
            import IPython
            IPython.embed()
            raise ValueError()

        if torch.any(torch.isnan(x)):
            print('problems with x_form')
            import IPython
            IPython.embed()
            raise ValueError()

        return h_y, bits_list, h_x_rounded

    def get_q_params(self):
        params = []
        params.append(self.h_synthesis_1.kernel) 
        params.append(self.h_synthesis_2.kernel)
        return params

class StandaloneInitShapeEmb(nn.Module):
    def __init__(self, prob_model='Gaussian', iQ=1):
        super(StandaloneInitShapeEmb, self).__init__()
        global SEED2
        global seed_ptr

        ch = 4# 1a_2 1a_5
        print('Bottleneck channel: ', ch)
        # ch = 2 # 1a_3 test
        
        self.h_analysis_2 = IConv3d(
            in_channels=16,
            out_channels=ch,
            kernel_size=3,
            stride=1,
            bias=True,
            padding=1,
            SEED=SEED2[seed_ptr:]
        )
        seed_ptr += self.h_analysis_2.offset
        self.gdn_2 = GDN3d(ch)
        print('Bottleneck shape: ', (ch, 2, 2, 2))

        h1_sigma = torch.nn.Parameter(torch.ones(
            (1, ch, 2, 2, 2), dtype=torch.float32, requires_grad=True))
        self.register_parameter('h1_sigma', h1_sigma)

        h1_mu = torch.nn.Parameter(torch.zeros(
            (1, ch, 2, 2, 2), dtype=torch.float32, requires_grad=True))
        self.register_buffer('h1_mu', h1_mu)
        if prob_model == 'Gaussian':
            self.gauss_model = GaussianModel(iQ)
        elif prob_model == 'Laplacian':
            self.gauss_model = LaplaceModel(iQ)
        self.h_synthesis_1 = QConvTranspose3d(ch, 8, 5, 2, True, padding=2, output_padding=1, SEED=SEED2[seed_ptr:])
        seed_ptr += self.h_synthesis_1.offset
        self.igdn_1 = IGDN3d(8)

        if iQ == 1:
            self.bypass_round = bypass_round
        elif iQ == 16:
            self.bypass_round = bypass_round16
        self.Q = 1 / iQ
    
    def forward(self, x, q, x_emb_init):
        h_x = self.gdn_2(self.h_analysis_2(x))
        
        noise = torch.rand_like(h_x) - 0.5
        h_x_noisy = h_x + noise * self.Q
        h_x_rounded = self.bypass_round(h_x)

        if q == 1:
            h_x_form = h_x_noisy
        elif q == 2:
            h_x_form = h_x_rounded
        elif q == 0:
            h_x_form = h_x_noisy
        else:
            raise NotImplementedError
        
        h_likelihood = self.gauss_model(h_x_form, torch.abs(self.h1_sigma), self.h1_mu)

        h_y = self.igdn_1(self.h_synthesis_1(h_x_rounded+x_emb_init, q))

        bits_list = [h_likelihood, torch.tensor(0).float()]

        if torch.any(torch.isnan(h_likelihood)):
            print('problems with bits')
            import IPython
            IPython.embed()
            raise ValueError()

        if torch.any(torch.isnan(x)):
            print('problems with x_form')
            import IPython
            IPython.embed()
            raise ValueError()

        return h_y, bits_list
    
    def eval(self, x, q, x_emb_init):
        h_x = self.gdn_2(self.h_analysis_2(x))
        
        noise = torch.rand_like(h_x) - 0.5
        h_x_noisy = h_x + noise * self.Q
        h_x_rounded = self.bypass_round(h_x)

        if q == 1:
            h_x_form = h_x_noisy
        elif q == 2:
            h_x_form = h_x_rounded
        elif q == 0:
            h_x_form = h_x_noisy
        else:
            raise NotImplementedError
        
        h_likelihood = self.gauss_model(h_x_form, torch.abs(self.h1_sigma), self.h1_mu)

        h_y = self.igdn_1(self.h_synthesis_1(h_x_rounded+x_emb_init, q))

        bits_list = [h_likelihood, torch.tensor(0).float()]

        if torch.any(torch.isnan(h_likelihood)):
            print('problems with bits')
            import IPython
            IPython.embed()
            raise ValueError()

        if torch.any(torch.isnan(x)):
            print('problems with x_form')
            import IPython
            IPython.embed()
            raise ValueError()

        return h_y, bits_list, h_x_rounded

    def get_q_params(self):
        params = []
        params.append(self.h_synthesis_1.kernel)
        return params

class ZeroStartStandaloneInitShapeEmb(nn.Module):
    def __init__(self, prob_model='Gaussian', iQ=1):
        super(ZeroStartStandaloneInitShapeEmb, self).__init__()
        global SEED2
        global seed_ptr

        ch = 4# 1a_2 1a_5
        print('Bottleneck channel: ', ch)
        # ch = 2 # 1a_3 test
        
        self.h_analysis_2 = IConv3d(
            in_channels=16,
            out_channels=ch,
            kernel_size=3,
            stride=1,
            bias=True,
            padding=1,
            SEED=SEED2[seed_ptr:],
            zero_bias=True
        )
        seed_ptr += self.h_analysis_2.offset
        self.gdn_2 = GDN3d(ch)
        print('Bottleneck shape: ', (ch, 2, 2, 2))

        h1_sigma = torch.nn.Parameter(torch.ones(
            (1, ch, 2, 2, 2), dtype=torch.float32, requires_grad=True))
        self.register_parameter('h1_sigma', h1_sigma)

        h1_mu = torch.nn.Parameter(torch.zeros(
            (1, ch, 2, 2, 2), dtype=torch.float32, requires_grad=True))
        self.register_buffer('h1_mu', h1_mu)
        if prob_model == 'Gaussian':
            self.gauss_model = GaussianModel(iQ)
        elif prob_model == 'Laplacian':
            self.gauss_model = LaplaceModel(iQ)
        self.h_synthesis_1 = QConvTranspose3d(ch, 8, 5, 2, True, padding=2, output_padding=1, SEED=SEED2[seed_ptr:], zero_bias=True)
        seed_ptr += self.h_synthesis_1.offset
        self.igdn_1 = IGDN3d(8)

        if iQ == 1:
            self.bypass_round = bypass_round
        elif iQ == 16:
            self.bypass_round = bypass_round16
        self.Q = 1 / iQ
    
    def forward(self, x, q, x_emb_init):
        h_x = self.gdn_2(self.h_analysis_2(x))
        
        noise = torch.rand_like(h_x) - 0.5
        h_x_noisy = h_x + noise * self.Q
        h_x_rounded = self.bypass_round(h_x)

        if q == 1:
            h_x_form = h_x_noisy
        elif q == 2:
            h_x_form = h_x_rounded
        elif q == 0:
            h_x_form = h_x_noisy
        else:
            raise NotImplementedError
        
        h_likelihood = self.gauss_model(h_x_form, torch.abs(self.h1_sigma), self.h1_mu)

        h_y = self.igdn_1(self.h_synthesis_1(h_x_rounded+x_emb_init, q))

        bits_list = [h_likelihood, torch.tensor(0).float()]

        if torch.any(torch.isnan(h_likelihood)):
            print('problems with bits')
            import IPython
            IPython.embed()
            raise ValueError()

        if torch.any(torch.isnan(x)):
            print('problems with x_form')
            import IPython
            IPython.embed()
            raise ValueError()

        return h_y, bits_list
    
    def eval(self, x, q, x_emb_init):
        h_x = self.gdn_2(self.h_analysis_2(x))
        
        noise = torch.rand_like(h_x) - 0.5
        h_x_noisy = h_x + noise * self.Q
        h_x_rounded = self.bypass_round(h_x)

        if q == 1:
            h_x_form = h_x_noisy
        elif q == 2:
            h_x_form = h_x_rounded
        elif q == 0:
            h_x_form = h_x_noisy
        else:
            raise NotImplementedError
        
        h_likelihood = self.gauss_model(h_x_form, torch.abs(self.h1_sigma), self.h1_mu)

        h_y = self.igdn_1(self.h_synthesis_1(h_x_rounded+x_emb_init, q))

        bits_list = [h_likelihood, torch.tensor(0).float()]

        if torch.any(torch.isnan(h_likelihood)):
            print('problems with bits')
            import IPython
            IPython.embed()
            raise ValueError()

        if torch.any(torch.isnan(x)):
            print('problems with x_form')
            import IPython
            IPython.embed()
            raise ValueError()

        return h_y, bits_list, h_x_rounded

    def get_q_params(self):
        params = []
        params.append(self.h_synthesis_1.kernel)
        return params

class HyperCoreLikelihoodModel(nn.Module):
    def __init__(self, in_channels, step_size=1, useGDN=False, iQ=1):
        super(HyperCoreLikelihoodModel, self).__init__()
        if useGDN:
            self.activation = GDN3d(in_channels)
        else:
            self.activation = NotActivation()
        self.gaussian_model = GaussianModel(step_size)
        
        if iQ == 1:
            self.bypass_round = bypass_round
        elif iQ == 16:
            self.bypass_round = bypass_round16
        else:
            raise NotImplementedError
        self.Q = 1/iQ
    
    def forward(self, x, mu, sigma, q):
        h_x = self.activation(x)

        noise = torch.rand_like(h_x) - 0.5
        h_x_noisy = h_x + noise * self.Q
        h_x_rounded = self.bypass_round(h_x)

        if q == 1:
            h_x_form = h_x_noisy
        elif q == 2:
            h_x_form = h_x_rounded
        elif q == 0:
            h_x_form = h_x_noisy
        else:
            raise NotImplementedError
        
        h_likelihood = self.gaussian_model(h_x_form, torch.abs(sigma), mu)

        if torch.any(torch.isnan(h_likelihood)):
            print('problems with bits')
            import IPython
            IPython.embed()
            raise ValueError()

        if torch.any(torch.isnan(x)):
            print('problems with x_form')
            import IPython
            IPython.embed()
            raise ValueError()

        return h_x_rounded, h_likelihood

meta_map = {'shape_emb_module.h_synthesis_1.kernel_init': 'meta_net.vars.0',
'shape_emb_module.h_synthesis_1.b_init': 'meta_net.vars.1',
'shape_emb_module.h_synthesis_2.kernel_init': 'meta_net.vars.5',
'shape_emb_module.h_synthesis_2.b_init': 'meta_net.vars.6',
'conv0_cls.weight': 'meta_net.vars.7',
'conv0_cls.bias': 'meta_net.vars.8',
'up1.kernel_init': 'meta_net.vars.9',
'up1.b_init': 'meta_net.vars.10',
'conv1.kernel_init': 'meta_net.vars.11',
'conv1.b_init': 'meta_net.vars.12',
'conv1_cls.weight': 'meta_net.vars.13',
'conv1_cls.bias': 'meta_net.vars.14',
'up2.kernel_init': 'meta_net.vars.15',
'up2.b_init': 'meta_net.vars.16',
'conv2.kernel_init': 'meta_net.vars.17',
'conv2.b_init': 'meta_net.vars.18',
'conv2_cls.kernel_init': 'meta_net.vars.19',
'conv2_cls.b_init': 'meta_net.vars.20'}

init_map = {
    'shape_emb_module.h_analysis_2.kernel_init': 'shape_emb_module.h_analysis_2.kernel_init',
    'shape_emb_module.h_analysis_2.b_init': 'shape_emb_module.h_analysis_2.b_init',
    'shape_emb_module.h_synthesis_1.kernel_init': 'shape_emb_module.h_synthesis_1.kernel_init',
    'shape_emb_module.h_synthesis_1.b_init': 'shape_emb_module.h_synthesis_1.b_init',
    'shape_emb_module.h_synthesis_2.kernel_init': 'shape_emb_module.h_synthesis_2.kernel_init',
    'shape_emb_module.h_synthesis_2.b_init': 'shape_emb_module.h_synthesis_2.b_init',
    'up1.kernel_init': 'up1.kernel_init',
    'up1.b_init': 'up1.b_init',
    'conv1.kernel_init': 'conv1.kernel_init',
    'conv1.b_init': 'conv1.b_init',
    'up2.kernel_init': 'up2.kernel_init',
    'up2.b_init': 'up2.b_init',
    'conv2.kernel_init': 'conv2.kernel_init',
    'conv2.b_init': 'conv2.b_init',
    'conv2_cls.kernel_init': 'conv2_cls.kernel_init',
    'conv2_cls.b_init': 'conv2_cls.b_init',
    'conv0_cls.kernel_init': 'conv0_cls.weight',
    'conv0_cls.b_init': 'conv0_cls.bias',
    'conv1_cls.kernel_init': 'conv1_cls.weight',
    'conv1_cls.b_init': 'conv1_cls.bias',
    'likelihood_model.sigma': 'likelihood_model.sigma',
    'likelihood_model.mu': 'likelihood_model.mu',
    'shape_emb_module.h1_sigma': 'shape_emb_module.h1_sigma',
    'shape_emb_module.h1_mu': 'shape_emb_module.h1_mu',
    'shape_emb_module.gdn_2.beta': 'shape_emb_module.gdn_2.beta',
    'shape_emb_module.gdn_2.gamma': 'shape_emb_module.gdn_2.gamma',
    'shape_emb_module.gdn_2.pedestal': 'shape_emb_module.gdn_2.pedestal',
    'shape_emb_module.igdn_1.beta': 'shape_emb_module.igdn_1.beta',
    'shape_emb_module.igdn_1.gamma': 'shape_emb_module.igdn_1.gamma',
    'shape_emb_module.igdn_1.pedestal': 'shape_emb_module.igdn_1.pedestal',
    }

class CompNet(nn.Module):
    def __init__(self, args, prob_model, param_model):
        super(CompNet, self).__init__()
        global SEED2
        global seed_ptr
        # global prob_model
        # global param_model

        channels=[8,16,8,8]
        if args.stat_latent:
            print('Using stat model for latent repr')
            self.shape_emb_module = StatCompShapeEmb(nLeaves=nLeaves, out_dim=channels[1], prob_model=prob_model)
            print('Using only stat shape emb')

            # self.shape_emb_module = DirectRandStatCompShapeEmb(nLeaves=nLeaves, out_dim=channels[1], prob_model=prob_model)
            # print('Using direct randomized shape emb')

            # self.shape_emb_module = RandStatCompShapeEmb(nLeaves=nLeaves, out_dim=channels[1], prob_model=prob_model)
            # print('Using randomized shape emb')
        else:
            print('Using learned entropy model for latent repr')
            self.shape_emb_module = GridRootlessShapeEmb(out_dim=channels[1], prob_model=prob_model)

        self.up1 = QConvTranspose3d(
            in_channels=channels[1],
            out_channels=channels[2],
            kernel_size=5,
            stride=2,
            bias=True,
            SEED=SEED2[seed_ptr:])
        seed_ptr += self.up1.offset

        self.conv1 = QConv3d(
            in_channels=channels[2],
            out_channels=channels[2],
            kernel_size=4,
            stride=1,
            bias=True,
            padding=0,
            SEED=SEED2[seed_ptr:])
        seed_ptr += self.conv1.offset

        self.up2 = QConvTranspose3d(
            in_channels=channels[2],
            out_channels=channels[3],
            kernel_size=5,
            stride=2,
            bias=True,
            SEED=SEED2[seed_ptr:])
        seed_ptr += self.up2.offset

        self.conv2 = QConv3d(
            in_channels=channels[3],
            out_channels=channels[3],
            kernel_size=4,
            stride=1,
            bias=True,
            padding=0,
            SEED=SEED2[seed_ptr:])
        seed_ptr += self.conv2.offset

        self.conv2_cls = QConv3d(
            in_channels=channels[3],
            out_channels=1,
            kernel_size=3,
            stride=1,
            bias=True,
            padding=1,
            SEED=SEED2[seed_ptr:])
        seed_ptr += self.conv2_cls.offset
        
        self.conv1_cls = IConv3d(
            in_channels=channels[2],
            out_channels=1,
            kernel_size=3,
            stride=1,
            bias=True,
            padding=1,
            SEED=SEED2[seed_ptr:])
        seed_ptr += self.conv1_cls.offset
        
        self.conv0_cls = IConv3d(
            in_channels=channels[1],
            out_channels=1,
            kernel_size=3,
            stride=1,
            bias=True,
            padding=1,
            SEED=SEED2[seed_ptr:])
        seed_ptr += self.conv0_cls.offset
    
        # self.relu = nn.ReLU()
        self.relu = nn.GELU()
        self.sigmoid = nn.Sigmoid()

        # self.likelihood_model = LikelihoodModel(step_size=1/16)
        if param_model == 'Gaussian':
            if args.stat_net:
                self.likelihood_model = StatGaussianLikelihoodModel(step_size=1/16)
            else:
                self.likelihood_model = GaussianLikelihoodModel(step_size=1/16)
        elif param_model == 'Laplacian':
            if args.stat_net:
                self.likelihood_model = StatLaplaceLikelihoodModel(step_size=1/16)
            else:
                self.likelihood_model = LaplaceLikelihoodModel(step_size=1/16)
        # print('Using single Laplacian likelihood model')

    def forward(self, x, q, x_emb_init=None):
        out, bits_list = self.shape_emb_module(x, q)
        out_cls_0 = self.sigmoid(self.conv0_cls(out))
        # print(out.shape)
        # print(out_cls_0.shape)
        # out = self.block0(out)
        out = self.relu(self.up1(out, q))
        # print(out.shape)
        out = self.relu(self.conv1(out, q))
        # print(out.shape)
        out_cls_1 = self.sigmoid(self.conv1_cls(out))
        # print(out.shape)
        # print(out_cls_1.shape)
        # out = self.block1(out)
        out = self.relu(self.up2(out, q))
        # print(out.shape)
        out = self.relu(self.conv2(out, q))
        # print(out.shape)
        # print(out.shape)
        out = self.conv2_cls(out, q)
        # print(out.shape)
        out = self.sigmoid(out)
        # print(out.shape)
        # quit()

        out_cls_list = [out_cls_0, out_cls_1, out]

        if torch.any(torch.isnan(out)):
            import IPython
            IPython.embed()
            raise ValueError()

        params = self.get_q_params()
        # net_bits = []
        # lklhs = []
        # for p in params:
        #     a, b = self.likelihood_model(p + (torch.rand_like(p)-0.5)*1/16)
        #     net_bits.append(a.sum())
        #     lklhs.append(b)
        # net_bits = torch.stack([self.likelihood_model(p + (torch.rand_like(p)-0.5)*1/16) for p in params])
        net_bits = torch.stack([self.likelihood_model(bypass_round16(p)) for p in params])
        # net_bits = torch.stack(net_bits)
        return out, out_cls_list, bits_list, net_bits
    
    def get_bits(self, x):
        out, bits_list = self.shape_emb_module.get_all_latent_and_bits(x)
        params = self.get_q_params()
        net_bits = torch.stack([self.likelihood_model(bypass_round16(p)) for p in params])
        # net_bits = torch.stack(net_bits)
        return bits_list, net_bits
    
    def eval(self, x, q, x_emb_init):
        out, bits_list, h_x_rounded = self.shape_emb_module.eval(x, q, x_emb_init)
        out_cls_0 = self.sigmoid(self.conv0_cls(out))
        # print(out.shape)
        # print(out_cls_0.shape)
        # out = self.block0(out)
        
        out = self.relu(self.conv1(self.relu(self.up1(out, q)), q))
        out_cls_1 = self.sigmoid(self.conv1_cls(out))
        # print(out.shape)
        # print(out_cls_1.shape)
        # out = self.block1(out)
        
        out = self.relu(self.conv2(self.relu(self.up2(out, q)), q))
        # print(out.shape)
        out = self.conv2_cls(out, q)
        out = self.sigmoid(out)
        # print(out.shape)
        # quit()

        out_cls_list = [out_cls_0, out_cls_1, out]

        if torch.any(torch.isnan(out)):
            import IPython
            IPython.embed()
            raise ValueError()

        params = self.get_q_params()

        net_bits = torch.stack([self.likelihood_model(bypass_round16(p)) for p in params])
        # net_bits = torch.stack(net_bits)
        return out, out_cls_list, bits_list, h_x_rounded
    
    def get_continuous_params(self):
        params = []
        for i in self.pos_emb.parameters():
            params.append(i)
        for i in self.conv0_cls.parameters():
            params.append(i)
        for i in self.conv1_cls.parameters():
            params.append(i)
        for i in self.parameters():
            if len(i.shape) == 1:
                params.append(i)
        return params

    def get_q_params(self):
        params = []

        params += self.shape_emb_module.get_q_params()
        params.append(self.up1.kernel)
        params.append(self.conv1.kernel)
        params.append(self.up2.kernel)
        params.append(self.conv2.kernel)
        params.append(self.conv2_cls.kernel)

        return params
    
    def load_init(self, meta_dict, meta_map):
        nd = {}
        for k in meta_map:
            nd[k] = meta_dict[meta_map[k]]
        rt = self.load_state_dict(nd, strict=False)
        print(rt)

class CompDecodeNet(nn.Module):
    def __init__(self, args, ch, prob_model, param_model):
        super(CompDecodeNet, self).__init__()
        global SEED2
        global seed_ptr

        channels=[8,16,8,8]

        self.h_synthesis_1 = QConvTranspose3d(ch, channels[0], 5, 2, True, padding=2, output_padding=1, SEED=SEED2[seed_ptr:])
        seed_ptr += self.h_synthesis_1.offset
        self.igdn_1 = IGDN3d(channels[0])

        self.h_synthesis_2 = QConvTranspose3d(channels[0], channels[1], 5, 2, True, padding=2, output_padding=1, SEED=SEED2[seed_ptr:])
        seed_ptr += self.h_synthesis_2.offset
        
        self.up1 = QConvTranspose3d(
            in_channels=channels[1],
            out_channels=channels[2],
            kernel_size=5,
            stride=2,
            bias=True,
            SEED=SEED2[seed_ptr:])
        seed_ptr += self.up1.offset

        self.conv1 = QConv3d(
            in_channels=channels[2],
            out_channels=channels[2],
            kernel_size=4,
            stride=1,
            bias=True,
            padding=0,
            SEED=SEED2[seed_ptr:])
        seed_ptr += self.conv1.offset

        self.up2 = QConvTranspose3d(
            in_channels=channels[2],
            out_channels=channels[3],
            kernel_size=5,
            stride=2,
            bias=True,
            SEED=SEED2[seed_ptr:])
        seed_ptr += self.up2.offset

        self.conv2 = QConv3d(
            in_channels=channels[3],
            out_channels=channels[3],
            kernel_size=4,
            stride=1,
            bias=True,
            padding=0,
            SEED=SEED2[seed_ptr:])
        seed_ptr += self.conv2.offset

        self.conv2_cls = QConv3d(
            in_channels=channels[3],
            out_channels=1,
            kernel_size=3,
            stride=1,
            bias=True,
            padding=1,
            SEED=SEED2[seed_ptr:])
        seed_ptr += self.conv2_cls.offset
        
        self.conv1_cls = IConv3d(
            in_channels=channels[2],
            out_channels=1,
            kernel_size=3,
            stride=1,
            bias=True,
            padding=1,
            SEED=SEED2[seed_ptr:])
        seed_ptr += self.conv1_cls.offset
        
        self.conv0_cls = IConv3d(
            in_channels=channels[1],
            out_channels=1,
            kernel_size=3,
            stride=1,
            bias=True,
            padding=1,
            SEED=SEED2[seed_ptr:])
        seed_ptr += self.conv0_cls.offset
    
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        if param_model == 'Gaussian':
            if args.stat_net:
                self.likelihood_model = StatGaussianLikelihoodModel(step_size=1/16)
            else:
                self.likelihood_model = GaussianLikelihoodModel(step_size=1/16)
        elif param_model == 'Laplacian':
            if args.stat_net:
                self.likelihood_model = StatLaplaceLikelihoodModel(step_size=1/16)
            else:
                self.likelihood_model = LaplaceLikelihoodModel(step_size=1/16)

    def forward(self, x_tilde, q):

        out = self.relu(self.h_synthesis_2(self.igdn_1(self.h_synthesis_1(x_tilde, q)), q))

        out_cls_0 = self.sigmoid(self.conv0_cls(out))
        # print(out.shape)
        # print(out_cls_0.shape)
        out = self.relu(self.up1(out, q))
        # print(out.shape)
        out = self.relu(self.conv1(out, q))
        # print(out.shape)
        out_cls_1 = self.sigmoid(self.conv1_cls(out))
        # print(out.shape)
        # print(out_cls_1.shape)
        out = self.relu(self.up2(out, q))
        # print(out.shape)
        out = self.relu(self.conv2(out, q))
        # print(out.shape)
        # print(out.shape)
        out = self.conv2_cls(out, q)
        # print(out.shape)
        out = self.sigmoid(out)
        # print(out.shape)
        # quit()

        out_cls_list = [out_cls_0, out_cls_1, out]

        if torch.any(torch.isnan(out)):
            import IPython
            IPython.embed()
            raise ValueError()

        params = self.get_q_params()
        net_bits = torch.stack([self.likelihood_model(bypass_round16(p)) for p in params])
        return out, out_cls_list, net_bits
    
    def get_bits(self, x):
        params = self.get_q_params()
        net_bits = torch.stack([self.likelihood_model(bypass_round16(p)) for p in params])
        return net_bits
    
    def get_q_params(self):
        params = []

        params.append(self.h_synthesis_1.kernel)
        params.append(self.h_synthesis_2.kernel)
        params.append(self.up1.kernel)
        params.append(self.conv1.kernel)
        params.append(self.up2.kernel)
        params.append(self.conv2.kernel)
        params.append(self.conv2_cls.kernel)

        return params

class CompDecoder(nn.Module):
    def __init__(self, args, param_model, in_channels=32, useIGDN=False):
        super(CompDecoder, self).__init__()
        global SEED2
        global seed_ptr

        channels=[8,16,8,8]

        self.up0 = QConvTranspose3d(
            in_channels=in_channels,
            out_channels=channels[1],
            kernel_size=5,
            stride=2,
            padding=2,
            output_padding=1,
            bias=True,
            SEED=SEED2[seed_ptr:])
        seed_ptr += self.up0.offset

        self.conv0 = QConvTranspose3d(
            in_channels=channels[1],
            out_channels=channels[1],
            kernel_size=5,
            stride=2,
            padding=2,
            output_padding=1,
            bias=True,
            SEED=SEED2[seed_ptr:])
        seed_ptr += self.conv0.offset

        self.up1 = QConvTranspose3d(
            in_channels=channels[1],
            out_channels=channels[2],
            kernel_size=5,
            stride=2,
            bias=True,
            SEED=SEED2[seed_ptr:])
        seed_ptr += self.up1.offset

        self.conv1 = QConv3d(
            in_channels=channels[2],
            out_channels=channels[2],
            kernel_size=4,
            stride=1,
            bias=True,
            padding=0,
            SEED=SEED2[seed_ptr:])
        seed_ptr += self.conv1.offset

        self.up2 = QConvTranspose3d(
            in_channels=channels[2],
            out_channels=channels[3],
            kernel_size=5,
            stride=2,
            bias=True,
            SEED=SEED2[seed_ptr:])
        seed_ptr += self.up2.offset

        self.conv2 = QConv3d(
            in_channels=channels[3],
            out_channels=channels[3],
            kernel_size=4,
            stride=1,
            bias=True,
            padding=0,
            SEED=SEED2[seed_ptr:])
        seed_ptr += self.conv2.offset

        self.conv2_cls = QConv3d(
            in_channels=channels[3],
            out_channels=1,
            kernel_size=3,
            stride=1,
            bias=True,
            padding=1,
            SEED=SEED2[seed_ptr:])
        seed_ptr += self.conv2_cls.offset
        
        self.conv1_cls = IConv3d(
            in_channels=channels[2],
            out_channels=1,
            kernel_size=3,
            stride=1,
            bias=True,
            padding=1,
            SEED=SEED2[seed_ptr:])
        seed_ptr += self.conv1_cls.offset
        
        self.conv0_cls = IConv3d(
            in_channels=channels[1],
            out_channels=1,
            kernel_size=3,
            stride=1,
            bias=True,
            padding=1,
            SEED=SEED2[seed_ptr:])
        seed_ptr += self.conv0_cls.offset
    
        # self.relu = nn.ReLU()
        self.relu = nn.GELU()
        self.sigmoid = nn.Sigmoid()

        if param_model == 'Gaussian':
            if args.stat_net:
                self.likelihood_model = StatGaussianLikelihoodModel(step_size=1/16)
            else:
                self.likelihood_model = GaussianLikelihoodModel(step_size=1/16)
        elif param_model == 'Laplacian':
            if args.stat_net:
                self.likelihood_model = StatLaplaceLikelihoodModel(step_size=1/16)
            else:
                self.likelihood_model = LaplaceLikelihoodModel(step_size=1/16)
        
        if useIGDN:
            self.activation = IGDN3d(channels[1])
        else:
            self.activation = nn.ReLU()

    def forward(self, x, q):
        out = self.activation(self.up0(x, q))
        out = self.relu(self.conv0(out, q))
        out_cls_0 = self.sigmoid(self.conv0_cls(out))
        out = self.relu(self.up1(out, q))
        out = self.relu(self.conv1(out, q))
        out_cls_1 = self.sigmoid(self.conv1_cls(out))
        out = self.relu(self.up2(out, q))
        out = self.relu(self.conv2(out, q))
        out = self.conv2_cls(out, q)
        out = self.sigmoid(out)

        out_cls_list = [out_cls_0, out_cls_1, out]

        if torch.any(torch.isnan(out)):
            import IPython
            IPython.embed()
            raise ValueError()

        params = self.get_q_params()
        net_bits = torch.stack([self.likelihood_model(bypass_round16(p)) for p in params])
        return out, out_cls_list, net_bits

    def get_q_params(self):
        params = []
        
        params.append(self.up0.kernel)
        params.append(self.conv0.kernel)
        params.append(self.up1.kernel)
        params.append(self.conv1.kernel)
        params.append(self.up2.kernel)
        params.append(self.conv2.kernel)
        params.append(self.conv2_cls.kernel)

        return params

class StraightCompNet(nn.Module):
    def __init__(self, args, prob_model, param_model):
        super(StraightCompNet, self).__init__()
        global SEED2
        global seed_ptr
        # global prob_model
        # global param_model

        channels=[8,16,8,8]
        if args.stat_latent:
            raise NotImplementedError()
        else:
            print('Using learned entropy model for latent repr')
            self.shape_emb_module = GridRootlessShapeEmb(out_dim=channels[1], prob_model=prob_model)

        self.up1 = QConv3d(
            in_channels=channels[1],
            out_channels=channels[2],
            kernel_size=3,
            stride=1,
            bias=True,
            padding=1,
            SEED=SEED2[seed_ptr:])
        seed_ptr += self.up1.offset

        self.conv1 = QConv3d(
            in_channels=channels[2],
            out_channels=channels[2],
            kernel_size=3,
            stride=1,
            bias=True,
            padding=1,
            SEED=SEED2[seed_ptr:])
        seed_ptr += self.conv1.offset

        self.up2 = QConv3d(
            in_channels=channels[2],
            out_channels=channels[3],
            kernel_size=3,
            stride=1,
            bias=True,
            padding=1,
            SEED=SEED2[seed_ptr:])
        seed_ptr += self.up2.offset

        self.conv2 = QConv3d(
            in_channels=channels[3],
            out_channels=channels[3],
            kernel_size=3,
            stride=1,
            bias=True,
            padding=1,
            SEED=SEED2[seed_ptr:])
        seed_ptr += self.conv2.offset

        self.conv2_cls = QConv3d(
            in_channels=channels[3],
            out_channels=1,
            kernel_size=3,
            stride=1,
            bias=True,
            padding=1,
            SEED=SEED2[seed_ptr:])
        seed_ptr += self.conv2_cls.offset
        
        self.conv1_cls = IConv3d(
            in_channels=channels[2],
            out_channels=1,
            kernel_size=3,
            stride=1,
            bias=True,
            padding=1,
            SEED=SEED2[seed_ptr:])
        seed_ptr += self.conv1_cls.offset
        
        self.conv0_cls = IConv3d(
            in_channels=channels[1],
            out_channels=1,
            kernel_size=3,
            stride=1,
            bias=True,
            padding=1,
            SEED=SEED2[seed_ptr:])
        seed_ptr += self.conv0_cls.offset
    
        # self.relu = nn.ReLU()
        self.relu = nn.GELU()
        self.sigmoid = nn.Sigmoid()

        # self.likelihood_model = LikelihoodModel(step_size=1/16)
        if param_model == 'Gaussian':
            if args.stat_net:
                self.likelihood_model = StatGaussianLikelihoodModel(step_size=1/16)
            else:
                self.likelihood_model = GaussianLikelihoodModel(step_size=1/16)
        elif param_model == 'Laplacian':
            if args.stat_net:
                self.likelihood_model = StatLaplaceLikelihoodModel(step_size=1/16)
            else:
                self.likelihood_model = LaplaceLikelihoodModel(step_size=1/16)
        # print('Using single Laplacian likelihood model')

    def forward(self, x, q, x_emb_init=None):
        out, bits_list = self.shape_emb_module(x, q)
        out = F.interpolate(out, (32,32,32), mode='trilinear', align_corners=True)
        out_cls_0 = self.sigmoid(self.conv0_cls(out))
        out = self.relu(self.up1(out, q))
        out = self.relu(self.conv1(out, q))
        out_cls_1 = self.sigmoid(self.conv1_cls(out))
        out = self.relu(self.up2(out, q))
        out = self.relu(self.conv2(out, q))
        out = self.conv2_cls(out, q)
        out = self.sigmoid(out)
        out_cls_list = [out_cls_0, out_cls_1, out]

        if torch.any(torch.isnan(out)):
            import IPython
            IPython.embed()
            raise ValueError()

        params = self.get_q_params()
        net_bits = torch.stack([self.likelihood_model(bypass_round16(p)) for p in params])
        return out, out_cls_list, bits_list, net_bits
    
    def get_bits(self, x):
        out, bits_list = self.shape_emb_module.get_all_latent_and_bits(x)
        params = self.get_q_params()
        net_bits = torch.stack([self.likelihood_model(bypass_round16(p)) for p in params])
        # net_bits = torch.stack(net_bits)
        return bits_list, net_bits
    
    def eval(self, x, q, x_emb_init):
        out, bits_list, h_x_rounded = self.shape_emb_module.eval(x, q, x_emb_init)
        out_cls_0 = self.sigmoid(self.conv0_cls(out))
        # print(out.shape)
        # print(out_cls_0.shape)
        # out = self.block0(out)
        
        out = self.relu(self.conv1(self.relu(self.up1(out, q)), q))
        out_cls_1 = self.sigmoid(self.conv1_cls(out))
        # print(out.shape)
        # print(out_cls_1.shape)
        # out = self.block1(out)
        
        out = self.relu(self.conv2(self.relu(self.up2(out, q)), q))
        # print(out.shape)
        out = self.conv2_cls(out, q)
        out = self.sigmoid(out)
        # print(out.shape)
        # quit()

        out_cls_list = [out_cls_0, out_cls_1, out]

        if torch.any(torch.isnan(out)):
            import IPython
            IPython.embed()
            raise ValueError()

        params = self.get_q_params()

        net_bits = torch.stack([self.likelihood_model(bypass_round16(p)) for p in params])
        # net_bits = torch.stack(net_bits)
        return out, out_cls_list, bits_list, h_x_rounded
    
    def get_continuous_params(self):
        params = []
        for i in self.pos_emb.parameters():
            params.append(i)
        for i in self.conv0_cls.parameters():
            params.append(i)
        for i in self.conv1_cls.parameters():
            params.append(i)
        for i in self.parameters():
            if len(i.shape) == 1:
                params.append(i)
        return params

    def get_q_params(self):
        params = []

        params += self.shape_emb_module.get_q_params()
        params.append(self.up1.kernel)
        params.append(self.conv1.kernel)
        params.append(self.up2.kernel)
        params.append(self.conv2.kernel)
        params.append(self.conv2_cls.kernel)

        return params
    
    def load_init(self, meta_dict, meta_map):
        nd = {}
        for k in meta_map:
            nd[k] = meta_dict[meta_map[k]]
        rt = self.load_state_dict(nd, strict=False)
        print(rt)

class AddModuloCompNet(nn.Module):
    def __init__(self, args, prob_model, param_model):
        super(AddModuloCompNet, self).__init__()
        global SEED2
        global seed_ptr
        # global prob_model
        # global param_model

        channels=[8,16,8,8]
        switch = nn.parameter.Parameter(torch.ones((3)).float(), requires_grad=True)
        self.register_parameter('switch', switch)
        
        self.shape_emb_module = StandaloneInitShapeEmb(prob_model=prob_model)

        self.h_synthesis_2 = QConvTranspose3d(8, channels[1]//2, 5, 2, True, padding=2, output_padding=1, SEED=SEED2[seed_ptr:])
        seed_ptr += self.h_synthesis_2.offset

        self.up1 = QConvTranspose3d(
            in_channels=channels[1],
            out_channels=channels[2],
            kernel_size=5,
            stride=2,
            bias=True,
            SEED=SEED2[seed_ptr:])
        seed_ptr += self.up1.offset

        self.conv1 = QConv3d(
            in_channels=channels[2]+4,
            out_channels=channels[2],
            kernel_size=4,
            stride=1,
            bias=True,
            padding=0,
            SEED=SEED2[seed_ptr:])
        seed_ptr += self.conv1.offset

        self.up2 = QConvTranspose3d(
            in_channels=channels[2],
            out_channels=channels[3],
            kernel_size=5,
            stride=2,
            bias=True,
            SEED=SEED2[seed_ptr:])
        seed_ptr += self.up2.offset

        self.conv2 = QConv3d(
            in_channels=channels[3]+4,
            out_channels=channels[3],
            kernel_size=4,
            stride=1,
            bias=True,
            padding=0,
            SEED=SEED2[seed_ptr:])
        seed_ptr += self.conv2.offset

        self.conv2_cls = QConv3d(
            in_channels=channels[3],
            out_channels=1,
            kernel_size=3,
            stride=1,
            bias=True,
            padding=1,
            SEED=SEED2[seed_ptr:])
        seed_ptr += self.conv2_cls.offset
        
        self.conv1_cls = IConv3d(
            in_channels=channels[2],
            out_channels=1,
            kernel_size=3,
            stride=1,
            bias=True,
            padding=1,
            SEED=SEED2[seed_ptr:])
        seed_ptr += self.conv1_cls.offset
        
        self.conv0_cls = IConv3d(
            in_channels=channels[1],
            out_channels=1,
            kernel_size=3,
            stride=1,
            bias=True,
            padding=1,
            SEED=SEED2[seed_ptr:])
        seed_ptr += self.conv0_cls.offset
    
        # self.relu = nn.ReLU()
        self.relu = nn.GELU()
        self.sigmoid = nn.Sigmoid()

        # self.likelihood_model = LikelihoodModel(step_size=1/16)
        if param_model == 'Gaussian':
            if args.stat_net:
                self.likelihood_model = StatGaussianLikelihoodModel(step_size=1/16)
            else:
                self.likelihood_model = GaussianLikelihoodModel(step_size=1/16)
        elif param_model == 'Laplacian':
            if args.stat_net:
                self.likelihood_model = StatLaplaceLikelihoodModel(step_size=1/16)
            else:
                self.likelihood_model = LaplaceLikelihoodModel(step_size=1/16)
        # print('Using single Laplacian likelihood model')

    def forward(self, x, q, ys, x_emb_init=None):
        out, bits_list = self.shape_emb_module(x, q, x_emb_init)
        out = self.h_synthesis_2(out, q)

        out = self.relu(torch.cat([out, ys[0]*self.switch[0]], 1))
        out_cls_0 = self.sigmoid(self.conv0_cls(out))
        # print(out.shape)
        # print(out_cls_0.shape)
        # out = self.block0(out)
        out = self.relu(self.up1(out, q))
        out = torch.cat([out, ys[1]*self.switch[1]], 1)
        out = self.relu(self.conv1(out, q))
        out_cls_1 = self.sigmoid(self.conv1_cls(out))
        # print(out.shape)
        # print(out_cls_1.shape)
        # out = self.block1(out)
        out = self.relu(self.up2(out, q))
        out = torch.cat([out, ys[2]*self.switch[2]], 1)
        out = self.relu(self.conv2(out, q))
        # print(out.shape)
        out = self.conv2_cls(out, q)
        out = self.sigmoid(out)
        # print(out.shape)
        # quit()

        out_cls_list = [out_cls_0, out_cls_1, out]

        if torch.any(torch.isnan(out)):
            import IPython
            IPython.embed()
            raise ValueError()

        params = self.get_q_params()
        
        net_bits = torch.stack([self.likelihood_model(bypass_round16(p)) for p in params])
        # net_bits = torch.stack(net_bits)
        return out, out_cls_list, bits_list, net_bits
    
    def eval(self, x, q, x_emb_init):
        out, bits_list, h_x_rounded = self.shape_emb_module.eval(x, q, x_emb_init)
        out_cls_0 = self.sigmoid(self.conv0_cls(out))
        # print(out.shape)
        # print(out_cls_0.shape)
        # out = self.block0(out)
        
        out = self.relu(self.conv1(self.relu(self.up1(out, q)), q))
        out_cls_1 = self.sigmoid(self.conv1_cls(out))
        # print(out.shape)
        # print(out_cls_1.shape)
        # out = self.block1(out)
        
        out = self.relu(self.conv2(self.relu(self.up2(out, q)), q))
        # print(out.shape)
        out = self.conv2_cls(out, q)
        out = self.sigmoid(out)
        # print(out.shape)
        # quit()

        out_cls_list = [out_cls_0, out_cls_1, out]

        if torch.any(torch.isnan(out)):
            import IPython
            IPython.embed()
            raise ValueError()

        params = self.get_q_params()

        net_bits = torch.stack([self.likelihood_model(bypass_round16(p)) for p in params])
        # net_bits = torch.stack(net_bits)
        return out, out_cls_list, bits_list, h_x_rounded
    
    def get_continuous_params(self):
        params = []
        for i in self.pos_emb.parameters():
            params.append(i)
        for i in self.conv0_cls.parameters():
            params.append(i)
        for i in self.conv1_cls.parameters():
            params.append(i)
        for i in self.parameters():
            if len(i.shape) == 1:
                params.append(i)
        return params

    def get_q_params(self):
        params = []

        params += self.shape_emb_module.get_q_params()
        params.append(self.up1.kernel)
        params.append(self.conv1.kernel)
        params.append(self.up2.kernel)
        params.append(self.conv2.kernel)
        params.append(self.conv2_cls.kernel)

        return params
    
    def load_init(self, meta_dict, meta_map):
        nd = {}
        for k in meta_map:
            nd[k] = meta_dict[meta_map[k]]
        rt = self.load_state_dict(nd, strict=False)
        print(rt)

class AddModuloCompNetv2(nn.Module):
    # With only one concatenation
    def __init__(self, args, prob_model, param_model):
        super(AddModuloCompNetv2, self).__init__()
        global SEED2
        global seed_ptr
        # global prob_model
        # global param_model

        channels=[8,16,8,8]
        switch = nn.parameter.Parameter(torch.ones((3)).float(), requires_grad=True)
        self.register_parameter('switch', switch)
        
        self.shape_emb_module = StandaloneInitShapeEmb(prob_model=prob_model)

        self.h_synthesis_2 = QConvTranspose3d(8, channels[1]//2, 5, 2, True, padding=2, output_padding=1, SEED=SEED2[seed_ptr:])
        seed_ptr += self.h_synthesis_2.offset

        self.up1 = QConvTranspose3d(
            in_channels=channels[1],
            out_channels=channels[2],
            kernel_size=5,
            stride=2,
            bias=True,
            SEED=SEED2[seed_ptr:])
        seed_ptr += self.up1.offset

        self.conv1 = QConv3d(
            in_channels=channels[2],
            out_channels=channels[2],
            kernel_size=4,
            stride=1,
            bias=True,
            padding=0,
            SEED=SEED2[seed_ptr:])
        seed_ptr += self.conv1.offset

        self.up2 = QConvTranspose3d(
            in_channels=channels[2],
            out_channels=channels[3],
            kernel_size=5,
            stride=2,
            bias=True,
            SEED=SEED2[seed_ptr:])
        seed_ptr += self.up2.offset

        self.conv2 = QConv3d(
            in_channels=channels[3],
            out_channels=channels[3],
            kernel_size=4,
            stride=1,
            bias=True,
            padding=0,
            SEED=SEED2[seed_ptr:])
        seed_ptr += self.conv2.offset

        self.conv2_cls = QConv3d(
            in_channels=channels[3],
            out_channels=1,
            kernel_size=3,
            stride=1,
            bias=True,
            padding=1,
            SEED=SEED2[seed_ptr:])
        seed_ptr += self.conv2_cls.offset
        
        self.conv1_cls = IConv3d(
            in_channels=channels[2],
            out_channels=1,
            kernel_size=3,
            stride=1,
            bias=True,
            padding=1,
            SEED=SEED2[seed_ptr:])
        seed_ptr += self.conv1_cls.offset
        
        self.conv0_cls = IConv3d(
            in_channels=channels[1],
            out_channels=1,
            kernel_size=3,
            stride=1,
            bias=True,
            padding=1,
            SEED=SEED2[seed_ptr:])
        seed_ptr += self.conv0_cls.offset
    
        # self.relu = nn.ReLU()
        self.relu = nn.GELU()
        self.sigmoid = nn.Sigmoid()

        # self.likelihood_model = LikelihoodModel(step_size=1/16)
        if param_model == 'Gaussian':
            if args.stat_net:
                self.likelihood_model = StatGaussianLikelihoodModel(step_size=1/16)
            else:
                self.likelihood_model = GaussianLikelihoodModel(step_size=1/16)
        elif param_model == 'Laplacian':
            if args.stat_net:
                self.likelihood_model = StatLaplaceLikelihoodModel(step_size=1/16)
            else:
                self.likelihood_model = LaplaceLikelihoodModel(step_size=1/16)
        # print('Using single Laplacian likelihood model')

    def forward(self, x, q, ys, x_emb_init=None):
        out, bits_list = self.shape_emb_module(x, q, x_emb_init)
        out = self.h_synthesis_2(out, q)

        out = self.relu(torch.cat([out, ys[0]*self.switch[0]], 1))
        out_cls_0 = self.sigmoid(self.conv0_cls(out))
        # print(out.shape)
        # print(out_cls_0.shape)
        # out = self.block0(out)
        out = self.relu(self.up1(out, q))
        out = self.relu(self.conv1(out, q))
        out_cls_1 = self.sigmoid(self.conv1_cls(out))
        # print(out.shape)
        # print(out_cls_1.shape)
        # out = self.block1(out)
        out = self.relu(self.up2(out, q))
        out = self.relu(self.conv2(out, q))
        # print(out.shape)
        out = self.conv2_cls(out, q)
        out = self.sigmoid(out)
        # print(out.shape)
        # quit()

        out_cls_list = [out_cls_0, out_cls_1, out]

        if torch.any(torch.isnan(out)):
            import IPython
            IPython.embed()
            raise ValueError()

        params = self.get_q_params()
        
        net_bits = torch.stack([self.likelihood_model(bypass_round16(p)) for p in params])
        # net_bits = torch.stack(net_bits)
        return out, out_cls_list, bits_list, net_bits
    
    def eval(self, x, q, x_emb_init):
        out, bits_list, h_x_rounded = self.shape_emb_module.eval(x, q, x_emb_init)
        out_cls_0 = self.sigmoid(self.conv0_cls(out))
        # print(out.shape)
        # print(out_cls_0.shape)
        # out = self.block0(out)
        
        out = self.relu(self.conv1(self.relu(self.up1(out, q)), q))
        out_cls_1 = self.sigmoid(self.conv1_cls(out))
        # print(out.shape)
        # print(out_cls_1.shape)
        # out = self.block1(out)
        
        out = self.relu(self.conv2(self.relu(self.up2(out, q)), q))
        # print(out.shape)
        out = self.conv2_cls(out, q)
        out = self.sigmoid(out)
        # print(out.shape)
        # quit()

        out_cls_list = [out_cls_0, out_cls_1, out]

        if torch.any(torch.isnan(out)):
            import IPython
            IPython.embed()
            raise ValueError()

        params = self.get_q_params()

        net_bits = torch.stack([self.likelihood_model(bypass_round16(p)) for p in params])
        # net_bits = torch.stack(net_bits)
        return out, out_cls_list, bits_list, h_x_rounded
    
    def get_continuous_params(self):
        params = []
        for i in self.pos_emb.parameters():
            params.append(i)
        for i in self.conv0_cls.parameters():
            params.append(i)
        for i in self.conv1_cls.parameters():
            params.append(i)
        for i in self.parameters():
            if len(i.shape) == 1:
                params.append(i)
        return params

    def get_q_params(self):
        params = []

        params += self.shape_emb_module.get_q_params()
        params.append(self.up1.kernel)
        params.append(self.conv1.kernel)
        params.append(self.up2.kernel)
        params.append(self.conv2.kernel)
        params.append(self.conv2_cls.kernel)

        return params
    
    def load_init(self, meta_dict, meta_map):
        nd = {}
        for k in meta_map:
            nd[k] = meta_dict[meta_map[k]]
        rt = self.load_state_dict(nd, strict=False)
        print(rt)

class ZeroStartAddModuloCompNetv2(nn.Module):
    # With only one concatenation
    def __init__(self, args, prob_model, param_model):
        super(ZeroStartAddModuloCompNetv2, self).__init__()
        global SEED2
        global seed_ptr
        # global prob_model
        # global param_model

        channels=[8,16,8,8]
        switch = nn.parameter.Parameter(torch.ones((3)).float(), requires_grad=True)
        self.register_parameter('switch', switch)
        
        self.shape_emb_module = ZeroStartStandaloneInitShapeEmb(prob_model=prob_model)

        self.h_synthesis_2 = QConvTranspose3d(8, channels[1]//2, 5, 2, True, padding=2, output_padding=1, SEED=SEED2[seed_ptr:], zero_bias=True)
        seed_ptr += self.h_synthesis_2.offset

        self.up1 = QConvTranspose3d(
            in_channels=channels[1],
            out_channels=channels[2],
            kernel_size=5,
            stride=2,
            bias=True,
            SEED=SEED2[seed_ptr:])
        seed_ptr += self.up1.offset

        self.conv1 = QConv3d(
            in_channels=channels[2],
            out_channels=channels[2],
            kernel_size=4,
            stride=1,
            bias=True,
            padding=0,
            SEED=SEED2[seed_ptr:])
        seed_ptr += self.conv1.offset

        self.up2 = QConvTranspose3d(
            in_channels=channels[2],
            out_channels=channels[3],
            kernel_size=5,
            stride=2,
            bias=True,
            SEED=SEED2[seed_ptr:])
        seed_ptr += self.up2.offset

        self.conv2 = QConv3d(
            in_channels=channels[3],
            out_channels=channels[3],
            kernel_size=4,
            stride=1,
            bias=True,
            padding=0,
            SEED=SEED2[seed_ptr:])
        seed_ptr += self.conv2.offset

        self.conv2_cls = QConv3d(
            in_channels=channels[3],
            out_channels=1,
            kernel_size=3,
            stride=1,
            bias=True,
            padding=1,
            SEED=SEED2[seed_ptr:])
        seed_ptr += self.conv2_cls.offset
        
        self.conv1_cls = IConv3d(
            in_channels=channels[2],
            out_channels=1,
            kernel_size=3,
            stride=1,
            bias=True,
            padding=1,
            SEED=SEED2[seed_ptr:])
        seed_ptr += self.conv1_cls.offset
        
        self.conv0_cls = IConv3d(
            in_channels=channels[1],
            out_channels=1,
            kernel_size=3,
            stride=1,
            bias=True,
            padding=1,
            SEED=SEED2[seed_ptr:])
        seed_ptr += self.conv0_cls.offset
    
        # self.relu = nn.ReLU()
        self.relu = nn.GELU()
        self.sigmoid = nn.Sigmoid()

        # self.likelihood_model = LikelihoodModel(step_size=1/16)
        if param_model == 'Gaussian':
            if args.stat_net:
                self.likelihood_model = StatGaussianLikelihoodModel(step_size=1/16)
            else:
                self.likelihood_model = GaussianLikelihoodModel(step_size=1/16)
        elif param_model == 'Laplacian':
            if args.stat_net:
                self.likelihood_model = StatLaplaceLikelihoodModel(step_size=1/16)
            else:
                self.likelihood_model = LaplaceLikelihoodModel(step_size=1/16)
        # print('Using single Laplacian likelihood model')

    def forward(self, x, q, y1, x_emb_init=None):
        out, bits_list = self.shape_emb_module(x, q, x_emb_init)
        out = self.h_synthesis_2(out, q)

        # For DEBUG #
        # print(torch.abs(out).sum())
        #############

        out = self.relu(torch.cat([out, y1*self.switch[0]], 1))
        out_cls_0 = self.sigmoid(self.conv0_cls(out))
        # print(out.shape)
        # print(out_cls_0.shape)
        # out = self.block0(out)
        out = self.relu(self.up1(out, q))
        out = self.relu(self.conv1(out, q))
        out_cls_1 = self.sigmoid(self.conv1_cls(out))
        # print(out.shape)
        # print(out_cls_1.shape)
        # out = self.block1(out)
        out = self.relu(self.up2(out, q))
        out = self.relu(self.conv2(out, q))
        # print(out.shape)
        out = self.conv2_cls(out, q)
        out = self.sigmoid(out)
        # print(out.shape)
        # quit()

        out_cls_list = [out_cls_0, out_cls_1, out]

        if torch.any(torch.isnan(out)):
            import IPython
            IPython.embed()
            raise ValueError()

        params = self.get_q_params()
        
        net_bits = torch.stack([self.likelihood_model(bypass_round16(p)) for p in params])
        # net_bits = torch.stack(net_bits)
        return out, out_cls_list, bits_list, net_bits
    
    def eval(self, x, q, x_emb_init):
        out, bits_list, h_x_rounded = self.shape_emb_module.eval(x, q, x_emb_init)
        out_cls_0 = self.sigmoid(self.conv0_cls(out))
        # print(out.shape)
        # print(out_cls_0.shape)
        # out = self.block0(out)
        
        out = self.relu(self.conv1(self.relu(self.up1(out, q)), q))
        out_cls_1 = self.sigmoid(self.conv1_cls(out))
        # print(out.shape)
        # print(out_cls_1.shape)
        # out = self.block1(out)
        
        out = self.relu(self.conv2(self.relu(self.up2(out, q)), q))
        # print(out.shape)
        out = self.conv2_cls(out, q)
        out = self.sigmoid(out)
        # print(out.shape)
        # quit()

        out_cls_list = [out_cls_0, out_cls_1, out]

        if torch.any(torch.isnan(out)):
            import IPython
            IPython.embed()
            raise ValueError()

        params = self.get_q_params()

        net_bits = torch.stack([self.likelihood_model(bypass_round16(p)) for p in params])
        # net_bits = torch.stack(net_bits)
        return out, out_cls_list, bits_list, h_x_rounded
    
    def get_continuous_params(self):
        params = []
        for i in self.pos_emb.parameters():
            params.append(i)
        for i in self.conv0_cls.parameters():
            params.append(i)
        for i in self.conv1_cls.parameters():
            params.append(i)
        for i in self.parameters():
            if len(i.shape) == 1:
                params.append(i)
        return params

    def get_q_params(self):
        params = []

        params += self.shape_emb_module.get_q_params()
        params.append(self.up1.kernel)
        params.append(self.conv1.kernel)
        params.append(self.up2.kernel)
        params.append(self.conv2.kernel)
        params.append(self.conv2_cls.kernel)

        return params
    
    def load_init(self, meta_dict, meta_map):
        nd = {}
        for k in meta_map:
            nd[k] = meta_dict[meta_map[k]]
        rt = self.load_state_dict(nd, strict=False)
        print(rt)
    
    def get_aux_params(self):
        return list(self.conv0_cls.parameters())

class TempAddModuloCompNet(nn.Module):
    def __init__(self, args, prob_model, param_model):
        super(TempAddModuloCompNet, self).__init__()
        global SEED2
        global seed_ptr
        # global prob_model
        # global param_model

        channels=[8,16,8,8]
        switch = nn.parameter.Parameter(torch.zeros((3)).float(), requires_grad=False)
        self.register_buffer('switch', switch)
        
        self.shape_emb_module = StandaloneInitShapeEmb(prob_model=prob_model)

        self.h_synthesis_2 = QConvTranspose3d(8, channels[1]//2, 5, 2, True, padding=2, output_padding=1, SEED=SEED2[seed_ptr:])
        seed_ptr += self.h_synthesis_2.offset

        self.up1 = QConvTranspose3d(
            in_channels=channels[1],
            out_channels=channels[2],
            kernel_size=5,
            stride=2,
            bias=True,
            SEED=SEED2[seed_ptr:])
        seed_ptr += self.up1.offset

        self.conv1 = QConv3d(
            in_channels=channels[2]+4,
            out_channels=channels[2],
            kernel_size=4,
            stride=1,
            bias=True,
            padding=0,
            SEED=SEED2[seed_ptr:])
        seed_ptr += self.conv1.offset

        self.up2 = QConvTranspose3d(
            in_channels=channels[2],
            out_channels=channels[3],
            kernel_size=5,
            stride=2,
            bias=True,
            SEED=SEED2[seed_ptr:])
        seed_ptr += self.up2.offset

        self.conv2 = QConv3d(
            in_channels=channels[3]+4,
            out_channels=channels[3],
            kernel_size=4,
            stride=1,
            bias=True,
            padding=0,
            SEED=SEED2[seed_ptr:])
        seed_ptr += self.conv2.offset

        self.conv2_cls = QConv3d(
            in_channels=channels[3],
            out_channels=1,
            kernel_size=3,
            stride=1,
            bias=True,
            padding=1,
            SEED=SEED2[seed_ptr:])
        seed_ptr += self.conv2_cls.offset
        
        self.conv1_cls = IConv3d(
            in_channels=channels[2],
            out_channels=1,
            kernel_size=3,
            stride=1,
            bias=True,
            padding=1,
            SEED=SEED2[seed_ptr:])
        seed_ptr += self.conv1_cls.offset
        
        self.conv0_cls = IConv3d(
            in_channels=channels[1],
            out_channels=1,
            kernel_size=3,
            stride=1,
            bias=True,
            padding=1,
            SEED=SEED2[seed_ptr:])
        seed_ptr += self.conv0_cls.offset
    
        # self.relu = nn.ReLU()
        self.relu = nn.GELU()
        self.sigmoid = nn.Sigmoid()

        # self.likelihood_model = LikelihoodModel(step_size=1/16)
        if param_model == 'Gaussian':
            if args.stat_net:
                self.likelihood_model = StatGaussianLikelihoodModel(step_size=1/16)
            else:
                self.likelihood_model = GaussianLikelihoodModel(step_size=1/16)
        elif param_model == 'Laplacian':
            if args.stat_net:
                self.likelihood_model = StatLaplaceLikelihoodModel(step_size=1/16)
            else:
                self.likelihood_model = LaplaceLikelihoodModel(step_size=1/16)
        # print('Using single Laplacian likelihood model')

    def forward(self, x, q, ys, x_emb_init=None):
        out, bits_list = self.shape_emb_module(x, q, x_emb_init)
        out = self.h_synthesis_2(out, q)

        out = self.relu(torch.cat([out, ys[0]*self.switch[0]], 1))
        out_cls_0 = self.sigmoid(self.conv0_cls(out))
        # print(out.shape)
        # print(out_cls_0.shape)
        # out = self.block0(out)
        out = self.relu(self.up1(out, q))
        out = torch.cat([out, ys[1]*self.switch[1]], 1)
        out = self.relu(self.conv1(out, q))
        out_cls_1 = self.sigmoid(self.conv1_cls(out))
        # print(out.shape)
        # print(out_cls_1.shape)
        # out = self.block1(out)
        out = self.relu(self.up2(out, q))
        out = torch.cat([out, ys[2]*self.switch[2]], 1)
        out = self.relu(self.conv2(out, q))
        # print(out.shape)
        out = self.conv2_cls(out, q)
        out = self.sigmoid(out)
        # print(out.shape)
        # quit()

        out_cls_list = [out_cls_0, out_cls_1, out]

        if torch.any(torch.isnan(out)):
            import IPython
            IPython.embed()
            raise ValueError()

        params = self.get_q_params()
        
        net_bits = torch.stack([self.likelihood_model(bypass_round16(p)) for p in params])
        # net_bits = torch.stack(net_bits)
        return out, out_cls_list, bits_list, net_bits
    
    def eval(self, x, q, x_emb_init):
        out, bits_list, h_x_rounded = self.shape_emb_module.eval(x, q, x_emb_init)
        out_cls_0 = self.sigmoid(self.conv0_cls(out))
        # print(out.shape)
        # print(out_cls_0.shape)
        # out = self.block0(out)
        
        out = self.relu(self.conv1(self.relu(self.up1(out, q)), q))
        out_cls_1 = self.sigmoid(self.conv1_cls(out))
        # print(out.shape)
        # print(out_cls_1.shape)
        # out = self.block1(out)
        
        out = self.relu(self.conv2(self.relu(self.up2(out, q)), q))
        # print(out.shape)
        out = self.conv2_cls(out, q)
        out = self.sigmoid(out)
        # print(out.shape)
        # quit()

        out_cls_list = [out_cls_0, out_cls_1, out]

        if torch.any(torch.isnan(out)):
            import IPython
            IPython.embed()
            raise ValueError()

        params = self.get_q_params()

        net_bits = torch.stack([self.likelihood_model(bypass_round16(p)) for p in params])
        # net_bits = torch.stack(net_bits)
        return out, out_cls_list, bits_list, h_x_rounded
    
    def get_continuous_params(self):
        params = []
        for i in self.pos_emb.parameters():
            params.append(i)
        for i in self.conv0_cls.parameters():
            params.append(i)
        for i in self.conv1_cls.parameters():
            params.append(i)
        for i in self.parameters():
            if len(i.shape) == 1:
                params.append(i)
        return params

    def get_q_params(self):
        params = []

        params += self.shape_emb_module.get_q_params()
        params.append(self.up1.kernel)
        params.append(self.conv1.kernel)
        params.append(self.up2.kernel)
        params.append(self.conv2.kernel)
        params.append(self.conv2_cls.kernel)

        return params
    
    def load_init(self, meta_dict, meta_map):
        nd = {}
        for k in meta_map:
            nd[k] = meta_dict[meta_map[k]]
        rt = self.load_state_dict(nd, strict=False)
        print(rt)

class BaseCodeLatentNet(nn.Module):
    def __init__(self) -> None:
        super(BaseCodeLatentNet, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.res1 = InceptionResNetDense(8)
        self.conv2 = nn.Conv3d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.up1 = nn.ConvTranspose3d(in_channels=16, out_channels=8, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv3d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=0)
        self.deliver1 = nn.Conv3d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1)

        self.up2 = nn.ConvTranspose3d(in_channels=8, out_channels=8, kernel_size=5, stride=2, padding=0)
        self.deliver2 = nn.Conv3d(in_channels=8, out_channels=4, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv3d(in_channels=8, out_channels=8, kernel_size=4, stride=1, padding=0)

        self.up3 = nn.ConvTranspose3d(in_channels=8, out_channels=8, kernel_size=5, stride=2, padding=0)
        self.deliver3 = nn.Conv3d(in_channels=8, out_channels=4, kernel_size=3, stride=1, padding=1)

        self.relu = nn.ReLU()
        self.pad = nn.ConstantPad3d((1, 0, 1, 0, 1, 0),0)
        
    def forward(self, x):
        out = self.relu(self.conv2(self.res1(self.relu(self.conv1(x)))))
        out = self.conv3(self.pad(self.relu(self.up1(out))))
        y1 = self.deliver1(out)
        # print(y1.shape)
        out = self.up2(self.relu(out))
        y2  = self.deliver2(out)
        out = self.conv4(self.relu(out))
        # print(y2.shape)
        out = self.up3(self.relu(out))
        y3 = self.deliver3(out)
        # print(y3.shape)
        return y1, y2, y3

class BaseCodeLatentNetv2(nn.Module):
    def __init__(self) -> None:
        super(BaseCodeLatentNetv2, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.res1 = InceptionResNetDense(8)
        self.conv2 = nn.Conv3d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.up1 = nn.ConvTranspose3d(in_channels=16, out_channels=8, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv3d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=0)
        self.deliver1 = nn.Conv3d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1)

        self.relu = nn.ReLU()
        self.pad = nn.ConstantPad3d((1, 0, 1, 0, 1, 0),0)
        
    def forward(self, x):
        out = self.relu(self.conv2(self.res1(self.relu(self.conv1(x)))))
        out = self.conv3(self.pad(self.relu(self.up1(out))))
        y1 = self.deliver1(out)
        
        return y1

class BaseCodeLatentNetPred(nn.Module):
    def __init__(self) -> None:
        super(BaseCodeLatentNetPred, self).__init__()
        self.transform = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            InceptionResNetDense(8),
            nn.LeakyReLU(),
            nn.Conv3d(in_channels=8, out_channels=32, kernel_size=5, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.ConvTranspose3d(in_channels=32, out_channels=32, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose3d(in_channels=32, out_channels=32, kernel_size=5, stride=2, padding=2)
        )

        self.deliver_mu = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.deliver_sigma = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0)

        self.relu = nn.ReLU()
        self.pad = nn.ConstantPad3d((1, 0, 1, 0, 1, 0),0)
        
    def forward(self, x):
        out = self.transform(x)
        mu = self.deliver_mu(out)
        sigma = self.deliver_sigma(out)
        
        return mu, sigma

class LameBaseCodeLatentNet(nn.Module):
    def __init__(self) -> None:
        super(LameBaseCodeLatentNet, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.res1 = InceptionResNetDense(8)
        self.conv2 = nn.Conv3d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.up1 = nn.ConvTranspose3d(in_channels=16, out_channels=8, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv3d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=0)
        self.deliver1 = nn.Conv3d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1)

        self.up2 = nn.ConvTranspose3d(in_channels=8, out_channels=8, kernel_size=5, stride=2, padding=0)
        self.deliver2 = nn.Conv3d(in_channels=8, out_channels=4, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv3d(in_channels=8, out_channels=8, kernel_size=4, stride=1, padding=0)

        self.up3 = nn.ConvTranspose3d(in_channels=8, out_channels=8, kernel_size=5, stride=2, padding=0)
        self.deliver3 = nn.Conv3d(in_channels=8, out_channels=4, kernel_size=3, stride=1, padding=1)

        self.relu = nn.ReLU()
        self.pad = nn.ConstantPad3d((1, 0, 1, 0, 1, 0),0)
        
    def forward(self, x):
        out = self.relu(self.conv2(self.res1(self.relu(self.conv1(x)))))
        out = self.conv3(self.pad(self.relu(self.up1(out))))
        y1 = self.deliver1(out)
        y2 = y1
        y3 = y1
        # print(y3.shape)
        return y1, y2, y3

class FeedForwardEncoder(nn.Module):
    def __init__(self, out_channels=32) -> None:
        super(FeedForwardEncoder, self).__init__()
        self.transform = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=8, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.Conv3d(8, 16, 5, 2, 2),
            nn.LeakyReLU(),
            nn.Conv3d(16, 32, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Conv3d(32, 32, 5, 2, 2),
            nn.LeakyReLU(),
            InceptionResNetDense(32),
            nn.Conv3d(32, out_channels, 5, 2, 2)
        )
    def forward(self, x):
        out = self.transform(x)
        return out

class NotActivation(nn.Module):
    def __init__(self) -> None:
        super(NotActivation, self).__init__()
    
    def forward(self, x):
        return x

class DSepCompShapeEmb(nn.Module):
    def __init__(self, nLeaves, out_dim, prob_model='Gaussian'):
        super(DSepCompShapeEmb, self).__init__()
        global SEED2
        global seed_ptr

        emb = nn.parameter.Parameter(torch.ones((nLeaves,16,2,2,2), dtype=torch.float32), requires_grad=True)
        print('Embedding shape: ', emb.shape)
        self.register_parameter('emb', emb)
        ch = 4 # 1a_2 1a_5
        print('Bottleneck channel: ', ch)
        # ch = 2 # 1a_3 test
        
        self.h_analysis_2 = nn.Conv3d(
            in_channels=16,
            out_channels=ch,
            kernel_size=3,
            stride=1,
            bias=True,
            padding=1
        )
        self.gdn_2 = GDN3d(ch)
        # self.gdn_2 = nn.GELU()
        # self.gdn_2 = NotActivation()
        print('Bottleneck shape: ', (ch, 2, 2, 2))

        h1_sigma = torch.nn.Parameter(torch.ones(
            (1, ch, 2, 2, 2), dtype=torch.float32, requires_grad=True))
        self.register_parameter('h1_sigma', h1_sigma)

        h1_mu = torch.nn.Parameter(torch.zeros(
            (1, ch, 2, 2, 2), dtype=torch.float32, requires_grad=True))
        self.register_buffer('h1_mu', h1_mu)
        if prob_model == 'Gaussian':
            self.gauss_model = GaussianModel()
        elif prob_model == 'Laplacian':
            self.gauss_model = LaplaceModel()
        self.h_synthesis_1 = QDSepConvTransposed3d(ch, 32, 5, 2, True, padding=2, output_padding=1, SEED=SEED2[seed_ptr:], activation=NotActivation())
        seed_ptr += self.h_synthesis_1.offset
        self.igdn_1 = IGDN3d(32)
        # self.igdn_1 = nn.GELU()

        self.h_synthesis_2 = QDSepConvTransposed3d(32, out_dim, 5, 2, True, padding=2, output_padding=1, SEED=SEED2[seed_ptr:], activation=NotActivation())
        seed_ptr += self.h_synthesis_2.offset
        # self.igdn_2 = IGDN3d(out_dim)
        # self.igdn_2 = nn.ReLU()
        self.igdn_2 = nn.GELU()
    
    def forward(self, idx, q):
        x = self.emb[idx]

        # h_x = self.gdn_1(self.h_analysis_1(x))
        # print(h_x.shape)
        h_x = self.gdn_2(self.h_analysis_2(x))
        # print(h_x.shape)
        
        noise = torch.rand_like(h_x) - 0.5
        h_x_noisy = h_x + noise
        h_x_rounded = bypass_round(h_x)

        if q == 1:
            h_x_form = h_x_noisy
        elif q == 2:
            h_x_form = h_x_rounded
        elif q == 0:
            h_x_form = h_x_noisy
        else:
            raise NotImplementedError
        
        h_likelihood = self.gauss_model(h_x_form, torch.abs(self.h1_sigma), self.h1_mu)

        h_y = self.igdn_2(self.h_synthesis_2(self.igdn_1(self.h_synthesis_1(h_x_rounded, q)), q))

        bits_list = [h_likelihood, torch.tensor(0).float()]

        if torch.any(torch.isnan(h_likelihood)):
            print('problems with bits')
            import IPython
            IPython.embed()
            raise ValueError()

        if torch.any(torch.isnan(x)):
            print('problems with x_form')
            import IPython
            IPython.embed()
            raise ValueError()

        return h_y, bits_list
    

    def eval(self, idx, q):
        x = self.emb[idx]

        # h_x = self.gdn_1(self.h_analysis_1(x))
        # print(h_x.shape)
        h_x = self.gdn_2(self.h_analysis_2(x))
        # print(h_x.shape)
        
        noise = torch.rand_like(h_x) - 0.5
        h_x_noisy = h_x + noise
        h_x_rounded = bypass_round(h_x)

        if q == 1:
            h_x_form = h_x_noisy
        elif q == 2:
            h_x_form = h_x_rounded
        elif q == 0:
            h_x_form = h_x_noisy
        else:
            raise NotImplementedError
        
        h_likelihood = self.gauss_model(h_x_form, torch.abs(self.h1_sigma), self.h1_mu)

        h_y = self.igdn_2(self.h_synthesis_2(self.igdn_1(self.h_synthesis_1(h_x_rounded, q)), q))

        bits_list = [h_likelihood, torch.tensor(0).float()]

        if torch.any(torch.isnan(h_likelihood)):
            print('problems with bits')
            import IPython
            IPython.embed()
            raise ValueError()

        if torch.any(torch.isnan(x)):
            print('problems with x_form')
            import IPython
            IPython.embed()
            raise ValueError()

        return h_y, bits_list, h_x_rounded
    
    def get_q_params(self):
        params = []
        params += self.h_synthesis_1.get_q_params() 
        params += self.h_synthesis_2.get_q_params()
        return params

class DSepStatCompShapeEmb(nn.Module):
    def __init__(self, nLeaves, out_dim, prob_model='Gaussian'):
        super(DSepStatCompShapeEmb, self).__init__()
        global SEED2
        global seed_ptr

        emb = nn.parameter.Parameter(torch.ones((nLeaves,16,2,2,2), dtype=torch.float32), requires_grad=True)
        print('Embedding shape: ', emb.shape)
        self.register_parameter('emb', emb)
        ch = 4 # 1a_2 1a_5
        print('Bottleneck channel: ', ch)
        # ch = 2 # 1a_3 test
        
        self.h_analysis_2 = nn.Conv3d(
            in_channels=16,
            out_channels=ch,
            kernel_size=3,
            stride=1,
            bias=True,
            padding=1
        )
        self.gdn_2 = GDN3d(ch)
        # self.gdn_2 = nn.GELU()
        # self.gdn_2 = NotActivation()
        print('Bottleneck shape: ', (ch, 2, 2, 2))

        h1_sigma = torch.nn.Parameter(torch.ones(
            (1, ch, 2, 2, 2), dtype=torch.float32, requires_grad=True))
        self.register_parameter('h1_sigma', h1_sigma)

        h1_mu = torch.nn.Parameter(torch.zeros(
            (1, ch, 2, 2, 2), dtype=torch.float32, requires_grad=True))
        self.register_parameter('h1_mu', h1_mu)
        if prob_model == 'Gaussian':
            self.gauss_model = GaussianModel()
        elif prob_model == 'Laplacian':
            self.gauss_model = LaplaceModel()
        self.h_synthesis_1 = QDSepConvTransposed3d(ch, 32, 5, 2, True, padding=2, output_padding=1, SEED=SEED2[seed_ptr:], activation=NotActivation())
        seed_ptr += self.h_synthesis_1.offset
        self.igdn_1 = IGDN3d(32)
        # self.igdn_1 = nn.GELU()

        self.h_synthesis_2 = QDSepConvTransposed3d(32, out_dim, 5, 2, True, padding=2, output_padding=1, SEED=SEED2[seed_ptr:], activation=NotActivation())
        seed_ptr += self.h_synthesis_2.offset
        # self.igdn_2 = IGDN3d(out_dim)
        # self.igdn_2 = nn.ReLU()
        self.igdn_2 = nn.GELU()
    
    def forward(self, idx, q):
        x = self.emb[idx]

        # h_x = self.gdn_1(self.h_analysis_1(x))
        # print(h_x.shape)
        h_x = self.gdn_2(self.h_analysis_2(x))
        # print(h_x.shape)
        
        noise = torch.rand_like(h_x) - 0.5
        h_x_noisy = h_x + noise
        h_x_rounded = bypass_round(h_x)

        if q == 1:
            h_x_form = h_x_noisy
        elif q == 2:
            h_x_form = h_x_rounded
        elif q == 0:
            h_x_form = h_x_noisy
        else:
            raise NotImplementedError
        
        h_likelihood = self.gauss_model(h_x_form, torch.abs(self.h1_sigma.detach()), self.h1_mu.detach())

        h_y = self.igdn_2(self.h_synthesis_2(self.igdn_1(self.h_synthesis_1(h_x_rounded, q)), q))

        bits_list = [h_likelihood, torch.tensor(0).float()]

        if torch.any(torch.isnan(h_likelihood)):
            print('problems with bits')
            import IPython
            IPython.embed()
            raise ValueError()

        if torch.any(torch.isnan(x)):
            print('problems with x_form')
            import IPython
            IPython.embed()
            raise ValueError()

        return h_y, bits_list
    

    def eval(self, idx, q):
        x = self.emb[idx]

        # h_x = self.gdn_1(self.h_analysis_1(x))
        # print(h_x.shape)
        h_x = self.gdn_2(self.h_analysis_2(x))
        # print(h_x.shape)
        
        noise = torch.rand_like(h_x) - 0.5
        h_x_noisy = h_x + noise
        h_x_rounded = bypass_round(h_x)

        if q == 1:
            h_x_form = h_x_noisy
        elif q == 2:
            h_x_form = h_x_rounded
        elif q == 0:
            h_x_form = h_x_noisy
        else:
            raise NotImplementedError
        
        h_likelihood = self.gauss_model(h_x_form, torch.abs(self.h1_sigma), self.h1_mu)

        h_y = self.igdn_2(self.h_synthesis_2(self.igdn_1(self.h_synthesis_1(h_x_rounded, q)), q))

        bits_list = [h_likelihood, torch.tensor(0).float()]

        if torch.any(torch.isnan(h_likelihood)):
            print('problems with bits')
            import IPython
            IPython.embed()
            raise ValueError()

        if torch.any(torch.isnan(x)):
            print('problems with x_form')
            import IPython
            IPython.embed()
            raise ValueError()

        return h_y, bits_list, h_x_rounded
    
    def update_stats(self):
        with torch.no_grad():
            feat = self.h_analysis_2(self.emb)
            qfeat = torch.round(feat)
            scale, mu = torch.std_mean(qfeat, 0)
            self.h1_sigma[...] = scale + 1e-6
            self.h1_mu[...] = mu

    def get_q_params(self):
        params = []
        params += self.h_synthesis_1.get_q_params() 
        params += self.h_synthesis_2.get_q_params()
        return params

class DSepCompNet(nn.Module):
    def __init__(self, bsize, in_dim, nLeaves):
        super(DSepCompNet, self).__init__()
        global SEED2
        global seed_ptr
        global prob_model
        global param_model
        self.relu = nn.GELU()

        channels=[32,64,32,32]
        # channels=[16,32,16,16]
        print("Channels: ", channels)

        if args.stat_latent:
            print('Using stat model for latent repr')
            self.shape_emb_module = DSepStatCompShapeEmb(nLeaves=nLeaves, out_dim=channels[1], prob_model=prob_model)
        else:
            print('Using regular entropy model for latent repr')
            self.shape_emb_module = DSepCompShapeEmb(nLeaves=nLeaves, out_dim=channels[1], prob_model=prob_model)
        noact = NotActivation()

        self.up1 = QDSepConvTransposed3d(
            in_channels=channels[1],
            out_channels=channels[2],
            kernel_size=5,
            stride=2,
            bias=True,
            SEED=SEED2[seed_ptr:], activation=noact)
        seed_ptr += self.up1.offset

        self.conv1 = QDSepConv3d(
            in_channels=channels[2],
            out_channels=channels[2],
            kernel_size=4,
            stride=1,
            bias=True,
            padding=0,
            SEED=SEED2[seed_ptr:], activation=noact)
        seed_ptr += self.conv1.offset

        self.up2 = QDSepConvTransposed3d(
            in_channels=channels[2],
            out_channels=channels[3],
            kernel_size=5,
            stride=2,
            bias=True,
            SEED=SEED2[seed_ptr:], activation=noact)
        seed_ptr += self.up2.offset

        self.conv2 = QDSepConv3d(
            in_channels=channels[3],
            out_channels=channels[3],
            kernel_size=4,
            stride=1,
            bias=True,
            padding=0,
            SEED=SEED2[seed_ptr:], activation=noact)
        seed_ptr += self.conv2.offset

        self.conv2_cls = QDSepConv3d(
            in_channels=channels[3],
            out_channels=1,
            kernel_size=3,
            stride=1,
            bias=True,
            padding=1,
            SEED=SEED2[seed_ptr:], activation=noact)
        seed_ptr += self.conv2_cls.offset
        
        self.conv1_cls = nn.Conv3d(
            in_channels=channels[2],
            out_channels=1,
            kernel_size=3,
            stride=1,
            bias=True,
            padding=1)
        
        self.conv0_cls = nn.Conv3d(
            in_channels=channels[1],
            out_channels=1,
            kernel_size=3,
            stride=1,
            bias=True,
            padding=1)
    
        # self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        # self.likelihood_model = LikelihoodModel(step_size=1/16)
        if param_model == 'Gaussian':
            if args.stat_net:
                self.likelihood_model = StatGaussianLikelihoodModel(step_size=1/16)
            else:
                self.likelihood_model = GaussianLikelihoodModel(step_size=1/16)
        elif param_model == 'Laplacian':
            if args.stat_net:
                self.likelihood_model = StatLaplaceLikelihoodModel(step_size=1/16)
            else:
                self.likelihood_model = LaplaceLikelihoodModel(step_size=1/16)
        # print('Using single Laplacian likelihood model')

    def forward(self, x, q):
        out, bits_list = self.shape_emb_module(x, q)
        out_cls_0 = self.sigmoid(self.conv0_cls(out))
        # print(out.shape)
        # print(out_cls_0.shape)
        # out = self.block0(out)
        
        out = self.relu(self.conv1(self.relu(self.up1(out, q)), q))
        out_cls_1 = self.sigmoid(self.conv1_cls(out))
        # print(out.shape)
        # print(out_cls_1.shape)
        # out = self.block1(out)
        
        out = self.relu(self.conv2(self.relu(self.up2(out, q)), q))
        # print(out.shape)
        out = self.conv2_cls(out, q)
        out = self.sigmoid(out)
        # print(out.shape)
        # quit()

        out_cls_list = [out_cls_0, out_cls_1, out]

        if torch.any(torch.isnan(out)):
            import IPython
            IPython.embed()
            raise ValueError()

        params = self.get_q_params()
        # net_bits = []
        # lklhs = []
        # for p in params:
        #     a, b = self.likelihood_model(p + (torch.rand_like(p)-0.5)*1/16)
        #     net_bits.append(a.sum())
        #     lklhs.append(b)
        # net_bits = torch.stack([self.likelihood_model(p + (torch.rand_like(p)-0.5)*1/16) for p in params])
        net_bits = torch.stack([self.likelihood_model(bypass_round16(p)) for p in params])
        # net_bits = torch.stack(net_bits)
        return out, out_cls_list, bits_list, net_bits
    
    def eval(self, x, q):
        # emb = self.pos_emb(x)
        # y = emb
        # for l in self.fc_layers:
        #     y = l(y, q)
        
        # feat = torch.reshape(y, (y.shape[0], 8, 4, 4, 4))
        # # x = ME.SparseTensor(features=feat, coordinates=self.init_coords, tensor_stride=8)
        # x = feat
        # #
        # out = self.relu(self.conv0(self.relu(self.up0(x, q)), q))
        out, bits_list, h_x_rounded = self.shape_emb_module.eval(x, q)
        out_cls_0 = self.sigmoid(self.conv0_cls(out))
        # print(out.shape)
        # print(out_cls_0.shape)
        # out = self.block0(out)
        
        out = self.relu(self.conv1(self.relu(self.up1(out, q)), q))
        out_cls_1 = self.sigmoid(self.conv1_cls(out))
        # print(out.shape)
        # print(out_cls_1.shape)
        # out = self.block1(out)
        
        out = self.relu(self.conv2(self.relu(self.up2(out, q)), q))
        # print(out.shape)
        out = self.conv2_cls(out, q)
        out = self.sigmoid(out)
        # print(out.shape)
        # quit()

        out_cls_list = [out_cls_0, out_cls_1, out]

        if torch.any(torch.isnan(out)):
            import IPython
            IPython.embed()
            raise ValueError()

        return out, out_cls_list, bits_list, h_x_rounded
    
    def get_q_params(self):
        params = []

        params += self.shape_emb_module.get_q_params()
        params += self.up1.get_q_params()
        params += self.conv1.get_q_params()
        params += self.up2.get_q_params()
        params += self.conv2.get_q_params()
        params += self.conv2_cls.get_q_params()

        return params

class QMaskedConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, bias, padding=0, iQ=16, SEED=None, zero_bias=False):
        super(QMaskedConv3d, self).__init__()
        self.padding = padding
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.bias = bias
        
        np_mask = np.zeros((out_channels, in_channels,
                kernel_size, kernel_size, kernel_size), dtype=np.float32)
        assert kernel_size % 2 == 1
        half_k = kernel_size // 2
        np_mask[:,:,:half_k,:,:] = 1
        np_mask[:,:,half_k,:half_k,:] = 1
        np_mask[:,:,half_k,half_k,:half_k] = 1
        self.register_buffer('mask', torch.from_numpy(np_mask))
        
        kernel = nn.parameter.Parameter(torch.zeros(
                (out_channels, in_channels,
                kernel_size, kernel_size, kernel_size)
            ),
            requires_grad=True
        )
        b = nn.parameter.Parameter(torch.zeros((out_channels)).float(), requires_grad=True)

        off1 = in_channels * out_channels * (kernel_size**3)
        kernel_seed = torch.from_numpy(SEED[:off1].reshape(kernel.shape)).float()
        kernel_init = get_kaiming_init_from_seed(kernel, kernel_seed)
        self.register_buffer('kernel_init', kernel_init)
        self.register_parameter('kernel', kernel)

        if self.bias:
            off2 = out_channels + off1
            b_seed = torch.from_numpy(SEED[off1:off2].reshape(b.shape)).float()
            init_bound = 1/np.sqrt(in_channels)

            if zero_bias:
                b_init = torch.zeros_like(b_seed)
            else:
                b_init = (b_seed - 0.5) * 2 * init_bound

            self.register_buffer('b_init', b_init)
            self.register_parameter('b', b)
        else:
            off2 = off1

        self.Q = 1/iQ
        self.iQ = iQ
        self.offset = off2

    def forward(self, x, q):
        kernel = self.kernel
        if self.bias:
            b = self.b + self.b_init
        else:
            b = None
        if q == 1:
            k_rounded = kernel + (torch.rand_like(kernel)-0.5)*self.Q
        elif q == 2:
            k_rounded = bypass_round16(kernel)
        else:
            k_rounded = kernel
        k_rounded = k_rounded + self.kernel_init
        k_rounded = k_rounded * self.mask
        out = F.conv3d(x, k_rounded, b, self.stride, self.padding)
        return out

class PredNet(nn.Module):
    def __init__(self, in_dim, out_dim, iQ):
        super(PredNet, self).__init__()
        global seed_ptr
        global SEED2
        self.conv1 = QMaskedConv3d(in_dim, 8, 3, 1, True, 1, SEED=SEED2[seed_ptr:], iQ=iQ)
        seed_ptr += self.conv1.offset
        self.conv2 = QMaskedConv3d(8, out_dim*2, 3, 1, True, 1, SEED=SEED2[seed_ptr:], iQ=iQ)
        seed_ptr += self.conv2.offset
        # self.conv3 = QMaskedConv3d(8, 8, 3, 1, True, 1, SEED=SEED2[seed_ptr:], iQ=iQ)
        # seed_ptr += self.conv3.offset
        # self.conv4 = QMaskedConv3d(8, out_dim*2, 3, 1, True, 1, SEED=SEED2[seed_ptr:], iQ=iQ)
        # seed_ptr += self.conv4.offset
        self.out_dim = out_dim

    def forward(self, x, q):
        out = self.conv1(x, q)
        out = F.leaky_relu(out)
        out = self.conv2(out, q)
        # out = F.leaky_relu(out)
        # out = self.conv3(out, q)
        # out = F.leaky_relu(out)
        # out = self.conv4(out, q)
        mu = out[:,:self.out_dim,...]
        sigma = torch.abs(out[:,self.out_dim:,...]) + 1e-6
        return mu, sigma
    
    def get_q_params(self):
        params = []
        params.append(self.conv1.kernel)
        params.append(self.conv2.kernel)
        return params

class CtxPredLikelihoodModel(nn.Module):
    def __init__(self, in_channels, step_size=1, iQ=1):
        super(CtxPredLikelihoodModel, self).__init__()
        self.gaussian_model = GaussianModel(step_size)
        self.pred_net = PredNet(in_channels, in_channels, iQ)

        if iQ == 1:
            self.bypass_round = bypass_round
        elif iQ == 16:
            self.bypass_round = bypass_round16
        else:
            raise NotImplementedError
        self.Q = 1/iQ
    
    def forward(self, x, q, likelihood_model):
        noise = torch.rand_like(x) - 0.5
        h_x_noisy = x + noise * self.Q
        h_x_rounded = self.bypass_round(x)

        mu, sigma = self.pred_net(h_x_rounded, q)

        if q == 1:
            h_x_form = h_x_noisy
        elif q == 2:
            h_x_form = h_x_rounded
        elif q == 0:
            h_x_form = h_x_noisy
        else:
            raise NotImplementedError
        
        h_likelihood = self.gaussian_model(h_x_form, torch.abs(sigma), mu)
        net_bits = self.get_net_bits(likelihood_model)

        if torch.any(torch.isnan(h_likelihood)):

            print('problems with bits')
            import IPython
            IPython.embed()
            raise ValueError()

        if torch.any(torch.isnan(x)):
            print('problems with x_form')
            import IPython
            IPython.embed()
            raise ValueError()

        return h_x_rounded, h_likelihood, net_bits
    
    def get_net_bits(self, likelihood_model):
        params = self.pred_net.get_q_params()
        net_bits = torch.stack([likelihood_model(bypass_round16(p)) for p in params])
        return net_bits

class GridRootlessShapeEncoder(nn.Module):
    def __init__(self, in_channels=8, out_channels=4):
        super(GridRootlessShapeEncoder, self).__init__()
        global SEED2
        global seed_ptr
        

        self.h_analysis_2 = IConv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            bias=True,
            padding=0,
            SEED=SEED2[seed_ptr:]
        )
        seed_ptr += self.h_analysis_2.offset
        self.gdn_2 = GDN3d(out_channels)
    
    def forward(self, x, q):
        h_x = self.gdn_2(self.h_analysis_2(x))
        return h_x

class QuantGaussianLikelihood(nn.Module):
    """Entropy coder module, with signaled parameters."""
    def __init__(self, in_channels, step_size=1, iQ=1, assume_zero_mean=False):
        super(QuantGaussianLikelihood, self).__init__()
        self.gaussian_model = GaussianModel(step_size)
        if iQ == 1:
            self.bypass_round = bypass_round
        elif iQ == 16:
            self.bypass_round = bypass_round16
        else:
            raise NotImplementedError
        self.Q = 1/iQ
        self.assume_zero_mean = assume_zero_mean

        sigma = torch.nn.Parameter(
            torch.ones((1,in_channels,1,1,1), dtype=torch.float32, requires_grad=True))
        self.register_parameter('sigma', sigma)
        mu = torch.nn.Parameter(
            torch.zeros((1,in_channels,1,1,1), dtype=torch.float32, requires_grad=True))
        if assume_zero_mean:
            self.register_buffer('mu', mu)
        else:
            self.register_parameter('mu', mu)
    
    def forward(self, x, mode='train'):
        # Quantize
        noise = torch.rand_like(x) - 0.5
        # nosie: [-1/2 \delta, 1/2 \delta]
        x_noisy = x + noise * self.Q
        x_rounded = self.bypass_round(x)

        if mode == 'train':
            x_form = x_noisy
        elif mode == 'eval':
            x_form = x_rounded
        likelihood = self.gaussian_model(x_form, torch.abs(self.sigma), self.mu)

        # if torch.any(torch.isnan(likelihood)):
        #     print('problems with bits')
        #     import IPython
        #     IPython.embed()
        #     raise ValueError()

        # if torch.any(torch.isnan(x)):
        #     print('problems with x_form')
        #     import IPython
        #     IPython.embed()
        #     raise ValueError()

        return x_rounded, likelihood
    
    def get_bits(self):
        nbits = np.prod(self.sigma.shape) * 32
        if not self.assume_zero_mean:
            nbits += np.prod(self.mu.shape) * 32
        return nbits
    
    def get_bits_empirical(self, x_all):
        x_rounded = self.bypass_round(x_all)
        sigma = torch.std(x_all)
        mu = torch.mean(x_all)
        likelihood = self.gaussian_model(x_rounded, sigma, mu)
        return x_rounded, likelihood.sum(), sigma, mu

class SpatioQuantGaussianLikelihood(nn.Module):
    """Entropy coder module, with signaled parameters."""
    def __init__(self, in_channels, step_size=1, iQ=1, assume_zero_mean=False):
        super(SpatioQuantGaussianLikelihood, self).__init__()
        self.gaussian_model = GaussianModel(step_size)
        self.assume_zero_mean = assume_zero_mean
        self.Q = 1/iQ
        sigma = torch.nn.Parameter(
            torch.ones((1,in_channels,2,2,2), dtype=torch.float32, requires_grad=True))
        self.register_parameter('sigma', sigma)
        mu = torch.nn.Parameter(
            torch.zeros((1,in_channels,2,2,2), dtype=torch.float32, requires_grad=True))
        if assume_zero_mean:
            self.register_buffer('mu', mu)
        else:
            self.register_parameter('mu', mu)
    
    def forward(self, x, mode='train'):
        # Quantize
        noise = torch.rand_like(x) - 0.5
        # nosie: [-1/2 \delta, 1/2 \delta]
        x_noisy = x + noise * self.Q
        x_rounded = self.bypass_round(x)

        if mode == 'train':
            x_form = x_noisy
        elif mode == 'eval':
            x_form = x_rounded
        likelihood = self.gaussian_model(x_form, torch.abs(self.sigma), self.mu)

        return x_rounded, likelihood
    
    def get_bits(self):
        nbits = np.prod(self.sigma.shape) * 32
        if not self.assume_zero_mean:
            nbits += np.prod(self.mu.shape) * 32
        return nbits

class SingleLayerLatentGen(nn.Module):
    """A form of latent generator"""
    def __init__(self, in_channels=8, out_channels=4):
        super(SingleLayerLatentGen, self).__init__()
        global SEED2
        global seed_ptr
        self.h_analysis_2 = IConv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            bias=True,
            padding=0,
            SEED=SEED2[seed_ptr:]
        )
        seed_ptr += self.h_analysis_2.offset
        self.gdn_2 = GDN3d(out_channels)

    def forward(self, x):
        h_x = self.gdn_2(self.h_analysis_2(x))
        return h_x

class SpatioSingleLayerLatentGen(nn.Module):
    """A form of latent generator"""
    def __init__(self, in_channels=8, out_channels=4):
        super(SpatioSingleLayerLatentGen, self).__init__()
        global SEED2
        global seed_ptr

        self.h_analysis_2 = IConv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            bias=True,
            padding=1,
            SEED=SEED2[seed_ptr:]
        )
        seed_ptr += self.h_analysis_2.offset
        self.gdn_2 = GDN3d(out_channels)
    
    def forward(self, x):
        h_x = self.gdn_2(self.h_analysis_2(x))
        return h_x


class GDNLatentGen(nn.Module):
    """A form of latent generator"""
    def __init__(self, in_channels=8):
        super(GDNLatentGen, self).__init__()
        self.gdn = GDN3d(in_channels)
    
    def forward(self, x):
        h_x = self.gdn_2(x)
        return h_x

class CompDecoder(nn.Module):
    """A form of reconstructor"""
    def __init__(self, args, param_model, in_channels=4, useIGDN=False, channels=(8,16,8,8)):
        super(CompDecoder, self).__init__()
        global SEED2
        global seed_ptr

        self.channels = channels
        self.useIGDN = useIGDN
        
        if useIGDN:
            self.activation = IGDN3d(channels[0])
        else:
            self.activation = nn.ReLU()

        self.up0 = QConvTranspose3d(
            in_channels=in_channels,
            out_channels=channels[0],
            kernel_size=5,
            stride=2,
            padding=2,
            output_padding=1,
            bias=True,
            SEED=SEED2[seed_ptr:])
        seed_ptr += self.up0.offset

        self.conv0 = QConvTranspose3d(
            in_channels=channels[0],
            out_channels=channels[1],
            kernel_size=5,
            stride=2,
            padding=2,
            output_padding=1,
            bias=True,
            SEED=SEED2[seed_ptr:])
        seed_ptr += self.conv0.offset

        self.up1 = QConvTranspose3d(
            in_channels=channels[1],
            out_channels=channels[2],
            kernel_size=5,
            stride=2,
            bias=True,
            SEED=SEED2[seed_ptr:])
        seed_ptr += self.up1.offset

        self.conv1 = QConv3d(
            in_channels=channels[2],
            out_channels=channels[2],
            kernel_size=4,
            stride=1,
            bias=True,
            padding=0,
            SEED=SEED2[seed_ptr:])
        seed_ptr += self.conv1.offset

        self.up2 = QConvTranspose3d(
            in_channels=channels[2],
            out_channels=channels[3],
            kernel_size=5,
            stride=2,
            bias=True,
            SEED=SEED2[seed_ptr:])
        seed_ptr += self.up2.offset

        self.conv2 = QConv3d(
            in_channels=channels[3],
            out_channels=channels[3],
            kernel_size=4,
            stride=1,
            bias=True,
            padding=0,
            SEED=SEED2[seed_ptr:])
        seed_ptr += self.conv2.offset

        self.conv2_cls = QConv3d(
            in_channels=channels[3],
            out_channels=1,
            kernel_size=3,
            stride=1,
            bias=True,
            padding=1,
            SEED=SEED2[seed_ptr:])
        seed_ptr += self.conv2_cls.offset
        
        self.conv1_cls = IConv3d(
            in_channels=channels[2],
            out_channels=1,
            kernel_size=3,
            stride=1,
            bias=True,
            padding=1,
            SEED=SEED2[seed_ptr:])
        seed_ptr += self.conv1_cls.offset
        
        self.conv0_cls = IConv3d(
            in_channels=channels[1],
            out_channels=1,
            kernel_size=3,
            stride=1,
            bias=True,
            padding=1,
            SEED=SEED2[seed_ptr:])
        seed_ptr += self.conv0_cls.offset
    
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.likelihood_model = GaussianLikelihoodModel(step_size=1/16)

    def forward(self, x, q):
        out = self.activation(self.up0(x, q))
        out = self.relu(self.conv0(out, q))
        out_cls_0 = self.sigmoid(self.conv0_cls(out))
        out = self.relu(self.up1(out, q))
        out = self.relu(self.conv1(out, q))
        out_cls_1 = self.sigmoid(self.conv1_cls(out))
        out = self.relu(self.up2(out, q))
        out = self.relu(self.conv2(out, q))
        out = self.conv2_cls(out, q)
        out = self.sigmoid(out)

        out_cls_list = [out_cls_0, out_cls_1, out]

        # if torch.any(torch.isnan(out)):
        #     import IPython
        #     IPython.embed()
        #     raise ValueError()

        params = self.get_q_params()
        net_bits = torch.stack([self.likelihood_model(bypass_round16(p)) for p in params])
        return out, out_cls_list, net_bits

    def get_q_params(self):
        params = []
        
        params.append(self.up0.kernel)
        params.append(self.conv0.kernel)
        params.append(self.up1.kernel)
        params.append(self.conv1.kernel)
        params.append(self.up2.kernel)
        params.append(self.conv2.kernel)
        params.append(self.conv2_cls.kernel)

        return params
    
    def get_bits(self):
        params = self.get_q_params()
        net_bits = torch.stack([self.likelihood_model(bypass_round16(p)) for p in params])
        aux_bits = sum([self.channels[i]*2 for i in (1,2,3)]) * 32 + 32 + (self.channels[1] ** 2 + self.channels[1]) * 32
        nbits = net_bits.sum().item() + aux_bits
        return nbits


class CompStraightDecoder(nn.Module):
    """A form of reconstructor"""
    def __init__(self, args, param_model, in_channels=4, useIGDN=False):
        super(CompStraightDecoder, self).__init__()
        global SEED2
        global seed_ptr

        channels=[8,16,8,8]
        self.channels = channels
        self.useIGDN = useIGDN
        
        if useIGDN:
            self.activation = IGDN3d(channels[0])
        else:
            self.activation = nn.ReLU()

        self.up0 = QConvTranspose3d(
            in_channels=in_channels,
            out_channels=channels[0],
            kernel_size=5,
            stride=2,
            padding=2,
            output_padding=1,
            bias=True,
            SEED=SEED2[seed_ptr:])
        seed_ptr += self.up0.offset

        self.conv0 = QConvTranspose3d(
            in_channels=channels[0],
            out_channels=channels[1],
            kernel_size=5,
            stride=2,
            padding=2,
            output_padding=1,
            bias=True,
            SEED=SEED2[seed_ptr:])
        seed_ptr += self.conv0.offset

        self.up1 = QConv3d(
            in_channels=channels[1],
            out_channels=channels[2],
            kernel_size=3,
            stride=1,
            bias=True,
            padding=1,
            SEED=SEED2[seed_ptr:])
        seed_ptr += self.up1.offset

        self.conv1 = QConv3d(
            in_channels=channels[2],
            out_channels=channels[2],
            kernel_size=3,
            stride=1,
            bias=True,
            padding=1,
            SEED=SEED2[seed_ptr:])
        seed_ptr += self.conv1.offset

        self.up2 = QConv3d(
            in_channels=channels[2],
            out_channels=channels[3],
            kernel_size=3,
            stride=1,
            bias=True,
            padding=1,
            SEED=SEED2[seed_ptr:])
        seed_ptr += self.up2.offset

        self.conv2 = QConv3d(
            in_channels=channels[3],
            out_channels=channels[3],
            kernel_size=3,
            stride=1,
            bias=True,
            padding=1,
            SEED=SEED2[seed_ptr:])
        seed_ptr += self.conv2.offset

        self.conv2_cls = QConv3d(
            in_channels=channels[3],
            out_channels=1,
            kernel_size=3,
            stride=1,
            bias=True,
            padding=1,
            SEED=SEED2[seed_ptr:])
        seed_ptr += self.conv2_cls.offset
        
        self.conv1_cls = IConv3d(
            in_channels=channels[2],
            out_channels=1,
            kernel_size=3,
            stride=1,
            bias=True,
            padding=1,
            SEED=SEED2[seed_ptr:])
        seed_ptr += self.conv1_cls.offset
        
        self.conv0_cls = IConv3d(
            in_channels=channels[1],
            out_channels=1,
            kernel_size=3,
            stride=1,
            bias=True,
            padding=1,
            SEED=SEED2[seed_ptr:])
        seed_ptr += self.conv0_cls.offset
    
        self.relu = nn.GELU()
        # self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.likelihood_model = GaussianLikelihoodModel(step_size=1/16)

    def forward(self, x, q):
        out = self.activation(self.up0(x, q))
        out = self.relu(self.conv0(out, q))
        out = F.interpolate(out, (32,32,32), mode='trilinear', align_corners=True)
        out_cls_0 = self.sigmoid(self.conv0_cls(out))
        out = self.relu(self.up1(out, q))
        out = self.relu(self.conv1(out, q))
        out_cls_1 = self.sigmoid(self.conv1_cls(out))
        out = self.relu(self.up2(out, q))
        out = self.relu(self.conv2(out, q))
        out = self.conv2_cls(out, q)
        out = self.sigmoid(out)
        out_cls_list = [out_cls_0, out_cls_1, out]

        if torch.any(torch.isnan(out)):
            import IPython
            IPython.embed()
            raise ValueError()

        params = self.get_q_params()
        net_bits = torch.stack([self.likelihood_model(bypass_round16(p)) for p in params])
        return out, out_cls_list, net_bits

    def get_q_params(self):
        params = []
        
        params.append(self.up0.kernel)
        params.append(self.conv0.kernel)
        params.append(self.up1.kernel)
        params.append(self.conv1.kernel)
        params.append(self.up2.kernel)
        params.append(self.conv2.kernel)
        params.append(self.conv2_cls.kernel)

        return params
    
    def get_bits(self):
        params = self.get_q_params()
        net_bits = torch.stack([self.likelihood_model(bypass_round16(p)) for p in params])
        aux_bits = sum([self.channels[i]*2 for i in (1,2,3)]) * 32 + 32 + (self.channels[1] ** 2 + self.channels[1]) * 32
        nbits = net_bits.sum().item() + aux_bits
        return nbits


class CompAllStraightDecoder(nn.Module):
    """A form of reconstructor"""
    def __init__(self, args, param_model, in_channels=4, useIGDN=False):
        super(CompAllStraightDecoder, self).__init__()
        global SEED2
        global seed_ptr

        channels=[8,16,8,8]
        self.channels = channels
        self.useIGDN = useIGDN
        
        if useIGDN:
            self.activation = IGDN3d(channels[0])
        else:
            self.activation = nn.ReLU()

        self.up0 = QConv3d(
            in_channels=in_channels,
            out_channels=channels[0],
            kernel_size=5,
            stride=1,
            padding=2,
            bias=True,
            SEED=SEED2[seed_ptr:])
        seed_ptr += self.up0.offset

        self.conv0 = QConv3d(
            in_channels=channels[0],
            out_channels=channels[1],
            kernel_size=5,
            stride=1,
            padding=2,
            bias=True,
            SEED=SEED2[seed_ptr:])
        seed_ptr += self.conv0.offset

        self.up1 = QConv3d(
            in_channels=channels[1],
            out_channels=channels[2],
            kernel_size=3,
            stride=1,
            bias=True,
            padding=1,
            SEED=SEED2[seed_ptr:])
        seed_ptr += self.up1.offset

        self.conv1 = QConv3d(
            in_channels=channels[2],
            out_channels=channels[2],
            kernel_size=3,
            stride=1,
            bias=True,
            padding=1,
            SEED=SEED2[seed_ptr:])
        seed_ptr += self.conv1.offset

        self.up2 = QConv3d(
            in_channels=channels[2],
            out_channels=channels[3],
            kernel_size=3,
            stride=1,
            bias=True,
            padding=1,
            SEED=SEED2[seed_ptr:])
        seed_ptr += self.up2.offset

        self.conv2 = QConv3d(
            in_channels=channels[3],
            out_channels=channels[3],
            kernel_size=3,
            stride=1,
            bias=True,
            padding=1,
            SEED=SEED2[seed_ptr:])
        seed_ptr += self.conv2.offset

        self.conv2_cls = QConv3d(
            in_channels=channels[3],
            out_channels=1,
            kernel_size=3,
            stride=1,
            bias=True,
            padding=1,
            SEED=SEED2[seed_ptr:])
        seed_ptr += self.conv2_cls.offset
        
        self.conv1_cls = IConv3d(
            in_channels=channels[2],
            out_channels=1,
            kernel_size=3,
            stride=1,
            bias=True,
            padding=1,
            SEED=SEED2[seed_ptr:])
        seed_ptr += self.conv1_cls.offset
        
        self.conv0_cls = IConv3d(
            in_channels=channels[1],
            out_channels=1,
            kernel_size=3,
            stride=1,
            bias=True,
            padding=1,
            SEED=SEED2[seed_ptr:])
        seed_ptr += self.conv0_cls.offset
    
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.likelihood_model = GaussianLikelihoodModel(step_size=1/16)

    def forward(self, x, q):
        out = F.interpolate(x, (32,32,32), mode='trilinear', align_corners=True)
        out = self.activation(self.up0(out, q))
        out = self.relu(self.conv0(out, q))
        out_cls_0 = self.sigmoid(self.conv0_cls(out))
        out = self.relu(self.up1(out, q))
        out = self.relu(self.conv1(out, q))
        out_cls_1 = self.sigmoid(self.conv1_cls(out))
        out = self.relu(self.up2(out, q))
        out = self.relu(self.conv2(out, q))
        out = self.conv2_cls(out, q)
        out = self.sigmoid(out)

        out_cls_list = [out_cls_0, out_cls_1, out]

        if torch.any(torch.isnan(out)):
            import IPython
            IPython.embed()
            raise ValueError()

        params = self.get_q_params()
        net_bits = torch.stack([self.likelihood_model(bypass_round16(p)) for p in params])
        return out, out_cls_list, net_bits

    def get_q_params(self):
        params = []
        
        params.append(self.up0.kernel)
        params.append(self.conv0.kernel)
        params.append(self.up1.kernel)
        params.append(self.conv1.kernel)
        params.append(self.up2.kernel)
        params.append(self.conv2.kernel)
        params.append(self.conv2_cls.kernel)

        return params
    
    def get_bits(self):
        params = self.get_q_params()
        net_bits = torch.stack([self.likelihood_model(bypass_round16(p)) for p in params])
        aux_bits = sum([self.channels[i]*2 for i in (1,2,3)]) * 32 + 32 + (self.channels[1] ** 2 + self.channels[1]) * 32
        nbits = net_bits.sum().item() + aux_bits
        return nbits
