import torch
from torch import nn
import math
# from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

######### helpers

class Lambda(nn.Module):
    def __init__(self, f): super().__init__(); self.f=f
    def forward(self, x): return self.f(x)

class Flatten(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, x): return x.view(x.size(0), -1)

class ReflectionPad3d(torch.nn.modules.padding._ReflectionPadNd):
    def __init__(self, padding):
        super(ReflectionPad3d, self).__init__()
        self.padding = torch.nn.modules.utils._ntuple(6)(padding)
        raise NotImplementedError("TODO: 5D padding not implemented")

class ZPad3D(nn.Module):
    def __init__(self, padding, mode="reflect"):
        super(ZPad3D, self).__init__()
        self.padding = torch.nn.modules.utils._ntuple(4)((padding, padding, 0, 0))
        self.mode = mode

    def forward(self,x):
        if x.shape[0]!=1:
            raise NotImplementedError("TODO: 5D padding with batch size > 1 not implemented")
        return F.pad(x[0], self.padding, self.mode).unsqueeze(0)

class XYPad3D(nn.Module):
    def __init__(self, padding, mode="constant"):
        super(XYPad3D, self).__init__()
        self.padding = torch.nn.modules.utils._ntuple(4)((0, 0, padding, padding, padding, padding))
        self.mode = mode

    def forward(self,x):
        if x.shape[0]!=1:
            raise NotImplementedError("TODO: 5D padding with batch size > 1 not implemented")
        return F.pad(x[0], self.padding, self.mode).unsqueeze(0)


class PartialConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        """ arXiv:1804.07723 Image Inpainting for Irregular Holes Using Partial Convolutions
        TODO: feather mask: so that original bleeds through (exponentially)"""

        super(PartialConv2d, self).__init__(*args, **kwargs)

        self.patch_size = float(self.kernel_size[0] * self.kernel_size[1])

        self.weight_mask = torch.ones(1, 1, self.kernel_size[0], self.kernel_size[1])

        self.update_mask = None
        self.inv_support = None

    def forward(self, x, mask, return_mask=False):
        assert len(x.shape) == 4

        with torch.no_grad():
            if self.weight_mask.type() != x.type():
                self.weight_mask = self.weight_mask.to(x)

            self.update_mask = F.conv2d(mask, self.weight_mask, bias=None, stride=self.stride,
                                        padding=self.padding, dilation=self.dilation, groups=1)

            self.inv_support = self.patch_size/(self.update_mask + 1e-4)
            self.update_mask = torch.clamp(self.update_mask, 0, 1)
            self.inv_support = torch.mul(self.inv_support, self.update_mask)

        if self.bias is None:
            output = torch.mul(super(PartialConv2d, self).forward(torch.mul(x, mask)), self.inv_support)
        else:
            bias = self.bias.view(1, self.out_channels, 1, 1)
            output = torch.mul(super(PartialConv2d, self).forward(torch.mul(x, mask)) - bias, self.mask_ratio) + bias
            output = torch.mul(output, self.update_mask)

        output[mask > 0.5] = X[mask > 0.5]  # don't change values inside original mask

        if return_mask:
            return output, self.update_mask
        return output


class InplaneLowpassFilter3D(nn.Module):
    def __init__(self, channels=1, param=3, kernel_size=9, mode="reflect", filter_type="sinc-hamming", **kwargs):
        super(InplaneLowpassFilter3D, self).__init__()
        self.mode = mode
        self.padding = torch.nn.modules.utils._ntuple(4)((0, 0, kernel_size // 2, kernel_size // 2))

        ks = kernel_size
        assert ks > 0, ks
        assert ks % 2 == 1, ks
        Z = 1
        if filter_type.lower() == "gaussian":
            blurrrs = torch.tensor([0]).double()
            Z = blurrrs.shape[-1]
            param = blurrrs + param

            x_coord = torch.arange(kernel_size)
            x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
            y_grid = x_grid.t()
            xy_grid = torch.stack([x_grid, y_grid], dim=-1)

            mean = (kernel_size - 1) / 2.
            variance = param ** 2.
            variance = variance.view(1, 1, -1)

            gaussian_kernel = (1. / (2. * math.pi * variance)) * torch.exp(
                -torch.sum((xy_grid - mean) ** 2., dim=-1).double().unsqueeze(2) / (2 * variance))
            kernel = gaussian_kernel / torch.sum(gaussian_kernel, dim=[0, 1])

        elif filter_type.lower().startswith('sinc'):
            # sinc window
            sinc_bw = param
            t = torch.arange(-ks // 2 + 1, ks // 2 + 1).float()
            kernel = torch.sin(2 * math.pi * sinc_bw * t) / (2 * math.pi * sinc_bw * t)
            kernel[kernel != kernel] = 1
            if filter_type.endswith("hamming"):
                # apply hamming window
                if ks > 1:
                    kernel *= 0.54 - 0.46 * torch.cos((2 * math.pi * (t + ks // 2)) / (ks - 1))
            kernel = kernel[:, None] * kernel[None, :]
        elif filter_type.lower() == 'butterworth':
            order = kwargs.pop('butterworth_order', 3)
            if not 0 < param <= 1.0:
                raise ValueError('Cutoff frequency must be between 0 and 1.0')

            x = np.linspace(-0.5, 0.5, ks)
            y = np.linspace(-0.5, 0.5, ks)
            if len(x) == 1:
                x[:] = 1.
            if len(y) == 1:
                y[:] = 1.

            distance = np.sqrt((x ** 2)[None] + (y ** 2)[:, None])
            f = 1 / (1.0 + (distance / param) ** (2 * order))
            import scipy.fftpack as fftpack
            from scipy.fftpack import ifftshift, fftshift
            f = fftshift(fftpack.ifft2(ifftshift(f))).real
            kernel = torch.from_numpy(f)
        else:
            assert 0, filter_type
        # kernel = kernel.float()
        kernel /= kernel.sum()
        kernel = kernel.view(1, 1, kernel_size, kernel_size, Z)
        kernel = kernel.repeat(channels, 1, 1, 1, 1)

        self.filter = nn.Conv3d(in_channels=channels, out_channels=channels,
                                kernel_size=(kernel_size, kernel_size, 1),
                                stride=(1, 1, 1), padding=0, groups=channels,
                                bias=False)
        self.filter.weight.data = kernel
        self.filter.weight.requires_grad = False

    def forward(self, x):
        shape = x.shape
        shape_padded = [s for s in shape]
        shape_padded[-2] += self.padding[-1] + self.padding[-2]
        shape_padded[-3] += self.padding[-1] + self.padding[-2]
        xp = F.pad(x.view(shape[0] * shape[1], *shape[2:]), self.padding, self.mode)  # pad y
        xp = F.pad(xp.view(1, shape[0] * shape[1], shape[2], -1), self.padding, self.mode)  # pad x
        return self.filter(xp.view(*shape_padded))


class InplaneGaussian(nn.Module):
    """ TODO replace by InplaneLowpassFilter """
    def __init__(self, channels=1, blur_sigma=3, kernel_size=9, mode="reflect"):
        super(InplaneGaussian, self).__init__()
        self.mode = mode
        self.padding = torch.nn.modules.utils._ntuple(4)((0, 0, kernel_size // 2, kernel_size // 2))

        assert kernel_size % 2 == 1, kernel_size

        blurrrs = torch.tensor([0]).double()
        Z = blurrrs.shape[-1]
        sigma = blurrrs + blur_sigma  # 3 * torch.sigmoid(blurrrs) + 1e-6

        x_coord = torch.arange(kernel_size)
        x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1)

        mean = (kernel_size - 1) / 2.
        variance = sigma ** 2.
        variance = variance.view(1, 1, -1)

        gaussian_kernel = (1. / (2. * math.pi * variance)) * torch.exp(
            -torch.sum((xy_grid - mean) ** 2., dim=-1).double().unsqueeze(2) / (2 * variance))
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel, dim=[0, 1])
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size, Z)
        gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1, 1)

        self.gaussian = nn.Conv3d(in_channels=channels, out_channels=channels,
                                  kernel_size=(kernel_size, kernel_size, 1),
                                  stride=(1, 1, 1), padding=0, groups=channels,
                                  bias=False)
        self.gaussian.weight.data = gaussian_kernel
        self.gaussian.weight.requires_grad = False

    def forward(self, x):
        shape = x.shape
        shape_padded = [s for s in shape]
        shape_padded[-2] += self.padding[-1] + self.padding[-2]
        shape_padded[-3] += self.padding[-1] + self.padding[-2]
        xp = F.pad(x.view(shape[0] * shape[1], *shape[2:]), self.padding, self.mode)  # pad y
        xp = F.pad(xp.view(1, shape[0] * shape[1], shape[2], -1), self.padding, self.mode)  # pad x
        return self.gaussian(xp.view(*shape_padded))


class MaskFill3D(nn.Conv3d):
    def __init__(self, *args, **kwargs):
        """ fills in holes in mask by local averaging using valid points

        inspired by arXiv:1804.07723 Image Inpainting for Irregular Holes Using Partial Convolutions """

        super(MaskFill3D, self).__init__(*args, **kwargs)

        ks = self.kernel_size[:3]
        self.patch_size = float(ks[0] * ks[1] * ks[2])

        # mask filter:
        self.weight_mask = torch.ones(1, 1, ks[0], ks[1], ks[2]) # * self.patch_size

        # filter to fill values:
        filter_type = kwargs.pop('type', 'rbf')
        self.bias = None
        if filter_type == 'average':
            self.weight.data.fill_(1.0 / self.patch_size) #  = torch.ones(1, self.in_channels, *ks)
        elif filter_type == 'rbf':
            sx, sy, sz = kwargs.pop('rbf_scale', (1.0, 1.0, 1.0))
            sigma = kwargs.pop('rbf_sigma', 1.0)

            x = sx * np.linspace(-0.5, 0.5, ks[0])
            y = sy * np.linspace(-0.5, 0.5, ks[1])
            z = sz * np.linspace(-0.5, 0.5, ks[2])

            distance_sq = (x**2)[:, None, None] + (y**2)[None, :, None] + (z**2)[None, None, :]
            rbf = np.exp(distance_sq/(-2*sigma**2))
            rbf /= rbf.sum()
            self.weight.data = torch.from_numpy(rbf.astype(np.float32)).view(1, 1, *rbf.shape)
        else:
            assert 0, filter_type
        self.weight.requires_grad = False

        self.update_mask = None
        self.inv_support = None

    def forward(self, x, mask, return_mask=True):
        assert len(x.shape) == 5

        with torch.no_grad():
            if self.weight_mask.type() != x.type():
                self.weight_mask = self.weight_mask.to(x)

            self.update_mask = F.conv3d(mask, self.weight_mask, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1)

            self.inv_support = self.patch_size/(self.update_mask + 1e-4)
            self.update_mask = torch.clamp(self.update_mask, 0, 1)
            self.inv_support = torch.mul(self.inv_support, self.update_mask)

            if self.bias is None:
                output = torch.mul(super(MaskFill3D, self).forward(torch.mul(x, mask)), self.inv_support)
            else:
                bias = self.bias.view(1, self.out_channels, 1, 1)
                output = torch.mul(super(MaskFill3D, self).forward(torch.mul(x, mask)) - bias, self.mask_ratio) + bias
                output = torch.mul(output, self.update_mask)

            output[mask > 0.5] = x[mask > 0.5]  # don't change values inside original mask
            if return_mask:
                return output, self.update_mask
            return output


class SliceLowPassFilter(nn.Module):
    def __init__(self, channels=1, kernel_size=9, padding="reflect", filter_type="gaussian", gauss_std=1, sinc_bw=0.25):
        """
        returns image lowpass-filtered in slice direction (last dimension)

        :param channels: integer
        :param kernel_size: integer
        :param padding: one of ["reflect", "replicate", ...]
        :param filter_type: one of gauss, sinc, sinc_hamming
        :param gauss_std: sigma of Gaussian filter
        :param sinc_bw: bandwith of sinc filter
        """
        super(SliceLowPassFilter, self).__init__()
        self.padding_mode = padding
        self.padding = kernel_size // 2
        self.type = filter_type

        assert kernel_size % 2 == 1, kernel_size

        if self.type.startswith("sinc"):
            ks = kernel_size
            assert ks > 0, ks
            assert ks % 2 == 1, ks
            # sinc window
            t = torch.arange(-ks // 2 + 1, ks // 2 + 1).float()
            kernel = torch.sin(2 * math.pi * sinc_bw * t) / (2 * math.pi * sinc_bw * t)
            kernel[kernel != kernel] = 1
            if self.type.endswith("hamming"):
                # apply hamming window
                if ks > 1:
                    kernel *= 0.54 - 0.46 * torch.cos((2 * math.pi * (t + ks // 2)) / (ks - 1))
        elif self.type.startswith("gauss"):
            sigma = torch.tensor([gauss_std]).double()
            coord = torch.arange(kernel_size)
            mean = (kernel_size - 1) / 2.
            variance = sigma ** 2.
            kernel = torch.exp(-(coord.float() - mean).double() ** 2. / (2. * variance))
        else:
            assert 0, self.type
        kernel = kernel.double()
        kernel /= kernel.sum()

        self.conv = nn.Conv3d(in_channels=channels, out_channels=channels, kernel_size=(1, 1, kernel_size),
                                  stride=1, padding=0, groups=channels, bias=False)
        self.conv.weight.data = kernel.repeat(channels, 1, 1, 1, 1)
        # kernel.expand_as(gaussian.weight.data) changes stride of dim 0
        self.conv.weight.requires_grad = False

    def forward(self, x):
        shape = x.shape
        assert len(shape) == 5, shape
        shape_padded = [s for s in shape]
        shape_padded[-1] += 2 * self.padding
        p = torch.nn.modules.utils._ntuple(4)((self.padding, self.padding, 0, 0))
        if self.padding_mode == "replicate": # replicate padding is implemented for 4D input
            xp = F.pad(x.view(x.shape[0] * x.shape[1], *x.shape[2:]), p, self.padding_mode).view(*shape_padded)
        else: # reshape to b * c, xy, z
            xp = F.pad(x.view(1, shape[0] * shape[1], -1, shape[-1]), p, self.padding_mode).view(*shape_padded)
        return self.conv(xp)


class WeightedSliceFilter(nn.Module):
    def __init__(self, channels=1, kernel_size=9, kernel_width=1, weight_smooth=6, padding="reflect",
                 filter_type="sinc-hamming", gauss_std=1.5, sinc_bw=0.15):
        """
        Handy to low-pass filter log field in slice direction using a (weighted) mask image for attention via
        weighted convolution.

        only provide mask if x is log-transformed multiplicative field

        :param channels: integer
        :param kernel_size: integer
        :param kernel_width: unused
        :param weight_smooth: weight mask Gaussian inplane smoothing STD, off if set to zero
        :param padding: one of ["reflect", "replicate", ...]
        :param filter_type: one of gauss, sinc, sinc_hamming
        :param gauss_std: sigma of Gaussian filter
        :param sinc_bw: bandwith of sinc filter
        """
        super(WeightedSliceFilter, self).__init__()
        self.padding_mode = padding
        self.padding = kernel_size // 2
        self.type = filter_type

        assert kernel_size % 2 == 1, kernel_size
        assert kernel_width == 1, "TODO, kernel_width > 1"

        if self.type.startswith("sinc"):
            import math
            ks = kernel_size
            assert ks > 0, ks
            assert ks % 2 == 1, ks
            # sinc window
            t = torch.arange(-ks // 2 + 1, ks // 2 + 1).float()
            kernel = torch.sin(2 * math.pi * sinc_bw * t) / (2 * math.pi * sinc_bw * t)
            kernel[kernel != kernel] = 1
            if self.type.endswith("hamming"):
                # apply hamming window
                if ks > 1:
                    kernel *= 0.54 - 0.46 * torch.cos((2 * math.pi * (t + ks // 2)) / (ks - 1))
        elif self.type.startswith("gauss"):
            sigma = torch.tensor([gauss_std]).double()
            coord = torch.arange(kernel_size)
            mean = (kernel_size - 1) / 2.
            variance = sigma ** 2.
            kernel = torch.exp(-(coord.float() - mean).double() ** 2. / (2. * variance))
        else:
            assert 0, self.type
        kernel = kernel.double()
        kernel /= kernel.sum()

        self.conv = nn.Conv3d(in_channels=channels, out_channels=channels,
                              kernel_size=(kernel_width, kernel_width, kernel_size),
                              stride=1, padding=0, groups=channels, bias=False)
        self.conv.weight.data = kernel.repeat(channels, 1, 1, 1, 1)
        # kernel.expand_as(gaussian.weight.data) changes stride of dim 0
        self.conv.weight.requires_grad = False

        self.mask_conv = nn.Conv3d(in_channels=channels, out_channels=channels,
                                   kernel_size=(kernel_width, kernel_width, kernel_size),
                                   stride=1, padding=0, groups=channels, bias=False)
        torch.nn.init.constant_(self.mask_conv.weight, 1.0 / (kernel_width * kernel_width * kernel_size))
        self.mask_conv.weight.requires_grad = False
        self.mask_conv.double()

        self.weight_filter = None
        if weight_smooth:
            import math
            ks = int(math.ceil(weight_smooth) * 2 + 1)
            assert ks % 2 == 1, ks
            self.weight_filter = InplaneGaussian(channels=1, blur_sigma=weight_smooth, kernel_size=ks, mode="reflect")

    def forward(self, x, mask=None, outside_fill=1e-3, ret_support=False):
        shape = x.shape
        if mask is None:
            mask = torch.ones_like(x, dtype=x.dtype, device=x.device)
        else:
            assert mask.shape == x.shape, (mask.shape, x.shape)
            mask = mask.clone()
            if self.weight_filter:
                with torch.no_grad():
                    if outside_fill > 0:
                        mask.masked_fill_(mask < outside_fill, outside_fill)
                    mask = self.weight_filter(mask)

        assert len(shape) == 5, shape
        shape_padded = [s for s in shape]
        shape_padded[-1] += 2 * self.padding

        p = torch.nn.modules.utils._ntuple(4)((self.padding, self.padding, 0, 0))
        # "zero" pad mask (image is mirrored, mask is not)
        mp = F.pad(mask.view(mask.shape[0] * mask.shape[1], *mask.shape[2:]), p, 'constant', outside_fill)
        mp = mp.view(*mask.shape[:-1], shape_padded[-1])

        if self.padding_mode == "replicate": # replicate padding is implemented for 4D input
            xp = F.pad(x.view(x.shape[0] * x.shape[1], *x.shape[2:]), p, self.padding_mode).view(*shape_padded)
        else: # reshape to b * c, xy, z
            xp = F.pad(x.view(1, shape[0] * shape[1], -1, shape[-1]), p, self.padding_mode).view(*shape_padded)

        with torch.no_grad():
            support = self.mask_conv(mp)

        if outside_fill <= 0:
            outside = support == 0
            support = support.masked_fill_(outside, 1.0)

        output = self.conv(xp * mp)
        output = output / support
        if ret_support:
            return output, support
        return output

def weights_init(m):
    if isinstance(m, torch.nn.Conv3d) or isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            with torch.no_grad():
                m.bias.zero_()


######### layers / blocks


class SeparableConv3d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False,preventRedundantParams=False):
        super(SeparableConv3d,self).__init__()

        if kernel_size > 1 or not preventRedundantParams:
            self.conv = nn.Conv3d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
            self.pointwise = nn.Conv3d(in_channels,out_channels,1,1,0,1,1,bias=bias)
        else:
            self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias)
            self.pointwise = lambda x: x
    def forward(self,x):
        x = self.conv(x)
        x = self.pointwise(x)
        return x


class StackedZDilatedSeparableConv3d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,zdilations=[1,2,3],bias=False,grouped=True):
        super(StackedZDilatedSeparableConv3d,self).__init__()
        self.zdilations = zdilations
        if grouped:
            groups=in_channels
        else:
            groups=1
        for idal, dilation in enumerate(self.zdilations):
            setattr(self,'conv%i'%idal, nn.Conv3d(in_channels,in_channels,kernel_size,stride,[padding, padding, padding+dilation-1],[1,1,dilation],groups=groups,bias=bias))
        self.pointwise = nn.Conv3d(in_channels*len(self.zdilations),out_channels,1,1,0,1,1,bias=bias)

    def forward(self,x):
        x = torch.cat([getattr(self,'conv%i'%idal)(x) for idal in range(len(self.zdilations))],1)
        x = self.pointwise(x)
        return x


class StackedMirroredZDilatedSeparableConv3d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,xypadding=0,zdilations=[1,2,3],bias=False,grouped=True):
        # zdilation = [1, 2, 3]
        # stride = 1
        # k1 = 3
        # d = max(zdilation)
        # ((k1 + (k1 - 1) * (d - 1)) // stride - 1) // 2
        super(StackedMirroredZDilatedSeparableConv3d,self).__init__()
        self.zdilations = zdilations
        self.zpad = max(zdilations)-1 + kernel_size//2
        if grouped:
            groups=in_channels
        else:
            groups=1
        for idal, dilation in enumerate(self.zdilations):
            pad = dilation-1 + kernel_size//2
            setattr(self,'pad%i'%idal, ZPad3D(pad, mode="reflect"))
            setattr(self,'conv%i'%idal, nn.Conv3d(in_channels,in_channels,kernel_size,stride,padding=(xypadding, xypadding, 0),
                                                  dilation=(1,1,dilation),groups=groups,bias=bias))
        self.pointwise = nn.Conv3d(in_channels*len(self.zdilations),out_channels,1,1,0,1,1,bias=bias)

    def forward(self,x):
        # for idal in range(len(self.zdilations)):
        #       print (getattr(self,'conv%i'%idal)(getattr(self,'pad%i'%idal)(x)).shape)
        x = torch.cat([getattr(self,'conv%i'%idal)(getattr(self,'pad%i'%idal)(x)) for idal in range(len(self.zdilations))],1)
        x = self.pointwise(x)
        return x


class AdaptiveConcatPool3d(nn.Module):
    def __init__(self, sz=None):
        super().__init__()
        sz = sz or (1,1)
        self.ap = nn.AdaptiveAvgPool3d(sz)
        self.mp = nn.AdaptiveMaxPool3d(sz)
    def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], 1)


class ConcatPool3d(nn.Module):
    def __init__(self, sz=None):
        super().__init__()
        sz = sz or (1,1)
        self.ap = nn.AvgPool3d(sz)
        self.mp = nn.MaxPool3d(sz)
    def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], 1)


class ZeroMaskedLayerNorm(nn.Module):
    # calculates the mean and std over all channels combined.
    # mean is subtracted from all channels. also normalised by global std if std==True
    # the mask for stats is derived from first channel being 0
    # beta and gamma are per-channel.
    def __init__(self, num_features, eps=1e-5, std=True, affine=False, masked=True):
        super(ZeroMaskedLayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps
        self.masked = masked
        self.std = std

        self.maskdim = 0
        self.maskval = 0.0

        if self.affine:
            assert self.std, 'affine needs std'
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def __masked_stats(self, x):
        B = x.shape[0]
        shape = [B] + [1] * (x.dim() - 1)
        mean = x.new(*shape)
        std = x.new(*shape)
        for b in range(B):
            mask = x[b,self.maskdim] != self.maskval
            mean[b] = torch.masked_select(x[b], mask).mean().view(*shape[1:])
            if torch.isnan(mean[b]).any(): # empty mask
                mean[b].fill_(0.0)
                std[b].fill_(1.0)
                continue
            std[b] = torch.masked_select(x[b], mask).std().view(*shape[1:])
        return mean.view(*shape), std.view(*shape)

    def __masked_mean(self, x):
        B = x.shape[0]
        shape = [B] + [1] * (x.dim() - 1)
        mean = x.new(*shape)
        for b in range(B):
            mask = x[b,self.maskdim] != self.maskval
            mean[b] = torch.masked_select(x[b], mask).mean().view(*shape[1:])
            if torch.isnan(mean[b]).any(): # empty mask
                mean[b].fill_(0.0)
                continue
        return mean.view(*shape)

    def forward(self, x):
        if not self.masked:
            shape = [-1] + [1] * (x.dim() - 1)
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            if self.std:
                std = x.view(x.size(0), -1).std(1).view(*shape)
        else:
            if self.std:
                mean, std = self.__masked_stats(x)
            else:
                mean = self.__masked_mean(x)
        if not self.std:
            return x - mean
        y = (x - mean) / (std + self.eps)
        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            y = self.gamma.view(*shape) * y + self.beta.view(*shape)
        return y


class ConvBlock(nn.Module):
    def __init__(self, input_dim ,output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero', zdilation=1, grouped=True, preventRedundantParams=False):
        super(ConvBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = ReflectionPad3d(padding)
        elif pad_type == 'replicate':
            self.pad = ZPad3D(padding, mode="replicate")
        elif pad_type == 'zero':
            self.pad = torch.nn.modules.padding.ConstantPad3d(padding, 0)
        elif pad_type == 'zreflect':
            self.pad = ZPad3D(padding, mode="reflect")
        elif pad_type == 'none-zreflect':
            self.pad = None
            # assert not isinstance(zdilation, int), 'TODO, workaround: wrap dilation in list {}'.format(zdilation)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm3d(norm_dim) # TODO inplace activation and BN: https://github.com/mapillary/inplace_abn
        elif norm == 'in':
            self.norm = nn.InstanceNorm3d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        if isinstance(zdilation, int):
            zdilation = [zdilation]
        #     TODO: self.conv = SeparableConv3d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias, dilation=zdilation)
        if self.pad is None:
            self.conv = StackedMirroredZDilatedSeparableConv3d(input_dim, output_dim, kernel_size, stride,
                                                               xypadding=padding, bias=self.use_bias,
                                                               zdilations=zdilation, grouped=grouped)
        else:
            if len(zdilation) == 1:
                self.conv = SeparableConv3d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias,
                                            dilation=zdilation, preventRedundantParams=preventRedundantParams)
            else:
                self.conv = StackedZDilatedSeparableConv3d(input_dim, output_dim, kernel_size, stride,
                                                           bias=self.use_bias, zdilations=zdilation, grouped=grouped)


    def forward(self, x):
        if self.pad is not None:
            x = self.pad(x)
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, dim, norm='in', activation='relu', pad_type='zero', n_conv = 2):
        super(ResBlock, self).__init__()

        model = []
        for i in range(n_conv - 1):
            model += [ConvBlock(dim, dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)]
        model += [ConvBlock(dim, dim, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out


class ResBlocks(nn.Module):
    def __init__(self, num_blocks, dim, norm='in', activation='relu', pad_type='zero', res_n_conv=2, se_reduction=4):
        super(ResBlocks, self).__init__()
        self.model = []
        for i in range(num_blocks):
            self.model += [ResBlock(dim, norm=norm, activation=activation, pad_type=pad_type, n_conv=res_n_conv)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


def logsumexp(inputs, dim=None, keepdim=False):
    """Numerically stable logsumexp.

    Args:
        inputs: A Variable with any shape.
        dim: An integer.
        keepdim: A boolean.

    Returns:
        Equivalent of log(sum(exp(inputs), dim=dim, keepdim=keepdim)).
    """
    # For a 1-D array x (any array along a single dimension),
    # log sum exp(x) = s + log sum exp(x - s)
    # with s = max(x) being a common choice.
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs
