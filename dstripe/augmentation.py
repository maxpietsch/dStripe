import numpy as np
import random

class SampleCopySourceToTarget(object):
    """ copy target to source with stripe pattern imprinted
    """
    def __init__(self):
        pass

    def __call__(self, **sample):
        sample_out = dict()

        sample_out['source'] = sample['source']
        sample_out['target'] = sample['source']

        for k in list(set(sample.keys()) - set(['source', 'target'])):
            sample_out[k] = sample[k]

        return sample_out


class SamplePairedRandIntensityScale(object):
    def __init__(self, scale_range=(0.8,1.2)):
        self.scale_range = scale_range
        self.__mult = scale_range[1] - scale_range[0]
        self.__offset = 1.0 - self.__mult / 2

    def __call__(self, **sample):
        sample_out = dict()
        scale = (random.random() * self.__mult + self.__offset)
        sample_out['source'] = sample['source'] * scale
        sample_out['target'] = sample['target'] * scale

        for k in list(set(sample.keys()) - set(sample_out.keys())):
            sample_out[k] = sample[k]

        return sample_out


class SampleIntensityClamp(object):
    def __init__(self, percentiles=(0.1,99.9)):
        self.percentiles = percentiles

    def __call__(self, **sample):
        sample_out = dict()

        sample_out['source'] = np.clip(sample['source'], np.percentile(sample['source'],self.percentiles[0]),np.percentile(sample['source'],self.percentiles[1]))
        sample_out['target'] = np.clip(sample['target'], np.percentile(sample['source'],self.percentiles[0]),np.percentile(sample['source'],self.percentiles[1]))

        for k in list(set(sample.keys()) - set(sample_out.keys())):
            sample_out[k] = sample[k]

        return sample_out

class SampleCentralCrop4D(object):
    """ Spatial crop of 4D image in centre, jiggle centre by +- jiggle voxels (uniform)
    """
    def __init__(self, imsize=(90, 92, 56), jiggle=(1,1,1)):
        self.imsize = np.asarray(imsize)
        self.jiggle = np.asarray(jiggle)
        assert len(self.imsize) == 3
        assert len(self.jiggle) == 3
        assert np.all(self.jiggle >= 0)
        self.dojiggle = np.any(self.jiggle>0)

    def __call__(self, **sample):
        sample_out = dict()

        shape_in = np.asarray(sample['source'].shape[1:])
        centre =  shape_in // 2
        assert len(centre) == 3

        if self.dojiggle:
            centre += [np.random.randint(-self.jiggle[0], self.jiggle[0]+1, dtype=int),
                        np.random.randint(-self.jiggle[1], self.jiggle[1]+1, dtype=int),
                        np.random.randint(-self.jiggle[2], self.jiggle[2]+1, dtype=int)]

        r = self.imsize // 2
        lo, hi = centre - r, centre + self.imsize - r

        # limit crop to image
        lo[lo<0] = 0
        if np.any(hi>shape_in):
            hi[hi>shape_in] = shape_in[hi>shape_in]

        sample_out['source'] = sample['source'][...,lo[0]:hi[0],lo[1]:hi[1],lo[2]:hi[2]]
        for k in ['target', 'mask_source', 'mask_target']:
            if k in sample:
                sample_out[k] = sample[k][...,lo[0]:hi[0],lo[1]:hi[1],lo[2]:hi[2]]

        for k in list(set(sample.keys()) - set(sample_out.keys())):
            sample_out[k] = sample[k]

        return sample_out

# https://github.com/fastai/fastai/blob/master/fastai/transforms.py

# def lighting(im, b, c):
#     """ Adjust image balance and contrast """
#     if b==0 and c==1: return im
#     mu = np.average(im)
#     return np.clip((im-mu)*c+mu+b,0.,1.).astype(np.float32)

# def cutout(im, n_holes, length):
#     """ Cut out n_holes number of square holes of size length in image at random locations. Holes may overlap. """
#     r,c,*_ = im.shape
#     mask = np.ones((r, c), np.int32)
#     for n in range(n_holes):
#         y = np.random.randint(0, r)
#         x = np.random.randint(0, c)

#         y1 = int(np.clip(y - length / 2, 0, r))
#         y2 = int(np.clip(y + length / 2, 0, r))
#         x1 = int(np.clip(x - length / 2, 0, c))
#         x2 = int(np.clip(x + length / 2, 0, c))
#         mask[y1: y2, x1: x2] = 0.

#     mask = mask[:,:,None]
#     im = im * mask
#     return im

# class TfmType(IntEnum):
#     """ Type of transformation.
#     Parameters
#         IntEnum: predefined types of transformations
#             NO:    the default, y does not get transformed when x is transformed.
#             PIXEL: x and y are images and should be transformed in the same way.
#                    Example: image segmentation.
#             COORD: y are coordinates (i.e bounding boxes)
#             CLASS: y are class labels (same behaviour as PIXEL, except no normalization)
#     """
#     NO = 1
#     PIXEL = 2
#     COORD = 3
#     CLASS = 4

# class Transform():
#     """ A class that represents a transform.
#     All other transforms should subclass it. All subclasses should override
#     do_transform.
#     Arguments
#     ---------
#         tfm_y : TfmType
#             type of transform
#     """
#     def __init__(self, tfm_y=TfmType.NO):
#         self.tfm_y=tfm_y
#         self.store = threading.local()

#     def set_state(self): pass
#     def __call__(self, x, y):
#         self.set_state()
#         x,y = ((self.transform(x),y) if self.tfm_y==TfmType.NO
#                 else self.transform(x,y) if self.tfm_y in (TfmType.PIXEL, TfmType.CLASS)
#                 else self.transform_coord(x,y))
#         return x, y

#     def transform_coord(self, x, y): return self.transform(x),y

#     def transform(self, x, y=None):
#         x = self.do_transform(x,False)
#         return (x, self.do_transform(y,True)) if y is not None else x

#     @abstractmethod
#     def do_transform(self, x, is_y): raise NotImplementedError


# class RandomLighting(Transform):
#     def __init__(self, b, c, tfm_y=TfmType.NO):
#         super().__init__(tfm_y)
#         self.b,self.c = b,c

#     def set_state(self):
#         self.store.b_rand = rand0(self.b)
#         self.store.c_rand = rand0(self.c)

#     def do_transform(self, x, is_y):
#         if is_y and self.tfm_y != TfmType.PIXEL: return x
#         b = self.store.b_rand
#         c = self.store.c_rand
#         c = -1/(c-1) if c<0 else c+1
#         x = lighting(x, b, c)
#         return x

# class RandomDihedral(CoordTransform):
#     """
#     Rotates images by random multiples of 90 degrees and/or reflection.
#     Please reference D8(dihedral group of order eight), the group of all symmetries of the square.
#     """
#     def set_state(self):
#         self.store.rot_times = random.randint(0,3)
#         self.store.do_flip = random.random()<0.5
#
#     def do_transform(self, x, is_y):
#         x = np.rot90(x, self.store.rot_times)
#         return np.fliplr(x).copy() if self.store.do_flip else x


# class RandomFlip(CoordTransform):
#     def __init__(self, tfm_y=TfmType.NO, p=0.5):
#         super().__init__(tfm_y=tfm_y)
#         self.p=p

#     def set_state(self): self.store.do_flip = random.random()<self.p
#     def do_transform(self, x, is_y): return np.fliplr(x).copy() if self.store.do_flip else x


def _is_tensor_image(img):
    return torch.is_tensor(img) and img.ndimension() == 3

def normalize(tensor, mean, std):
    """Normalize a tensor image with mean and standard deviation.
    .. note::
        This transform acts in-place, i.e., it mutates the input tensor.
    See :class:`~torchvision.transforms.Normalize` for more details.
    Args:
        tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channely.
    Returns:
        Tensor: Normalized Tensor image.
    """
    if not _is_tensor_image(tensor):
        raise TypeError('tensor is not a torch image.')

    # This is faster than using broadcasting, don't change without benchmarking
    for t, m, s in zip(tensor, mean, std):
        t.sub_(m).div_(s)
    return tensor

class Info(object):
    def __init__(self):
        pass

    def __call__(self, **sample):
        print (sample.keys())
        for k in sample:
            print(k, sample[k].shape)
        return sample

class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``
    .. note::
        This transform acts in-place, i.e., it mutates the input tensor.
    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized Tensor image.
        """
        return normalize(tensor, self.mean, self.std)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class SampleCopyToTarget(object):
    """ copy source to target
    """
    def __init__(self):
        pass


    def __call__(self, **sample):
        assert 'target' not in sample, sample.keys()
        sample['target'] = sample['source'].copy()
        return sample


def makeStripes1D(im, bias_noise):
    pattern = np.ones(im.shape[-1])
    v_idx = np.cumsum(np.ones(im.shape[-1]))-1
    for p in bias_noise:
        assert isinstance(p,tuple) and len(p) > 1, 'requires tuple with bias and scale factor'
        m = len(p) - 1
        ppattern = np.ones(im.shape[-1])
        for mm in range(m):
            idx = np.mod(v_idx,m) == mm
    #         print (m, p[mm+1], idx)
            ppattern[idx] *= np.random.rand(idx.sum()) * p[mm+1] + p[0]
        pattern *= ppattern
    return np.maximum(pattern, 1e-5)


def makeStripesJointN1D(im, bias_noise, step=16):
    pattern = np.ones(im.shape[-1])
    v_idx = np.cumsum(np.ones(im.shape[-1]))-1
    for p in bias_noise:
        assert isinstance(p,tuple) and len(p) > 1, 'requires tuple with bias and scale factor'
        m = len(p) - 1
        ppattern = np.ones(im.shape[-1])
        for mm in range(m):
            idx = np.mod(v_idx,m) == mm
            # print (m, p[mm+1], idx)
            for sl in range(step):
                idxx = np.logical_and(idx, np.mod(np.arange(im.shape[-1]),step)==sl)
                ppattern[idxx] *= np.random.randn(1) * p[mm+1] + p[0]
        pattern *= ppattern
    return np.maximum(pattern, 1e-5)

def makeMBSlope1D(im, bias_noise, step=16):
    bias, noise = bias_noise
    pattern = np.ones(im.shape[-1])
    N_packs = int(np.ceil(float(pattern.size) / step))
    for n in range(N_packs):
        start, stop = int(n*step), int(min(pattern.size + 1, (n+1)*step))
        pattern[start:stop] = bias + (1 if random.random() < 0.5 else -1) * np.random.rand(1) * noise * np.linspace(-1, 1, stop-start, endpoint=True)
    return pattern

class StripedSource(object):
    """ stripe pattern imprinted on source
    """
    def __init__(self, bias_noise=[(1, 0.1)], bias_noise_pack=(1, 0.1), mode='1D_16-slice-pack', masked=True, decenter=False, copy_source2target=False):
        self.bias_noise = bias_noise
        self.bias_noise_pack = bias_noise_pack
        self.mode = mode
        self.masked = masked
        self.decenter = decenter
        self.copy_s2t = copy_source2target

    def __call__(self, **sample):
        sample_out = dict()
        if self.copy_s2t:
            sample_out['target'] = sample['source']
        else:
            sample_out['target'] = sample['target']
        if self.mode == '1D':
            pattern = makeStripes1D(sample['source'], self.bias_noise)
        elif self.mode =='1D_16-slice-pack':
            pattern = makeMBSlope1D(sample['source'], self.bias_noise_pack) *\
                      makeStripesJointN1D(sample['source'], self.bias_noise)
            pattern = np.maximum(pattern, 1e-2)
            pattern /= pattern.mean()
            if self.masked:
                assert 'mask_source' in sample, 'StripedSource masked=True requires source mask'
                pattern = sample['mask_source'] * pattern + 1.0 * np.logical_not(sample['mask_source'])
        else:
            assert 0, self.mode

        if self.decenter:
            imin = sample['source'].min()
            sample_out['source'] = (sample['source'] + imin) * pattern - imin
        else:
            sample_out['source'] = sample['source'] * pattern

        for k in list(set(sample.keys()) - set(sample_out.keys())):
            sample_out[k] = sample[k]

        return sample_out

class StripedSourceCleanTarget(object):
    """ Move source to target, copy target to source with stripe pattern imprinted
    """
    def __init__(self, bias_noise = [(1, 0.01), (1, 0.01, 0.01), (1, 0.2, -0.05, -0.05)], masked=False):
        self.bias_noise = bias_noise
        self.mode = '1D'
        self.masked = masked

    def __call__(self, **sample):
        sample_out = dict()

        sample_out['target'] = sample['source']
        if self.mode == '1D':
            if self.masked:
                sample_out['source'] = sample['source'].copy()
                assert 'mask_source' in sample, 'StripedSourceCleanTarget masked=True requires source mask'
                pattern = makeStripes1D(sample['source'], self.bias_noise)
                sample_out['source'] = sample['source'] * (sample['mask_source'] * pattern + 1.0 * np.logical_not(sample['mask_source']))
            else:
                sample_out['source'] = sample['source'] * makeStripes1D(sample['source'], self.bias_noise)
        else:
            assert 0, self.mode

        for k in list(set(sample.keys()) - set(sample_out.keys())):
            sample_out[k] = sample[k]

        return sample_out

class SampleRandomCrop4D(object):
    """Crop randomly the image in a sample.
    Args:
        output_size (tuple or int): Desired output size. If int, cuboid crop.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size, output_size)
        else:
            assert len(output_size) == 3
            self.output_size = output_size

    def __call__(self, **sample):
        source = sample['source']
        target = None
        if 'target' in sample:
            target = sample['target']
            assert np.all(source.shape == target.shape)

        h_i, w_i, d_i = source.shape[1:4]
        h_o, w_o, d_o = self.output_size

        top = 0
        left = 0
        front = 0
        if h_i > h_o:
            top = np.random.randint(0, h_i - h_o)
        if w_i > w_o:
            left = np.random.randint(0, w_i - w_o)
        if d_i > d_o:
            front = np.random.randint(0, d_i - d_o)

        sample_out = {'source': source[ :,
                                        top: top + h_o,
                                        left: left + w_o,
                                        front: front + d_o]}

        if target is not None:
            sample_out['target'] = target[ :,
                                        top: top + h_o,
                                        left: left + w_o,
                                        front: front + d_o]

        for k in list(set(sample.keys()) - set(sample_out.keys())):
            sample_out[k] = sample[k]

        return sample_out
