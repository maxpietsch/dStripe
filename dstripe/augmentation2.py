#
# designed to be compatible with batchgenerators
#
#
# ________________ AUGMENTATIONS _______________

import random, math
import numpy as np
from builtins import range
# from batchgenerators.augmentations.utils import get_range_val
# from scipy.ndimage import gaussian_filter
# from batchgenerators.augmentations.noise_augmentations import augment_gaussian_blur


# def augment_rician_noise(data, noise_variance=(0, 0.1)):
#     for sample_idx in range(data.shape[0]):
#         sample = data[sample_idx]
#         variance = random.uniform(noise_variance[0], noise_variance[1])
#         sample = np.sqrt(
#             (sample + np.random.normal(0.0, variance, size=sample.shape)) ** 2 +
#             np.random.normal(0.0, variance, size=sample.shape) ** 2)
#         data[sample_idx] = sample
#     return data


def interleave_pattern(nstripes, interleave=3):
    m = interleave
    v_idx = np.cumsum(np.ones(nstripes)) - 1
    for mm in range(m):
        idx = np.mod(v_idx, m) == mm
        # print(m, mm, idx[:10])
        yield idx


def rice_stripe_pattern(nstripes, noise_variance=(0.1, 0.1), centre_variance=(0.8, 1.2), epsilon=1e-5):
    variance = random.uniform(*noise_variance)
    centre = random.uniform(*centre_variance)
    # print(centre, variance)
    return np.maximum(np.sqrt(np.random.normal(loc=centre, scale=variance, size=nstripes) ** 2), epsilon)


def affine_block_pattern(nstripes, block_size=16, centre_range=(1., 1.), slope_range=(0.1, 0.2), flip_slope=True):
    slope_range = [i/block_size for i in slope_range]
    pattern = np.ones(nstripes)
    slope_sign = 1
    if flip_slope:
        slope_sign = np.random.choice([-1,1])
    for block in range(math.ceil(float(nstripes)/block_size)):
        start, stop = block*block_size, min((block+1)*block_size, nstripes)
        nsl = stop - start
        if nsl > 1:
            centre = np.random.uniform(*centre_range)
            slope = slope_sign * np.random.uniform(*slope_range)
            pattern[start:stop] = np.linspace(0, nsl*slope, num=nsl)
            pattern[start:stop] += centre - pattern[start:stop].mean()
    return pattern

def augment_rot90(data, rotations, planes):
    """ applies random 90 degree rotation
    planes: Rotation direction is from the first towards the second axis.
    rotations: Number of times the array is rotated by 90 degrees

    """
    assert len(planes) == len(rotations)
    for axes in planes:
        assert len(axes) == 2
    for i, nrot in enumerate(rotations):
        ax0, ax1 = planes[i]
        data = np.rot90(data, k=nrot, axes=(ax0, ax1))
    return data


def augment_flip(data, flips):
    """ flips image, returns copy of input (positive strides, in order)

    """
    assert len(flips)
    for flip in flips:
        data = np.flip(data, flip)
    return data.copy()

# ________________ TRANSFORMS _______________

from batchgenerators.transforms.abstract_transforms import AbstractTransform


# ________________ utility ________________

class DebugInfoTransform(AbstractTransform):
    """ Debug

    """

    def __init__(self):
        pass

    def __call__(self, **data_dict):
        print("DebugTransform:", data_dict.keys())
        return data_dict


class GetState(AbstractTransform):
    def __call__(self, **sample):
        nrs = np.random.get_state()
        rs = random.getstate()
        print('<---------------- state:', np.random.choice(1000), random.random(), np.random.rand(),
                      np.sum(rs[1][:-1]), np.sum(rs[1][-1]),
                      np.sum(nrs[1][:-1]), np.sum(nrs[1][-1]),
              '------------------------>')
        return sample


class DebugLoadTransform(AbstractTransform):
    """ Debug

    """

    def __init__(self, data_key="data", label_key="seg"):
        self.data_key = data_key
        self.label_key = label_key

    def __call__(self, **data_dict):
        data = data_dict.get(self.data_key, None)
        seg = data_dict.get(self.label_key, None)

        print("DebugLoadTransform:", data.shape)
        import time
        t = time.time
        u = t() + 10
        while t() < u: 1

        return data_dict


class DebugTransform(AbstractTransform):
    """ Debug

    """

    def __init__(self):
        pass

    def __call__(self, **data_dict):
        assert data_dict is not None
        print("DebugTransform:", 'idx:', data_dict.get('idx', "none"), data_dict.keys())
        # import time
        # t = time.time
        # u = t() + 10
        # while t() < u:
        #     1

        return data_dict


class KeyRename(AbstractTransform):
    """Utility compatability function. Converts keys between "data" (and "seg") and "source" (and "target")
    """

    def __init__(self, keys_map=(['data', 'source'], ['seg', 'target'])):
        self.keys_map = keys_map

    def __call__(self, **data_dict):
        assert self.keys_map[0][0] in data_dict.keys() or self.keys_map[0][1] in data_dict.keys(), data_dict.keys()
        source = 1 - int(self.keys_map[0][0] in data_dict.keys())
        target = 1 - source

        for keys in self.keys_map:
            if keys[source] in data_dict:
                data_dict[keys[target]] = data_dict.pop(keys[source])

        return data_dict

class RndChoiceTransform(AbstractTransform):
    """Applies a transformation from a list of transformations, optionally also no transformation

    Args:
        transforms: The transformations (or composed transformations)

    """

    def __init__(self, *transforms, identity_transform=False):
        self.identity_transform = identity_transform
        self.transforms = transforms

    def __call__(self, **data_dict):
        chc = np.random.choice(len(self.transforms) + self.identity_transform)
        if chc == len(self.transforms):
            return data_dict

        return self.transforms[chc](**data_dict)

# ________________ stripes ________________


class GeneralStripeTransform(AbstractTransform):
    """Applies correlated multiplicative slice pattern
    Args:

    """

    def __init__(self, interleave=3, block_size=16, copy2target=True, power=1., demean=True,
                 centre_variance=(0.9, 1.1), noise_variance=(0.05, 0.05),
                 block_slope_range=(0., 0.3), apply_block_to_target=False,
                 data_key="source", modes=None):
        self.data_key = data_key
        self.copy2target = copy2target
        self.noise_variance = noise_variance
        self.centre_variance = centre_variance
        self.interleave = interleave
        self.block_size = block_size
        self.block_slope_range = block_slope_range
        self.apply_block_to_target = apply_block_to_target
        self.flip_slope = True
        self.degmean = demean
        self.power = power
        self.modes = modes

        assert centre_variance[0] <= centre_variance[1], centre_variance
        assert block_slope_range[0] <= block_slope_range[1], block_slope_range
        assert noise_variance[0] <= noise_variance[1], noise_variance

        if self.apply_block_to_target:
            assert self.block_size > 2, 'apply_block_to_target requires block_size > 2'

    def __call__(self, **data_dict):
        if self.copy2target:
            assert 'target' not in data_dict, (data_dict.keys(), data_dict.get('augment_log', None))
            data_dict['target'] = data_dict[self.data_key].copy()

        sample = data_dict[self.data_key]

        # check mode is specified, if so, only perform if mode matches or unspecified
        if self.modes is not None and len(self.modes):
            if 'mode' in data_dict:  # mode augmentation used (otherwise assume default mode)
                assert len(set(data_dict['mode'])) == 1, data_dict['mode']

                if data_dict['mode'][0] is not None and data_dict['mode'][0] not in self.modes:
                    # print('mode:', data_dict['mode'], 'not applied:', self.__repr__())
                    return data_dict
            elif 'IS' not in self.modes:  # default mode... TODO make sure mode is always recorded
                return data_dict
            # print('mode:', data_dict['mode'] if 'mode' in data_dict else 'None', 'applied:', self.__repr__())

        nstripes = sample[0].shape[-1]
        data_dict['pattern'] = []
        for sample_idx in range(sample.shape[0]):
            pattern_block = np.ones(nstripes)
            if self.interleave > 1:
                pattern = np.ones(nstripes)
                for sel in interleave_pattern(nstripes, interleave=self.interleave):
                    pattern[sel] *= rice_stripe_pattern(nstripes=sel.sum(),
                                                        centre_variance=self.centre_variance,
                                                        noise_variance=self.noise_variance)
            else:
                pattern = rice_stripe_pattern(nstripes=nstripes,
                                              centre_variance=self.centre_variance,
                                              noise_variance=self.noise_variance)
            if self.block_size > 2:
                pattern_block = affine_block_pattern(nstripes, block_size=self.block_size, centre_range=(1., 1.),
                                                          slope_range=self.block_slope_range,
                                                          flip_slope=self.flip_slope)
            if self.power != 1.0:
                pattern = np.power(pattern, self.power)
                pattern_block = np.power(pattern_block, self.power)
            if self.degmean:
                pattern /= np.exp(np.log(pattern).mean())
                pattern_block /= np.exp(np.log(pattern_block).mean())

            data_dict[self.data_key][sample_idx] *= pattern * pattern_block
            if self.apply_block_to_target:
                assert 'target' in data_dict, 'apply_block_to_target requires target'
                data_dict['target'] *= pattern_block
                data_dict['pattern'].append(pattern)
            else:
                data_dict['pattern'].append(pattern * pattern_block)

        return data_dict

    def __repr__(self):
        params = '(noise_variance={0}, centre_variance={1}, degmean:{2}'.format(
            self.noise_variance, self.centre_variance, self.degmean)
        if self.interleave > 1:
            params = params + ', interleave={0}'.format(self.interleave)
        if self.block_size > 1:
            params = params + ', block_size={0}, block_slope_range={1}'.format(self.block_size, self.block_slope_range)
        if self.power != 1.0:
            params = params + ', power={0}'.format(self.power)
        if self.modes is not None and len(self.modes):
            params = params + ', modes={0}'.format(self.modes)
        return self.__class__.__name__ + params + ')'


class AxialSliceDir(AbstractTransform):
    """ random rotation that changes the slice direction (rotate AP or LR direction onto SI direction)
    one per sample in batch to allow non-cuboid image shape"""

    def __init__(self, data_fields=('source', 'target', 'mask_source', 'mask_target'), modes=('AP', 'LR')):
        self.data_fields = data_fields
        if isinstance(modes, str):
            modes = tuple([modes])
        self.modes = modes

    def __call__(self, **data_dict):
        batchsize = data_dict[self.data_fields[0]].shape[0]
        data_tmp = dict()
        if 'augment_log' not in data_dict:
            data_dict['augment_log'] = [[] for i in range(batchsize)]

        assert len(self.modes)
        if len(self.modes) == 1:
            mode = self.modes[0]
        else:
            mode = random.choice(self.modes)

        if 'mode' not in data_dict:
            data_dict['mode'] = []
        data_dict['mode'].append(mode)

        if mode.upper() == 'AP':
            planes, rotations = [[-1, -2]], [random.choice([-3, -1, 1, 3])]
        elif mode.upper() == 'LR':
            planes, rotations = [[-1, -3]], [random.choice([-3, -1, 1, 3])]
        else:
            assert 0, "mode not understood, require (('AP'), ('LR')): {}".format(self.modes)

        for isample in range(batchsize):
            data_dict['augment_log'][isample].append(('augment_rot90', rotations, planes))
        for data_field in self.data_fields:
            if not data_field in data_dict:
                continue
            data_tmp[data_field] = []
            for isample in range(batchsize):
                data_tmp[data_field].append(augment_rot90(data_dict[data_field][isample], rotations, planes))
            data_dict[data_field] = np.stack(data_tmp[data_field], 0)
        return data_dict

class RandomDihedralSliceDirPreserving(AbstractTransform):
    """
    Rotates 3D image by random multiples of 90 degrees, then applies reflection. Preserves 'slice' axis (not up/down if flipped)
    """

    def __init__(self, data_fields=('source', 'target', 'mask_source', 'mask_target'),
                 flip_slice_dir=True, flip_non_slice_dir=True, determinant_preserving=False):
        self.flip_slice_dir = flip_slice_dir
        self.flip_non_slice_dir = flip_non_slice_dir
        self.data_fields = data_fields
        self.determinant_preserving = determinant_preserving

    def __call__(self, **data_dict):
        batch_size = data_dict[self.data_fields[0]].shape[0]
        if 'augment_log' not in data_dict:
            data_dict['augment_log'] = [[] for i in range(batch_size)]

        planes, rotations = [[-2, -3]], [random.randint(-3, 3)]
        flips = []
        if self.flip_slice_dir and random.random() < 0.5:
            flips.append(-1)
        if self.flip_non_slice_dir:
            if random.random() < 0.5:
                flips.append(-2)
            if random.random() < 0.5:
                flips.append(-3)
        if len(flips):
            if self.determinant_preserving and len(flip) % 2 != 0:
                flips = random.shuffle(flips)[:-2]
        for isample in range(batch_size):
            data_dict['augment_log'][isample].append(('augment_rot90', rotations, planes))
            if len(flips):
                data_dict['augment_log'][isample].append(('flips', flips))
        for data_field in self.data_fields:
            if data_field not in data_dict:
                continue
            data_temp = []
            for isample in range(batch_size):
                data_temp.append(augment_rot90(data_dict[data_field][isample], rotations, planes))
                if len(flips):
                    data_temp[-1] = augment_flip(data_temp[-1], tuple(flips))
            data_dict[data_field] = np.array(data_temp)
        return data_dict


class UndoDihedral(AbstractTransform):
    def __init__(self, data_fields=('source', 'target', 'output', 'mask_source', 'mask_target')):
        self.data_fields = data_fields


    def __call__(self, **data_dict):
        batch_size = data_dict[self.data_fields[0]].shape[0]
        if 'augment_log' not in data_dict:
            return data_dict

        for isample in range(batch_size):
            operations = []
            auglog = data_dict['augment_log']
            for op in auglog[isample]:
                assert isinstance(op[0], str), op
                if op[0] == 'augment_rot90':
                    assert len(op) == 3, (op, auglog)
                    operations.append((op[0], [-r for r in op[1]], op[2]))  # invert number of rotations
                elif op[0] == 'flips':
                    assert len(op) == 2, (op, auglog)
                    operations.append((op[0], [f for f in op[1][::-1]]))  # reverse order of flips
                else:
                    print(self.__class__.__name__+' not understood:', op)
            operations = operations[::-1]  # invert order
            break  # TODO check other samples
#         print(operations, data_dict['augment_log'][isample])
        for data_field in self.data_fields:
            if data_field not in data_dict:
                continue
            data_temp = []
            for isample in range(batch_size):
                tmp = data_dict[data_field][isample]
                for op in operations:
                    # print(data_field, isample, op)
                    if op[0] == 'augment_rot90':
                        tmp = augment_rot90(tmp, op[1], op[2])
                    elif op[0] == 'flips':
                        tmp = augment_flip(tmp, tuple(op[1]))
                    else:
                        assert 0, op
                data_temp.append(tmp)

            if isinstance(tmp, np.ndarray):
                data_dict[data_field] = np.array(data_temp)
            elif isinstance(tmp, torch.Tensor):
                data_dict[data_field] = torch.stack(data_temp)
            else:
                assert 0, type(tmp)

        return data_dict

class TensorToNumpy(AbstractTransform):
    def __init__(self, keys=None):
        """Utility function
        """
        if not isinstance(keys, (list, tuple)):
            keys = [keys]
        self.keys = keys

    def __call__(self, **data_dict):

        if self.keys is None:
            for key, val in data_dict.items():
                if isinstance(val, torch.tensor):
                    data_dict[key] = val.data.numpy()
        else:
            for key in self.keys:
                data_dict[key] = val.data.numpy()

        return data_dict
