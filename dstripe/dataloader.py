import os
import sys
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from numpy import expand_dims
import pprint
import random

from utils.mif import load_mrtrix
import sklearn
from sklearn.model_selection import train_test_split


class SampleCropToMask4D(object):
    """ Spatial crop of 4D image in centre, jiggle centre by +- jiggle voxels (uniform)
    """

    def __init__(self):
        pass

    def __call__(self, **sample):
        sample_out = dict()

        # sample_out['source'] = sample['source'][...,lo[0]:hi[0],lo[1]:hi[1],lo[2]:hi[2]]
        # if 'target' in sample:
        #     sample_out['target'] = sample['target'][...,lo[0]:hi[0],lo[1]:hi[1],lo[2]:hi[2]]

        for k in list(set(sample.keys()) - set(['source', 'target'])):
            sample_out[k] = sample[k]

        assert 0, 'TODO'
        return sample_out


class SampleVolumeExtract(object):
    def __init__(self, volumes=slice(None), volgroups=None):
        self.slice = [Ellipsis, volumes]
        if volgroups is not None:
            self.slice.append(volgroups)
        self.slice = tuple(self.slice)

    def __call__(self, **sample):
        sample_out = dict()
        sample_out['source'] = sample['source'][slice(self.slice)]
        if len(sample_out['source'].shape) < 4:
            np.expand_dims(sample_out['source'], 4)
        if 'target' in sample:
            sample_out['target'] = sample['target'][slice(self.slice)]
            if len(sample_out['target'].shape) < 4:
                np.expand_dims(sample_out['target'], 4)

        for k in list(set(sample.keys()) - set(['source', 'target'])):
            sample_out[k] = sample[k]

        return sample_out


class SampleToTensor4D(object):
    """Convert a ``numpy.ndarray`` to tensor without rescaling (as opposed to torchvision.transforms.ToTensor).

    Converts numpy.ndarray (C x H x W x D) to a torch.FloatTensor of same shape.

    .. note::
        This transform does not act in-place, i.e., it does not mutate the input.
    """

    def __init__(self, keys=('source', 'target', 'mask_source', 'mask_target')):
        self.keys = keys

    def __call__(self, **sample):
        """
        Args:
            sample dict{source: numpy.ndarray, target: numpy.ndarray or not existent}: Images to be converted to tensor.

        Returns:
            dict: Converted images.
        """
        sample_out = dict()
        for key in self.keys:
            if key not in sample:
                continue
            if isinstance(sample[key], np.ndarray):
                sample_out[key] = torch.from_numpy(sample[key]).float()
            else:
                raise TypeError(key+' type:' + str(type(sample[key])))

        for k in list(set(sample.keys()) - set(self.keys)):
            sample_out[k] = sample[k]

        return sample_out


class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, **img):
        for t in self.transforms:
            img = t(**img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class ZCANormalise(object):
    def __init__(self, epsilon=1e-5):
        self.ZCAMatrix = None
        self.epsilon = epsilon
        self.mean = None

    def fit(self, X):
        """
        Function to compute ZCA whitening matrix (aka Mahalanobis whitening).
        INPUT:  X: [M x N] matrix.
            Rows: Variables
            Columns: Observations
        OUTPUT: ZCAMatrix: [M x M] matrix
        """
        self.mean = X.mean(1)[..., None]

        # Covariance matrix [column-wise variables]: Sigma = (X-mu)' * (X-mu) / N
        sigma = np.cov(X, rowvar=True)  # [M x M]
        # Singular Value Decomposition. X = U * np.diag(S) * V
        U, S, V = np.linalg.svd(sigma)
        # U: [M x M] eigenvectors of sigma.
        # S: [M x 1] eigenvalues of sigma.
        # V: [M x M] transpose of U
        # Whitening constant: prevents division by zero

        # ZCA Whitening matrix: U * Lambda * U'
        self.ZCAMatrix = np.dot(
            U, np.dot(np.diag(1.0/np.sqrt(S + self.epsilon)), U.T))  # [M x M]

    def transform(self, X):
        assert self.ZCAMatrix is not None
        return np.dot(self.ZCAMatrix, X - self.mean)

    def transform_inverse(self, X):
        raise NotImplementedError('TODO: transform_inverse')

    @property
    def scale_offset(self):
        raise NotImplementedError('TODO: scale_offset')


class MeanStdNormalise(object):
    def __init__(self, epsilon=1e-5):
        self.epsilon = epsilon
        self.std = None
        self.mean = None

    def fit(self, X):
        assert len(X.shape) == 2
        self.mean = X.mean(1, keepdims=True)
        self.std = X.std(1, keepdims=True)
        # self.mean = X.contiguous().view(X.shape[0],-1).mean(1,keepdim=True)
        # self.std = X.contiguous().view(X.shape[0], -1).std(1, keepdim=True)

    def transform(self, X):
        assert self.mean is not None
        return (X - self.mean) / (self.std + self.epsilon)

    def transform_inverse(self, X):
        return X * (self.std + self.epsilon) + self.mean

    @property
    def scale_offset(self):
        return self.std + self.epsilon, self.mean


class MinMaxNormalise(object):
    def __init__(self):
        self.min = None
        self.max = None

    def fit(self, X):
        assert len(X.shape) == 2
        self.min = X.min(1, keepdims=True)
        self.max = X.max(1, keepdims=True)

    def transform(self, X):
        assert self.min is not None
        assert self.max is not None
        return (X - self.min) / (self.max - self.min)

    def transform_inverse(self, X):
        return X * (self.max - self.min) + self.min

    @property
    def scale_offset(self):
        return self.max - self.min, self.min


class ConstNormalise(object):
    def __init__(self, lo=0, hi=0.95):
        self.lo = lo
        self.hi = hi

    def fit(self, X):
        pass

    def transform(self, X):
        return (X - self.lo) / (self.hi - self.lo)

    def transform_inverse(self, X):
        return X * (self.hi - self.lo) + self.lo

    @property
    def scale_offset(self):
        return self.hi - self.lo, self.lo


class PercentileNormalise(object):
    def __init__(self, p=(0, 98), lo_zero=False):
        self.lo = None
        self.hi = None
        self.p = p
        self.lo_zero = lo_zero
        assert p[0] < p[1], p

    def fit(self, X):
        assert len(X.shape) == 2
        if not self.lo_zero:
            self.lo = np.percentile(X, self.p[0], axis=1)
        else:
            self.lo = 0
        self.hi = np.percentile(X, self.p[1], axis=1)

    def transform(self, X):
        assert self.lo is not None
        assert self.hi is not None
        return (X - self.lo) / (self.hi - self.lo)

    def transform_inverse(self, X):
        return X * (self.hi - self.lo) + self.lo

    @property
    def scale_offset(self):
        return self.hi - self.lo, self.lo


class SHPairedDataset4D(torch.utils.data.Dataset):
    """4D dataset."""

    def __init__(self, root_dir, metadata, paired=True, transform=None, load_mask=True, cropped_to_mask=False, normalise='none',
                 import_functions=None, data_postproc=None):
        """
        Args:
            root_dir (string): Directory containing all images
            metadata (list): list of dicts with source and target file paths relative to root_dir, optional additional metadata
            transform (callable, optional): Optional transform to be applied on a sample.
        """

        self.metadata = []
        self.root_dir = root_dir
        self.paired = paired
        self.transform = transform

        if normalise == 'zca':
            self.normalise = ZCANormalise()
        elif normalise == 'meanstd':
            self.normalise = MeanStdNormalise()
        elif normalise == 'minmax':
            self.normalise = MinMaxNormalise()
        elif normalise == 'percentile':
            self.normalise = PercentileNormalise(p=(0, 98))
        elif normalise == 'none':
            self.normalise = None
        else:
            self.normalise = normalise

        # self.imdim = imdim

        self.import_functions = import_functions
        if self.import_functions is not None:
            print('SHPairedDataset4D import_functions:',
                  str(self.import_functions))
        self.data_postproc = data_postproc
        if self.data_postproc is not None:
            print('SHPairedDataset4D data_postproc:', str(self.data_postproc))

        self._pp = pprint.PrettyPrinter(indent=4, width=160, depth=None)
        self.log = lambda x: print(x) if isinstance(
            x, str) else self._pp.pprint(x)

        # pre-load images into RAM
        self.imagedata = dict()
        self.log('loading data, cropped to mask: {}'.format(cropped_to_mask))
        whats = ['source']
        if self.paired:
            whats += ['target']
        idx = 0
        for md in metadata:
            self.log(md)

            imdata = {}
            for what in whats:
                assert what in md, (md.keys(), self.paired)
                importer = None if self.import_functions is None or what not in self.import_functions else self.import_functions[
                    what]
                impath = os.path.join(self.root_dir, md[what])
                msk = None
                if load_mask or cropped_to_mask:
                    assert 'mask_'+what in md, md.keys()
                    msk = os.path.join(self.root_dir, md['mask_'+what])

                # image = load_mrtrix(impath)
                # if hasattr(image, 'grad'):
                #     md[what+'_grad'] = image.grad
                imdata.update({what: self.__load_mif(
                    impath, mask=msk if cropped_to_mask else None, fun=importer)})
                imdata[what+"_file"] = impath
                if load_mask:
                    imdata.update(
                        {'mask_'+what: self.__load_mif(msk, mask=msk if cropped_to_mask else None)})
            assert len(self.metadata) == idx, (len(self.metadata), idx)
            import types
            if isinstance(self.data_postproc, types.FunctionType) or callable(self.data_postproc):
                gen = self.data_postproc(imagedata=imdata, metadata=md)
                if isinstance(gen, types.GeneratorType):
                    for imdat, mdat in gen:
                        self.imagedata[idx] = imdat
                        self.metadata.append(mdat)
                        idx += 1
                else:
                    assert isinstance(
                        gen, dict) and 'imdata' in gen and 'metadata' in gen, str(gen)
                    self.imagedata[idx] = gen['imdata']
                    self.metadata.append(gen['metadata'])
                    idx += 1
            else:
                assert self.data_postproc is None, str(self.data_postproc)
                self.imagedata[idx] = imdata
                self.metadata.append(md)
                idx += 1

            assert len(self.metadata) == idx and len(
                self.imagedata) == idx, (len(self.metadata), len(self.imagedata), idx)

        self.log('loading of {} images in {} shards done'.format(
            len(metadata), len(self.metadata)))

    def __len__(self):
        assert len(self.metadata) == len(self.imagedata)
        return len(self.imagedata)

    def fit_normalise(self, masked=True):
        Xs = []
        if 'target' in self.imagedata[0]:
            print('target not used for normalisation')
        for sample in self.imagedata.values():
            im = sample['source']
            X = im.reshape(im.shape[0], -1)
            if masked and 'mask_source' in sample and sample['mask_source'] is not None:
                m = sample['mask_source']
                Xs.append(X[:, m.ravel() > 0.5])
            else:
                Xs.append(X)

        self.normalise.fit(np.concatenate(Xs, axis=1, out=None))

    def apply_normalise(self):
        for sample in self.imagedata.values():
            for what in ['source', 'target']:
                if what not in sample:
                    continue
                shp = sample[what].shape
                sample[what] = self.normalise.transform(
                    sample[what].reshape(shp[0], -1)).reshape(shp)

    def __bbox2_3D(self, img, pad=1):
        # https://stackoverflow.com/questions/31400769/bounding-box-of-numpy-array
        shape = img.shape
        r = np.any(img, axis=(1, 2))
        c = np.any(img, axis=(0, 2))
        z = np.any(img, axis=(0, 1))

        rmin, rmax = np.where(r)[0][[0, -1]]
        cmin, cmax = np.where(c)[0][[0, -1]]
        zmin, zmax = np.where(z)[0][[0, -1]]
        if pad:
            return max(0, rmin-pad), min(rmax+pad, shape[0]-1), max(0, cmin-pad), min(cmax+pad, shape[1]-1), max(0, zmin-pad), min(zmax+pad, shape[2]-1)

        return rmin, rmax, cmin, cmax, zmin, zmax

    def __load_mif(self, path, mask=None, fun=None):
        assert path is not None
        assert os.path.isfile(path), 'can not find image %s' % (path)

        im = load_mrtrix(path).data
        if len(im.shape) == 3:
            im = im[..., None]
        if fun is not None:
            im = fun(im)

        assert len(im.shape) == 4, im.shape

        if mask is not None:
            assert os.path.isfile(mask), 'can not find mask %s' % (mask)
            mask = load_mrtrix(mask).data
            if len(mask.shape) == 3:
                mask = mask[..., None]
            xmin, xmax, ymin, ymax, zmin, zmax = self.__bbox2_3D(mask)
            return im[xmin:xmax+1, ymin:ymax+1, zmin:zmax+1].transpose((3, 0, 1, 2)).copy()

        # (H x W x D x C)--> (C x H x W x D), C-contiguous
        return im.transpose((3, 0, 1, 2)).copy()

    def __getitem__(self, idx):
        if idx not in self.imagedata:
            raise Exception('key %i not in imagedata (%i, %i)' % (
                idx, min(self.imagedata.keys()), max(self.imagedata.keys())))

        sample = self.imagedata[idx]
        sample['idx'] = idx

        # self.log(['before transform',sample.keys()])

        if self.transform is not None:
            try:
                sample = self.transform(**sample)
            except TypeError:
                print("failed:", self.transform, sample.keys())
                raise
            # self.log(['after transform',sample.keys()])

        return sample


def get_tra_val_loader(root_dir, metadata, transform=None, batch_size=2, split=True, paired=True, shuffle=False,
                       num_workers=4, val_ratio=0.2, pin_memory=False, cropped_to_mask=False,
                       import_functions=None, data_postproc=None):
    """function for loading and returning training and validation Dataloader
    :return:
        if split the data set then returns:
        - train_loader: Dataloader for training
        - valid_loader: Dataloader for validation
        else returns:
        - dataloader: Dataloader of all the data set
    """

    def worker_init_fn(worker_id):
        """
        set different numpy random seed for each worker by offset from initial seed
        see https://github.com/pytorch/pytorch/issues/5059
        :param worker_id:
        """

        seed = (np.random.get_state()[
                1][0] + worker_id + torch.initial_seed()) % 4294967295 # unsigned 32bit integer
        np.random.seed(seed)
        random.seed(seed)

    if split:
        train_md, val_md = train_test_split(metadata, test_size=val_ratio)

        train_transformed_dataset = SHPairedDataset4D(root_dir=root_dir,
                                                      metadata=train_md,
                                                      paired=paired,
                                                      transform=transform,
                                                      cropped_to_mask=cropped_to_mask,
                                                      import_functions=import_functions,
                                                      data_postproc=data_postproc)

        val_transformed_dataset = SHPairedDataset4D(root_dir=root_dir,
                                                    metadata=val_md,
                                                    paired=paired,
                                                    transform=transform,
                                                    cropped_to_mask=cropped_to_mask,
                                                    import_functions=import_functions,
                                                    data_postproc=data_postproc)

        train_loader = DataLoader(train_transformed_dataset, batch_size=batch_size, shuffle=shuffle,
                                  num_workers=num_workers, pin_memory=pin_memory, worker_init_fn=worker_init_fn)
        val_loader = DataLoader(val_transformed_dataset, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=pin_memory, worker_init_fn=worker_init_fn)
        return train_loader, val_loader
    else:
        train_transformed_dataset = SHPairedDataset4D(root_dir=root_dir,
                                                      metadata=metadata,
                                                      paired=paired,
                                                      transform=transform,
                                                      cropped_to_mask=cropped_to_mask,
                                                      import_functions=import_functions,
                                                      data_postproc=data_postproc)
        train_loader = DataLoader(train_transformed_dataset, batch_size=batch_size, shuffle=shuffle,
                                  num_workers=num_workers, pin_memory=pin_memory, worker_init_fn=worker_init_fn)
        return train_loader


class TransformTensorDataset(torch.utils.data.Dataset):
    """
    A simple dataset - loads the tensor that are passed in input. This is the same as
    torch.utils.data.TensorDataset except that you can add transformations to your data and target tensor.
    Target tensor can also be None, in which case it is not returned.
    """

    def __init__(self, data_tensor, label_tensor, transforms=None):
        self.data_tensor = data_tensor
        self.label_tensor = label_tensor
        assert data_tensor.size(0) == label_tensor.size(0)

        if transforms is None:
            transforms = []

        if not isinstance(transforms, list):
            transforms = [transforms]

        self.transforms = transforms

    def __getitem__(self, index):

        data_tensor = self.data_tensor[index]
        for transform in self.transforms:
            data_tensor = transform(data_tensor)
        return data_tensor, self.label_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)
