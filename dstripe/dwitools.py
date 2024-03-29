import copy
import numpy as np
import os


def balanced_sample_maker(X, y, class_sample_size=10, shuffle=False, random_seed=None):
    uniq_levels = np.unique(y)
    uniq_counts = {level: sum(y == level) for level in uniq_levels}

    if random_seed is not None:
        np.random.seed(random_seed)

    groupby_levels = {}
    for ii, level in enumerate(uniq_levels):
        obs_idx = [idx for idx, val in enumerate(y) if val == level]
        groupby_levels[level] = obs_idx

    # undersampling
    sample_size = min(class_sample_size, min(uniq_counts.values()))
    balanced_copy_idx = []
    for group in sorted(groupby_levels.keys()):
        under_sample_idx = sorted(np.random.choice(
            groupby_levels[group], size=sample_size, replace=False).tolist())
        balanced_copy_idx += under_sample_idx
    if shuffle:
        np.random.shuffle(balanced_copy_idx)

    return X[balanced_copy_idx, :], y[balanced_copy_idx]


def get_all(arr):
    assert arr.ndim == 4
    return arr


def get_b0l0(arr):
    assert arr.ndim == 5
    a = np.expand_dims(arr[..., 0, 0], 4)
    assert a.ndim == 4
    return a


def get_l0(arr):
    assert arr.ndim == 5
    a = arr[..., 0]
    assert a.ndim == 4
    return a


def get_n_bbalanced(arr, n=20, gradient=None):
    assert arr.ndim == 4, arr.shape
    # TODO: read dwischeme from metadata
    if gradient is None:
        if arr.shape[3] == 300:
            grad = np.round(np.loadtxt(os.path.join(os.path.dirname(
                os.path.abspath(__file__)), 'dhcp300.txt'))[:, 3]).astype(int)
        elif arr.shape[3] == 288:
            grad = np.round(np.loadtxt(os.path.join(os.path.dirname(
                os.path.abspath(__file__)), 'hcp288.txt'))[:, 3]).astype(int)
        else:
            raise NotImplementedError(
                "TODO: use grad from metadata " + str(arr.shape))
    else:
        assert gradient.shape[0] == arr.shape[0], (gradient.shape, arr.shape)
    X = np.arange(arr.shape[3])[:, None]
    y = grad[:, None]
    X_sel, y_sel = balanced_sample_maker(
        X, y, class_sample_size=n//len(np.unique(y)))
    arr = arr[..., X_sel.ravel()]
    assert arr.ndim == 4, arr.shape
    return {'imdata': arr, 'metadata': {'b': y[X_sel.ravel()].tolist(), 'volume': X_sel.ravel().tolist()}}


def b0stats(im, md):
    b0 = im['source'][0]
    msk = im['mask_source'][0] > 0.5
    md['b0stats'] = (np.min(b0[msk]), np.percentile(b0[msk], 99))
    return im, md


def b0matchnormalise_fun(im, md):
    assert 'mask_source' in im, im.keys()
    assert 'source' in im, im.keys()
    v = md['b0stats'][1] / np.percentile(im['source'][im['mask_source'] > 0.5], 99)
    im['source'] *= v
    md['scaled'] = v


def normalise_fun(im, md, p=99):
    for k in ['source', 'mask_source']:
        assert k in im, im.keys()
        assert isinstance(im[k], np.ndarray), (k, type(im[k]), im[k].keys())
    v = 1.0 / np.percentile(im['source'][im['mask_source'] > 0.5], p)
    im['source'] *= v
    md['scaled'] = v
    return im, md


def normalise_fun_affine(im, md):
    assert 'mask_source' in im, im.keys()
    assert 'source' in im, im.keys()
    o = np.percentile(im['source'][im['mask_source'] > 0.5], 1)
    im['source'] -= o
    md['offset'] = o
    v = 1.0 / np.percentile(im['source'][im['mask_source'] > 0.5], 99)
    im['source'] *= v
    md['scaled'] = v
    return im, md


def split_by_vol(imagedata, metadata, vols_=None, normalise_fun=normalise_fun, preproc_fun=None):
    def gen():
        if preproc_fun is not None:
            imdata, mdata = preproc_fun(imagedata, metadata)
        else:
            imdata, mdata = imagedata, metadata
        vols = vols_
        if vols is None:
            vols = range(imdata['source'].shape[0])
        if vols == 'any':
            vols = [np.random.randint(imdata['source'].shape[0], size=1)]
        for vol in vols:
            md = copy.deepcopy(mdata)
            md['vol'] = vol
            im = dict()
            im['vol'] = vol
            for k in imdata.keys():
                if isinstance(imdata[k], np.ndarray) and imdata[k].shape[0] > 1:
                    assert imdata[k].shape[0] > vol, (imdata[k].shape, vol)
                    im[k] = np.expand_dims(imdata[k][vol], 0)
                    assert im[k].ndim == 4, (k, imdata[k], im[k].shape)
                else:
                    im[k] = imdata[k]
            if normalise_fun is not None:
                im, md = normalise_fun(im, md)
            yield im, md
    if vols_ == 'any':
        return next(gen())
    return gen()


class VolumeLoader(object):
    def __init__(self, normalise=normalise_fun, preproc=None):
        self.preproc_fun = preproc
        self.normalise_fun = normalise

    def __call__(self, imagedata=None, metadata=None):
        if self.preproc_fun is not None:
            imdata, mdata = self.preproc_fun(imagedata, metadata)
        else:
            imdata, mdata = imagedata, metadata
        vol = np.random.randint(imdata['source'].shape[0], size=1)[0]
        md = copy.deepcopy(mdata)
        md['vol'] = vol
        im = {'vol': vol}
        for k in imdata.keys():
            if isinstance(imdata[k], np.ndarray) and imdata[k].shape[0] > 1:
                assert imdata[k].shape[0] > vol, (imdata[k].shape, vol)
                im[k] = np.expand_dims(imdata[k][vol], 0)
                assert im[k].ndim == 4, (k, imdata[k], im[k].shape)
            else:
                im[k] = imdata[k]
            if k+'_md' in metadata:
                b = metadata[k+'_md'].get('b', None)
                if b is not None:
                    im['b'] = b[vol]
                v = metadata[k + '_md'].get('volume', None)
                if v is not None:  # volumes were preselected
                    im['vol'] = v[vol]
        # {'imdata': arr, 'metadata': {'b': bs[X_sel.ravel()[vol]], 'volume': vol}}
        if self.normalise_fun is not None:
            im, md = self.normalise_fun(im, md)
        return {'imdata': im, 'metadata': md}


def get_one_bbalanced(arr):
    assert arr.ndim == 4, arr.shape
    # TODO: read dwischeme from metadata
    bs = np.round(np.loadtxt(os.path.join(os.path.dirname(
        os.path.abspath(__file__)), 'dhcp300.txt'))[:arr.shape[3], 3]).astype(int)
    assert arr.shape[3] == 300, arr.shape
    X = np.arange(arr.shape[3])[:, None]
    y = bs[:, None]
    X_sel, y_sel = balanced_sample_maker(X, y, class_sample_size=1)
    vol = np.random.randint(len(X_sel), size=1)[0]
    arr = arr[..., X_sel.ravel()[vol]].reshape(*arr.shape[:3], 1)
    assert arr.ndim == 4, arr.shape
    return {'imdata': arr, 'metadata': {'b': [bs[X_sel.ravel()[vol]]], 'volume': [X_sel.ravel()[vol]]}}
