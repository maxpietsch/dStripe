#!/usr/bin/env python3
#
#   APPLY AND EVALUATE DWI STRIPE REMOVAL
#
#   input: DWI amplitude file
#   output: stripe correction field, corrected data
#
#   Author:  Max Pietsch
#            King's College London
#            maximilian.pietsch@kcl.ac.uk
#
import os, shutil, argparse, sys, pprint
import multiprocessing
import multiprocessing as mp
from multiprocessing.pool import Pool
import numpy as np
from scipy import signal
import scipy.signal.windows
import scipy.fftpack
from scipy.fftpack import fft, fftshift, ifft, ifftshift
from scipy.linalg import logm, expm
import scipy.ndimage

import utils.mif
import torch
from models.layers import InplaneLowpassFilter3D, SliceLowPassFilter
from collections import defaultdict, deque

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

global verbose
verbose = True


def eprint(*args, **kwargs):
    global verbose
    if kwargs.pop('verbose', verbose):
        print(*args, file=sys.stderr, **kwargs)


def timing(f):
    import time
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        eprint('{:s} function took {:.3f} ms'.format(f.__name__, (time2 - time1) * 1000.0))
        return ret
    return wrap


def show_memusage(device=0):
    import torch
    try:
        import gpustat
    except ImportError:
        raise ImportError("pip install gpustat")

    gpu_stats = gpustat.GPUStatCollection.new_query()
    item = gpu_stats.jsonify()["gpus"][device]
    eprint(
        "gpustats {}/{}; pytorch (MB) allocated: {:.1f} cached: {:.1f} max allocated: {:.1f} max cached: {:.1f}".format(
            item["memory.used"], item["memory.total"],
            torch.cuda.memory_allocated(device=device) * 1e-6, torch.cuda.memory_cached(device=device) * 1e-6,
            torch.cuda.max_memory_allocated(device=device) * 1e-6, torch.cuda.max_memory_cached(device=device) * 1e-6))

def get_device(args):
    cuda = False
    device = 'cpu'
    gpu_ids = []
    if args.device.lower() != 'cpu':
        cuda = True
        device = 'cuda'
        gpu_ids = [0]
        assert torch.cuda.is_available(), args.device
        if args.device.isdigit():
            if "CUDA_VISIBLE_DEVICES" in os.environ and os.environ["CUDA_VISIBLE_DEVICES"] != args.device:
                eprint("warning: not setting GPU device to {} as CUDA_VISIBLE_DEVICES is already set to {}".format(
                    args.device, os.environ["CUDA_VISIBLE_DEVICES"]), verbose=True)
            else:
                gpu_ids = [int(args.device)]
        elif torch.cuda.device_count() > 1:
            try:
                gpu_ids = [int(d) for d in args.device.split(',') if 0 <= int(d) < torch.cuda.device_count()]
            except ValueError:
                gpu_ids = list(range(torch.cuda.device_count()))
            torch.cuda.set_device(gpu_ids[0])
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpu_ids))
    return cuda, device, gpu_ids


# def mirrorpadz(im_o, nmirr=3):
#     if nmirr > 0:
#         zdim = [len(im_o.shape) - 1]
#         out = torch.empty(*im_o.shape[:-1], im_o.shape[-1] + 2 * nmirr, dtype=im_o.dtype, device=im_o.device)
#         out[..., :nmirr] = torch.flip(im_o[..., 1:nmirr + 1], zdim)
#         out[..., nmirr:-nmirr] = im_o
#         out[..., -nmirr:] = torch.flip(im_o[..., -nmirr - 1:-1], zdim)
#         return out
#     return im_o

def mirrorpadz(im_o, nmirr=3, mode='reflect', value=0):
    import torch.nn.functional as F
    if nmirr > 0:
        shape = im_o.shape
        assert len(shape) == 5, shape
        shape_padded = [s for s in shape]
        shape_padded[-1] += 2 * nmirr

        p = torch.nn.modules.utils._ntuple(4)((nmirr, nmirr, 0, 0))
        return F.pad(im_o.view(1, shape[0] * shape[1], -1, shape[-1]), p, mode, value=value).view(*shape_padded)
    return im_o

def mirrorpadz_inv(im_m, nmirr):
    if nmirr > 0:
        return im_m[..., nmirr:-nmirr]
    return im_m


def undo_datashapechange(data, nmirr, rotations):
    _data = mirrorpadz_inv(np.squeeze(data), nmirr)
    if rotations is not None:
        for rotation in rotations[::-1]:
            _data = np.rot90(_data, k=rotation[3], axes=rotation[:2][::-1])
    return _data


def lie2tr(r):
    L = np.zeros((4, 4))
    L[:3, 3] = r[:3]
    L[2, 1] = r[3]
    L[1, 2] = -r[3]
    L[0, 2] = r[4]
    L[2, 0] = -r[4]
    L[1, 0] = r[5]
    L[0, 1] = -r[5]
    T = expm(L)
    return T


def butter2d_lp(size, cutoff, n=3):
    """2D Butterworth lowpass filter

           size : tuple
           cutoff : cutoff frequency of the filter (0 - 1.0)
           n : int, filter order, increase for sharpness
       """
    if not 0 < cutoff <= 1.0:
        raise ValueError('Cutoff frequency must be between 0 and 1.0')

    if not isinstance(n, int):
        raise ValueError('n must be an integer >= 1')

    rows, cols = size
    assert rows > 0, size
    assert cols > 0, size
    x = np.linspace(-0.5, 0.5, rows)
    y = np.linspace(-0.5, 0.5, cols)
    if len(x) == 1:
        x[:] = 1.
    if len(y) == 1:
        y[:] = 1.

    distance = np.sqrt((y ** 2)[None] + (x ** 2)[:, None])
    return 1 / (1.0 + (distance / cutoff) ** (2 * n))


def cabs(ctensor):
    return torch.sqrt(torch.pow(ctensor[...,0], 2) + torch.pow(ctensor[...,1], 2))

def cdiv(a, b):
    # a / b = a * b^* / (b * b^*)
    return a * torch.mul(b, torch.tensor([1,-1], dtype=b.dtype)) / torch.sum(b * b, -1, keepdim=True)

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def clear_header(image):
    from collections import defaultdict
    image.header.pop('pe_scheme', None)
    image.comments = []
    image.header = defaultdict(list)
    image.grad = None
    return image


if __name__ == '__main__':

    #########################
    nmirr = 17  # TODO: revise use of nmirr and nfixed to prevent edge artefacts
    flip_dims = [[4], [3], [3,4], [2,3,4], [2], [2,4]]
    # nfixed = 4
    nsamples = None
    mode = 'upsample'
    upsample_from = 16  # neural network layer extent before linear upsampling
    #########################

    parser = argparse.ArgumentParser(description='Evaluate destripe network.')
    parser.add_argument('params', type=str, help='json parameter file defining network')
    parser.add_argument('dwi', type=str, help='input amplitude data')
    parser.add_argument('mask', type=str, help='input mask (required for intensity normalisation), needs to share trafo & grid with dwi')
    parser.add_argument('--motion', type=str, help='per pack motion parameters to apply to mask. packs per slice determined by --slice_native_mb')
    parser.add_argument('--outdir', type=str, default='~/tmp/')

    parser.add_argument('--slice_native', type=str2bool, help='input are slice-native volumes, outputs slice native field'
                                                              ' (do not use unless ssp omitted from dwirecon_project)', default=False)
    parser.add_argument('--slice_ssp', type=str, help='slice profile to apply to field (approximate SR destriping)', default='')
    parser.add_argument('--ssp_field_scale', action='store_true', help='scale field by centre of slice profile, not via convolution (approximate SR destriping)')
    parser.add_argument('--slice_native_mb', type=int, help='multiband factor', default=4)
    parser.add_argument('--slice_native_vols', type=int, help='number of volumes of non-repeated DWI', default=300)
    parser.add_argument('--slice_native_vol_offset', type=int, help='volume offset, non-zero if not first chunk', default=0)
    parser.add_argument('--slice_native_slice_buffer', type=str, help='temporary slice buffer used across chunks', default='')

    parser.add_argument('--checkpoint', default='', help='use specific weights instead of last (or best) state. full path to checkpoint')
    parser.add_argument('--batch_size', type=int, default=1, help='number of volumes to process in parallel')
    parser.add_argument('-d', '--device', type=str, default='cpu', help='cpu or gpu number')
    parser.add_argument('--precision', type=str, default='float32', help='float32 or float64')

    parser.add_argument('--npass', type=int, default=3, help='number of recursion passes input --> field')
    parser.add_argument('--nmirr', type=int, default=nmirr, help='nmirr')
    parser.add_argument('--attention', action='store_true', help='use mask as attention filter')
    parser.add_argument('--aug', action='store_true', help='for testing')
    parser.add_argument('--butterworth_samples_cutoff', type=float, default=44./64, help='highpass filter cutoff. default: 44./64')
    # parser.add_argument('--nfixed', type=int, default=nfixed, help='nfixed')

    parser.add_argument('--debug', action='store_true', help='more checks')
    parser.add_argument('--write_corrected', type=str2bool, default=False)
    parser.add_argument('--write_field', type=str2bool, default=True)
    parser.add_argument('--verbose', type=str2bool, default=False)
    args = parser.parse_args()

    verbose = args.verbose
    eprint("=" * 100)
    for k in dir(args):
        if k.startswith('_'): continue
        eprint((k + ':').rjust(15, ' '), getattr(args, k))
    eprint("-" * 100)

    verbose = args.verbose
    debug = args.debug
    write_field = args.write_field
    write_corrected = args.write_corrected

    runtime_total = defaultdict(float)

    #########################
    # Butterworth lowpass filter parameters
    bw_order = 4
    bw_samples_cutoff = float(args.butterworth_samples_cutoff) # 36. / 64
    bw_freq_cutoff = 1. - bw_samples_cutoff
    #########################

    #  _______________________ compute setup
    cuda, device, gpu_ids = get_device(args)
    if cuda and len(gpu_ids) >= 1:
        torch.cuda.set_device(gpu_ids[0])
    batch_size  = args.batch_size

    nmirr = args.nmirr
    # nfixed = args.nfixed

    precision = args.precision.lower()
    assert precision in ['float32', 'float64'], args.precision.lower()
    # float64:
    # real    24m32.258s
    # user    8m35.898s
    # sys     16m30.945s
    # float32:
    # real    6m25.371s
    # user    3m9.313s
    # sys     4m27.400s

    def arr2prec(array, precision=precision):
        return array.asdtype(precision, copy=False)


    def ten2prec(tensor, precision=precision):
        if precision == 'float32':
            return tensor.float()
        elif precision == 'float64':
            return tensor.double()
        else:
            assert 0, precision

    def to_gpu(f):
        if cuda:
            try:
                if len(gpu_ids) > 1:
                    return torch.nn.DataParallel(f, device_ids=gpu_ids).cuda()
                return f.cuda(*gpu_ids)
            except Exception as e:
                print('__Python VERSION:', sys.version)
                print('__pyTorch VERSION:', torch.__version__)
                print('__CUDA VERSION')
                print('__CUDNN VERSION:', torch.backends.cudnn.version())
                print('__Number CUDA Devices:', torch.cuda.device_count())
                print('__Devices')
                print('Active CUDA Device: GPU', torch.cuda.current_device())
                print('Available devices ', torch.cuda.device_count())
                print('Current cuda device ', torch.cuda.current_device())
                print('os.environ:')
                for k, v in os.environ.items():
                    print(k, ':', v)
                raise
        return f

    #  _______________________ data setup

    params_in = os.path.expanduser(args.params)
    assert os.path.isfile(params_in), params_in

    MODEL = os.path.split(params_in.replace('.pth.tar.json', ''))[1]
    eprint('model:'.rjust(15, ' '), MODEL)

    assert os.path.isfile(args.dwi), args.dwi
    dwi_shape = utils.mif.load_mrtrix(args.dwi, memmap=True).shape
    assert os.path.isfile(args.mask), args.mask
    if args.ssp_field_scale:
        assert args.slice_ssp
    slice_ssp = args.slice_ssp if args.slice_ssp else None
    if slice_ssp:
        assert os.path.isfile(slice_ssp), slice_ssp
        slice_ssp = np.loadtxt(slice_ssp).ravel()[None, None, :]
    ssp_field_scale_factor = slice_ssp.ravel()[slice_ssp.size // 2] if args.ssp_field_scale else None
    datadir = os.path.dirname(os.path.commonprefix([args.dwi, args.mask]))
    dwi_file = os.path.relpath(args.dwi, datadir)
    mask_file = os.path.relpath(args.mask, datadir)
    datadir, subses = os.path.split(datadir)
    subses = [subses]

    outdir = os.path.expanduser(args.outdir)
    valpath = os.path.join(outdir, 'eval_' + MODEL) + '/'

    eprint('valpath:'.rjust(15, ' '), valpath)
    eprint('datadir:'.rjust(15, ' '), datadir)
    eprint('dwi_file:'.rjust(15, ' '), dwi_file)
    eprint('mask_file:'.rjust(15, ' '), mask_file)
    eprint('outdir:'.rjust(15, ' '), outdir)
    eprint('subses:'.rjust(15, ' '), subses)

    n_slices = dwi_shape[2]
    n_volumes = dwi_shape[3]
    mb = int(args.slice_native_mb)
    n_packs = n_slices // mb
    slice_buffer = None
    if args.slice_native:
        vol_offset = int(args.slice_native_vol_offset)
        out_shape = [dwi_shape[0], dwi_shape[1], n_slices, args.slice_native_vols]
        # slice_buffer stores 2 fields:
        # [each slice in native space, native but factored out slice above and below in corresponding native volume]
        slice_buffer = np.memmap(args.slice_native_slice_buffer,
                                 dtype=np.float32,
                                 offset=0,
                                 mode=('r+' if os.path.isfile(args.slice_native_slice_buffer) else 'w+'),
                                 shape=tuple(out_shape))
    # inplane FFT parameters
    upsample_to = max(*dwi_shape[:2])
    pad_fft = 4 * upsample_to // upsample_from

    # through-plane FFT
    # edge_feather = 3
    # edge_weight = np.linspace(0, 1., edge_feather)
    # eps = 0.5
    # pad = n_slices // 2
    # sinc_bw = 0.5 / float(n_slices)
    # ks = n_slices + 2 * pad
    # t = torch.arange(-ks // 2 + 1, ks // 2 + 1).numpy()
    # sinc = np.sinc(2.0 * sinc_bw * t)

    # D'Antona, Gabriele, and A. Ferrero, "Digital Signal Processing for Measurement Systems", Springer Media, 2006,
    # p. 70 DOI:10.1007/0-387-28666-7.
    # flattop = scipy.signal.windows.flattop(ks)
    # fft_filter_bias = ifftshift(sinc * flattop)

    # mask
    I_mask = utils.mif.load_mrtrix(args.mask)
    mask = I_mask.data

    # filters for mask --> attention
    attention_eps = 0.95
    attention_power = 1.0
    ip_param = 9
    tp_param = 3
    ip_filter = to_gpu(InplaneLowpassFilter3D(channels=1, param=ip_param, kernel_size=31, mode="reflect",
                                       filter_type="gaussian").float())
    tp_filter = to_gpu(SliceLowPassFilter(channels=1, kernel_size=2 * ((3 * tp_param) // 2) + 1, padding="constant",
                                   filter_type="gaussian", gauss_std=tp_param).float())

    structure_d = np.zeros((3, 3, 3), np.bool)
    structure_d[:, :, 1] = True

    structure_e = np.zeros((3, 3, 3), dtype=np.bool)
    structure_e[1, 1, :] = True
    n_erode = 6

    #  low pass filter for rough bias field estimation
    # lp_ip_filter = InplaneLowpassFilter3D(channels=1, param=5, kernel_size=15, mode="reflect",
    #                                       filter_type="gaussian").float().to(device)
    # gauss_std = 11
    # lp_tp_filter = SliceLowPassFilter(channels=1, kernel_size=2 * ((3 * gauss_std) // 2) + 1, padding="reflect",
    #                                   filter_type="gaussian", gauss_std=gauss_std).float().to(device)
    class SliceHighPass3D(torch.nn.Module):
        def __init__(self, fft_filter, ks, ip_gauss_std=5, tp_gauss_std=11):
            super(SliceHighPass3D, self).__init__()

            self.lp_ip_filter = InplaneLowpassFilter3D(channels=1, param=ip_gauss_std,
                                                       kernel_size=2 * ((3 * ip_gauss_std) // 2) + 1, mode="reflect",
                                                       filter_type="gaussian")

            self.lp_tp_filter = SliceLowPassFilter(channels=1,
                                                   kernel_size=2 * ((3 * tp_gauss_std) // 2) + 1, padding="reflect",
                                                   filter_type="gaussian", gauss_std=tp_gauss_std).float() # .to(device)
            self.fft_filter = torch.nn.Parameter(fft_filter, requires_grad=False)
            self.ks = ks

        def forward(self, tensor):
            if len(tensor.shape) == 6:
                assert tensor.shape[-1] == 2, tensor.shape
                tensor, attention_weights = torch.split(tensor, 1, -1)
                tensor = tensor.view(*tensor.shape[:-1])
                attention_weights = attention_weights.view(*attention_weights.shape[:-1])
                tensor_lp = self.lp_ip_filter(self.lp_tp_filter(tensor))
                tensor = torch.exp(torch.log(tensor / tensor_lp) * attention_weights) * tensor_lp
            assert len(tensor.shape) == 5, tensor.shape
            return tensor / torch.irfft(torch.rfft(tensor, 1) * self.fft_filter, 1, signal_sizes=[self.ks])

    # volume-to-volume motion parameters (transform mask relative to dwi)
    voxel2scanner = None
    scanner2voxel = None
    motion = []
    if args.motion:
        mot = np.loadtxt(args.motion)[::n_packs]
        assert mot.shape[0] == dwi_shape[3], (mot.shape, dwi_shape, n_packs)
        voxel2scanner = I_mask.transform.dot(np.diag([I_mask.vox[0], I_mask.vox[1], I_mask.vox[2], 1]))
        scanner2voxel = np.linalg.inv(voxel2scanner)
        for vol in range(n_volumes):
            motion.append(np.dot(scanner2voxel, np.dot(lie2tr(mot[vol]), voxel2scanner)))

    ## model
    npass = args.npass
    assert npass > 0, npass

    # ________________________ TODO clean up this json/tar mess:
    checkpoint = params_in[:-5]
    if args.checkpoint:
        checkpoint = args.checkpoint
    assert os.path.isfile(checkpoint), checkpoint
    eprint('checkpoint:'.rjust(15, ' '), checkpoint)
    params = params_in + '_val'

    eprint("=" * 100)

    if not os.path.isdir(valpath):
        os.makedirs(valpath)

    shutil.copy2(params_in, params)
    # ________________________ /TODO clean up

    poverride_dict = {"valpath": valpath, "datadir": datadir}
    poverride_dict["subs_val"] = subses
    poverride_dict['meta_data_val'] = [{'source': '{}/{}'.format(x, dwi_file) if x else dwi_file,
                                        'mask_source': '{}/{}'.format(x, mask_file) if x else mask_file} for
                                       x in poverride_dict["subs_val"]]
    poverride_dict['subs_train'] = None
    poverride_dict['batches_per_epoch'] = np.inf
    poverride_dict['batch_size'] = batch_size
    # pprint.pprint(poverride_dict)

    # ______________________ load network

    from train_stripes import *

    Net = DeStripeNetwork(quiet=not verbose)
    Net.import_functions = {'source': get_all, 'target': get_all}
    Net.num_workers = 1
    Net.data_postproc = split_by_vol
    Net.predict_val(params=params,
                    checkpoint=checkpoint,
                    transforms_val=SampleToTensor4D(), nsamples=0, poverride_dict=poverride_dict)
    model = Net.model['snet']
    model.field_filter = None  # here we use FFT filter instead. TODO: build FFT into model
    model.return_field = True
    model.return_x = False
    model.mode = mode

    model = to_gpu(ten2prec(model))

    # ______________________ data loader

    datagen = Net.val_loader
    metadata = Net.p.meta_data_val
    if nsamples is None:
        try:
            nsamples = len(datagen)  # pytorch dataloader
        except TypeError:
            nsamples = len(datagen.dataset)  # batchgenerators data augmentation

    assert nsamples == n_volumes, (nsamples, n_volumes)

    # ______________________ evaluate

    # global scale and offset
    if datagen.dataset.normalise is not None:
        scale_offset = np.array([float(x) for x in datagen.dataset.normalise.scale_offset])
        eprint('scale_offset:', scale_offset)
        assert scale_offset[0] != 0, scale_offset
    else:
        scale_offset = np.array([1.0, 0])

    with torch.no_grad():
        # setup butterworth FFT filter for DC and bias field removal
        sos = scipy.signal.butter(bw_order, bw_freq_cutoff, analog=False, btype='lowpass', output='sos')
        ks = n_slices + 2 * nmirr
        impulse = np.zeros(ks)
        impulse[ks // 2] = 1
        imp_ff = signal.sosfiltfilt(sos, impulse)
        bias_filter = cabs(cdiv(torch.rfft(torch.tensor(imp_ff.copy(), dtype=torch.float32), 1),
                                torch.rfft(torch.tensor(impulse.copy(), dtype=torch.float32), 1))).view(1, 1, 1, -1, 1)
        # bias_filter = bias_filter.to(device)

        slice_highpass = to_gpu(SliceHighPass3D(bias_filter, ks=n_slices + 2 * nmirr, ip_gauss_std=5, tp_gauss_std=11).float())

        d_imout = dict()
        n_vols_in_samples = 0  # TODO: make sure we iterate over all volumes once, irrespective of batch size!
        for ibatch, sample in enumerate(datagen):
            if verbose: print('\nbatch %i' % ibatch, len(sample['vol']), nsamples, sample['vol'], '\n')
            n_vols_in_samples += len(sample['vol'])
            assert n_vols_in_samples <= n_volumes, (n_vols_in_samples, n_volumes)
            print_stem = '[' + str(n_vols_in_samples) + '/' + str(nsamples) + '] '
            eprint(print_stem, end='\r', verbose=True)

            # _________ batch processing on CPU or GPU
            # pprint.pprint([(k, sample[k].shape if (hasattr(sample[k],'shape') and k.endswith('source')) else sample[k]) for k in sample.keys()])
            batch_sample_idx = np.asanyarray(sample['idx'])
            batch_vols = np.asanyarray(sample['vol'])
            batch_size_this = len(batch_vols)
            assert len(np.unique(batch_vols)) == batch_size_this, batch_vols
            # if batch_size_this > 1:
            #     assert all([v % 2 == 1 for v in batch_vols]) or all([v % 2 == 0 for v in batch_vols]), batch_vols
            assert len(np.unique(batch_sample_idx)) == len(batch_sample_idx), batch_sample_idx

            # _____ image normalisation _____
            batch_image_offset = np.asanyarray([Net.val_loader.dataset.metadata[idx].get('offset', 0.0) for idx in batch_sample_idx])
            batch_image_scale = np.asanyarray([Net.val_loader.dataset.metadata[idx].get('scaled', 1.0) for idx in batch_sample_idx])

            # _____ attention _____
            attention = None
            t_highpass_filter_and_attention = 0
            if args.attention:
                hpf_start = time.time()
                attention = np.zeros([batch_size_this, 1]+list(dwi_shape[:3]), np.float32)
                for ii, isample in enumerate(batch_sample_idx):
                    if motion:
                        msk = scipy.ndimage.affine_transform(mask, motion[isample][:3, :3],
                                                                               offset=motion[isample][:3, 3], order=0)
                    else:
                        msk = mask.astype(np.float32)
                    # dilate mask inplane so that true inside mask after Gaussian inplane filter by dilating by full width at tenth of maximum of Gaussian
                        # erode through-plane
                        msk = scipy.ndimage.morphology.binary_erosion(msk, structure=structure_e, iterations=n_erode)
                        # dilate mask inplane so that true inside mask after Gaussian inplane filter by dilating by full width at tenth of maximum of Gaussian
                        msk = scipy.ndimage.morphology.binary_dilation(msk, structure=structure_d, iterations=int(
                            np.ceil(2 * np.sqrt(2 * np.log(10)) * np.sqrt(ip_param))))
                    attention[ii, 0] = msk.astype(np.float32)

                # inplane smooth
                attention = torch.from_numpy(attention).to(device)
                # feather mask through-plane (not constrained) and inplane (extend outside mask only)
                attention = ip_filter(attention)
                attention = mirrorpadz(attention, nmirr=nmirr, mode='constant') # TODO can we pad less?
                attention = tp_filter(attention)
                attention = attention_eps * attention + (1.0 - attention_eps)
                if attention_power != 1.0:
                    attention = torch.exp(torch.log(attention) * attention_power)
                attention = ten2prec(attention)
                t_highpass_filter_and_attention += time.time() - hpf_start
                # print('attention prep took %is' % (time.time() - hpf_start))

            # __________ destripe _________
            start = time.time()
            S = mirrorpadz(ten2prec(sample['source']), nmirr=nmirr, mode='reflect').to(device)

            assert ks == S.shape[-1], (ks, S.shape[-1])
            # DC and bias field filter (iterative destriping: FFT filter frequency remains valid!)
            dfields = deque(maxlen=2)  # [previous iteration's normalised field, current field]
            if args.aug:
                dfields.append(torch.flip(model(torch.flip(S, flip_dims[0])), flip_dims[0]))
            else:
                dfields.append(model(S))
            hpf_start = time.time()
            dfields[0] = slice_highpass(torch.stack([dfields[0], attention], dim=-1).to(device) if attention is not None
                                        else dfields[0])  # TODO find more memory efficient solution
            t_highpass_filter_and_attention += time.time() - hpf_start
            ipass = 0
            for ipass in range(1, npass):
                if args.aug:
                    dfields.append(torch.flip(model(torch.flip(S * dfields[-1], flip_dims[ipass % len(flip_dims)])), flip_dims[ipass % len(flip_dims)]))
                else:
                    dfields.append(model(S * dfields[-1]))
                hpf_start = time.time()
                dfields[1] = slice_highpass(torch.stack([dfields[0] * dfields[1], attention], dim=-1).to(device) if attention is not None
                                            else dfields[0] * dfields[1])  # TODO find more memory efficient solution
                t_highpass_filter_and_attention += time.time() - hpf_start
            if len(dfields) > 1:
                f = dfields.popleft()
                del f

            # _________ volume-wise processing on CPU
            batch_S = mirrorpadz_inv(S.cpu(), nmirr=nmirr).numpy().astype(np.float32, copy=False)
            batch_field = mirrorpadz_inv(dfields.pop().cpu(), nmirr=nmirr).numpy().astype(np.float32, copy=False)

            t_network = time.time() - start
            start = time.time()

            for idx_batch in range(batch_size_this):
                vol = batch_vols[idx_batch]
                sample_idx = batch_sample_idx[idx_batch]
                image_offset = batch_image_offset[idx_batch]
                image_scale = batch_image_scale[idx_batch]
                orig = sample["source_file"][idx_batch]
                assert os.path.isfile(orig), orig
                assert image_scale > 0, image_scale
                valstem = os.path.splitext(os.path.join(Net.p.valpath,
                                                        os.path.relpath(sample["source_file"][idx_batch],
                                                                        Net.p.datadir).replace('/', '-')))[0]

                S = None
                field = np.squeeze(batch_field[idx_batch])

                im_out = d_imout.get(orig, utils.mif.load_mrtrix(orig, read_data=False))
                if orig not in d_imout:
                    d_imout = dict()  # keep one image reference only
                    im_out.vox = im_out.vox[:3]
                    im_out = clear_header(im_out)
                    d_imout[orig] = im_out

                im_out.header['scale_offset'] = scale_offset.tolist()
                im_out.header['image_scale_offset'] = [image_scale, image_offset]

                # _____________ write output (matching projected data -- without slice profile)

                if write_field:
                    im_out.data = field  # * nanmask
                    sout = valstem + '_field_vol_%s_%i.mif' % (str(int(vol)).zfill(5), ipass)
                    utils.mif.save_mrtrix(sout, im_out)
                    eprint(print_stem + sout if verbose else '... ' + sout[-140:], end='\r', verbose=True)

                if write_corrected:
                    im_out.data = (np.squeeze(batch_S[idx_batch]) * np.squeeze(field)).astype(np.float32, copy=False)
                    if scale_offset[0] != 1.0:
                        im_out.data *= scale_offset[0]
                    if scale_offset[1] != 0:
                        im_out.data += scale_offset[1]
                    if image_scale != 1.0:
                        im_out.data /= image_scale
                    if image_offset != 0:
                        im_out.data += image_offset
                    sout = valstem + '_fieldapplied_vol_%s_%i.mif' % (str(int(vol)).zfill(5), ipass)
                    utils.mif.save_mrtrix(sout, im_out)
                    eprint(print_stem + sout if verbose else '... ' + sout[-140:], end='\r', verbose=True)

                # _____________ convolve stripe field with slice profile

                if slice_ssp is not None:
                    if ssp_field_scale_factor is None:
                        field = scipy.ndimage.convolve(np.squeeze(field), slice_ssp, mode='nearest').reshape(*field.shape)
                    else:
                        field = 1.0 + (field - 1.0) * ssp_field_scale_factor  # scale offset from 1 by scale factor
                        field = field.astype(np.float32, copy=False)

                if slice_buffer is not None:
                    dat = field
                    volume = vol + vol_offset
                    sl = [volume % n_packs + n_packs * i for i in range(mb)]  # all native slices
                    slice_buffer[..., sl, volume // n_packs] = dat[..., sl]

                # _____________________ FFT filter inplane artefacts
                # aggregate and filter if all data in all packs is processed
                # TODO: move this out of batch loop (if last batch does not contain last volume)
                # TODO: check we've covered all volumes
                if slice_buffer is not None and (volume + 1) // n_packs == args.slice_native_vols:
                    t_ipfft = time.time()
                    out = np.array(slice_buffer[:])

                    # _____________________ FFT filter inplane artefacts
                    eprint('\nFFT filter inplane artefacts\n', verbose=True)
                    cutoff_sr = 0.5 / upsample_from
                    cutoff_frequency = 2.0 * cutoff_sr

                    def fft_filter(args):
                        im, bw2, pad_fft, ii = args
                        _im = np.pad(im, pad_fft, mode='constant', constant_values=1)
                        im = scipy.fftpack.ifft2(scipy.fftpack.fft2(_im) * bw2).real[pad_fft:-pad_fft, pad_fft:-pad_fft]
                        return im

                    global Output
                    Output = out.reshape(*out.shape[:2], out.shape[2] * out.shape[3])
                    bw2 = scipy.fftpack.ifftshift(
                        butter2d_lp((Output.shape[0] + 2 * pad_fft, Output.shape[1] + 2 * pad_fft),
                                    cutoff_frequency))

                    def fft_worker(ii):
                        im = np.pad(Output[..., ii], pad_fft, mode='reflect')
                        return ii, scipy.fftpack.ifft2(scipy.fftpack.fft2(im) * bw2).real[pad_fft:-pad_fft,
                                   pad_fft:-pad_fft]

                    def write_result(result):
                        global Output
                        ii, result = result
                        Output[..., ii] = result  # Output modified only by the main process

                    try:
                        pool = mp.Pool()
                        for i in range(Output.shape[-1]):
                            pool.apply_async(fft_worker, args=(i,), callback=write_result)
                    finally:
                        pool.close()
                        pool.join()

                    runtime_total['inplane_FFT'] += time.time() - t_ipfft

                    out = Output.reshape(*out_shape)

                    im_out.data = out
                    sout = valstem + '_field_native.mif'
                    utils.mif.save_mrtrix(sout, im_out)
                    eprint(sout, verbose=True)

            t_postproc = time.time() - start

            runtime_total['network'] += t_network
            runtime_total['highpass_filter_and_attention'] += t_highpass_filter_and_attention
            runtime_total['postproc_IO'] += t_postproc

            # if ibatch % 5 == 0:
            #     eprint('\ntotal runtimes:\n' + pprint.pformat(runtime_total), verbose=True)

            eprint(print_stem, end='\r', verbose=True)
            sys.stderr.flush()
    if slice_buffer is not None:
        eprint('\nflushing slice buffer', verbose=True)
        del slice_buffer
    eprint('\ntotal runtimes:\n' + pprint.pformat(runtime_total), verbose=True)
    eprint('', verbose=True)
