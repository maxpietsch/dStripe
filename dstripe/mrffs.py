#!/usr/bin/env python
#
#   Field Frequency Sanitiser for DWI stripe correction
#
#   Author:  Max Pietsch
#            King's College London
#            maximilian.pietsch@kcl.ac.uk
#

#   __________ Initialisation __________

# Make the corresponding MRtrix3 Python libraries available
import os, sys, socket, shutil, pprint, glob, subprocess, shlex, inspect # psutil
import multiprocessing
import multiprocessing as mp
if not sys.version_info >= (3, 5):
    raise Exception('requires python version >= 3.5')

host = socket.gethostname()
import numpy as np
from scipy import fftpack
import torch
import utils.mif
from models.layers import MaskFill3D, WeightedSliceFilter, InplaneLowpassFilter3D
from tqdm import tqdm
#   __________ Setup __________
# HOME = os.path.expanduser('~') + '/'
# os.environ['PATH'] = HOME+'/anaconda3/envs/py3ml3D/bin/:'+HOME+'/mrtrix3_mrreg_standalone/bin:'+os.environ['PATH']

### setup mrtrix
# p = subprocess.Popen(['which', 'mrinfo'], stdout=subprocess.PIPE)
# line = p.stdout.readline().decode().strip()
# lib_folder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(line)[0], os.pardir, 'lib')))
# if not os.path.isdir(lib_folder):
#     sys.stderr.write('Unable to locate MRtrix3 Python libraries in {} on (host:{})\n'.format(lib_folder, host))
#     print('\n'.join(sys.path))
#     sys.exit(1)
# sys.path.insert(0, lib_folder)
# from mrtrix3 import app, image, path, run, file


def console(string):
    print(string)

def exists(path):
    return os.path.exists(path)


def mkdir_p(path):
    if not os.path.isdir(path):
        # console('mkdir_p: ' + path)
        os.makedirs(path, exist_ok=True)


def check_input(path, dir=False):
    if not dir and os.path.isfile(path):
        # console('check file: ' + path)
        return
    elif dir and os.path.isdir(path):
        # console('check dir: ' + path)
        return
    raise IOError(('file' if not dir else 'dir') + ' not found: ' + path)

class LogExceptions(object):
    def __init__(self, callable):
        self.__callable = callable

    def __call__(self, *args, **kwargs):
        try:
            result = self.__callable(*args, **kwargs)

        except Exception as e:
            error(traceback.format_exc())
            raise

        return result

from multiprocessing.pool import Pool
class LoggingPool(Pool):
    def apply_async(self, func, args=(), kwds={}, callback=None):
        return Pool.apply_async(self, LogExceptions(func), args, kwds, callback)

def butter2d_lp(size, cutoff, n=3):
    """2D Butterworth lowpass filter

           size : tuple
           cutoff : cutoff frequency of the filter (0 - 1.0)
           n : int, filter order, increase for sharpness
       """
    if not 0 < cutoff <= 1.0:
        raise ValueError('Cutoff frequency must be between 0 and 1.0')

    if not isinstance(n, int):
        raise ValueError ('n must be an integer >= 1')

    rows, cols = size
    assert rows > 0, size
    assert cols > 0, size
    x = np.linspace(-0.5, 0.5, rows)
    y = np.linspace(-0.5, 0.5, cols)
    if len(x) == 1:
        x[:] = 1.
    if len(y) == 1:
        y[:] = 1.

    distance = np.sqrt((y**2)[None] + (x**2)[:, None])
    return 1 / (1.0 + (distance / cutoff)**(2*n))

import argparse
parser = argparse.ArgumentParser(description='Field Frequency Sanitiser for DWI stripe correction')

parser.add_argument('destriped',  help='amplitude data without stripes but with bias field OR destripe field (if -destriped_is_field)')
parser.add_argument('reference',  help='amplitude data with stripes but without bias field')
parser.add_argument('mask',  help='brain mask')
parser.add_argument('output',  help='sanitised multiplicative field')
parser.add_argument('-grad',  help='gradient table in MRtrix format')

options = parser
options.add_argument('-no_zclean', help='do not remove bias field: stop after Tikhonov regularised field estimation and inplane filter (for spred)', action='store_true')
options.add_argument('-no_xyscrub', help='no inplane filter (for destriped_is_field and clean fields)', action='store_true')
options.add_argument('-no_xyfilter', help='no inplane filter (for destriped_is_field and clean fields)', action='store_true')
options.add_argument('-tikhonov_delta', help=' ', type=float, default=1e-4)
options.add_argument('-destriped_is_field', help=' ', action='store_true')
options.add_argument('-upsample_from', help='neural network layer extent before linear upsampling', type=int, default=16)
options.add_argument('-device', help='"cpu" or GPU number', default="cpu")
options.add_argument('-no_fft', help='Convolution instead of FFT filtering. Worse performance and memory footprint.', action='store_true')
options.add_argument('-nthreads', help='nthreads', type=int)
options.add_argument('-force', help='force', action='store_true')
options.add_argument('-debug', help='debug', action='store_true')

global force

def write_output(IM, Output, location, stride_reference=None, **kwargs):
    global force
    local_force = kwargs.pop('force', force)
    if isinstance(local_force, str):
        local_force = [local_force]
    if stride_reference is not None and not np.allclose(IM.data.strides, Output.strides):
        assert 0, "TODO"
        # # np.lib.stride_tricks.as_strided(X1, shape=X.shape, strides=X1.strides)[:] = X
        # console("strides changed")
        # if not app.tempDir:
        #     app.makeTempDir()
        # app.gotoTempDir()
        # IM.data = Output
        # IM.save(app.tempDir + '/result.mif')
        # cmd = ['mrconvert', app.tempDir + '/result.mif', '-strides', stride_reference, location] + local_force
        # run.command(' '.join(cmd))
    else:
        IM.data = Output
        console("saving image to " + location)
        IM.save(location)
    if np.isnan(Output).any():
        raise RuntimeError("output containts NaN")


if __name__ == '__main__':
    args = parser.parse_args()

    do_fft = not args.no_fft
    force = ['-force'] if args.force else []
    debug = args.debug
    nthreads = 32 if args.nthreads is None else int(args.nthreads)
    if nthreads != 32:
        console("nthreads: %i" % nthreads)

    if args.device.lower() == 'cpu':
        cuda = False
        device = 'cpu'
    else:
        cuda = True
        if args.device.isdigit():
            if "CUDA_VISIBLE_DEVICES" in os.environ and os.environ["CUDA_VISIBLE_DEVICES"] != args.device:
                console("warning: not setting GPU device to {} as CUDA_VISIBLE_DEVICES is already set to {}".format(
                    args.device, os.environ["CUDA_VISIBLE_DEVICES"]))
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = args.device
        device = 'cuda'

    console("________________________________________________________________")
    for k in sorted(dir(args)):
        if k.startswith('_') or getattr(args, k) is None or k in ['debug']:
            continue
        console((k + ': ').rjust(21, ' ') + str(getattr(args, k)))
    console(('host: ').rjust(21, ' ') + str(host))
    if debug:
        console("warning: writing debug output")
    console("________________________________________________________________")

    check_input(args.destriped, dir=False)

    if not force and exists(args.output):
        raise RuntimeError(args.output+" exists, use -force to overwrite")

    # __________________________________ load data
    global D, R, mask, Output, d_high_intensity, bvalues, success
    success = True

    do_estimate_field = not args.destriped_is_field
    if not do_estimate_field:
        console("input is stripe field, skipping stripe field estimation")

    check_input(args.reference, dir=False)
    R = utils.mif.load_mrtrix(args.reference).data  # memory map
    if np.isnan(R).any():
        raise RuntimeError("reference contains NaN")
    if (R == 0).all():
        raise RuntimeError("reference is zero ")

    IM = utils.mif.load_mrtrix(args.destriped)  # will be used for output
    if args.grad:
        grad = np.loadtxt(args.grad)
    else:
        grad = IM.grad
        if grad is None:
            raise RuntimeError('no gradient table found in ' + args.destriped)
    bvalues = np.round(grad[:, 3] / 50., 0) * 50
    bs = sorted(np.unique(bvalues).tolist())
    if debug:
        console("b values: " + str(bs))

    D = IM.data  # memory map
    assert len(D.shape) == 4, D.shape
    assert D.shape[3] == len(bvalues), (D.shape, grad.shape)
    assert np.all(R.shape == D.shape), (R.shape, D.shape)
    if np.isnan(D).any():
        raise RuntimeError("destriped contains NaN")
    if (D == 0).all():
        raise RuntimeError("destriped is zero ")
    if (D == 1).all():
        app.warn("destripe is ones")
    slices, vols = D.shape[2:4]
    if debug:
        console('%i slices, %i volumes' %(slices, vols))

    mask = np.squeeze(utils.mif.load_mrtrix(args.mask).data) > 0.5
    assert len(mask.shape) == 3 and np.all(D.shape[:3] == mask.shape), (D.shape, mask.shape)
    if mask.sum() == 0:
        raise RuntimeError("mask is empty")

    d_high_intensity = dict()
    _masked = R[mask]
    for b in bs:
        d_high_intensity[b] = np.percentile(_masked[..., bvalues == b], 99)
    _masked = None

    # console('reference 99 percentile intensity [b]:\n' + str(pprint.pformat(d_high_intensity, indent=4)))

    mask_slice_means = mask.mean(axis=(0, 1))

    tikhonov_delta = float(args.tikhonov_delta)
    if debug:
        console('Tikhonov delta: %f' % tikhonov_delta)
    upsample_from = int(args.upsample_from)
    MaskFill3D_ks = (5, 5, 1)
    upsample_to = max(*mask.shape[:2])
    pad_fft = 4 * upsample_to // upsample_from

    if not do_estimate_field:
        Output = D.astype(np.float32, copy=False).copy()
        if not args.no_xyscrub:
            # mark untrusted data with -1
            Output[Output < 0.1] = -1
            Output[Output > 10] = -1
    else:
        # __________________________________ Tikhonov regularised field estimation, marks untrusted data with -1
        Output = np.ones_like(D, dtype=np.float32) * -1
        console("estimating multiplicative field")

        def field_fit_worker(ii, tikhonov_delta):
            global D, R, mask, d_high_intensity, bvalues, success
            if not success:
                return ii, 0
            destriped = D[..., ii].astype(np.float32, copy=False)
            source = R[..., ii].astype(np.float32, copy=False)
            high_intensity = d_high_intensity[bvalues[ii]]
            assert high_intensity > 0, high_intensity
            assert tikhonov_delta > 0, tikhonov_delta
            intensity_cutoff = max(1e-6, 10.0 * tikhonov_delta * high_intensity)

            # ||s * x - d||^2 + || \delta * x  - \delta ||^2
            # x = \frac{s * d + \delta^2}{s^2 + \delta^2}
            d = (tikhonov_delta * high_intensity) ** 2
            assert d > 0, d
            ft = ((source * destriped + d) / (source ** 2 + d)).astype(np.float32)

            outside = np.logical_or(source <= intensity_cutoff, destriped <= intensity_cutoff)
            ft[outside] = -1
            if (ft[~outside] <= 0).any():
                success = False
            if (~outside).sum() == 0:
                success = False
            return ii, ft

        if nthreads <= 1:
            for vol in range(vols):
                _, Output[..., vol] = field_fit_worker(vol, tikhonov_delta)
        else:
            def write_result(result):
                global Output
                _ii, result = result
                Output[..., _ii] = result

            try:
                # pool = mp.Pool()
                multiprocessing.log_to_stderr()
                pool = LoggingPool(nthreads)
                for vol in range(vols):
                    pool.apply_async(field_fit_worker, args=(vol, tikhonov_delta, ), callback=write_result)
            finally:
                pool.close()
                pool.join()

        if not success:
            raise RuntimeError("oops")
        if np.isnan(Output).any():
            raise RuntimeError("Output contains NaN")
        if (Output == 0).any():
            raise RuntimeError("Output contains zeros")

        if debug:
            write_output(IM, Output, args.output + '_ft0.mif', stride_reference=None, force=True)

    # __________________________________ interpolate / extrapolate field in untrusted / low intensity regions
    if not args.no_xyscrub:
        console("Train yourself to let go of everything you fear to lose.")
        assert MaskFill3D_ks[2] == 1,  "TODO interpolation across slices requires loop over volumes"

        # iterate over slices so that each volume gets the same number of iterations irrespective of intensity

        progress = tqdm(total=slices, desc="sanitising multiplicative field", leave=True)
        for sl in range(slices):
            progress.update()
            if debug:
                console('slice %i' % sl)
            chunk = Output[..., sl, :]  # x, y, v
            with torch.no_grad():
                Inside = torch.from_numpy((chunk != -1).astype(np.float32)).reshape(1, 1, *chunk.shape[:3]).contiguous().to(device)
                for _vol, _inside in enumerate(Inside.view(-1, vols).sum(0)):
                    if _inside == 0:  # don't trust anything, fill all with 1s
                        app.warn('slice %i, volume %i has no valid field data' % (sl, _vol))
                        Inside.view(-1, vols)[:, _vol] = 1.0
                        chunk[..., _vol] = 0
                    continue

                # slice-level brain mask
                BM = torch.from_numpy(mask[..., sl].astype(np.float32, copy=True))
                BM = BM.reshape(*mask.shape[:2], 1).expand(1, 1, -1, -1, vols).contiguous().to(device)
                mask_mean = mask[..., sl].mean()
                # slice-level log field
                chunk[chunk == -1] = 1.0
                if np.isnan(chunk).any():
                    raise RuntimeError("chunk contains NaN")
                if (chunk == 0).any():
                    raise RuntimeError("chunk contains zeros")
                if (chunk == -1).any():
                    raise RuntimeError("chunk contains -1 ")
                if (chunk < 0).any():
                    raise RuntimeError("chunk contains negative values")
                X = torch.log(torch.from_numpy(chunk.astype(np.float32, copy=True)).reshape(1, 1, *chunk.shape[:3])).to(device)

                # fill at least 5 times or until brain mask has no non-inside voxels
                pc = MaskFill3D(in_channels=1, out_channels=1,
                                kernel_size=MaskFill3D_ks, padding=tuple(ks//2 for ks in MaskFill3D_ks)).to(device)
                it = -1
                while True:
                    # stop if full coverage
                    if int(Inside.sum()) == int(np.product(Inside.shape)):
                        console("converged at iteration %i (nothing to fill in) %f %f %g" % (
                            it, torch.mul(Inside, BM).mean(), mask_mean,  torch.mul(Inside, BM).mean() - mask_mean))
                        break
                    it += 1
                    if it > 4 and (mask_slice_means[sl] == 0 or torch.abs(torch.mul(Inside, BM).mean() - mask_mean) < 0.5 / np.product(BM.shape)):
                        console("converged at iteration %i (brain mask filled in) %g" % (it, torch.mul(Inside, BM).mean() - mask_mean))
                        break
                    X, Inside = pc(X, Inside, return_mask=True)
                X = X.cpu().numpy()
                Output[..., sl, :] = np.squeeze(np.exp(X))
                if (Output[..., sl, :] <= 0).any():
                    raise RuntimeError("Output slice %i non-positive" % sl)

        if debug:
            write_output(IM, Output, args.output + '_ft1.mif', stride_reference=None, force=True)

    if not args.no_xyfilter:
        # __________________________________ remove upsampling and extrapolation artefacts
        cutoff_sr = 0.5 / upsample_from
        cutoff_frequency = 2.0 * cutoff_sr

        if do_fft:
            console("FFT filtering of inplane field artefacts")
            import warnings
            warnings.simplefilter(action='ignore', category=FutureWarning)

            def fft_filter(args):
                im, bw2, pad_fft, ii = args
                _im = np.pad(im, pad_fft, mode='constant', constant_values=1)
                im = fftpack.ifft2(fftpack.fft2(_im) * bw2).real[pad_fft:-pad_fft, pad_fft:-pad_fft]
                return im

            Output = Output.reshape(*Output.shape[:2], slices * vols)
            bw2 = fftpack.ifftshift(butter2d_lp((Output.shape[0]+2*pad_fft, Output.shape[1]+2*pad_fft), cutoff_frequency))

            def fft_worker(ii):
                im = np.pad(Output[..., ii].copy(), pad_fft, mode='reflect')
                return ii, fftpack.ifft2(fftpack.fft2(im) * bw2).real[pad_fft:-pad_fft, pad_fft:-pad_fft]

            if nthreads <= 1:
                for ii in range(slices * vols):
                    _, Output[..., ii] = fft_worker(ii)
            else:
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

            Output = Output.reshape(*Output.shape[:2], slices, vols)
        else:
            console("Convolutionial filtering of inplane field artefacts")

            kernel_size = min(upsample_to // 4, 41)
            kernel_size += kernel_size % 2 - 1

            ipfilter = InplaneLowpassFilter3D(channels=1, param=cutoff_frequency, kernel_size=kernel_size, mode="reflect",
                                              filter_type="sinc").float().to(device)
            for vol in range(vols):
                X = torch.from_numpy(Output[..., vol].astype(np.float32, copy=False)).unsqueeze(0).unsqueeze(0).to(device)
                with torch.no_grad():
                    Output[..., vol] = np.squeeze(ipfilter(X).cpu().numpy())
        if debug:
            write_output(IM, Output, args.output + '_ft2.mif', stride_reference=None, force=True)

    if (Output <= 0).any():
        raise RuntimeError("Output non-positive")

    if not args.no_zclean:
        # ____________________________ remove bias field using mask as "attention" mechanism, lower trust edge slices

        console("filtering out bias field")
        # refine mask based on mean intensity in lowest shell
        global m1
        m1 = np.logical_and(mask, R[..., bvalues == min(bs)].mean(-1) > 0.5 * d_high_intensity[min(bs)]).astype(np.float32)
        if m1.sum() < 1:
            raise RuntimeError("m1 empty")
        eps = 0.5
        # if not do_fft:
        #     ks = 9
        #     param = 0.1
        #     ftype = 'sinc'
        #     slfilter = WeightedSliceFilter(channels=1, gauss_std=param, sinc_bw=param, kernel_size=ks, kernel_width=1,
        #                                    weight_smooth=6, padding="constant", filter_type=ftype).float().to(device)
        #
        #     m1 = torch.from_numpy(eps + (1.0-eps) * m1).view(1, 1, *mask.shape).contiguous().to(device)
        #
        #     # filter bias field in log space using mask (without padding) as attention mechanism (1.0) vs outside (eps)
        #     progress = app.progressBar("Conv filtering of bias field", target=vols)
        #     for vol in range(vols):
        #         progress.increment()
        #         console(str(vol))
        #         f = torch.from_numpy(np.log(Output[..., vol])).view(*m1.shape).contiguous().float().to(device)
        #         Output[..., vol] /= np.squeeze(np.exp(slfilter(f, m1, eps, ret_support=False).cpu().numpy()))
        #     progress.done()
        # else:
        p = slices // 2
        edge_feather = 3
        sinc_bw = 0.5 / float(slices)

        from scipy import signal
        import scipy.signal.windows
        import scipy.fftpack as fftpack
        from scipy.fftpack import fft, fftshift, ifft, ifftshift
        ks = slices + 2 * p
        t = torch.arange(-ks // 2 + 1, ks // 2 + 1).numpy()
        sinc = np.sinc(2.0 * sinc_bw * t)

        # D'Antona, Gabriele, and A. Ferrero, "Digital Signal Processing for Measurement Systems", Springer Media, 2006,
        # p. 70 DOI:10.1007/0-387-28666-7.
        flattop = scipy.signal.windows.flattop(ks)
        global fft_filter2
        fft_filter2 = ifftshift(sinc * flattop)

        # blur mask inplane
        ip_filter = InplaneLowpassFilter3D(channels=1, param=6, kernel_size=31,
                                           mode="reflect", filter_type="gaussian").float().to(device)
        # mask is prior (weight: [eps, 1])
        m1 = ip_filter(torch.from_numpy((1.0 - eps) * m1).view(1, 1, *mask.shape).contiguous().to(device)).cpu().numpy()
        m1 = np.squeeze(m1).reshape(-1, slices)

        def fft_worker2(ii, eps, edge_feather, slices):
            global Output, fft_filter2, m1, success
            if not success:
                return ii, 0
            try:
                I = np.log(Output[..., ii].reshape(-1, slices))
                im_shape = Output.shape[:-1]
                if edge_feather:
                    I[..., :edge_feather] *= np.linspace(0, 1., edge_feather)
                    I[..., -edge_feather:] *= np.linspace(0, 1., edge_feather)[::-1]

                I *= eps + m1
                if p > 0:
                    I = np.pad(I, ((0, 0), (p, p)), 'reflect')
                I = np.exp(fftpack.ifft(fftpack.fft(I) * fft_filter2).real)
                if p > 0:
                    I = I[:, p:-p]
                I = I.reshape(*im_shape)
            except:  # TODO this does not work when multiprocessing
                success = False
                raise
            return ii, I

        def write_result2(args):
            global Output, progress_bar
            _ii, result = args
            Output[..., _ii] /= result
            progress_bar.increment()
            return

        global progress_bar
        progress_bar = app.progressBar("FFT filtering of bias field", target=vols)

        assert Output.shape[-1] == vols, Output.shape
        if nthreads <= 1:
            for ii in range(vols):
                write_result2(fft_worker2(ii, eps, edge_feather, slices))
            progress_bar.done()
        else:
            try:
                pool = Pool()
                for ii in range(vols):
                    pool.apply_async(fft_worker2, args=(ii, eps, edge_feather, slices, ), callback=write_result2)
            finally:
                pool.close()
                pool.join()
            progress_bar.done()

        if not success:
            raise RuntimeError("oops2")

    write_output(IM, Output, args.output) #, stride_reference=args.destriped)


























































