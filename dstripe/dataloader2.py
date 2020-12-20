import os, pprint
import torch
import numpy as np
from abc import ABCMeta
import random, logging
from multiprocessing import Process
from multiprocessing import Queue as MPQueue

from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.dataloading.data_loader import SlimDataLoaderBase

from utils.mif import load_mrtrix

class MRLoader(SlimDataLoaderBase):
    def __init__(self, batch_size, root_dir, metadata,
                 paired=True,  # load source and target
                 load_mask=True,
                 cropped_to_mask=False,
                 global_normalise='none',  # global intensity normalisation applied to all data equally after loading
                 load_np_funs=None,  # at loading time: dict of functions applied to numpy image (before cropping)
                 load_sample_funs=None,  # at loading time: applied to sample dictionary (after cropping)
                 batch_sample_transform=None,
                 # during batching: applied to sample (i.e. cropping of irregularly sized data, volume selection...)
                 number_of_threads=None,
                 shuffle=True,
                 memmap=False):
        """
        If you use MultiThreadedAugmenter you will need to set and use number_of_threads_in_multithreaded. See
        multithreaded_dataloading in examples!
        :param batch_size: will be stored in self.batch_size for use in self.generate_train_batch()
        :param root_dir (string): Directory containing all images
        :param metadata (list): list of dicts with source and optional target file paths relative to root_dir
        :param paired: data is paired (source and target files)
        :param number_of_threads_in_multithreaded: will be stored in self.number_of_threads_in_multithreaded.
        None per default. If you wish to iterate over all your training data only once per epoch, you must coordinate
        your Dataloaders and you will need this information
        :param transform (callable, optional)
        """
        __metaclass__ = ABCMeta
        super(MRLoader, self).__init__(None, batch_size, number_of_threads_in_multithreaded=number_of_threads)

        # load data (not into _data)
        self.metadata = []
        self._memmap_original_metadata = []
        self.root_dir = root_dir
        self.paired = paired
        self.transform = batch_sample_transform
        self.shuffle = shuffle

        # memmap_dir = os.path.expanduser('~/tmp')

        if global_normalise == 'zca':
            from dataloader import ZCANormalise
            self.normalise = ZCANormalise()
        elif global_normalise == 'meanstd':
            from dataloader import MeanStdNormalise
            self.normalise = MeanStdNormalise()
        elif global_normalise == 'minmax':
            from dataloader import MinMaxNormalise
            self.normalise = MinMaxNormalise()
        elif global_normalise == 'percentile':
            from dataloader import PercentileNormalise
            self.normalise = PercentileNormalise(p=(0, 98))
        elif global_normalise == 'none':
            self.normalise = None
        else:
            self.normalise = normalise

        self.import_functions = load_np_funs
        if self.import_functions is not None:
            print(self.__class__.__name__+' load_np_funs:', str(self.import_functions))
        self.data_postproc = load_sample_funs
        if self.data_postproc is not None:
            print(self.__class__.__name__+' load_sample_funs:', str(self.data_postproc))

        self._pp = pprint.PrettyPrinter(indent=4, width=160, depth=None)
        self.log = lambda x: print(x) if isinstance(x, str) else self._pp.pprint(x)

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
                importer = None if self.import_functions is None or what not in self.import_functions else self.import_functions[what]
                impath = os.path.join(self.root_dir, md[what])
                msk = None
                if load_mask or cropped_to_mask:
                    assert 'mask_' + what in md, md.keys()
                    msk = os.path.join(self.root_dir, md['mask_' + what])
                if not memmap:  # load images into RAM
                    mif = self.__load_mif(impath, mask=msk if cropped_to_mask else None, fun=importer, name=what)
                    imdata.update({what: mif[what]})  # overwrite imdata[what]
                    if what+'_md' in mif:
                        md[what+'_md'] = mif[what+'_md']
                    imdata[what + "_file"] = impath
                    # TODO bbox to self.metadata
                    if load_mask:
                        if 'mask' in mif:
                            assert cropped_to_mask
                            imdata.update({'mask_' + what: mif['mask']})
                        else:
                            assert not cropped_to_mask
                            imdata.update({'mask_' + what: self.__load_mif(msk, mask=msk if cropped_to_mask else None)['im']})

                else:  # memory map images
                    mif = self.__load_mif(impath, mask=msk if cropped_to_mask else None, fun=importer, memmap=memmap, name=what)
                    imdata.update({what: mif[what], '_load_functions_' + what: mif['load_functions'], '_memmap_' + what: mif['memmap']})
                    if what+'_md' in mif:
                        md[what+'_md'] = mif[what+'_md']
                    imdata[what + "_file"] = impath
                    # TODO bbox to self.metadata
                    if load_mask:
                        if 'mask' in mif:
                            assert cropped_to_mask
                            imdata.update({'mask_' + what: mif['mask']})
                        else:
                            assert not cropped_to_mask
                            imdata.update({'mask_' + what: self.__load_mif(msk, mask=msk if cropped_to_mask else None)['im']})

            assert len(self.metadata) == idx, (len(self.metadata), idx)

            if not memmap:  # post-process
                for _im, _md in self.__postproc(imdata, md):
                    self.imagedata[idx] = _im
                    self.metadata.append(_md)
                    idx += 1
            else:  # memory mapped input file --> process on the fly
                self.imagedata[idx] = imdata
                self.metadata.append(md)
                self._memmap_original_metadata.append(md)
                idx += 1

            assert len(self.metadata) == idx and len(self.imagedata) == idx, (len(self.metadata), len(self.imagedata), idx)

        self.log('loading of {} images in {} shards done'.format(len(metadata), len(self.metadata)))

        # DKFZ batchgenerator, multi threading
        self.num_restarted = 0
        self.current_position = 0
        self.was_initialized = False
        self.data_order = np.arange(self.__len__())

    def __len__(self):
        assert len(self.metadata) == len(self.imagedata)
        return len(self.imagedata)

    # class GetMMappedData(object):
    #     def __init__(self, idx, memmap):
    #         self.idx = idx
    #         self.memmap = memmap
    #
    #     def __getitem__(self, *args):
    #         return self.memmap[idx, *args]

    def __postproc(self, imdata, md):
        """ do not use generators on memory mapped data """
        import types
        if isinstance(self.data_postproc, types.FunctionType) or callable(self.data_postproc):
            gen = self.data_postproc(imagedata=imdata, metadata=md)
            if isinstance(gen, types.GeneratorType):
                result = []
                for imdat, mdat in gen:
                    result.append((imdat, mdat))
                return result
            else:
                assert isinstance(gen, dict) and 'imdata' in gen and 'metadata' in gen, str(gen)
                return [(gen['imdata'], gen['metadata'])]
        else:
            assert self.data_postproc is None, str(self.data_postproc)
            return [(imdata, md)]

    def memmap_old(self, idx, whats, memmap_dir, n_data):
        for what in whats:
            if what not in self._memmap:
                import tempfile
                with tempfile.NamedTemporaryFile(dir=memmap_dir) as ntf:
                    self._memmap[what] = np.memmap(ntf,
                                                   mode='w+',
                                                   shape=tuple(list([n_data] + self.imagedata[idx][what].shape)),
                                                   dtype=self.imagedata[idx][what].dtype)
                    print("created memmap for idx %i %s at %s" % (idx, what, ntf.name))
            self._memmap[what][idx, :] = self.imagedata[idx][what][:]
            self.imagedata[idx][what] = None  # TODO

    def fit_normalise(self, masked=True):
        if self.normalise is None:
            return
        assert len(self._memmap_original_metadata) == 0, "TODO"
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
        if self.normalise is None:
            return
        assert len(self._memmap_original_metadata) == 0, "TODO"
        for sample in self.imagedata.values():
            for what in ['source', 'target']:
                if what not in sample:
                    continue
                shp = sample[what].shape
                sample[what] = self.normalise.transform(sample[what].reshape(shp[0], -1)).reshape(shp)

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
            return max(0, rmin - pad), min(rmax + pad, shape[0] - 1), max(0, cmin - pad), \
                   min(cmax + pad, shape[1] - 1), max(0, zmin - pad), min(zmax + pad, shape[2] - 1)

        return rmin, rmax, cmin, cmax, zmin, zmax

    def __mask_image(self, im, mask, name='im', pad=1):
        if len(mask.shape) == 3:
            mask = mask[..., None]
        bbox = self.__bbox2_3D(mask, pad=pad)
        xmin, xmax, ymin, ymax, zmin, zmax = bbox
        return {name: self.__numpy2torch_transpose(im[xmin:xmax + 1, ymin:ymax + 1, zmin:zmax + 1]),
                "mask": self.__numpy2torch_transpose(mask[xmin:xmax + 1, ymin:ymax + 1, zmin:zmax + 1]),
                "bbox": bbox}

    def __load_mif(self, path, mask=None, fun=None, pad=1, name='im', grad=False, memmap=False):
        assert path is not None
        assert os.path.isfile(path), 'can not find image %s' % (path)

        ret = {name: None}  # stores image (H x W x D x C) --> (C x H x W x D), C-contiguous

        im = load_mrtrix(path, memmap=memmap).data
        if grad:
            assert hasattr(im, 'grad')
            ret[name+'_grad'] = im.grad.copy()

        if not memmap:
            if len(im.shape) == 3:
                im = im[..., None]
            if fun is not None:
                im = fun(im)
                if isinstance(im, dict):
                    d = im
                    im = d['imdata']
                    ret[name+'_md'] = d['metadata']

            assert len(im.shape) == 4, im.shape

            if mask is not None:
                assert os.path.isfile(mask), 'can not find mask %s' % mask
                mask = load_mrtrix(mask).data
                ret.update(self.__mask_image(im, mask, name, pad=pad))
            else:
                bbox = 0, im.shape[0], 0, im.shape[1], 0, im.shape[2]
                ret.update({name: self.__numpy2torch_transpose(im), "bbox": bbox})
        else:  # memory map
            functions = []
            if fun is not None:
                functions.append(fun)

            if mask is not None:
                raise NotImplementedError("TODO: masked memmap")
            else:
                bbox = 0, im.shape[0], 0, im.shape[1], 0, im.shape[2]
                functions.append(self.__numpy2torch_transpose)
                ret.update({"bbox": bbox, "load_functions": functions, "memmap": im})
        return ret


    def __numpy2torch_transpose(self, x):
        return x.transpose((3, 0, 1, 2)).copy()

    # def set_thread_id(self, thread_id):
    #     self.thread_id = thread_id
    #
    # def __iter__(self):
    #     return self
    #
    # def __next__(self):
    #     return self.generate_train_batch()

    def reset(self):
        self.current_position = self.thread_id
        self.was_initialized = True
        rs = np.random.RandomState(self.num_restarted)
        if self.shuffle:
            rs.shuffle(self.data_order)
        self.num_restarted += 1

    def __getitem__(self, idx):
        if idx not in self.imagedata:
            raise Exception('key %i not in imagedata (%i, %i)' % (idx, min(self.imagedata.keys()), max(self.imagedata.keys())))

        sample = self.imagedata[idx]

        if self._memmap_original_metadata:
            # load memory mapped file, pre- and post-process, save metadata
            import copy
            md = copy.deepcopy(self._memmap_original_metadata[idx])
            csample = dict()
            toload = []
            for k in sample.keys():
                if '_load_functions_'+k in sample:
                    toload.append(k)
                    assert sample[k] is None
                if k.startswith('_load_functions_') or '_load_functions_'+k in sample:
                    continue
                csample[k] = sample[k]
            assert len(toload), "memory mapping but nothing to load"
            for k in toload:
                im = sample['_memmap_' + k]
                assert im is not None
                assert len(sample.get('_load_functions_'+k, [])), sample.get('_load_functions_'+k, [])
                for f_load in sample.get('_load_functions_'+k, []):
                    im = f_load(im)
                    if isinstance(im, dict):
                        md[k + '_md'] = im['metadata']
                        im = im['imdata']
                csample[k] = im
            sample, md = self.__postproc(csample, md)[0]
            self.metadata[idx] = md

        sample['idx'] = idx
        if self.transform is not None:
            sample = self.transform(**sample)

        return sample

    def _batch_joiner(self, items):
        keys = items[0].keys()
        res = {}
        for k in keys:
            c = []
            for i in items:
                c.append(i[k])
            if isinstance(items[0][k], np.ndarray):
                res[k] = np.stack(c, axis=0)
            elif isinstance(items[0][k], torch.Tensor):
                res[k] = torch.stack(c, dim=0)
            elif isinstance(items[0][k], (list, str, dict)):
                res[k] = c
            elif np.isscalar(items[0][k]):
                res[k] = np.array(c)
            elif isinstance(items[0][k], tuple):
                res[k] = tuple(c)
            else:
                raise ValueError("key %s: don't know how to join instances of %s to a batch" % (k, str(type(items[0][k]))))
        return res

    def generate_train_batch(self):
        ''' Generate the batch. Make sure images in data have the same size (in self.__getitem__(idx) or here) and
          that batch size (self.batch_size) is correct
        '''
        if not self.was_initialized:
            self.reset()
        assert self.batch_size > 0, self.batch_size
        assert self.was_initialized
        data = []
        idxx = []
        while len(data) < self.batch_size:
            idx = self.current_position
            idxx.append(idx)
            if self.number_of_threads_in_multithreaded is not None:
                self.current_position = idx + self.number_of_threads_in_multithreaded
            else:
                self.current_position += 1
            if idx < self.__len__():
                data.append(self.__getitem__(self.data_order[idx]))
            else:
                # print(idx, idxx, self.number_of_threads_in_multithreaded, self.__len__())
                self.reset()
                raise StopIteration

        return self._batch_joiner(data)


def mr_producer(queue, data_loader, transform, thread_id, seed):
    """ same as in MultiThreadedAugmenter but also sets seed for random"""
    np.random.seed(seed)
    random.seed(seed)
    data_loader.set_thread_id(thread_id)
    while True:
        for item in data_loader:
            if transform is not None:
                item = transform(**item)
            queue.put(item)
        queue.put("end")


class MRMultiThreadedAugmenter(MultiThreadedAugmenter):
    @property
    def dataset(self):  # compatibility with torch
        return self.generator

    def _start(self):
        if len(self._threads) == 0:
            logging.debug("starting workers")
            self._queue_loop = 0
            self._end_ctr = 0

            for i in range(self.num_processes):
                logging.debug("starting worker %i" % i)
                self._queues.append(MPQueue(self.num_cached_per_queue))
                self._threads.append(Process(target=mr_producer, args=(self._queues[i], self.generator, self.transform, i, self.seeds[i])))
                self._threads[-1].daemon = True
                self._threads[-1].start()
        else:
            logging.debug("MultiThreadedGenerator Warning: start() has been called but workers are already running")