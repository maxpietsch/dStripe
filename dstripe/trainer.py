import os, sys
import pprint
import torch
import numpy as np
import numpy as np
from utils.params import *
import utils
from dataloader import *
from dataloader2 import *
from augmentation import *
from torch.autograd import Variable
from collections import defaultdict

def overrides(interface_class):
    def overrider(method):
        assert method.__name__ in dir(interface_class), method.__name__
        return method
    return overrider

class NetworkTrainer(object):
    def __init__(self, datadir='.', outdir='/tmp/', epochs=250, learning_rate = 1e-4,
                 meta_data_train=None, meta_data_val=None, meta_data_test=None,
                 cuda=True, max_loss=np.inf, quiet=False, **kwargs):

        self.p = Params(dict())
        self.p.cuda = cuda
        self.p.learning_rate = learning_rate
        self.p.datadir = datadir
        self.p.outdir = outdir
        self.p.checkpoint = None

        self.p.epochs = epochs

        self.p.max_loss = max_loss

        self.p.checkmemory = False
        self.p.verbose = False

        self.model = dict()
        self.optimiser = None
        self.scheduler = None

        self.import_functions = None
        self.data_postproc= None
        self.num_workers = 2
        self.num_cached_per_queue = 2
        self.pin_memory = False

        self.quiet = quiet

        try:
            self.p.basename = os.path.splitext(os.path.basename(__file__))[0]
            self.p.show = False
        except:
            self.p.basename = 'ipythontest'
            self.p.show = True

        self._train_loss = -1
        self._val_loss = -1
        self.bestloss = np.inf
        self._model_status = None
        self.epoch = 0
        self.history = defaultdict(list)

        self.p.meta_data_train = meta_data_train
        self.p.meta_data_val = meta_data_val
        self.p.meta_data_test = meta_data_test

        self.p.paired = False
        self.p.test_data_cropped = False
        self.p.val_data_cropped = False
        self.p.train_data_cropped = False

        self.weight_scheduler = None

        self.p.__dict__.update(kwargs)

    def _train_step(self, *args, **kwargs):
        raise NotImplementedError("_train_step should be implemented in child classes")

    def _val_step(self, *args, **kwargs):
        raise NotImplementedError("_val_step should be implemented in child classes")

    def _run_validation(self, *args, **kwargs):
        # raise NotImplementedError("_val_step should be implemented in child class")
        for isample, sample in enumerate(self.val_loader):
            S = Variable(sample['source'])
            T = Variable(sample['target'])
            if self.p.cuda:
                S = S.cuda()

            output = self._forward(S).cpu()

            target_val_file = os.path.join(self.valpath, os.path.split(checkpoint)[1] + str(self.epoch) + '__' +
                                           self.p.meta_data_val[isample]['target'].replace('/', '-'))
            print(target_val_file)

            Iout = utils.mif.Image()
            Itarget = utils.mif.load_mrtrix(os.path.join(self.p.datadir, self.p.meta_data_val[isample]['target']))
            Iout.empty_as(Itarget)
            Iout.vox = Itarget.vox

            assert Iout.data is None
            output = np.squeeze(output.data.numpy()).transpose(1, 2, 3, 0)
            Iout.data = output
            Iout.save(target_val_file)

            if isample + 1 == nsamples:
                break

    def make_model(self, *args, **kwargs):
        raise NotImplementedError("make_model should be implemented in child class")

    def _run_training(self, epochs, *args, **kwargs):
        raise NotImplementedError("_run_training should be implemented in child class")

    def _forward(self, S):
        raise NotImplementedError("_forward should be implemented in child class")
        # return self.model['decoder'](self.model['encoder'](S))

    def load_data(self,**kwargs):
        if self.p.meta_data_train is not None:
            print('loading training data')
            self.__load_training_data(**kwargs)

        if self.p.meta_data_test is not None:
            print('loading test data')
            self.__load_test_data(**kwargs)

        if self.p.meta_data_val is not None:
            print('loading validation data')
            self.__load_validation_data(**kwargs)

    def __load_data(self, what, transforms):
        assert what in ['train', 'val', 'test'], what
        short = what
        assert self.p.dict['meta_data_'+short] is not None, 'meta_data_'+short+' missing'

        if isinstance(transforms, dict):
            transform = transforms.get('transforms_'+short, None)
        else:
            transform = transforms

        shuffle = what == 'train'
        if 'dataloader' in self.p.dict:
            version = self.p.dataloader
        else:
            version = 'get_tra_val_loader'
        if version == 'get_tra_val_loader':
            self.__dict__[short+'_loader'] = get_tra_val_loader(self.p.datadir,
                                                                self.p.dict['meta_data_'+short], transform=transform,
                                                                batch_size=self.p.dict.get('batch_size', 1), split=False, paired=self.p.paired,
                                                                shuffle=shuffle, num_workers=self.num_workers,
                                                                val_ratio=0.2, pin_memory=self.pin_memory,
                                                                cropped_to_mask=self.p.dict[short+'_data_cropped'],
                                                                import_functions=self.import_functions,
                                                                data_postproc=self.data_postproc)
        elif version == 'MRLoader':
            assert self.num_workers >= 1, self.num_workers
            # from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
            # class MRMultiThreadedAugmenter(MultiThreadedAugmenter):
            #     @property
            #     def dataset(self):  # compatibility with torch
            #         return self.generator
            from dataloader import SampleToTensor4D
            from dataloader2 import MRMultiThreadedAugmenter

            loader = MRLoader(batch_size=self.p.dict.get('batch_size', 1),
                              root_dir=self.p.datadir,
                              metadata=self.p.dict['meta_data_'+short],
                              paired=self.p.paired,
                              load_mask=True,
                              cropped_to_mask=self.p.dict[short+'_data_cropped'],
                              global_normalise='none',
                              load_np_funs=self.import_functions,
                              load_sample_funs=self.data_postproc,
                              batch_sample_transform=None,
                              number_of_threads=self.num_workers,
                              shuffle=shuffle,
                              memmap=self.p.dict.get('memmap', False))

            self.__dict__[short + '_loader'] = MRMultiThreadedAugmenter(loader,
                                                                        transform=transform,
                                                                        num_processes=self.num_workers,
                                                                        num_cached_per_queue=self.num_cached_per_queue,
                                                                        seeds=[random.randint(0, 4294967295) for _ in
                                                                               range(self.num_workers)])  # random seed
        else:
            assert 0, version

    def __load_training_data(self, transforms=None):
        self.__load_data('train', transforms)

    def __load_validation_data (self, transforms=None):
        self.__load_data('val', transforms)

    def __load_test_data (self, transforms=None):
        self.__load_data('test', transforms)

    def info(self):
        if self.quiet:
            return
        print('==================== parameters: ====================')
        dmp = json.dumps(self.p.__dict__, indent=4, sort_keys=True)
        print(dmp)
        # pprint.pprint(dmp)
        for k in ['import_functions','data_postproc']:
            if k in self.__dict__ and self.__dict__[k] is not None:
                print(k+':')
                pprint.pprint(self.__dict__[k])

    def parse_params(self, js="params.json", poverride_dict=None):
        print('loading:', js)
        params = Params(js)
        pupd = []
        pset = []
        if poverride_dict is not None:
            for k in poverride_dict.keys():
                params.dict[k] = poverride_dict[k]
        for key, val in params.dict.items():
            if key in self.p.__dict__:
                pupd.append(key)
            else:
                pset.append(key)

            if isinstance(val, dict) and key in self.p.__dict__:
                self.p.__dict__[key].update(val) # update dict
            else:
                setattr(self.p, key, val) # overwrite

        if self.quiet:
            return
        print('==================== updated parameters: ====================')
        pprint.pprint({a:params.dict[a] for a in pupd})
        # json.dumps({a:params.dict[a] for a in pupd}, indent=4, sort_keys=True)
        print('====================== new parameters: ======================')
        pprint.pprint({a:params.dict[a] for a in pset})
        # json.dumps({a:params.dict[a] for a in pset}, indent=4, sort_keys=True)

        # assert self.encoder_params['dim'] == self.decoder_params['dim'], self.info()

    def _normalise(self, normalise):
        def matprint(mat, fmt="g", headers=[]):
            assert len(mat.shape) == 2, mat.shape
            col_maxes = [max([len(("{:" + fmt + "}").format(x)) for x in col]) for col in mat.T]
            if headers:
                assert len(headers) == mat.shape[1], (len(headers), mat.shape)
                col_maxes = [max(len(h),cm) for h, cm in zip(headers,col_maxes)]
                for i,h in enumerate(headers):
                    print(("{:" + str(col_maxes[i]) + "}").format(headers[i]), end="  ")
                print("")
            for x in mat:
                for i, y in enumerate(x):
                    print(("{:" + str(col_maxes[i]) + fmt + "}").format(y), end="  ")
                print("")

        def report_image_stats(loader='train_loader'):
            if self.quiet:
                return
            if loader not in self.__dict__:
                return
            res = []
            print(loader)
            dset = self.__dict__[loader].dataset
            for isample in range(len(dset.imagedata)):
                sample = dset[isample]
                assert sample['source'] is not None, sample.keys()
                if 'mask_source' in sample and sample['mask_source'] is not None:
                    X = sample['source'][sample['mask_source']>0.5]
                else:
                    X = sample['source']
                res.append([sample.get('vol', -1),
                            sample.get('b', -1),
                            np.mean(X),
                            np.min(X),
                            np.percentile(X, 95),
                            np.percentile(X, 99),
                            np.max(X)])
            matprint(np.array(res), headers='vol b mean min p95 p99 max'.split())
            if self.__dict__[loader].dataset.normalise is not None:
                print('scale_offset', self.__dict__[loader].dataset.normalise.scale_offset)

        if normalise is None:
            if not self.p.dict['memmap']:
                report_image_stats(loader='train_loader')
                report_image_stats(loader='val_loader')
                report_image_stats(loader='test_loader')
            return

        print('normalising:', normalise, end=' ')
        normalise_params = {}
        if normalise == 'meanstd':
            Normalise = MeanStdNormalise
        elif normalise == 'minmax':
            Normalise = MinMaxNormalise
        elif normalise == 'percentile':
            Normalise = PercentileNormalise
        elif normalise == 'percentile-scale':
            Normalise = PercentileNormalise
            normalise_params['lo_zero'] = True
        else:
            assert 0, 'normalise not understood {}'.format(normalise)

        if 'train_loader' in self.__dict__:
            print('training data')
            self.train_loader.dataset.normalise = Normalise(**normalise_params)
            self.train_loader.dataset.fit_normalise(masked=True)
            self.train_loader.dataset.apply_normalise()
            report_image_stats(loader='train_loader')
        if 'val_loader' in self.__dict__:
            print('validation data')
            self.val_loader.dataset.normalise = Normalise(**normalise_params)
            self.val_loader.dataset.fit_normalise(masked=True)
            self.val_loader.dataset.apply_normalise()
            report_image_stats(loader='val_loader')
        if 'test_loader' in self.__dict__:
            print('test data')
            self.test_loader.dataset.normalise = Normalise(**normalise_params)
            self.test_loader.dataset.fit_normalise(masked=True)
            self.test_loader.dataset.apply_normalise()
            report_image_stats(loader='test_loader')
        print('')

    def training(self, params, epochs=None, checkpoint=None,
                 transforms_train=SampleToTensor4D(),
                 transforms_val=SampleToTensor4D(),
                 normalise=None,
                 weight_scheduler=None,
                 continue_from_checkpoint=False,
                 override=False,
                 poverride_dict=None,
                 find_lr=False):
        self.parse_params(params, poverride_dict)
        if normalise is not None:
            assert normalise in ['meanstd', 'minmax', 'percentile', 'percentile-scale'], normalise
        self.p.normalise_used_in_training = normalise

        if epochs is not None:
            self.p.epochs = epochs

        if not continue_from_checkpoint:
            if checkpoint is not None:
                assert not os.path.isdir(checkpoint), checkpoint
                self.p.checkpoint = checkpoint
            else:
                self.p.checkpoint = params.rstrip('.json')
        else:
            if checkpoint is not None:
                self.p.checkpoint = checkpoint
            assert os.path.isfile(self.p.checkpoint), self.p.checkpoint

        assert transforms_val is not None
        assert transforms_train is not None

        self.info()
        pprint.pprint(transforms_train)
        pprint.pprint(transforms_val)
        # P = Params(self.__dict__)
        self.p.save(self.p.checkpoint+'.json')
        print(self.p.checkpoint+'.json')

        self.make_model()
        if continue_from_checkpoint:
            self._load_checkpoint(self.p.checkpoint, override)

        if weight_scheduler is not None:
            self.weight_scheduler = weight_scheduler
            self.weight_scheduler.init(self)  # needs to be after self.make_model()!
            print('weight scheduler:', self.weight_scheduler)

        self.__load_training_data(transforms=transforms_train)
        self.__load_validation_data(transforms=transforms_val)
        self._normalise(normalise)
        print(self.p.checkpoint)
        if not self.quiet:
            print(self.model)
        if find_lr:
            self.find_lr()
        self._run_training(self.p.epochs)

    from functools import partial

    def find_lr(self):
        raise NotImplemented

    def continue_training(self, *args, **kwargs):
        self.training(*args, **kwargs, continue_from_checkpoint=True)

    # def continue_training(self,
    #                       params,
    #                       epochs=None,
    #                       checkpoint=None,
    #                       transforms_train=SampleToTensor4D(),
    #                       transforms_val=SampleToTensor4D(),
    #                       normalise=None,
    #                       override=False,
    #                       poverride_dict=None,
    #                       weight_scheduler=None):
    #     self.parse_params(params, poverride_dict)
    #     if normalise is not None:
    #         assert normalise in ['meanstd', 'minmax', 'percentile', 'percentile-scale'], normalise
    #     self.p.normalise_used_in_training = normalise
    #
    #     if epochs is not None:
    #         self.p.epochs = epochs
    #
    #     if checkpoint is not None:
    #         assert not os.path.isdir(checkpoint), checkpoint
    #         self.p.checkpoint = checkpoint
    #
    #     assert os.path.isfile(self.p.checkpoint), self.p.checkpoint
    #
    #     self.info()
    #     pprint.pprint(transforms_train)
    #     pprint.pprint(transforms_val)
    #     # P = Params(self.__dict__)
    #     self.p.save(self.p.checkpoint+'.json')
    #     print(self.p.checkpoint+'.json')
    #
    #     self.make_model()
    #     self._load_checkpoint(self.p.checkpoint, override)
    #
    #     if weight_scheduler is not None:
    #         self.weight_scheduler = weight_scheduler
    #         self.weight_scheduler.init(self)  # needs to be after self.make_model()!
    #         print('weight scheduler:', self.weight_scheduler)
    #
    #     self.__load_training_data(transforms=transforms_train)
    #     self.__load_validation_data(transforms=transforms_val)
    #     if 'normalise_used_in_training' not in self.p.dict:
    #         self.p.normalise_used_in_training = None
    #     self._normalise(self.p.normalise_used_in_training)
    #     print(self.p.checkpoint)
    #     self._run_training(self.p.epochs)

    def predict(self,
                params,
                epochs=None,
                checkpoint=None,
                transforms=SampleToTensor4D(),
                suffix='',
                npass=1,
                output_orig=False,
                nsamples=3,
                normalise=None,
                post_trafo=None,
                override=False,
                poverride_dict=None,
                seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.parse_params(params, poverride_dict)
        if normalise is not None:
            assert normalise in ['meanstd', 'minmax', 'percentile', 'percentile-scale'], normalise
        self.p.normalise_used_in_training = normalise

        if epochs is not None:
            self.p.epochs = epochs

        if checkpoint is not None:
            assert not os.path.isdir(checkpoint), checkpoint
            self.p.checkpoint = checkpoint

        assert os.path.isfile(self.p.checkpoint), self.p.checkpoint

        self.info()
        pprint.pprint(transforms)
        self.p.save(self.p.checkpoint+'.json')
        print(self.p.checkpoint+'.json')

        self.make_model()
        self._load_checkpoint(self.p.checkpoint, override)
        self.__load_validation_data(transforms=None)
        if 'normalise_used_in_training' not in self.p.dict:
            self.p.normalise_used_in_training = None
        self._normalise(self.p.normalise_used_in_training)
        print(self.p.checkpoint)
        self._predict(what='val', nsamples=nsamples, transforms=transforms, suffix=suffix, npass=npass,
                      set_train=False, output_orig=output_orig, post_trafo=post_trafo)

    def _eval(self):
        self._model_status = 'eval'
        for k, part in self.model.items():
            if hasattr(part, 'eval'):
                if self.p.verbose:
                    print('setting', k,'to eval mode')
                part.eval()

    def _train(self):
        self._model_status = 'train'
        for k, part in self.model.items():
            if hasattr(part, 'train'):
                if self.p.verbose:
                    print('setting', k, 'to train mode')
                part.train()

    def _predict(self, what='val', nsamples=3):
        if what == 'val':
            datagen = self.val_loader
            metadata = self.p.meta_data_val
        elif what == 'test':
            datagen = self.test_loader
            metadata = self.p.meta_data_test
        else:
            assert 0, what
        raise NotImplementedError("_predict should be defined in child class")

    def predict_val(self, params, checkpoint, valpath=None, cuda=False, transforms_val=SampleToTensor4D(), nsamples=3, poverride_dict=None):
        self.parse_params(params, poverride_dict=poverride_dict)
        if valpath is not None:
            assert os.path.isdir(valpath), valpath
            self.p.valpath = valpath
        assert os.path.isfile(checkpoint), checkpoint
        assert self.p.valpath is not None and os.path.isdir(self.p.valpath), self.p.valpath
        self.p.checkpoint = checkpoint
        self.p.cuda = cuda

        self.info()

        self.make_model()
        self._load_checkpoint()
        self.__load_validation_data(transforms=transforms_val)
        if 'normalise_used_in_training' not in self.p.dict:
            self.p.normalise_used_in_training = None
        self._normalise(self.p.normalise_used_in_training)
        self.p.save(os.path.join(self.p.valpath,os.path.split(self.p.checkpoint)[1] + '.json_val'))
        print(os.path.join(self.p.valpath,os.path.split(self.p.checkpoint)[1] + '.json_val'))
        self._eval()
        if nsamples > 0 :
            self._predict('val', nsamples=nsamples)

    def predict_test(self, params, checkpoint, valpath=None, cuda=False, transforms_test=SampleToTensor4D(), poverride_dict=None):
        self.parse_params(params, poverride_dict=poverride_dict)
        if valpath is not None:
            assert os.path.isdir(valpath), valpath
            self.p.valpath = valpath
        assert os.path.isdir(self.p.valpath), self.p.valpath
        assert os.path.isfile(checkpoint), checkpoint
        self.p.checkpoint = checkpoint
        self.p.cuda = cuda

        self.make_model()
        self._load_checkpoint()
        self.__load_test_data(transforms=transforms_test)
        assert 'normalise_used_in_training' not in self.p.dict, 'TODO'
        if 'normalise_used_in_training' not in self.p.dict:
            self.p.normalise_used_in_training = None
        self._normalise(self.p.normalise_used_in_training)
        self._eval()

        assert os.path.isdir(self.test_loader.dataset.root_dir)

        self._predict('test', nsamples=3)


    def _load_checkpoint (self, checkpoint=None, override=False):
        if checkpoint is not None:
            self.p.checkpoint = checkpoint

        assert self.p.checkpoint is not None
        assert os.path.isfile(self.p.checkpoint), self.p.checkpoint

        print("=> loading checkpoint '{}'".format(self.p.checkpoint))
        if self.p.cuda:
            chckpnt = torch.load(self.p.checkpoint)
        else:
            chckpnt = torch.load(self.p.checkpoint, map_location=lambda storage, loc: storage)
        self.epoch = chckpnt['epoch']
        self.bestloss = chckpnt['loss']
        if 'history' in chckpnt:
            self.history = chckpnt['history']
        if override:
            self.bestloss += np.inf

        assert self.optimiser is not None

        for k in chckpnt.keys():
            if k.endswith('_state_dict') and not k.startswith('optimiser'):
                if self.p.verbose:
                    print('loading '+k)
                assert k.replace('_state_dict','') in self.model, self.model.keys()
                self.model[k.replace('_state_dict','')].load_state_dict(chckpnt[k])

        self.optimiser.load_state_dict(chckpnt['optimiser_state_dict'])

        if 'scheduler.last_batch_iteration' in chckpnt:
            assert self.scheduler is not None and hasattr(self.scheduler, 'last_batch_iteration'), self.scheduler
            self.scheduler.last_batch_iteration = chckpnt['scheduler.last_batch_iteration']

        print("=> loaded checkpoint '{}' (epoch {}, best loss {})".format(self.p.checkpoint, self.epoch, self.bestloss))

    def _save_checkpoint (self, is_best=False, record_epoch=lambda x: x in [0, 1, 2, 3, 5, 10, 20, 30, 40, 50] or
                                                                      x % 50 == 0):
        assert self.p.checkpoint is not None
        if os.path.isfile(self.p.checkpoint):
            if self.p.verbose:
                print('overwriting: ' + self.p.checkpoint)

        chckpnt = dict()
        for k, v in self.model.items():
            if hasattr(v, 'state_dict'):
                chckpnt[k+'_state_dict'] = v.state_dict()

        chckpnt['epoch'] = self.epoch
        chckpnt['loss'] = self.bestloss
        chckpnt['optimiser_state_dict'] = self.optimiser.state_dict()
        if self.scheduler is not None and hasattr(self.scheduler, 'last_batch_iteration'):
                chckpnt['scheduler.last_batch_iteration'] = self.scheduler.last_batch_iteration
        chckpnt['history'] = self.history
        chckpnt['parameters'] = self.p.dict

        torch.save(chckpnt, self.p.checkpoint)
        if is_best:
            import shutil
            shutil.copyfile(self.p.checkpoint, self.p.checkpoint+'_best')
        if record_epoch(self.epoch):
            import shutil
            shutil.copyfile(self.p.checkpoint, self.p.checkpoint + '_epoch%08i' % self.epoch)

    # def screenshot(self, callback=lambda **kwargs: None):
    #     if callable(callback):
    #         kwargs = {'iteration': self.iteration, 'error': self.err, 'X': self.X, 'Y': self.TY}
    #         callback(**kwargs)
    #     pass

    def _screenshot(self, S, T, O, M=None, E=None, fstem='', show=True, reverse=False):
        return  # "TODO import matplotlib and plots"
        n = 1
        if 'screenshot_Nc' in self.p.dict:
            Nc = self.p['screenshot_Nc']
        else:
            Nc = min(S.shape[1], 20)
        sl = S.shape[2] // 2 + 5
        comparison = [torch.cat([S[:n, i, sl].cpu() for i in range(Nc)]),
                      torch.cat([T[:n, i, sl].cpu() for i in range(Nc)]),
                      torch.cat([O[:n, i, sl].cpu() for i in range(Nc)])]
        rows = 5
        divcentre = [None, None, None, 0, 0]
        title = 'S, T, O, S - T, O - T'
        if E is not None:
            comparison.append(torch.cat([E[:n, i, sl].cpu() for i in range(Nc)]))
            rows += 1
            divcentre = [None] + divcentre
            title = 'S, T, O, E, S - T, O - T'
        comparison.append(torch.cat([S[:n, i, sl].cpu() - T[:n, i, sl].cpu() for i in range(Nc)]))
        comparison.append(torch.cat([O[:n, i, sl].cpu() - T[:n, i, sl].cpu() for i in range(Nc)]))
        if M is not None:
            M = None # TODO
            # M = torch.cat([M[:n, i, sl].cpu() for _ in range(rows) for i in range(Nc)]).numpy()

        # clamp = (np.percentile(S.numpy(),1),np.percentile(S.numpy(),99))
        clamp = (None, None)
        plots(np.squeeze(torch.cat(comparison).data), figsize=(40, float(rows) / 3 * 4), rows=rows,
              divcentre=divcentre, divvclamp=(-0.8, 0.8), vclamp=clamp, mask=M)

        plt.suptitle(title+' epoch {}, train loss {:.4g}, val loss {:.4g}'.format(
            self.epoch, self._train_loss, self._val_loss), y=0.95)

        if fstem:
            plt.savefig(fstem + '%s.png' % ('' if not reverse else '_r'), bbox_inches='tight', pad_inches=0, dpi=140)
        if show:
            plt.show()
        plt.close()

if __name__ == '__main__':

    class MyNetwork(NetworkTrainer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

            self.p.datadir='.'

            self.p.datafile = 'dwi.mif'

            self.p.subs_train = "ABCDEF"
            self.p.subs_val = 'IJK'

            self.p.meta_data_train = [{'source': '{}/prisma/st/{}'.format(x, self.p.datafile_s),
                                'mask_source': '{}/prisma/st/{}'.format(x, 'mask.mif'),
                                'mask_target': '{}/prisma/st/{}'.format(x, 'mask.mif'),  # using prisma mask on purpose
                                'target': '{}/connectom/st/{}'.format(x, self.p.datafile_t)} for x in
                               list(self.p.subs_train)]
            self.p.meta_data_val = [{'source': '{}/prisma/st/{}'.format(x, self.p.datafile_s),
                              'target': '{}/connectom/st/{}'.format(x, self.p.datafile_t)} for x in list(self.p.subs_val)]
            self.p.meta_data_test = None

        @overrides(NetworkTrainer)
        def _train_step(self, *args, **kwargs):
            raise NotImplementedError("should be defined in child classes")

        @overrides(NetworkTrainer)
        def _val_step(self, *args, **kwargs):
            raise NotImplementedError("should be defined in child class")

        @overrides(NetworkTrainer)
        def make_model(self, *args, **kwargs):
            raise NotImplementedError("should be implemented in child class")

        @overrides(NetworkTrainer)
        def _run_training(self, *args, **kwargs):
            raise NotImplementedError("should be defined in child class")

        @overrides(NetworkTrainer)
        def _forward(self, S):
            raise NotImplementedError("should be defined in child class")
            return self.model['decoder'](self.model['encoder'](S))


    Net = MyNetwork()
    Net.train(params=dict())
