#!/usr/bin/env python


import glob, os, sys

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.checkpoint
import time
from trainer import *
import utils
from utils.params import Params
from dataloader import *
from augmentation import *
from functools import partial

from dataloader import Compose
# from torchvision.transforms import Compose
# from metrics import *
# from trainer import save_checkpoint
# from models.mresnet import ContextEncoder, Decoder, SELayer3D, ConvBlock

def uniquefname(fname, root_dir):
    fname = os.path.join(root_dir, fname)
    f = glob.glob(fname)
    if len(f) == 1:
        return os.path.relpath(f[0], root_dir)
    assert 0, (fname, f)


class WeightSchedulerISwcZero(object):
    def __init__(self, isweight=0):
        self.initialised = False
        self.isweight = isweight

    def init(self, obj):
        self.destripe_weight = obj.destripe_weight
        self.constancy_weight = obj.constancy_weight
        self.initialised = True

    def __call__(self, sample):
        if not self.initialised:
            raise Exception(self.__name__+" is not initialised")

        if 'mode' in sample:  # no mode augmentation
            assert len(set(sample['mode'])) == 1, sample['mode']

        if 'mode' not in sample or sample['mode'][0] is None or sample['mode'][0] == 'IS':
            w = max(self.constancy_weight + self.destripe_weight - self.isweight, 0.0)
            return w, self.isweight
        return self.destripe_weight, self.constancy_weight


########################################## globals ##########################################
DEVICE = 0

# RECON1DIR = '/projects/dhcp-pipeline-data/kcl/diffusion/ShardRecon01'
# DATADIR = os.path.expanduser('~/data/stripes/')

subs_dhcp_val = ["sub-CC00629XX19/ses-198100", "sub-CC00401XX05/ses-123900", "sub-CC00094AN13/ses-33500", "sub-CC00411XX07/ses-126200",
                 "sub-CC00143BN12/ses-47600", "sub-CC00629XX19/ses-182000", "sub-CC00238BN16/ses-80400", "sub-CC00245AN15/ses-82304",
                 "sub-CC00075XX10/ses-28400", "sub-CC00150AN02/ses-54800", "sub-CC00741XX16/ses-218900"]

subs_dhcp_train = ["sub-CC00693XX18/ses-201000", "sub-CC00218AN12/ses-85900", "sub-CC00513XX10/ses-150000",
                   "sub-CC00178XX14/ses-58600", "sub-CC00328XX15/ses-104800", "sub-CC00472XX11/ses-140000",
                    "sub-CC00568XX16/ses-198900", "sub-CC00650XX07/ses-218007", "sub-CC00227XX13/ses-76601",
                   "sub-CC00429XX17/ses-130900"]

assert len(set(subs_dhcp_val) - set(subs_dhcp_train)) == len(subs_dhcp_val)

########################################## /globals #########################################


from trainer import overrides, NetworkTrainer
# from models.netv1 import *
# from models.netv2 import *
from models.netv3 import *
from utils.learn import *
from dwitools import *

class DeStripeNetwork(NetworkTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.p.datadir = '/home/mp14/data/stripes/dwirecon_nofix'

        self.p.datafile = 'postmc-mssh.mif'  # 'unwarped.mif'

        subs_train = subs_dhcp_train
        subs_val = subs_dhcp_val

        self.p.meta_data_train = [{'source': '{}/{}'.format(x, self.p.datafile),
                                 'mask_source': '{}/{}'.format(x, 'mask.mif')} for x in subs_train]
        self.p.meta_data_val = [{'source': '{}/{}'.format(x, self.p.datafile),
                                 'mask_source': '{}/{}'.format(x, 'mask.mif')} for x in subs_val]
        self.p.meta_data_test = None

        self.p.model_params = {"n_block": 3, "input_dim": 1, "dim": 16, "updim": 2, "downsample": 3, "final_pool": 8,
                               "activ": 'relu', "pad_type": 'zero', "k1": 3, "k2": 1, "prep": 'none', "norm": 'none',
                               "bias": True, "custom_init": False, "mode": 1, "dobn": True, "version":1}

        self.p.epoch_modeswitch = 100

        self.p.learning_rate = 2.5e-3
        self.p.val_stride = 1

        self.p.screenshot = 5

        self.import_functions = {'source': get_b0l0, 'target':get_b0l0}
        self.data_postproc = None

        self.constancy_weight = 0.0
        self.destripe_weight = 1.0
        self.weight_scheduler = lambda x: (self.destripe_weight, self.constancy_weight)

        self.do_sure = False
        self.sure_perturbation = None

    @overrides(NetworkTrainer)
    def make_model(self, *args, **kwargs):
        if not self.quiet:
            print('+'*100)
            pprint.pprint(self.p.model_params)
            print('+' * 100)
        model_parames = copy.deepcopy(self.p.model_params)
        version = model_parames.pop('version', None)
        if not self.quiet:
            print("version: {}".format(version))
            print("checkpoint: {}".format(self.p.checkpoint))
            if 'data_settings' in self.p.dict:
                print("data_settings: {}".format(self.p.data_settings))
        if version is None or version == 1:
            self.model['snet'] = SNet(**model_parames)
        elif version == 2:
            self.model['snet'] = SNet2(**model_parames)
        elif version == 3:
            self.model['snet'] = SNet3(**model_parames)
        else:
            assert 0, 'network version not understood {}'.format(version)

        self.model['snet'] = self.model['snet'].double()

        self.optimiser = torch.optim.Adam(self.model['snet'].parameters(), self.p.learning_rate)

        if 'constancy' in self.p.dict:
            if 'weight' in self.p.constancy:
                self.constancy_weight = float(self.p.constancy['weight'])
            else:
                self.constancy_weight = 10.
        w = self.destripe_weight + self.constancy_weight
        self.destripe_weight /= w
        self.constancy_weight /= w
        print('destripe_weight:', self.destripe_weight)
        print('constancy_weight:', self.constancy_weight)
        if 'scheduler' in self.p.dict and self.p.scheduler == 'CyclicLR':
            self.scheduler = utils.learn.CyclicLR(self.optimiser,
                                                  base_lr=self.p.learning_rate,
                                                  max_lr=5*self.p.learning_rate,
                                                  step_size=80,  # 2-8 x training iterations in epoch
                                                  mode='triangular',
                                                  gamma=0.9, # has no effect
                                                  scale_fn=None,
                                                  scale_mode='cycle',
                                                  last_batch_iteration=-1)
            pprint.pprint('self.scheduler:' + str(self.scheduler))
        if self.p.cuda:
            self.model['snet'] = self.model['snet'].cuda()

        self.loss_func = nn.MSELoss()
        self.loss_func_cuda = nn.MSELoss()

    @overrides(NetworkTrainer)
    def _forward(self, S):
        return self.model['snet'](S)

    @overrides(NetworkTrainer)
    def _run_training(self, epochs, *args, **kwargs):
        print('before training:', end='')
        tval = time.time()
        val_loss, val_loss_normalised, is_best, images = self._val_step(screenshot=self.p.screenshot>0, record=True)
        print(' val loss {:.4e} val loss normalised: {:.4e}, took {:.1f}s {}'.format(val_loss, val_loss_normalised,
                                                                                      time.time() - tval,
                                                                                      '(%.3e)' % self.bestloss if not is_best else '*'))
        if images:
            self._screenshot(*images, fstem=self.p.checkpoint+'_last')
        if epochs != self.p.epochs:
            print('changing epochs from {} to {}'.format(self.p.epochs, epochs))
            self.p.epochs = epochs

        model = self.model['snet']
        previous_mode = model.mode
        for _ in range(self.epoch, self.p.epochs):
            if self.epoch < self.p.epoch_modeswitch:
                model.mode = 'global_avpool'
            else:
                model.mode = previous_mode
            self._train()
            loss = 0
            identity_loss = 0
            ttrain = time.time()
            textstem = 'epoch {} / {}'.format(self.epoch, self.p.epochs)
            nsamples = 0
            if 'batches_per_epoch' in self.p.dict:
                batches_per_epoch = self.p.batches_per_epoch
            else:
                batches_per_epoch = np.inf
            for batch_idx, sample in enumerate(self.train_loader):
                if batches_per_epoch <= batch_idx:
                    break
                nsamples += 1
                if batches_per_epoch < np.inf:
                    text = '{} [{}/{}]'.format(textstem, batch_idx, batches_per_epoch)
                else:
                    text = '{} [{}/{}]'.format(textstem, batch_idx, len(self.train_loader))
                try:
                    _lr = self.optimiser.state_dict()['param_groups'][0]['lr']
                except:
                    _lr = -1
                print(text , end="\r")
                loss_, identity_loss_ = self._train_step(sample)
                loss += loss_
                identity_loss += identity_loss_
                text = text + ' train loss: {:.4e} normalised: {:.4e} '.format(loss_, loss_/identity_loss_)
                if 'train_loss_con' in self.history:
                    text = text + 'loss_const: {:.2e} '.format(self.history['train_loss_con'][-1])
                if 'train_sure_div_term' in self.history:
                    text = text + 'sure_div: {:.2e} '.format(self.history['train_sure_div_term'][-1])
                if 'source_file' in sample:
                    text = text + 'source:...' + str(os.path.split(sample['source_file'][0])[0])[-40:]
                if 'vol' in sample:
                    text = text + ' vol: %i' % sample['vol'][0]
                if _lr > 0:
                    text = text + ' lr: {:.2e}'.format(_lr)
                text = '{:<180}'.format(text[:180])
                print(text, end="\r")
            print(text, end="\r")
            ttrain = time.time() - ttrain
            identity_loss /= nsamples
            loss /= nsamples
            text = '{}, loss {:.4e}, normalised loss: {:.4e} took {:.1f}s'.format(textstem, loss, loss / identity_loss, ttrain)
            print('{:<180}'.format(text[:180]), end='\r')
            tval = time.time()
            val_loss, val_loss_normalised, is_best, images = self._val_step(screenshot=self.p.screenshot > 0 and (self.epoch % self.p.screenshot == 0), record=True)
            print('{:<180}'.format('{} - val loss {:.4e} val loss normalised: {:.4e}, took {:.1f}s {}'.format(text, val_loss, val_loss_normalised,
                  time.time() - tval,  '(%.3e)' % self.bestloss if not is_best else '*')))
            if images:
                self._screenshot(*images, fstem=self.p.checkpoint+'_last')
            self.epoch += 1
            if loss > self.p.max_loss:
                print('stoppping. loss {:.4e} larger than max_loss: {:.4e}'.format(loss, self.p.max_loss))
                return

    @overrides(NetworkTrainer)
    def _predict(self, what='val', nsamples=3, transforms=None, suffix='', npass=1, set_train=False, output_orig=True, post_trafo=None):
        if what == 'val':
            datagen = self.val_loader
            metadata = self.p.meta_data_val
        elif what == 'test':
            datagen = self.test_loader
            metadata = self.p.meta_data_test
        else:
            assert 0, what

        if not set_train:
            self._eval()
        else:
            self._train()

        if transforms is not None:
            transforms_orig = datagen.dataset.transform
            datagen.dataset.transform = transforms
        with torch.no_grad():
            idxx = set()
            for _, sample in enumerate(datagen):
                idxx.add(sample['source_file'][0])
                isample = len(idxx) - 1
                if isample >= nsamples:
                    break
                S = Variable(sample['source']).double()
                if self.p.cuda:
                    S = S.cuda()
                # forward
                output = self._forward(S)
                for ipass in range(1, npass):
                    output = self._forward(output)
                if self.p.cuda:
                    output = output.cpu()
                # postprocess
                if post_trafo is not None:
                    sample['output'] = output
                    if 'augment_log' in sample:
                        sample['augment_log'] = sample.get('augment_log')[0]  # TODO HACK
                    sample = post_trafo(**sample)
                    output = torch.tensor(sample.pop('output'))  # TODO HACK
                    # _T = SampleToTensor4D()
                    # _T.data_fields = tuple(list(_T.data_fields) + ['output'])
                    # sample = _T(**sample)  # TODO HACK

                S = Variable(sample['source']).double()
                T = Variable(sample['target']).double()
                M = (sample['mask_source'] > 0.5).expand_as(S)

                mse_s2t_m = float(torch.nn.MSELoss()(S[M>0.5], T[M>0.5]))
                mse_o2t_m = float(torch.nn.MSELoss()(output[M>0.5], T[M>0.5]))

                output = np.squeeze(output.data.numpy())
                ndim = len(output.shape)
                if ndim == 3:
                    out_spatial_shape = output.shape
                else:
                    out_spatial_shape = output.shape[1:4]
                target = np.squeeze(T.data.numpy())
                source = np.squeeze(S.data.numpy())
                if ndim == 4:
                    output = output.transpose(1, 2, 3, 0)
                    target = target.transpose(1, 2, 3, 0)
                    source = source.transpose(1, 2, 3, 0)

                assert isample < len(metadata), (isample, len(metadata))
                out_file = os.path.join(self.p.valpath, os.path.split(self.p.checkpoint)[1] + '__' +
                                        '%04i' % (sample['vol'][0]) + '-' + suffix + '__' +
                                        metadata[isample]['source'].replace('/', '-'))
                print(out_file)
                source_file = os.path.join(datagen.dataset.root_dir, metadata[isample]['source'])

                Iout = utils.mif.Image()
                Isource = utils.mif.load_mrtrix(source_file)
                source_spatial_shape = Isource.data.shape[:3]
                if not source_spatial_shape == out_spatial_shape:
                    print('warning: shape has changed from '+str(source_spatial_shape)+' to '+str(out_spatial_shape))

                Iout.empty_as(Isource)
                if output_orig:
                    output = np.stack([output, target, source, target-output, target-source], ndim)
                    Iout.vox = tuple(list(Isource.vox[:ndim]) + [5])
                else:
                    Iout.vox = Isource.vox[:ndim]

                Iout.comments = ['source: ' + source_file,
                                 'checkpoint: ' + os.path.split(self.p.checkpoint)[1],
                                 'epoch: ' + str(self.epoch),
                                 'npass: %i' % npass,
                                 'transform: ' + ' '.join(str(datagen.dataset.transform).split())]
                Iout.header['mse_s2t_m'] = [str(mse_s2t_m)]
                Iout.header['mse_o2t_m'] = [str(mse_o2t_m)]
                if output_orig:
                    Iout.comments.append('volumes: output, target, source, target-output, target-source')
                assert Iout.data is None


                Iout.data = output
                Iout.save(out_file)

        if transforms is not None:
            datagen.dataset.transform = transforms_orig

    @overrides(NetworkTrainer)
    def _train_step(self, sample):
        self._train()
        destripe_weight, constancy_weight = self.weight_scheduler(sample)
        w = destripe_weight + constancy_weight
        assert np.abs(w - 1.) < 1e-6, ("weights not normalised wc:{}, wd:{}, mode:{}".format(self.constancy_weight,
                                                                                     self.destripe_weight,
                                                                                     sample.get('mode', ['IS'])[0]))

        if self.scheduler is not None:
            # Cyclical Learning Rates for Training Neural Networks: https://arxiv.org/abs/1506.01186
            self.scheduler.batch_step()
            self.history['train_lr'].append(self.scheduler.get_lr())
        for k in self.model.keys():
            try:
                self.model[k].train()
            except:
                pass
        if self.p.cuda:
            loss_fn = self.loss_func_cuda
        else:
            loss_fn = self.loss_func

        def togpu(x):
            if self.p.cuda:
                return x.cuda()
            return x

        self.optimiser.zero_grad()

        mode = sample.get('mode', ['IS'])[0]
        self.history['train_mode'].append(mode)
        if not self.do_sure or mode != 'IS' or self.epoch < -1:  # MSE loss
            try:
                S = Variable(sample['source'].double())
                T = Variable(sample['target'].double())
                M = (sample['mask_source'] > 0.5).expand_as(S)
            except:
                print(sample.keys())
                print(self.train_loader.dataset.transform)
                raise

            if self.p.cuda:
                S = S.pin_memory().cuda()
                T = T.pin_memory().cuda()

            output = self.model['snet'](S)

            loss = loss_fn(output[M > 0.5], T[M > 0.5]) * destripe_weight
            self.history['train_loss_s2t'].append(float(loss) / destripe_weight)
            self.history['train_loss_s2t_weighted'].append(float(loss))
            if constancy_weight > 0:
                previous_loss = float(loss)
                output = self.model['snet'](T)
                loss += constancy_weight * loss_fn(output[M > 0.5], T[M > 0.5])
                self.history['train_loss_con'].append((float(loss) - previous_loss) / constancy_weight)
                self.history['train_loss_con_weighted'].append(float(loss) - previous_loss)
                self.history['train_loss_con_epoch'].append(self.epoch)

            identity_loss = float(loss_fn(S[M > 0.5], T[M > 0.5]))
        else:  # SURE loss, ignores constancy weight
            try:
                # S = Variable(sample['source'].double())
                S = Variable(sample['source'].double())  # including stripe augmentation
                T = Variable(S.data.clone())  # source == target
                M = (sample['mask_source'] > 0.5).expand_as(T)

            except:
                print(sample.keys())
                print(self.train_loader.dataset.transform)
                raise

            ## MC SURE, followed by network update
            # estimate noise std to GT data, np.std((noise_target(images) - images).numpy())
            # sure_sigma = self.sure_sigma
            sure_sigma = togpu((sample['source'][M > 0.5] - sample['target'][M > 0.5]).std().double())

            epsilon = 1e-3  # * torch.max(images)
            sure_weight = 1.0  # min((1.0 + self.epoch) / 3., 1.)  # min(step/10., 1.) # warmup

            ####
            pert = Variable(torch.tensor(self.sure_perturbation(source=S.numpy().copy())['source']).double())
            # sure_perturbation_scale = self.sure_perturbation_scale
            sure_perturbation_scale = 1. / (np.squeeze(pert[M > 0.5].numpy()) - np.squeeze(S[M > 0.5])).std()

            # plots([np.squeeze(sample['source'].numpy())[45], np.squeeze(sample['target'].cpu().numpy())[45],
            #        np.squeeze(pert.numpy())[45]])
            # plt.show()
            # print('sure_sigma:', '%.1e' % sure_sigma, 'pert scale: %.1f' % float(sure_perturbation_scale), end=' ')
            ####

            perturbation = Variable(togpu(sure_perturbation_scale * (pert - S)))
            S = togpu(S)
            # T = togpu(T)  # use S instead
            output = self.model['snet'](S)

            delta_f = self.model['snet'](S + (perturbation * epsilon)) - output
            # print('delta_f', nn.MSELoss()(delta_f, output))
            # divergence term normalised by number of voxels
            sure_mc_div_norm = torch.mean((perturbation * delta_f)[M > 0.5]) / epsilon
            # take l2 norm of divergence term to prevent large fluctuations into negative values
            # sure_div_term = 2. * sure_sigma ** 2 * sure_mc_div_norm
            if 'do_sure_ns' in self.p.dict and self.p.do_sure_ns:
                sure_div_term = togpu(2. * sure_sigma ** 2 * sure_mc_div_norm)
            else:
                sure_div_term = togpu(2. * sure_sigma ** 2 * torch.sqrt(sure_mc_div_norm ** 2))

            mse_loss = loss_fn(output[M > 0.5], S[M > 0.5])
            sure_loss = mse_loss - sure_sigma ** 2 + sure_weight * sure_div_term
            loss = sure_loss
            # sure_sigma = self.sure_sigma # np.std((noise_target(images) - images).numpy())
            # sure_perturbation_scale = self.sure_perturbation_scale
            #
            # epsilon = 1e-3  # * torch.max(images)
            # sure_weight = 1.0  # min((1.0 + self.epoch) / 3., 1.)  # min(step/10., 1.) # warmup
            #
            # pert = torch.tensor(self.sure_perturbation(source=S.data.clone().numpy())['source'])
            # perturbation = Variable(togpu(sure_perturbation_scale * (pert - S)))
            # S = togpu(S)
            # # T = togpu(T)  # use S instead
            # output = self.model['snet'](S)
            #
            # delta_f = self.model['snet'](S + (perturbation * epsilon)) - output
            # # print('delta_f', nn.MSELoss()(delta_f, output))
            # # divergence term normalised by number of voxels
            # sure_mc_div_norm = torch.mean((perturbation * delta_f)[M > 0.5]) / epsilon
            # # sure_div_term = 2. * gt_sigma ** 2 * sure_mc_div_norm
            # # take l2 norm of divergence term to prevent large fluctuations into negative values
            # sure_div_term = 2. * sure_sigma ** 2 * torch.sqrt(sure_mc_div_norm ** 2)
            #
            # mse_loss = loss_fn(output[M > 0.5], S[M > 0.5])
            # sure_loss = mse_loss - sure_sigma ** 2 + sure_weight * sure_div_term
            # loss = sure_loss

            self.history['train_loss_s2t'].append(float(mse_loss) / destripe_weight)
            self.history['train_loss_s2t_weighted'].append(float(mse_loss))

            self.history['train_sure_div_term'].append(float(sure_div_term))
            self.history['train_sure_epoch'].append(self.epoch)
            # if constancy_weight > 0:
            #     previous_loss = float(loss)
            #     output = self.model['snet'](T)
            #     loss += constancy_weight * loss_fn(output[M > 0.5], T[M > 0.5])
            #     self.history['train_loss_con'].append((float(loss) - previous_loss) / constancy_weight)
            #     self.history['train_loss_con_weighted'].append(float(loss) - previous_loss)
            #     self.history['train_loss_con_epoch'].append(self.epoch)

            identity_loss = 1.0

        loss.backward()
        self.optimiser.step()
        loss = float(loss)
        self._train_loss = loss
        self.history['train_loss'].append(loss)

        self.history['train_loss_identity'].append(identity_loss)
        self.history['train_loss_norm'].append(loss / identity_loss)
        self.history['train_epoch'].append(self.epoch)
        return loss, identity_loss


    @overrides(NetworkTrainer)
    def find_lr(self):
        raise NotImplemented
        import utils.learn.Learner
        train_step = 5
        if self.cuda:
            learn = utils.learn.Learner(self.model['snet'], self.train_loader, train_step, self.loss_func_cuda,
                                        cuda=self.cuda)
        else:
            learn = utils.learn.Learner(self.model['snet'], self.train_loader, train_step, self.loss_func,
                                        cuda=self.cuda)
        learn.find_learning_rate(VOL=0, epochs_per_lr=1, learning_rates=np.logspace(-5, -1, 6))

    @overrides(NetworkTrainer)
    def _val_step(self, screenshot=False, record=True, **kwargs):
        self._eval()
        val_params = {'stride': self.p.val_stride, 'start': 0}
        val_params.update(kwargs)

        def enumerate2(xs, start=0, step=1):
            for x in xs:
                for _ in range(start):
                    x = next(xs)
                yield x
                for _ in range(step):
                    next(xs)
        images = {'S': [], 'T': [], 'output': [], 'M': []}
        with torch.no_grad():
            loss = 0
            loss_norm = 0
            cnt = 0
            for sample_num, sample in enumerate2(enumerate(self.val_loader), val_params['start'], val_params['stride']):
                cnt += 1
                try:
                    S = Variable(sample['source'].double())
                    T = Variable(sample['target'].double())
                    M = (sample['mask_source'] > 0.5).expand_as(S)
                except:
                    print(sample.keys())
                    print(self.val_loader.dataset.transform)
                    raise
                if self.p.cuda:
                    S = S.pin_memory().cuda()
                    T = T.pin_memory().cuda()

                output = self._forward(S)
                if self.p.cuda:
                    lf = self.loss_func_cuda
                else:
                    lf = self.loss_func
                loss_ = float(lf(output[M > 0.5], T[M > 0.5]))
                loss += loss_
                loss_norm += loss_ / float(lf(S[M > 0.5], T[M > 0.5]))
                if screenshot:
                    images['S'].append(S.cpu().numpy().copy())
                    images['T'].append(T.cpu().numpy().copy())
                    images['M'].append(M.cpu().numpy().copy())
                    images['output'].append(output.cpu().numpy().copy())
            loss /= cnt
            loss_norm /= cnt
        self._val_loss = loss
        if record:
            self.history['val_loss'].append(loss)
            self.history['val_loss_norm'].append(loss_norm)
            self.history['val_epoch'].append(self.epoch)
            is_best = self.bestloss > loss
            if is_best:
                self.bestloss = loss

            self._save_checkpoint(is_best)

        else:
            is_best = False

        def imvjoin(images):
            shape = list(images[0].shape)
            assert len(shape) == 5, 'requires batch of 4D images (5D array), got {}'.format(shape)
            for im in images:
                if im.shape != shape:
                    for d in range(len(shape)):
                        shape[d] = max(shape[d], im.shape[d])
            out = np.zeros((shape[0], len(images), *shape[2:]))
            for i, im in enumerate(images):
                out[:,i,:im.shape[2], :im.shape[3], :im.shape[4]] = im
            return torch.from_numpy(out)

        if screenshot:
            images = (imvjoin(images['S']), imvjoin(images['T']), imvjoin(images['output']), imvjoin(images['M']))
        else:
            images = None

        return loss, loss_norm, is_best, images

if __name__ == '__main__':

    import argparse, shutil

    device = 0

    parser = argparse.ArgumentParser(description='Train destripe network.')
    parser.add_argument('params', type=str, help='json parameter file')
    parser.add_argument('--data', type=int, default=None, help='data setting')
    parser.add_argument('--device', help='where to run neural network (default = {})'.format(device), default=device)
    parser.add_argument('-c', '--cont', action='store_true')
    parser.add_argument('--override', action='store_true', help='override network via --cont irrespective of best loss')

    args = parser.parse_args()

    print("=" * 50)
    print("params:", args.params)
    print("data:", args.data)
    print("continue:", args.cont)
    print("=" * 50)

    if args.device:
        device = args.device

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

    Net = DeStripeNetwork()

    params = os.path.expanduser(args.params)
    assert os.path.isfile(params), params
    assert '.json' in params, params
    shutil.copy(params, params+'_in')
    p = Params(params)

    data_settings = args.data
    if data_settings is None:
        if 'data_settings' in p.dict:
            data_settings = p.data_settings
        else:
            data_settings = 0

    if data_settings == 0:
        settings = {"params": '/tmp/trainer.pth.tar.json',
                    'transforms_val': Compose([
                     StripedSourceCleanTarget(bias_noise=[(1, 0.05), (1, 0.6, -0.6), (1, 0.2, -0.05, -0.05)]),
                     SampleToTensor4D()]),
                    'transforms_train': Compose([
                     StripedSourceCleanTarget(bias_noise=[(1, 0.05), (1, 0.6, -0.6), (1, 0.2, -0.05, -0.05)]),
                     SampleToTensor4D()]),
                    "epochs": 0,
                    "checkpoint": '/tmp/trainer_bogus.pth.tar'}
    elif data_settings == 1:
        settings = {'transforms_val': Compose([
            StripedSourceCleanTarget(bias_noise=[(1, 0.05), (1, 0.6, -0.6), (1, 0.2, -0.05, -0.05)]),
                        SampleToTensor4D()]),
                    'transforms_train': Compose([
                        StripedSourceCleanTarget(bias_noise=[(1, 0.05), (1, 0.6, -0.6), (1, 0.2, -0.05, -0.05)]),
                        SampleToTensor4D()]),
                    "epochs": 1000,
                    'checkpoint':os.path.splitext(params)[0]}
    elif data_settings == 2:
        settings = {'transforms_val': Compose([
            StripedSourceCleanTarget(bias_noise=[(1, 0.05), (1, 0.6, -0.6), (1, 0.2, -0.05, -0.05)]),
            SampleToTensor4D()]),
            'transforms_train': Compose([
                StripedSourceCleanTarget(bias_noise=[(1, 0.05), (1, 0.6, -0.6), (1, 0.2, -0.05, -0.05)]),
                SampleToTensor4D()]),
            "normalise":'minmax'}
    elif data_settings == 3:
        settings = {'transforms_val': Compose([StripedSource(bias_noise=[(1, 0.3)], bias_noise_pack=(1, 0.1),
                                                          mode='1D_16-slice-pack', masked=True, decenter=True, copy_source2target=True),
                                            SampleToTensor4D()]),
                 'transforms_train': Compose([StripedSource(bias_noise=[(1, 0.3)], bias_noise_pack=(1, 0.1),
                                                            mode='1D_16-slice-pack', masked=True, decenter=True, copy_source2target=True),
                                              SampleToTensor4D()]),
                 "epochs": None,
                 "normalise": 'meanstd'}
    elif data_settings == 4:
        settings = {'transforms_val': Compose([StripedSource(bias_noise=[(1, 0.3)], bias_noise_pack=(1, 0.1),
                                                          mode='1D_16-slice-pack', masked=True, decenter=True, copy_source2target=True),
                                            SampleToTensor4D()]),
                 'transforms_train': Compose([StripedSource(bias_noise=[(1, 0.3)], bias_noise_pack=(1, 0.1),
                                                            mode='1D_16-slice-pack', masked=True, decenter=True, copy_source2target=True),
                                              SampleToTensor4D()]),
                 "normalise": None}
    elif data_settings == 5:
        settings = {'transforms_val': Compose([StripedSourceCleanTarget(bias_noise=[(1, 0.05), (1, 0.6, -0.6), (1, 0.2, -0.05, -0.05)]),
                        SampleToTensor4D()]),
                    'transforms_train': Compose([StripedSourceCleanTarget(bias_noise=[(1, 0.05), (1, 0.6, -0.6), (1, 0.2, -0.05, -0.05)]),
                        SampleToTensor4D()]),
                    "normalise": None}
    elif data_settings == 6:
        settings = {'transforms_val': Compose([StripedSourceCleanTarget(bias_noise=[(1, 0.05), (1, 0.6, -0.6), (1, 0.2, -0.05, -0.05)]),
                        SampleToTensor4D()]),
                    'transforms_train': Compose([StripedSourceCleanTarget(bias_noise=[(1, 0.05), (1, 0.6, -0.6), (1, 0.2, -0.05, -0.05)]),
                        SampleToTensor4D()]),
                    "normalise": "minmax"}
        Net.import_functions = {'source': get_l0, 'target':get_l0}
        Net.data_postproc = split_by_vol
    elif data_settings == 7:
        settings = {'transforms_val': Compose([StripedSourceCleanTarget(bias_noise=[(1, 0.05), (1, 0.6, -0.6), (1, 0.2, -0.05, -0.05)]),
                        SampleToTensor4D()]),
                    'transforms_train': Compose([StripedSourceCleanTarget(bias_noise=[(1, 0.05), (1, 0.6, -0.6), (1, 0.2, -0.05, -0.05)]),
                        SampleToTensor4D()]),
                    "normalise": "percentile"}
        Net.import_functions = {'source': get_l0, 'target':get_l0}
        Net.data_postproc = split_by_vol
    elif data_settings == 8:
        settings = {'transforms_val': Compose(
            [StripedSourceCleanTarget(bias_noise=[(1, 0.05), (1, 0.6, -0.6), (1, 0.2, -0.05, -0.05)], masked=True),
             SampleToTensor4D()]),
                    'transforms_train': Compose(
                        [StripedSourceCleanTarget(bias_noise=[(1, 0.05), (1, 0.6, -0.6), (1, 0.2, -0.05, -0.05)], masked=True),
                         SampleToTensor4D()]),
                    "normalise": "percentile"}
        Net.import_functions = {'source': get_l0, 'target': get_l0}
        Net.data_postproc = split_by_vol
    elif data_settings == 9:
        settings = {'transforms_val': Compose([StripedSourceCleanTarget(bias_noise=[(1, 0.05), (1, 0.6, -0.6), (1, 0.2, -0.05, -0.05)]),
                        SampleToTensor4D()]),
                    'transforms_train': Compose([StripedSourceCleanTarget(bias_noise=[(1, 0.05), (1, 0.6, -0.6), (1, 0.2, -0.05, -0.05)]),
                        SampleToTensor4D()]),
                    "normalise": "percentile-scale"}
        Net.import_functions = {'source': get_l0, 'target':get_l0}
        Net.data_postproc = split_by_vol
    elif data_settings == 10:
        settings = {'transforms_val': Compose([StripedSourceCleanTarget(bias_noise=[(1, 0.05), (1, 0.6, -0.6), (1, 0.2, -0.05, -0.05)]),
                        SampleToTensor4D()]),
                    'transforms_train': Compose([StripedSourceCleanTarget(bias_noise=[(1, 0.05), (1, 0.6, -0.6), (1, 0.2, -0.05, -0.05)]),
                        SampleToTensor4D()]),
                    "normalise": "percentile-scale"}
        Net.import_functions = {'source': get_n_bbalanced, 'target': get_n_bbalanced}
        Net.data_postproc = split_by_vol
    elif data_settings == 11:
        settings = {'transforms_val': Compose([StripedSourceCleanTarget(bias_noise=[(1, 0.05), (1, 0.6, -0.6), (1, 0.2, -0.05, -0.05)]),
                        SampleToTensor4D()]),
                    'transforms_train': Compose([StripedSourceCleanTarget(bias_noise=[(1, 0.05), (1, 0.6, -0.6), (1, 0.2, -0.05, -0.05)]),
                        SampleToTensor4D()]),
                    "normalise": "percentile"}
        Net.import_functions = {'source': get_n_bbalanced, 'target': get_n_bbalanced}
        Net.data_postproc = split_by_vol
    elif data_settings == 12:
        settings = {'transforms_val': Compose([StripedSource(bias_noise=[(1, 0.3)],
                                                             bias_noise_pack=(1, 0.1),
                                                             mode='1D_16-slice-pack',
                                                             masked=True,
                                                             decenter=True,
                                                             copy_source2target=True), SampleToTensor4D()]),
                    'transforms_train': Compose([StripedSource(bias_noise=[(1, 0.3)],
                                                               bias_noise_pack=(1, 0.1),
                                                               mode='1D_16-slice-pack',
                                                               masked=True, decenter=True,
                                                               copy_source2target=True), SampleToTensor4D()]),
                    "normalise": "percentile-scale"}
    elif data_settings == 20.0:  # train & eval on IS without random rotation
        from augmentation2 import *
        settings = {'transforms_val': Compose([GeneralStripeTransform(interleave=3, block_size=16,
                                                                      centre_variance=(0.9, 1.1),
                                                                      noise_variance=(0.05, 0.05),
                                                                      block_slope_range=(0., 0.3), power=1),
                                               SampleToTensor4D()]),
                    'transforms_train': Compose([GeneralStripeTransform(interleave=3, block_size=16,
                                                                      centre_variance=(0.9, 1.1),
                                                                      noise_variance=(0.05, 0.05),
                                                                      block_slope_range=(0., 0.3), power=1),
                                               SampleToTensor4D()]),
                    "normalise": "percentile-scale"}
        Net.import_functions = {'source': get_n_bbalanced, 'target': get_n_bbalanced}
        Net.data_postproc = split_by_vol
    elif data_settings == 20.02:  # train & eval on IS without random rotation
        from augmentation2 import *
        settings = {'transforms_val': Compose([GeneralStripeTransform(interleave=3, block_size=16,
                                                                      centre_variance=(0.9, 1.1),
                                                                      noise_variance=(0.05, 0.05),
                                                                      block_slope_range=(0., 0.3), power=2),
                                               SampleToTensor4D()]),
                    'transforms_train': Compose([GeneralStripeTransform(interleave=3, block_size=16,
                                                                      centre_variance=(0.9, 1.1),
                                                                      noise_variance=(0.05, 0.05),
                                                                      block_slope_range=(0., 0.3), power=2),
                                               SampleToTensor4D()]),
                    "normalise": "percentile-scale"}
        Net.import_functions = {'source': get_n_bbalanced, 'target': get_n_bbalanced}
        Net.data_postproc = split_by_vol
    elif data_settings == 20.03:  # train & eval on IS without random rotation; power=3
        from augmentation2 import *
        settings = {'transforms_val': Compose([GeneralStripeTransform(interleave=3, block_size=16,
                                                                      centre_variance=(0.9, 1.1),
                                                                      noise_variance=(0.05, 0.05),
                                                                      block_slope_range=(0., 0.3), power=3),
                                               SampleToTensor4D()]),
                    'transforms_train': Compose([GeneralStripeTransform(interleave=3, block_size=16,
                                                                      centre_variance=(0.9, 1.1),
                                                                      noise_variance=(0.05, 0.05),
                                                                      block_slope_range=(0., 0.3), power=3),
                                               SampleToTensor4D()]),
                    "normalise": "percentile-scale"}
        Net.import_functions = {'source': get_n_bbalanced, 'target': get_n_bbalanced}
        Net.data_postproc = split_by_vol
    elif data_settings == 20.1:  # train & eval on IS; power=1
        from augmentation2 import *
        settings = {'transforms_val': Compose([RandomDihedralSliceDirPreserving(),
                                               GeneralStripeTransform(interleave=3, block_size=16,
                                                                      centre_variance=(0.9, 1.1),
                                                                      noise_variance=(0.05, 0.05),
                                                                      block_slope_range=(0., 0.3), power=1),
                                               RandomDihedralSliceDirPreserving(),
                                               SampleToTensor4D()]),
                    'transforms_train': Compose([RandomDihedralSliceDirPreserving(),
                                                 GeneralStripeTransform(interleave=3, block_size=16,
                                                                        centre_variance=(0.9, 1.1),
                                                                        noise_variance=(0.05, 0.05),
                                                                        block_slope_range=(0., 0.3), power=1),
                                                 SampleToTensor4D()]),
                    "normalise": "percentile-scale"}
        Net.import_functions = {'source': get_n_bbalanced, 'target': get_n_bbalanced}
        Net.data_postproc = split_by_vol
    elif data_settings == 20.2:  # train & eval on IS,AP; power=1
        from augmentation2 import *
        settings = {'transforms_val': Compose([RndChoiceTransform(AxialSliceDir(modes=('AP')), identity_transform=True),
                                               RandomDihedralSliceDirPreserving(),
                                               GeneralStripeTransform(interleave=3, block_size=16,
                                                                      centre_variance=(0.9, 1.1),
                                                                      noise_variance=(0.05, 0.05),
                                                                      block_slope_range=(0., 0.3), power=1),
                                               SampleToTensor4D()]),
                    'transforms_train': Compose([RndChoiceTransform(AxialSliceDir(modes=('AP')), identity_transform=True),
                                                 RandomDihedralSliceDirPreserving(),
                                                 GeneralStripeTransform(interleave=3, block_size=16,
                                                                        centre_variance=(0.9, 1.1),
                                                                        noise_variance=(0.05, 0.05),
                                                                        block_slope_range=(0., 0.3), power=1),
                                                 SampleToTensor4D()]),
                    "normalise": "percentile-scale"}
        Net.import_functions = {'source': get_n_bbalanced, 'target': get_n_bbalanced}
        Net.data_postproc = split_by_vol
    elif data_settings == 20.3:  # train & eval on IS,LR; power=1
        from augmentation2 import *
        settings = {'transforms_val': Compose([RndChoiceTransform(AxialSliceDir(modes=('LR')), identity_transform=True),
                                               RandomDihedralSliceDirPreserving(),
                                               GeneralStripeTransform(interleave=3, block_size=16,
                                                                      centre_variance=(0.9, 1.1),
                                                                      noise_variance=(0.05, 0.05),
                                                                      block_slope_range=(0., 0.3), power=1),
                                               SampleToTensor4D()]),
                    'transforms_train': Compose(
                        [RndChoiceTransform(AxialSliceDir(modes=('AP')), identity_transform=True),
                         RandomDihedralSliceDirPreserving(),
                         GeneralStripeTransform(interleave=3, block_size=16,
                                                centre_variance=(0.9, 1.1),
                                                noise_variance=(0.05, 0.05),
                                                block_slope_range=(0., 0.3), power=1),
                         SampleToTensor4D()]),
                    "normalise": "percentile-scale"}
        Net.import_functions = {'source': get_n_bbalanced, 'target': get_n_bbalanced}
        Net.data_postproc = split_by_vol
    elif data_settings == 20.4:  # train & eval on AP,LR
        from augmentation2 import *
        settings = {'transforms_val': Compose([RndChoiceTransform(AxialSliceDir(modes=('AP','LR')), identity_transform=False),
                                               RandomDihedralSliceDirPreserving(),
                                               GeneralStripeTransform(interleave=3, block_size=16,
                                                                      centre_variance=(0.9, 1.1),
                                                                      noise_variance=(0.05, 0.05),
                                                                      block_slope_range=(0., 0.3), power=1),
                                               SampleToTensor4D()]),
                    'transforms_train': Compose([RndChoiceTransform(AxialSliceDir(modes=('AP','LR')), identity_transform=False),
                                                RandomDihedralSliceDirPreserving(),
                                                GeneralStripeTransform(interleave=3, block_size=16,
                                                                       centre_variance=(0.9, 1.1),
                                                                       noise_variance=(0.05, 0.05),
                                                                       block_slope_range=(0., 0.3), power=1),
                                                 SampleToTensor4D()]),
                    "normalise": "percentile-scale"}
        Net.import_functions = {'source': get_n_bbalanced, 'target': get_n_bbalanced}
        Net.data_postproc = split_by_vol
    elif data_settings == 20.4:  # train & eval on AP,LR
        from augmentation2 import *
        settings = {'transforms_val': Compose([RndChoiceTransform(AxialSliceDir(modes=('AP','LR')), identity_transform=False),
                                               RandomDihedralSliceDirPreserving(),
                                               GeneralStripeTransform(interleave=3, block_size=16,
                                                                      centre_variance=(0.9, 1.1),
                                                                      noise_variance=(0.05, 0.05),
                                                                      block_slope_range=(0., 0.3), power=1),
                                               SampleToTensor4D()]),
                    'transforms_train': Compose([RndChoiceTransform(AxialSliceDir(modes=('AP','LR')), identity_transform=False),
                                                RandomDihedralSliceDirPreserving(),
                                                GeneralStripeTransform(interleave=3, block_size=16,
                                                                       centre_variance=(0.9, 1.1),
                                                                       noise_variance=(0.05, 0.05),
                                                                       block_slope_range=(0., 0.3), power=1),
                                                 SampleToTensor4D()]),
                    "normalise": "percentile-scale"}
        Net.import_functions = {'source': get_n_bbalanced, 'target': get_n_bbalanced}
        Net.data_postproc = split_by_vol
    elif data_settings == 20.5:  # train & eval: AP,LR,IS; test on AP,LR,IS
        from augmentation2 import *
        settings = {'transforms_val': Compose([RndChoiceTransform(AxialSliceDir(modes=('AP')),
                                                                  AxialSliceDir(modes=('LR')),
                                                                  identity_transform=True),
                                               RandomDihedralSliceDirPreserving(),
                                               GeneralStripeTransform(interleave=3, block_size=16,
                                                                      centre_variance=(0.9, 1.1),
                                                                      noise_variance=(0.05, 0.05),
                                                                      block_slope_range=(0., 0.3), power=1),
                                               SampleToTensor4D()]),
                    'transforms_train': Compose([RndChoiceTransform(AxialSliceDir(modes=('AP')),
                                                                    AxialSliceDir(modes=('LR')),
                                                                    identity_transform=True),
                                                 RandomDihedralSliceDirPreserving(),
                                                 GeneralStripeTransform(interleave=3, block_size=16,
                                                                        centre_variance=(0.9, 1.1),
                                                                        noise_variance=(0.05, 0.05),
                                                                        block_slope_range=(0., 0.3), power=1),
                                                 SampleToTensor4D()]),
                    "normalise": "percentile-scale"}
        Net.import_functions = {'source': get_n_bbalanced, 'target': get_n_bbalanced}
        Net.data_postproc = split_by_vol
    elif data_settings == 21:  # train & eval: AP,LR,IS; wc(IS)=0; power=1
        from augmentation2 import *
        settings = {'transforms_val': Compose([RndChoiceTransform(AxialSliceDir(modes=('AP')),
                                                                  AxialSliceDir(modes=('LR')),
                                                                  identity_transform=True),
                                               RandomDihedralSliceDirPreserving(),
                                               GeneralStripeTransform(interleave=3, block_size=16,
                                                                      centre_variance=(0.9, 1.1),
                                                                      noise_variance=(0.05, 0.05),
                                                                      block_slope_range=(0., 0.3), power=1),
                                               SampleToTensor4D()]),
                    'transforms_train': Compose([RndChoiceTransform(AxialSliceDir(modes=('AP')),
                                                                    AxialSliceDir(modes=('LR')),
                                                                    identity_transform=True),
                                                 RandomDihedralSliceDirPreserving(),
                                                 GeneralStripeTransform(interleave=3, block_size=16,
                                                                        centre_variance=(0.9, 1.1),
                                                                        noise_variance=(0.05, 0.05),
                                                                        block_slope_range=(0., 0.3), power=1),
                                                 SampleToTensor4D()]),
                    "weight_scheduler": WeightSchedulerISwcZero(),
                    "normalise": "percentile-scale"}
        Net.import_functions = {'source': get_n_bbalanced, 'target': get_n_bbalanced}
        Net.data_postproc = split_by_vol
    elif data_settings == 22:  # train & eval: AP,LR, no blocks; power=2
        from augmentation2 import *
        settings = {'transforms_val': Compose([RndChoiceTransform(AxialSliceDir(modes=('AP','LR')), identity_transform=False),
                                               RandomDihedralSliceDirPreserving(),
                                               GeneralStripeTransform(interleave=3, block_size=0,
                                                                      centre_variance=(0.9, 1.1),
                                                                      noise_variance=(0.05, 0.05),
                                                                      block_slope_range=(0., 0.3), power=2),
                                               SampleToTensor4D()]),
                    'transforms_train': Compose([RndChoiceTransform(AxialSliceDir(modes=('AP','LR')), identity_transform=False),
                                                RandomDihedralSliceDirPreserving(),
                                                GeneralStripeTransform(interleave=3, block_size=0,
                                                                       centre_variance=(0.9, 1.1),
                                                                       noise_variance=(0.05, 0.05),
                                                                       block_slope_range=(0., 0.3), power=2),
                                                 SampleToTensor4D()]),
                    "normalise": "percentile-scale"}
        Net.import_functions = {'source': get_n_bbalanced, 'target': get_n_bbalanced}
        Net.data_postproc = split_by_vol
    elif data_settings == 22.1:  # train & eval: AP,LR,IS; power=2
        from augmentation2 import *
        settings = {'transforms_val': Compose([RndChoiceTransform(AxialSliceDir(modes=('AP','LR')), identity_transform=True),
                                               RandomDihedralSliceDirPreserving(),
                                               GeneralStripeTransform(interleave=3, block_size=0,
                                                                      centre_variance=(0.9, 1.1),
                                                                      noise_variance=(0.05, 0.05),
                                                                      block_slope_range=(0., 0.3), power=2),
                                               SampleToTensor4D()]),
                    'transforms_train': Compose([RndChoiceTransform(AxialSliceDir(modes=('AP','LR')), identity_transform=True),
                                                RandomDihedralSliceDirPreserving(),
                                                GeneralStripeTransform(interleave=3, block_size=0,
                                                                       centre_variance=(0.9, 1.1),
                                                                       noise_variance=(0.05, 0.05),
                                                                       block_slope_range=(0., 0.3), power=2),
                                                 SampleToTensor4D()]),
                    "normalise": "percentile-scale"}
        Net.import_functions = {'source': get_n_bbalanced, 'target': get_n_bbalanced}
        Net.data_postproc = split_by_vol
    elif data_settings == 23:  # train: IS,AP,LR, eval: AP,LR no blocks, power=2, wc(IS)=0
        from augmentation2 import *

        settings = {
            'transforms_val': Compose([RndChoiceTransform(AxialSliceDir(modes=('AP', 'LR')), identity_transform=False),
                                       RandomDihedralSliceDirPreserving(),
                                       GeneralStripeTransform(interleave=3, block_size=0,
                                                              centre_variance=(0.9, 1.1),
                                                              noise_variance=(0.05, 0.05),
                                                              block_slope_range=(0., 0.3), power=2),
                                       SampleToTensor4D()]),
            'transforms_train': Compose(
                [RndChoiceTransform(AxialSliceDir(modes=('AP', 'LR')), identity_transform=True),
                 RandomDihedralSliceDirPreserving(),
                 GeneralStripeTransform(interleave=3, block_size=0,
                                        centre_variance=(0.9, 1.1),
                                        noise_variance=(0.05, 0.05),
                                        block_slope_range=(0., 0.3), power=2),
                 SampleToTensor4D()]),
            "normalise": "percentile-scale",
            "weight_scheduler": WeightSchedulerISwcZero()}
        Net.import_functions = {'source': get_n_bbalanced, 'target': get_n_bbalanced}
        Net.data_postproc = split_by_vol
    elif data_settings == 23.05:  # train: IS,AP,LR, eval: AP,LR no blocks, balanced IS,AP,LR; power=2, wc(IS)=0
        from augmentation2 import *

        settings = {
            'transforms_val': Compose([RndChoiceTransform(AxialSliceDir(modes=('AP', 'LR')), identity_transform=False),
                                       RandomDihedralSliceDirPreserving(),
                                       GeneralStripeTransform(interleave=3, block_size=0,
                                                              centre_variance=(0.9, 1.1),
                                                              noise_variance=(0.05, 0.05),
                                                              block_slope_range=(0., 0.3), power=2),
                                       SampleToTensor4D()]),
            'transforms_train': Compose(
                [RndChoiceTransform(AxialSliceDir(modes=('AP')),
                                    AxialSliceDir(modes=('LR')),
                                    identity_transform=True),
                 RandomDihedralSliceDirPreserving(),
                 GeneralStripeTransform(interleave=3, block_size=0,
                                        centre_variance=(0.9, 1.1),
                                        noise_variance=(0.05, 0.05),
                                        block_slope_range=(0., 0.3), power=2),
                 SampleToTensor4D()]),
            "normalise": "percentile-scale",
            "weight_scheduler": WeightSchedulerISwcZero()}
        Net.import_functions = {'source': get_n_bbalanced, 'target': get_n_bbalanced}
        Net.data_postproc = split_by_vol
    elif data_settings == 23.1:  # train on IS,AP,LR, eval on AP,LR stripes without blocks; wc(IS)=0; power=1
        from augmentation2 import *
        settings = {
            'transforms_val': Compose([RndChoiceTransform(AxialSliceDir(modes=('AP', 'LR')), identity_transform=False),
                                       RandomDihedralSliceDirPreserving(),
                                       GeneralStripeTransform(interleave=3, block_size=0,
                                                              centre_variance=(0.9, 1.1),
                                                              noise_variance=(0.05, 0.05),
                                                              block_slope_range=(0., 0.3), power=1),
                                       SampleToTensor4D()]),
            'transforms_train': Compose(
                [RndChoiceTransform(AxialSliceDir(modes=('AP', 'LR')), identity_transform=True),
                 RandomDihedralSliceDirPreserving(),
                 GeneralStripeTransform(interleave=3, block_size=0,
                                        centre_variance=(0.9, 1.1),
                                        noise_variance=(0.05, 0.05),
                                        block_slope_range=(0., 0.3), power=1),
                 SampleToTensor4D()]),
            "normalise": "percentile-scale",
            "weight_scheduler": WeightSchedulerISwcZero()}
        Net.import_functions = {'source': get_n_bbalanced, 'target': get_n_bbalanced}
        Net.data_postproc = split_by_vol
    elif data_settings == 24:  # train & eval on AP,LR,IS. noise2noise on IS, cw_is=0.0; power=2
        from augmentation2 import *

        settings = {
            'transforms_val': Compose([RndChoiceTransform(AxialSliceDir(modes=('AP', 'LR')), identity_transform=False),
                                       RandomDihedralSliceDirPreserving(),
                                       GeneralStripeTransform(interleave=3, block_size=0,
                                                              centre_variance=(0.9, 1.1),
                                                              noise_variance=(0.05, 0.05),
                                                              block_slope_range=(0., 0.3), power=2),
                                       SampleToTensor4D()]),
            'transforms_train': Compose(
                [RndChoiceTransform(AxialSliceDir(modes=('AP', 'LR')), identity_transform=True),
                 RandomDihedralSliceDirPreserving(),
                 GeneralStripeTransform(interleave=3, block_size=0,  # stripes on source data, clean target
                                        centre_variance=(0.9, 1.1),
                                        noise_variance=(0.05, 0.05),
                                        block_slope_range=(0., 0.3), power=2),
                 GeneralStripeTransform(interleave=3, block_size=0,  # stripes on IS target data
                                        centre_variance=(0.9, 1.1),
                                        noise_variance=(0.05, 0.05),
                                        block_slope_range=(0., 0.3),
                                        data_key="target", copy2target=False, apply_block_to_target=False, modes=['IS'],
                                        power=1),
                 SampleToTensor4D()]),
            "normalise": "percentile-scale",
            "weight_scheduler": WeightSchedulerISwcZero(isweight=0.0)}
        Net.import_functions = {'source': get_n_bbalanced, 'target': get_n_bbalanced}
        Net.data_postproc = split_by_vol

    elif data_settings == 30:  # train on AP LR IS (cw=0.5), cw_is=0.5; power=2
        from augmentation2 import *
        settings = {
            'transforms_val': Compose([RndChoiceTransform(AxialSliceDir(modes=('AP')),
                                                          AxialSliceDir(modes=('LR')),
                                                          identity_transform=False),
                                       RandomDihedralSliceDirPreserving(),
                                       GeneralStripeTransform(interleave=3, block_size=0,
                                                              centre_variance=(0.9, 1.1),
                                                              noise_variance=(0.05, 0.05),
                                                              block_slope_range=(0., 0.3), power=2),
                                       SampleToTensor4D()]),
            'transforms_train': Compose([RndChoiceTransform(AxialSliceDir(modes=('AP')),
                                                            AxialSliceDir(modes=('LR')),
                                                            identity_transform=True),
                                         RandomDihedralSliceDirPreserving(),
                                         GeneralStripeTransform(interleave=3, block_size=0,
                                                                centre_variance=(0.9, 1.1),
                                                                noise_variance=(0.05, 0.05),
                                                                block_slope_range=(0., 0.3), power=2),
                                         SampleToTensor4D()]),
            "normalise": "percentile-scale",
            "weight_scheduler": WeightSchedulerISwcZero(isweight=0.5)}
        Net.import_functions = {'source': get_n_bbalanced, 'target': get_n_bbalanced}
        Net.data_postproc = split_by_vol
    elif data_settings == 40:  # SURE: IS, MSE: AP, LR, power=2
        # source == target
        from augmentation2 import *

        settings = {
            'transforms_val': Compose([RndChoiceTransform(AxialSliceDir(modes=('AP', 'LR')), identity_transform=True),
                                       RandomDihedralSliceDirPreserving(),
                                       GeneralStripeTransform(interleave=3, block_size=0,
                                                              centre_variance=(0.9, 1.1),
                                                              noise_variance=(0.05, 0.05),
                                                              block_slope_range=(0., 0.3), power=2),
                                       SampleToTensor4D()]),
            'transforms_train': Compose(
                [RndChoiceTransform(AxialSliceDir(modes=('AP', 'LR')), identity_transform=True),
                 RandomDihedralSliceDirPreserving(),
                 GeneralStripeTransform(interleave=3, block_size=0,
                                        centre_variance=(0.9, 1.1),
                                        noise_variance=(0.05, 0.05),
                                        block_slope_range=(0., 0.3), power=2),
                 SampleToTensor4D()]),
            "normalise": "percentile-scale"}
        Net.import_functions = {'source': get_n_bbalanced, 'target': get_n_bbalanced}
        Net.data_postproc = split_by_vol

        Net.do_sure = True
        Net.sure_perturbation = GeneralStripeTransform(interleave=3, block_size=0,
                                                       centre_variance=(0.9, 1.1),
                                                       noise_variance=(0.05, 0.05),
                                                       block_slope_range=(0., 0.3), power=2)
    elif data_settings == 40.1:  # SURE: IS, eval on all, power=2
        # source == target
        from augmentation2 import *

        settings = {
            'transforms_val': Compose([RndChoiceTransform(AxialSliceDir(modes=('AP', 'LR')), identity_transform=True),
                                       RandomDihedralSliceDirPreserving(),
                                       GeneralStripeTransform(interleave=3, block_size=0,
                                                              centre_variance=(0.9, 1.1),
                                                              noise_variance=(0.05, 0.05),
                                                              block_slope_range=(0., 0.3), power=2),
                                       SampleToTensor4D()]),
            'transforms_train': Compose(
                [RandomDihedralSliceDirPreserving(),
                 GeneralStripeTransform(interleave=3, block_size=0,
                                        centre_variance=(0.9, 1.1),
                                        noise_variance=(0.05, 0.05),
                                        block_slope_range=(0., 0.3), power=2),
                 SampleToTensor4D()]),
            "normalise": "percentile-scale"}
        Net.import_functions = {'source': get_n_bbalanced, 'target': get_n_bbalanced}
        Net.data_postproc = split_by_vol

        Net.do_sure = True
        Net.sure_perturbation = GeneralStripeTransform(interleave=3, block_size=0,
                                                       centre_variance=(0.9, 1.1),
                                                       noise_variance=(0.05, 0.05),
                                                       block_slope_range=(0., 0.3), power=2)
    elif data_settings == 40.2:  # SURE IS, MSE: LR,AP balanced orientations, eval on all, power=2
        #  source == target
        from augmentation2 import *

        settings = {
            'transforms_val': Compose([RndChoiceTransform(AxialSliceDir(modes=('AP', 'LR')), identity_transform=True),
                                       RandomDihedralSliceDirPreserving(),
                                       GeneralStripeTransform(interleave=3, block_size=0,
                                                              centre_variance=(0.9, 1.1),
                                                              noise_variance=(0.05, 0.05),
                                                              block_slope_range=(0., 0.3), power=2),
                                       SampleToTensor4D()]),
            'transforms_train': Compose(
                [RndChoiceTransform(AxialSliceDir(modes=('AP')),
                                    AxialSliceDir(modes=('LR')),
                                    identity_transform=True),
                 RandomDihedralSliceDirPreserving(),
                 GeneralStripeTransform(interleave=3, block_size=0,
                                        centre_variance=(0.9, 1.1),
                                        noise_variance=(0.05, 0.05),
                                        block_slope_range=(0., 0.3), power=2),
                 SampleToTensor4D()]),
            "normalise": "percentile-scale"}
        Net.import_functions = {'source': get_n_bbalanced, 'target': get_n_bbalanced}
        Net.data_postproc = split_by_vol

        Net.do_sure = True
        Net.sure_perturbation = GeneralStripeTransform(interleave=3, block_size=0,
                                                       centre_variance=(0.9, 1.1),
                                                       noise_variance=(0.05, 0.05),
                                                       block_slope_range=(0., 0.3), power=2)
    elif data_settings == 40.21:  # for dHCP SHARD recon04 with SURE: IS; MSE: LR,AP balanced orientations, eval on all, power=2
        #  source == target
        from augmentation2 import *

        settings = {
            'transforms_val': Compose([RndChoiceTransform(AxialSliceDir(modes=('AP', 'LR')), identity_transform=True),
                                       RandomDihedralSliceDirPreserving(),
                                       GeneralStripeTransform(interleave=3, block_size=0,
                                                              centre_variance=(0.9, 1.1),
                                                              noise_variance=(0.05, 0.05),
                                                              block_slope_range=(0., 0.3), power=2),
                                       SampleToTensor4D()]),
            'transforms_train': Compose(
                [RndChoiceTransform(AxialSliceDir(modes=('AP')),
                                    AxialSliceDir(modes=('LR')),
                                    identity_transform=True),
                 RandomDihedralSliceDirPreserving(),
                 GeneralStripeTransform(interleave=3, block_size=0,
                                        centre_variance=(0.9, 1.1),
                                        noise_variance=(0.05, 0.05),
                                        block_slope_range=(0., 0.3), power=2),
                 SampleToTensor4D()]),
            "normalise": None}
        Net.import_functions = {'source': get_one_bbalanced}
        Net.data_postproc = VolumeLoader(normalise=normalise_fun)

        Net.do_sure = True
        Net.sure_perturbation = GeneralStripeTransform(interleave=3, block_size=0,
                                                       centre_variance=(0.9, 1.1),
                                                       noise_variance=(0.05, 0.05),
                                                       block_slope_range=(0., 0.3), power=2)

    elif data_settings == 40.3:  # for dHCP SHARD recon04 MSE: IS,LR,AP balanced orientations, eval on all, power=2
        #  source == target
        from augmentation2 import *

        settings = {
            'transforms_val': Compose([RndChoiceTransform(AxialSliceDir(modes=('AP', 'LR')), identity_transform=True),
                                       RandomDihedralSliceDirPreserving(),
                                       GeneralStripeTransform(interleave=3, block_size=0,
                                                              centre_variance=(0.9, 1.1),
                                                              noise_variance=(0.05, 0.05),
                                                              block_slope_range=(0., 0.3), power=2),
                                       SampleToTensor4D()]),
            'transforms_train': Compose(
                [RndChoiceTransform(AxialSliceDir(modes=('AP')),
                                    AxialSliceDir(modes=('LR')),
                                    identity_transform=True),
                 RandomDihedralSliceDirPreserving(),
                 GeneralStripeTransform(interleave=3, block_size=0,
                                        centre_variance=(0.9, 1.1),
                                        noise_variance=(0.05, 0.05),
                                        block_slope_range=(0., 0.3), power=2),
                 SampleToTensor4D()]),
            "normalise": None}
        Net.import_functions = {'source': get_one_bbalanced}
        Net.data_postproc = VolumeLoader(normalise=normalise_fun)

        Net.do_sure = False
        Net.sure_perturbation = GeneralStripeTransform(interleave=3, block_size=0,
                                                       centre_variance=(0.9, 1.1),
                                                       noise_variance=(0.05, 0.05),
                                                       block_slope_range=(0., 0.3), power=2)
    else:
        assert 0, (data_settings, args.data)

    print("=" * 50)
    print('data_settings:', data_settings)
    pprint.pprint(settings)
    p.data_settings = data_settings
    print("=" * 50)
    p.save(params)

    # if len(sys.argv) == 1:
    #     settings = settings1
    #     sys.exit(1)
    #     Net = DeStripeNetwork()
    #     Net.continue_training(**settings)
    #     Net.p.valpath = '/home/mp14/data/stripes/val'
    #     Net._predict('val', nsamples=3, transforms=None, suffix='_aug')
    #     Net._predict('val', nsamples=3, transforms=SampleToTensor4D(), suffix='_noaug')
    #     Net._predict('val', nsamples=3, transforms=SampleToTensor4D(), suffix='_noaug_train', set_train=True)
    if not args.cont:
        Net.training(params=params, **settings)
    else:
        Net.continue_training(params=params, **settings, checkpoint=params.rstrip('.json'), override=args.override)

    # Net.p.valpath = '/home/mp14/data/stripes/val'
    # Net._predict('val', nsamples=3, transforms=None, suffix='_aug')
    # Net._predict('val', nsamples=3, transforms=SampleToTensor4D(), suffix='_noaug')
    # Net._predict('val', nsamples=3, transforms=SampleToTensor4D(), suffix='_noaug_train', set_train=True)
