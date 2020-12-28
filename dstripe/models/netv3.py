from models.layers import *


class SNet3(nn.Module):
    def __init__(self, n_block, input_dim, dim=16, updim=2, downsample=None, final_pool=2,
                 activ='relu', pad_type='zero',
                 k1=3, k2=1,
                 prep='none', norm='none', bias=True,
                 custom_init=False,
                 mode=0,
                 field_filter='sinc_hamming', field_filter_options={},
                 inplane_filter=None, inplane_filter_options={},
                 dobn=False,
                 npass=1,
                 preventRedundantParams=False,
                 zdilation=[1, 2, 3],
                 inplane_first=True):
        super(SNet3, self).__init__()

        self.input_dim = input_dim
        if isinstance(mode, int):
            modes = {0: 'global_avpool', 1: 'upsample',
                     2: 'center', 3: 'center+', 4: 'none'}
            assert mode in modes, mode
            mode = modes[mode]
        self.mode = mode
        self.return_field = False
        self.return_x = True
        self.npass = npass

        self.inplane_first = inplane_first

        if prep == 'none':
            self.prep = None
        elif prep == 'demean':
            self.prep = ZeroMaskedLayerNorm(input_dim, std=False, affine=False)
        elif prep == 'whiten':
            self.prep = ZeroMaskedLayerNorm(input_dim, std=True, affine=False)
        else:
            assert 0, "Unsupported prep: {}".format(prep)

        if field_filter is None:
            self.field_filter = None
        elif field_filter == 'sinc_hamming':
            # TODO explicitly set "filter_type": "sinc_hamming", left as is due to legacy bug
            ff_opt = {"kernel_size": 9, "padding": 'reflect'}
            ff_opt.update(field_filter_options)
            self.field_filter = SliceLowPassFilter(channels=1, **ff_opt)
        elif field_filter.startswith('gauss'):
            ff_opt = {"kernel_size": 9, "padding": 'reflect',
                      "filter_type": "gaussian"}
            ff_opt.update(field_filter_options)
            self.field_filter = SliceLowPassFilter(channels=1, **ff_opt)
        else:
            assert 0, field_filter

        if inplane_filter is None:
            self.inplane_filter = None
        elif inplane_filter == "gauss":
            opt = {"blur_sigma": 3, "kernel_size": 9, "mode": "reflect"}
            opt.update(inplane_filter_options)
            self.inplane_filter = InplaneGaussian(channels=1, **opt)
        else:
            assert 0, inplane_filter

        self.model = []
        if self.prep:
            self.model = [self.prep]

        self.model += [SeparableConv3d(input_dim, dim, kernel_size=k1,
                                       stride=1, padding=k1//2, dilation=1, bias=bias)]
        dim0 = dim
        for iblock in range(1, n_block):
            if dobn:
                self.model += [nn.BatchNorm3d(dim0, eps=1e-05,
                                              momentum=0.1, affine=True, track_running_stats=True)]
            self.model += [ConvBlock(dim0, dim, k1, 1, k1//2, norm=norm,
                                     activation='none', pad_type=pad_type, zdilation=zdilation)]
            self.model += [ConvBlock(dim, dim, k1, 1, k1//2, norm=norm,
                                     activation=activ, pad_type=pad_type, zdilation=zdilation)]
            dim0 = dim
            if iblock < (n_block - 1) and downsample is not None:
                self.model += [ConcatPool3d(sz=(downsample, downsample, 1))]
                dim0 *= 2
                dim = int(dim * updim)
        if final_pool is not None:
            self.model += [AdaptiveConcatPool3d(
                sz=(final_pool, final_pool, None))]
            dim0 = 2 * dim
        self.model += [ConvBlock(dim0, dim, k2, 1, k2//2, norm=norm, activation='none', pad_type=pad_type,
                                 preventRedundantParams=preventRedundantParams)]
        self.model += [ConvBlock(dim, input_dim, k2, 1, k2//2, norm=norm, activation=activ, pad_type=pad_type,
                                 preventRedundantParams=preventRedundantParams)]
        self.model = nn.Sequential(*self.model)

        # self.b = nn.Parameter(torch.Tensor([0.9]).float())
        # self.register_buffer('b_const', self.b)

        if not custom_init:
            self.model.apply(self._weights_init)
        elif custom_init == 2:
            self.model.apply(self._weights_init2)
        elif custom_init == 3:
            self.model.apply(self._weights_init3)
        else:
            self.model.apply(custom_init)

    def _weights_init(self, m):
        if isinstance(m, torch.nn.Conv3d) or isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                with torch.no_grad():
                    m.bias.zero_()

    def _weights_init2(self, m):
        if isinstance(m, torch.nn.Conv3d) or isinstance(m, torch.nn.Linear):
            nn.init.kaiming_normal_(
                m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        if isinstance(m, nn.Linear):
            size = m.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            variance = np.sqrt(2.0/(fan_in + fan_out))
            m.weight.data.normal_(1.0, variance)

    def _weights_init3(self, m):
        if isinstance(m, torch.nn.Conv3d) or isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                with torch.no_grad():
                    # m.bias.zero_()
                    nn.init.uniform_(m.bias, a=1e-6, b=1e-3)

    # def train(self, mode=True):
    #     """
    #     Override the default train() to freeze the BN parameters
    #     """
    #     super(nn.Module, self).train(mode)
    #     if self.freeze_bn:
    #         print("Freezing Mean/Var of BatchNorm3d.")
    #         if self.freeze_bn_affine:
    #             print("Freezing Weight/Bias of BatchNorm3d.")
    #     if self.freeze_bn:
    #         for m in self.backbone.modules():
    #             if isinstance(m, nn.BatchNorm3d):
    #                 m.eval()
    #                 if self.freeze_bn_affine:
    #                     m.weight.requires_grad = False
    #                     m.bias.requires_grad = False

    def forward(self, x, eps=1.0e-4):
        shp = x.shape
        if self.mode == 'global_avpool':
            p = nn.AdaptiveAvgPool3d((1, 1, shp[-1]))
        elif self.mode == 'upsample':
            p = nn.Upsample(
                size=shp[-3:], mode='trilinear', align_corners=False)
        elif self.mode == 'center':
            def p(im): return im[..., im.shape[-3]//2, im.shape[-2]//2, :]
        elif self.mode == 'center+':
            def p(im):
                cx = im.shape[-3] // 2
                cy = im.shape[-2] // 2
                return nn.AdaptiveAvgPool3d((1, 1, shp[-1]))(im[..., cx-2:cx+3, cy-2:cy+3, :])
        elif self.mode == 'none':
            def p(x): return x
        else:
            assert 0, self.mode

        field = self.model(x)
        # restrict to [eps, 2+eps], 0 becomes 1.0+eps
        field = 2.0 * torch.sigmoid(field) + eps

        if self.inplane_first and self.inplane_filter:
            field = torch.exp(self.inplane_filter(torch.log(field)))

        if self.field_filter is not None:
            field = torch.exp(torch.log(field) -
                              self.field_filter(torch.log(field)))

        if not self.inplane_first and self.inplane_filter:
            field = torch.exp(self.inplane_filter(torch.log(field)))

        field = p(field)
        try:
            if self.return_field and self.return_x:
                return field, x * field
            elif self.return_x:
                return x * field
            else:
                assert self.return_field, (self.return_field, self.return_x)
                return field

        except:
            print(field.shape, x.shape)
            raise
