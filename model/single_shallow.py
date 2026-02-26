from torch import nn
import torch
from model.init_transforms import Transforms
from braindecode.models import *
from braindecode.models.util import to_dense_prediction_model, get_output_shape
from TCNetFusion.tcn_feature import TCNet_Fusion


class Single_Shallow(nn.Module):
    def __init__(self, adaptive_transform=False, model_type='shallow', in_chans=22, adapt_chans=22, n_filters_time=40, n_filters_spat=40, **kwargs):
        super(Single_Shallow, self).__init__()
        if model_type == 'shallow':
            self.net = ShallowFBCSPNet(in_chans=adapt_chans, n_filters_time=n_filters_time, n_filters_spat=n_filters_spat, **kwargs)
            to_dense_prediction_model(self.net)
        elif model_type == 'eeg':
            D = kwargs['D']
            F1 = kwargs['F1']
            self.net = EEGNetv4(in_chans=adapt_chans, F2=F1*D, **kwargs)
        elif model_type == 'tcnfusion':
            self.net = TCNet_Fusion(in_chans=adapt_chans, **kwargs)
        else:
            raise RuntimeError('model type error')

        self.transform = nn.Parameter(Transforms['M{}to{}'.format(in_chans, adapt_chans)], requires_grad=adaptive_transform)  # vv

    def forward(self, input):  # nvtc
        while (len(input.shape) < 4):
            input = input.unsqueeze(-1)
        input = torch.matmul(input.transpose(1, 3), self.transform).transpose(1, 3).contiguous()  # nctv

        output = self.net(input)

        return output


if __name__ == '__main__':
    import os
    from model.flops_count import get_model_complexity_info
    from thop import profile

    in_chans = 22
    adapt_chans = 22
    n_filters_time = 5
    n_filters_spat = 5
    input_window_samples = 1125
    F1 = 16
    D = 2
    os.environ['CUDA_VISIBLE_DEVICE'] = '0'
    # model = Single_Shallow(adaptive_transform=False, in_chans=in_chans, adapt_chans=adapt_chans, n_filters_time=n_filters_time, n_filters_spat=n_filters_spat, n_classes=4, input_window_samples=input_window_samples)
    model = Single_Shallow(adaptive_transform=False, in_chans=in_chans, adapt_chans=adapt_chans, model_type='tcnfusion', F1=F1, D=D, n_classes=4, input_window_samples=input_window_samples)
    # torch.save({'model': model.state_dict()}, '../../pretrain_models/single_shallow_chan{}.state'.format(adapt_chans),)
    dummy_data = torch.randn([1, in_chans, input_window_samples, 1])
    # a = model(dummy_data)
    # a.mean().backward()


    flops_count, params_count = get_model_complexity_info(model, (adapt_chans, input_window_samples, 1))
    print(f'flops: {flops_count}')
    print(f'parames: {params_count}')

    # print(gflops)  # 0.057 0.028 0.003
    # print(params)  # 0.046

    # flops, params = get_model_complexity_info(model, (3, num_t, 25, 1), as_strings=True)  # not support

    # print(flops)  # 0.16 gmac
    # print(params)  # 0.69 m
