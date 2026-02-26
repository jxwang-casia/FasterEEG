import numpy as np
from model.flops_count import get_model_complexity_info
from model.init_transforms import Transforms
from model.policy_layers import *
from braindecode.models import *
from braindecode.models.util import to_dense_prediction_model, get_output_shape
from TCNetFusion.tcn_feature import TCNet_Fusion

class Adapt_Shallow(nn.Module):
    def __init__(self, adapt_chans=[1, 11, 22], adapt_models = [5, 10, 20], tau=5., adaptive_transform=[True, True, True], policy_type='tconv',
                 args=None, tau_decay=-0.045,
                 pre_trains=None, tau_type='cos', init_num=2, model_type='shallow', **kwargs):
        super(Adapt_Shallow, self).__init__()

        self.transforms = nn.ParameterList(
            [nn.Parameter(Transforms['M{}to{}'.format(adapt_chans[-1], i)], requires_grad=adaptive_transform[ind]) for
             ind, i in enumerate(adapt_chans) for adapt_model in adapt_models])

        self.tau_decay = tau_decay
        self.tau_type = tau_type

        if model_type == 'shallow':
            self.nets = nn.ModuleList(
            [ShallowFEANet(in_chans=adapt_chan, n_filters_time=adapt_model, n_filters_spat=adapt_model, n_policy=len(adapt_chans)*len(adapt_models), out_policy=True, **kwargs)
             for adapt_chan in adapt_chans for adapt_model in adapt_models])  # j1m1 j1m2 j2m1 ...
            self.n_preds_per_input = self.nets[0].n_preds_per_input
        elif model_type == 'eeg':
            D = 2
            self.nets = nn.ModuleList(
            [EEGNet(in_chans=adapt_chan, F1=adapt_model, F2=adapt_model*D, n_policy=len(adapt_chans)*len(adapt_models), out_policy=True, **kwargs)
             for adapt_chan in adapt_chans for adapt_model in adapt_models])  # j1m1 j1m2 j2m1 ...
            self.n_preds_per_input = 1
        elif model_type == 'tcnfusion':
            self.nets = nn.ModuleList(
            [TCNet_Fusion(in_chans=adapt_chan, F1=adapt_model, n_policy=len(adapt_chans)*len(adapt_models), out_policy=True, **kwargs)
             for adapt_chan in adapt_chans for adapt_model in adapt_models])
            self.n_preds_per_input = 1
        else:
            raise RuntimeError("model type wrong")
        # self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        # self.conv_classifier=nn.Conv2d(
        #         kwargs['n_filters_spat'],
        #         kwargs['n_classes'],
        #         (self.final_conv_length, 1),
        #         bias=True,
        #     )

        self.tau = tau
        self.epoch = 0
        self.args = args
        self.adapt_chans = adapt_chans
        self.adapt_models = adapt_models
        self.num_action = len(adapt_chans)*len(adapt_models)
        self.policy_type = policy_type  # random fuse other
        self.seg = kwargs['input_window_samples']

        if pre_trains is not None:
            self.load(pre_trains, init_num)

        # # b c 1 t -> b 3 1 t
        # if policy_type == 'tconv':
        #     self.policy_net = Tconv(dim, self.num_action, k=policy_kernel, d=policy_dilate, init_type=init_type)
        # elif policy_type == 'tconv2':
        #     self.policy_net = Tconv2(dim, dim // 2, self.num_action, k=policy_kernel, d=policy_dilate,
        #                              init_type=init_type)
        # elif policy_type == 'transformer':
        #     self.policy_net = Transformer(dim, dim // 2, self.num_action)
        # elif policy_type == 'lstm':
        #     self.policy_net = Lstm(dim, dim // 2, self.num_action)
        # elif policy_type == 'tnet':
        #     self.policy_net = Temconv(dim, self.num_action, k=policy_kernel, d=policy_dilate, seg=seg, dim=dim,
        #                               init_type=init_type)
        # elif policy_type == 'random' or policy_type == 'fuse':
        #     self.policy_net = EmptyNet()
        # else:
        #     raise RuntimeError('No such policy net')

        self.get_gflops_table()

    def get_gflops_table(self):
        kwards = {
            'print_per_layer_stat': False,
            'as_strings': False
        }

        self.gflops_table = dict()
        self.gflops_table['net'] = [
            get_model_complexity_info(m, (j, self.seg, 1), **kwards)[0] / 1e9 for
            m, j in zip(self.nets, [j for j in self.adapt_chans for _ in self.adapt_models])]

        self.gflops_vector = torch.FloatTensor(self.gflops_table['net'])

        print("gflops_table: ")
        for i in range(len(self.adapt_chans)):
            print('net', self.adapt_chans[i], self.gflops_table['net'][i*len(self.adapt_models):(i + 1)*len(self.adapt_models)])

    def get_policy_usage_str(self, action_list, label_list):
        """

        :param action_list: [num_act, N]
        :param label_list: [N]
        :return:
        """
        action_list = np.concatenate(action_list, axis=1)
        label_list = np.concatenate(label_list)
        print(action_list)
        num_class = self.args.class_num
        num_action = self.num_action
        action_statistic = np.zeros([num_class, num_action])
        for i, l in enumerate(label_list):
            action_statistic[l] += action_list[:, i]
        action_statistic = action_statistic / (action_statistic.sum(0, keepdims=True) + 1e-6)
        top5 = [np.concatenate([
            np.argsort(action_statistic[:, i])[::-1][:4], np.sort(action_statistic[:, i])[::-1][:4]
        ]) for i in range(num_action)]  # num_act, 4  使用该行为最多的类别，与相应的使用率
        action_statistic_str = ''.join(
            '\nAction{}: {}, {}'.format(i, top5[i][:4].astype(np.int), top5[i][4:].astype(np.float16)) for i in
            range(num_action))
        print(action_statistic_str)

        actions_mean = action_list.mean(-1)  # num_act

        gflops = actions_mean * self.gflops_table['net']

        printed_str = '\nGflops: ' + str(gflops) \
                      + '\nActions: ' + str(actions_mean) \
                      + '\nGflops all: ' + str(sum(gflops)) \
                      + action_statistic_str
        return printed_str

    def train(self, mode=True):
        super(Adapt_Shallow, self).train(mode)
        if mode:
            self.epoch += 1
            for freeze_key in self.args.freeze_keys:
                if freeze_key[0] == 'policy_net' and freeze_key[1] >= self.epoch:
                    return
            if self.tau_type == 'linear':
                self.tau = self.tau * np.exp(self.tau_decay)
            elif self.tau_type == 'cos':
                self.tau = 0.01 + 0.5 * (self.tau - 0.01) * (1 + np.cos(np.pi * self.epoch / self.args.max_epoch))
            else:
                raise RuntimeError('no such tau type')
            print('current tau: ', self.tau)

    def forward(self, input):
        while (len(input.shape) < 4):
            input = input.unsqueeze(-1)
        n, v, t, c = input.shape
        # get input list with different size s b v t c
        input_list = downsample_input(input, self.transforms)
        # get input features s b c 1 t
        input_feats = get_input_feats(input_list, self.nets)
        # batch_size num_action 1 t
        if self.policy_type == 'random':
            prob, action = get_random_action(n, self.num_action)
        elif self.policy_type == 'fuse':
            prob, action = get_fixed_action(n, self.num_action)
        else:
            prob, action = get_litefeat_and_action(input_feats[0][1], self.tau)
        # print(prob[0, :, 0])
        # print(action[0, :, 0])
        # na
        action = action.permute(1, 0).unsqueeze(2).to(input.device)
        # a n c
        output = torch.stack([input_feat[0] for input_feat in input_feats])
        output = (action * output).sum(0)

        return output, action.view(self.num_action, n)

    def load_part(self, model, pretrained_dict):
        # del pretrained_dict['policy_net.weight']
        model_dict = model.state_dict()
        pretrained_dict = {k[4:]:v for k,v in pretrained_dict.items() if k[:4]=='net.'}
        # pretrained_dict = {k[len(key) + 1:]: v for k, v in pretrained_dict.items() if key in k}
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    def load(self, checkpoints_paths, init_num=5):
        assert len(checkpoints_paths) == len(self.nets)
        assert len(checkpoints_paths) != 0
        for i, path in enumerate(checkpoints_paths):
            state_dict = torch.load(path, map_location='cpu')['model']
            self.load_part(self.nets[i], state_dict)
            # self.load_part(self.joint_embeds[i], state_dict, 'joint_embed')
            # self.load_part(self.dif_embeds[i], state_dict, 'dif_embed')
            with torch.no_grad():
                self.transforms[i].data = state_dict['transform']
            # if i == init_num:
            #     self.load_part(self.fc, state_dict, 'fc')
            #     self.load_part(self.tem_net, state_dict, 'tem_net')
        print(checkpoints_paths)


if __name__ == '__main__':
    import os
    # from thop import profile
    import timeit



    num_js = [6, 12, 22]
    num_j = num_js[-1]
    num_t = 1125
    num_models = [5, 10, 20]
    os.environ['CUDA_VISIBLE_DEVICE'] = '0'

    pretrained = [
                 '../../work_dir/bci/shallow_id4_chanl6_timefilter5_transform-best.state',
                 '../../work_dir/bci/shallow_id4_chanl6_timefilter10_transform-best.state',
                 '../../work_dir/bci/shallow_id4_chanl6_timefilter20_transform-best.state',
                 '../../work_dir/bci/shallow_id4_chanl12_timefilter5_transform-best.state',
                 '../../work_dir/bci/shallow_id4_chanl12_timefilter10_transform-best.state',
                 '../../work_dir/bci/shallow_id4_chanl12_timefilter20_transform-best.state',
                 '../../work_dir/bci/shallow_id4_chanl22_timefilter5_transform-best.state',
                 '../../work_dir/bci/shallow_id4_chanl22_timefilter10_transform-best.state',
                 '../../work_dir/bci/shallow_id4_chanl22_timefilter20_transform-best.state',
    ]
    model = Adapt_Shallow(policy_type='tconv', tau=1e-5, pre_trains=pretrained, init_num=2,
                          adaptive_transform=[True, True, True], adapt_chans=num_js, adapt_models=num_models, n_classes=4,
                          input_window_samples=num_t, model_type='tcnfusion')
    # pretrained_dict = torch.load('../../work_dir/bci/adashallow_withtransformedsinglepre_transformfix5_policyfix10_alpha2_id6-best.state', map_location='cpu')['model']
    # model_dict = model.state_dict()
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # model_dict.update(pretrained_dict)
    # model.load_state_dict(model_dict)
    # for i, t in enumerate(model.transforms):
    #     model.transforms[i].data = pre['transforms.{}'.format(i)]
    model.eval()

    # verify that the "model.test" is the same with the "model.forward"
    dummy_data = torch.randn([1, num_j, num_t, 1])
    start = timeit.default_timer()
    o2, a2 = model(dummy_data)
    stop = timeit.default_timer()
    print('Time: ', stop - start)
    # o2.mean().backward()
    # o1 = model.test(dummy_data, labels)
    # o2, a2 = model(dummy_data)
    # print((o1 == o2).all())
    # print('finish')

    flops, params = get_model_complexity_info(model, (num_j, num_t, 1), as_strings=True)

    print(flops)  # 0.16 gmac
    print(params)  # 0.69 m

    # from dataset.vis import plot_skeleton, test_one, test_multi, plot_points
    # from dataset.ntu_skeleton import NTU_SKE, edge, edge1, edge9
    #
    # vid = 'S004C001P003R001A058'  # ntu60
    # data_path = "../../data/ntu60/CV/test_data.npy"
    # label_path = "../../data/ntu60/CV/test_label.pkl"
    #
    # kwards = {
    #     "window_size": 20,
    #     "final_size": 20,
    #     "random_choose": False,
    #     "center_choose": False,
    #     "rot_norm": True
    # }
    #
    # dataset = NTU_SKE(data_path, label_path, **kwards)
    # labels = open('../prepare/ntu/statistics/class_name.txt', 'r').readlines()
    #
    # save_paths = ['../../vis_results/adaskeleton/{}/frame{}'.format(vid, i) for i in range(num_t)]
    # test_one(dataset, plot_skeleton, lambda x: model.test(torch.from_numpy(x).unsqueeze(0))[1], vid=vid,
    #          edges=[edge1, edge9, edge], is_3d=True, pause=0.1, labels=labels, view=1)
    # test_multi(dataset, plot_skeleton, lambda x: model.test(x)[1], skip=1000,
    #          edges=[edge1, edge9, edge], is_3d=True, pause=1, labels=labels, view=1)
