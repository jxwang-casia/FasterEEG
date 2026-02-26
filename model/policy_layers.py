import torch
import torch.nn as nn
import torch.nn.functional as F


def get_random_action(batch_size, num_action):
    prob = torch.randn([batch_size, num_action])

    action = F.gumbel_softmax(prob, 1e-6, hard=True, dim=1)  # batch_size num_action 1 t

    return prob, action


def get_fixed_action(batch_size, num_action):
    prob = None

    action = torch.ones([batch_size, num_action])  # batch_size num_action 1 t

    return prob, action


def get_litefeat_and_action(input_feat, tau):
    prob = torch.log(F.softmax(input_feat, dim=1).clamp(min=1e-8))  # batch_size num_action

    action = F.gumbel_softmax(prob, tau, hard=True, dim=1)  # batch_size num_action

    return prob, action


def get_input_feats(input_list, models):
    output_list = []
    for input, model in zip(input_list, models):
        cls, pol = model(input)
        tmp = [cls, pol]
        output_list.append(tmp)
    return output_list


def downsample_input(input, transforms):
    # n c v t
    out_list = []
    for i, t in enumerate(transforms):
        out_list.append(torch.matmul(input.transpose(1, 3), t).transpose(1, 3).contiguous())
    return out_list