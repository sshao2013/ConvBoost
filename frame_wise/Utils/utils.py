import datetime
import os
from pathlib import Path
import numpy as np
import scipy
import torch
from sklearn.metrics import f1_score
from torch import nn


def valid_process(model, data_loader, DEVICE):
    y_pred_total = []
    y_total = []
    # clean hidden state for each new sequence
    for index, (sample, target) in enumerate(data_loader):
        inputs, targets = sample.to(DEVICE).float(), target.to(DEVICE).long()
        y_pred = model(inputs)
        y_total.append(targets)
        y_pred_total.append(y_pred)

    # concat the list of y to compute acc
    y_pred_total = torch.cat(y_pred_total, 0)
    y_total = torch.cat(y_total, 0)

    y_pred_all_argmax = np.argmax(y_pred_total.detach().cpu().numpy(), axis=-1)
    test_maf1 = f1_score(y_total.cpu(),y_pred_all_argmax, average='macro')

    # cal f1
    test_maf1 = f1_score(y_total.cpu(), y_pred_total.cpu().max(dim=-1)[1], average='macro')
    test_wf1 = f1_score(y_total.cpu(), y_pred_total.cpu().max(dim=-1)[1], average='weighted')
    return test_maf1, test_wf1


def frame_compute_acc(y_pred, y):
    max_index = y_pred.max(dim=-1)[1]
    accuracy = ((max_index == y).sum()).float() / y.shape[0]
    return accuracy.type(torch.FloatTensor).data.numpy()


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


def get_unlabel_data(unlabel_inputs, batchsize):
    indices = np.arange(len(unlabel_inputs))
    np.random.shuffle(indices)
    idx = int(np.random.randint(low=0, high=len(unlabel_inputs) - batchsize, size=1))
    excerpt = indices[idx:idx + batchsize]
    return unlabel_inputs[excerpt]


def init_weights_orthogonal(m):
    """
    Orthogonal initialization of layer parameters
    :param m:
    :return:
    """
    if type(m) == nn.LSTM:
        for name, param in m.named_parameters():
            if "weight_ih" in name:
                nn.init.orthogonal_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                param.data.fill_(0)

    elif type(m) == nn.Conv2d or type(m) == nn.Linear:
        nn.init.orthogonal_(m.weight)
        m.bias.data.fill_(0)

class MixUpLoss(nn.Module):
    """
    Mixup implementation heavily borrowed from https://github.com/fastai/fastai/blob/master/fastai/callbacks/mixup.py#L42
    Adapt the loss function `crit` to go with mixup.
    """

    def __init__(self, crit, reduction='mean'):
        super().__init__()
        if hasattr(crit, 'reduction'):
            self.crit = crit
            self.old_red = crit.reduction
            setattr(self.crit, 'reduction', 'none')
        self.reduction = reduction

    def forward(self, output, target):
        if len(target.size()) == 2:
            loss1, loss2 = self.crit(output, target[:, 0].long()), self.crit(output, target[:, 1].long())
            d = loss1 * target[:, 2] + loss2 * (1 - target[:, 2])
        else:
            d = self.crit(output, target)
        if self.reduction == 'mean':
            return d.mean()
        elif self.reduction == 'sum':
            return d.sum()
        return d

    def get_old(self):
        if hasattr(self, 'old_crit'):
            return self.old_crit
        elif hasattr(self, 'old_red'):
            setattr(self.crit, 'reduction', self.old_red)
            return self.crit


def mixup_data(x, y, x_unlabeled, y_unlabeled, alpha=0.8):
    """
    Returns mixed inputs, pairs of targets, and lambda
    """

    batch_size = x.shape[0]
    lam = np.random.beta(alpha, alpha, batch_size)
    # t = max(t, 1-t)
    lam = np.concatenate([lam[:, None], 1 - lam[:, None]], 1).max(1)
    # tensor and cuda version of lam
    lam = x.new(lam)

    shuffle = torch.randperm(batch_size).cuda()

    # x1, y1 = x_unlabeled, y_unlabeled
    x1, y1 = x[shuffle], y[shuffle]
    # out_shape = [bs, 1, 1]
    out_shape = [lam.size(0)] + [1 for _ in range(len(x1.shape) - 1)]

    # [bs, temporal, sensor]
    mixed_x = (x * lam.view(out_shape) + x1 * (1 - lam).view(out_shape))
    # [bs, 3]
    y_a_y_b_lam = torch.cat([y[:, None].float(), y1[:, None].float(), lam[:, None].float()], 1)

    return mixed_x, y_a_y_b_lam


def get_root_result_path(args):
    # create result folder
    div = ''
    div_count = 0
    if args.dynamic_data:
        div += 'dynamic-data_'
        div_count += 1
    if args.mixup:
        div += 'mixup_'
        div_count += 1
    if args.drop_channel:
        div += 'drop-channel_'
        div_count += 1
    if div_count == 0:
        div = 'noDiv_'
    elif div_count == 3:
        div = 'All-Div_'

    d_name = ''
    if args.dataset == 1:
        d_name = 'Opp'
    elif args.dataset == 2:
        d_name = 'Pamap2'
    elif args.dataset == 3:
        d_name = 'Gotov'
    root_result_path = './root_result_' + args.model_name + '_' + 'Channel-' + str(
        args.cnn_channel) + '_' + div + d_name + '_' + datetime.datetime.now().strftime(
        '%Y-%m-%d-%H-%M-%S') + '/'
    # root_result_path = './root_result_test/'
    if not os.path.exists(root_result_path):
        os.makedirs(root_result_path)
    return root_result_path

def get_project_root() -> Path:
    return Path(__file__).parent.parent