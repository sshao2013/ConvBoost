import math
import numpy as np
import torch
import scipy.io
from torch.utils.data import Dataset, DataLoader

from Utils.sliding_window import sliding_window


def data_loader_self(args, x, y):
    curr_win_len = args.win_size
    curr_win_step = args.win_step
    x_win, y_win = opp_sliding_window(x, y, curr_win_len, curr_win_step)
    return x_win, y_win


def opp_sliding_window(data_x, data_y, ws, ss):
    data_x = sliding_window(data_x, (ws, data_x.shape[1]), (ss, 1))
    data_y = np.argmax(data_y, axis=1)
    data_y = np.asarray([[i[-1]] for i in sliding_window(data_y, ws, ss)])
    return data_x.astype(np.float32), data_y.reshape(len(data_y)).astype(np.uint8)


class make_data_loader(Dataset):
    def __init__(self, samples, labels):
        self.samples = samples
        self.labels = labels

    def __getitem__(self, index):
        sample, target = self.samples[index], self.labels[index]
        return sample, target

    def __len__(self):
        return len(self.samples)
