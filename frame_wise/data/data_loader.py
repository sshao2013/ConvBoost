import numpy as np
import torch
from torch.utils.data import DataLoader

from Utils.makeDataLoader import data_loader_self, make_data_loader
from Utils.sliding_window import sliding_window
from data.data_preprocessing import loadingDB


def data_sliding_window(data_x, data_y, ws, ss):
    data_x = sliding_window(data_x, (ws, data_x.shape[1]), (ss, 1))
    data_y = np.asarray([[i[-1]] for i in sliding_window(data_y, ws, ss)])
    return data_x.astype(np.float32), data_y.reshape(len(data_y)).astype(np.uint8)




class HARDataLoader:
    def __init__(self, args):
        super(HARDataLoader, self).__init__()  # 1 is opp79, 2 is pamap2, 3 is skoda
        self.args = args
        if args.dataset == 1:
            self.train_x, self.valid_x, self.test_x, self.train_y, self.valid_y, self.test_y = loadingDB(args,
                '../../Dataset/OPP/',
                79)
            self.n_classes = 18
            self.DB = 79
        if args.dataset == 2:
            self.train_x, self.valid_x, self.test_x, self.train_y, self.valid_y, self.test_y = loadingDB(args,
                '../../Dataset/PAMAP2/',
                52)
            self.n_classes = 12
            self.DB = 52
        if args.dataset == 3:
            self.train_x, self.valid_x, self.test_x, self.train_y, self.valid_y, self.test_y = loadingDB(args,
                '../../Dataset/gotov/',
                9)
            self.n_classes = 16
            self.DB = 9


    def train_data(self):
        train_x = self.train_x
        train_y = self.train_y
        train_y = np.array(train_y)
        if self.args.dynamic_data:
            # epoch wise dynamic_data
            data_start = int(np.random.randint(low=0, high=self.args.win_step, size=1))
            train_x = train_x[data_start:, :]
            train_y = train_y[data_start:, :]
        x_train, y_train = data_loader_self(self.args, train_x, train_y)

        data_start = np.random.randint(low=0, high=x_train.shape[0], size=x_train.shape[0])
        x_train = x_train[data_start]
        y_train = y_train[data_start]

        data_set = make_data_loader(x_train, y_train)
        train_loader = DataLoader(data_set, batch_size=self.args.batch_size, shuffle=True, drop_last=True)
        return train_loader

    def test_data(self):
        test_x = self.test_x
        test_y = np.array(self.test_y)
        test_x, test_y = data_loader_self(self.args, test_x, test_y)
        data_set = make_data_loader(test_x, test_y)
        test_loader = DataLoader(data_set, batch_size=self.args.batch_size, shuffle=False, drop_last=True)
        return test_loader

    def valid_data(self):
        valid_x = self.valid_x
        valid_y = np.array(self.valid_y)
        valid_x, valid_y = data_loader_self(self.args, valid_x, valid_y)
        data_set = make_data_loader(valid_x, valid_y)
        valid_loader = DataLoader(data_set, batch_size=self.args.batch_size, shuffle=False, drop_last=True)
        return valid_loader
