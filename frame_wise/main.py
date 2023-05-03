import os.path as osp
import random
import sys
import numpy as np
import torch
from torch import nn

from Utils.logger import Logger
from Utils.utils import get_root_result_path, init_weights_orthogonal
from cfg_data import get_args
from data.data_loader import HARDataLoader
from ensemble import EnsembleTool
from models.model import select_model
from train import Trainer


def main(args, root_result_path, DEVICE):
    # get data
    data = HARDataLoader(args)

    model = select_model(data, args, DEVICE)
    model.apply(init_weights_orthogonal)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    trainer = Trainer(data, model, optimizer, criterion, args, root_result_path, DEVICE)
    trainer.train(args)

    # ensemble models
    ensemble_models = EnsembleTool(args, data.DB, criterion, root_result_path, DEVICE)
    ensemble_models.result_evaluate(data, model)


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed_all(seed)  # gpu
    torch.backends.cudnn.deterministic = True  # consistent results on the cpu and gpu


if __name__ == '__main__':

    args = get_args()
    root_result_path = get_root_result_path(args)

    # print in console and copy in log file as well
    if args.need_logs:
        sys.stdout = Logger(osp.join(root_result_path + 'logs/' + 'log.txt'))

    if torch.cuda.is_available():
        args.train_on_gpu = True
        print('Training on GPU')

    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    main(args, root_result_path, DEVICE)
