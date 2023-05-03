from models.cnn2d import CNN2D_3L
from models.convlstm import ConvLSTM
from models.rcnn import RCNN


def select_model(data, args, DEVICE):
    if args.model_name == 'CNN-3L':
        return CNN2D_3L(data.DB, data.n_classes, args.win_size, args.cnn_channel)
    elif args.model_name == 'DCL-3L':
        return ConvLSTM(data.DB, data.n_classes, args.cnn_channel)
    elif args.model_name == 'RCNN':
        return RCNN(data.DB, data.n_classes, DEVICE)
    else:
        raise 'Cannot find the correct model.'
