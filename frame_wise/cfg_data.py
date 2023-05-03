from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser(description='Ensemble')
    # dataset
    parser.add_argument('--dataset', type=int, default=1)  # 1 is opp79, 2 is pamap2, 3 is gotov

    # model parameter
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=int, default=0.5)
    parser.add_argument('--cnn_channel', type=int, default=256)
    parser.add_argument('--model_name', type=str, default='CNN-3L', help='CNN-3L, DCL-3L, RCNN')

    # ensemble settings
    parser.add_argument('--best_models', type=int, default=20)
    parser.add_argument('--noob_models', type=int, default=0)

    # epoch-wise settings
    parser.add_argument('--epoch_wise', type=bool, default=False)
    parser.add_argument('--dynamic_data', type=bool, default=False)
    parser.add_argument('--drop_channel', type=bool, default=False)
    parser.add_argument('--mixup', type=bool, default=False)

    # training settings
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--win_size', type=int, default=32)
    parser.add_argument('--win_step', type=int, default=16)
    parser.add_argument('--test_win', type=int, default=32)
    # parser.add_argument('--win_size', type=int, default=168)  # for pamap2 dataset
    # parser.add_argument('--win_step', type=int, default=32)  # for pamap2 dataset
    # parser.add_argument('--test_win', type=int, default=168)  # for pamap2 dataset
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--need_logs', type=bool, default=True)
    parser.add_argument('--train_on_gpu', type=bool, default=False)

    args = parser.parse_args()

    if args.epoch_wise:
        args.mixup = True
        args.dynamic_data = True
        args.drop_channel = True

    return args
