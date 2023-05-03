import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from Utils.utils import valid_process, frame_compute_acc, get_unlabel_data, mixup_data, MixUpLoss


class Trainer(object):
    def __init__(self, data, model, optimizer, criterion, args, root_result_path, DEVICE):
        super(Trainer, self).__init__()
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.data = data
        self.model_name = args.model_name
        self.results = np.empty([0, 4], dtype=np.float32)
        self.batch_size = args.batch_size
        self.valid_loader = data.valid_data()
        self.args = args
        self.noob_models = args.noob_models
        self.root_result_path = root_result_path
        self.DEVICE = DEVICE

    def train(self, args):
        self.frame_wise_train(args)

    def validation(self):

        with torch.no_grad():
            valid_f1, valid_f1_weighted = valid_process(self.model, self.valid_loader, self.DEVICE)
        return valid_f1, valid_f1_weighted

    def frame_wise_train(self, args):
        self.model_name = self.model_name + '_Framewise'

        if args.train_on_gpu:
            self.model = self.model.to(self.DEVICE)
            self.criterion = self.criterion.to(self.DEVICE)

        for epoch in range(args.epochs):
            train_loader= self.data.train_data()
            y_pred_total = []
            y_total = []
            train_loss = 0
            self.model.train()
            epoch_start_time = time.time()

            for index, (sample, target) in enumerate(train_loader):
                inputs, targets = sample.to(self.DEVICE).float(), target.to(self.DEVICE).long()

                if args.drop_channel:
                    # drop channel
                    for j in range(self.batch_size):
                        num_change = np.random.randint(0, int(inputs.shape[2] * 0.2))
                        dim_location_change = np.random.randint(0, inputs.shape[2] - num_change)
                        inputs[j, :, dim_location_change:dim_location_change + num_change] = 0

                # if args.train_on_gpu:
                #     inputs = torch.Tensor(inputs).cuda()
                #     if args.mixup:
                #         inputs_unlabel = torch.Tensor(inputs_unlabel).cuda()
                #     targets = torch.LongTensor(targets).cuda()

                self.optimizer.zero_grad()

                if args.mixup:
                    inputs, y_a_y_b_lam = mixup_data(inputs, targets, self.DEVICE, 0.8)

                y_pred = self.model(inputs)
                y_total.append(targets.detach().cpu())
                y_pred_total.append(y_pred.detach().cpu())

                if args.mixup:
                    criterion = MixUpLoss(self.criterion)
                    loss = criterion(y_pred, y_a_y_b_lam)
                else:
                    loss = self.criterion(y_pred, targets)

                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            y_pred_total = torch.cat(y_pred_total, 0)
            y_total = torch.cat(y_total, 0)
            train_acc_total = frame_compute_acc(y_pred_total, y_total)

            # validation part
            valid_f1, valid_f1_weighted = self.validation()

            print("Epoch {}/{}....\n".format(epoch + 1, args.epochs),
                  "Average Train Loss: {:.4f}".format(train_loss),
                  "Average Train Acc: {:.4f}\n".format(train_acc_total),
                  "Valid Mean F1: {:.4f}".format(valid_f1), "Valid Weighted F1: {:.4f}\n".format(valid_f1_weighted),
                  )

            epoch_end_time = int(time.time() - epoch_start_time)
            print('epoch {} end time: {:02d}:{:02d}:{:02d}'.format(epoch + 1, epoch_end_time // 3600,
                                                                   (epoch_end_time % 3600 // 60),
                                                                   epoch_end_time % 60))

            # save model & result
            if epoch >= self.noob_models:
                epoch_results = np.array(
                    [epoch, train_acc_total, valid_f1, valid_f1_weighted])
                self.results = np.float32(np.vstack((self.results, epoch_results)))
                if not os.path.exists(self.root_result_path + '/results/'):
                    os.makedirs(self.root_result_path + '/results/')
                np.save(
                    self.root_result_path + '/results/' + self.model_name + '_' + str(self.data.DB) + '.npy',
                    self.results)

                if not os.path.exists(self.root_result_path + 'saved_models/'):
                    os.makedirs(self.root_result_path + 'saved_models/')
                path = self.root_result_path + 'saved_models/{0}_{1}_{2}'.format(self.model_name,
                                                                                 str(self.data.DB), epoch)
                save_model(epoch, self.model.state_dict(), self.optimizer.state_dict(), path)
                print("Model saved to %s" % path)


def save_model(epoch, model_dict, opt_dict, path):
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model_dict,
        'optimizer_state_dict': opt_dict,
    }, path)
    print("Find new best model on Epoch: " + str(epoch + 1) + ". Model saved to %s" % path)