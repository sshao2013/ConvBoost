import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score


def bestM_prob(prob_M1, truth_M):
    ensemble_results = np.array(
        [prob_M1, truth_M])
    return ensemble_results


def show_result(i, truth_result, pred_result, truth, fused_pred):
    current_f1 = f1_score(truth_result, pred_result, average='macro')
    current_f1_weighted = f1_score(truth_result, pred_result, average='weighted')
    f1_fused = f1_score(truth, fused_pred, average='macro')
    f1_weighted = f1_score(truth, fused_pred, average='weighted')
    print(
        'curr_mean_f1 {:.3f} curr_weighted_f1 {:.3f} and sz {} fused_f1 {:.3f} fused_f1_weighted {:.3f}'.format(
            current_f1, current_f1_weighted, i + 1,
            f1_fused,
            f1_weighted))


class EnsembleTool(object):
    def __init__(self, args, DB, criterion, root_result_path, DEVICE):
        super(EnsembleTool, self).__init__()
        self.best_models = args.best_models
        self.DB = DB
        self.criterion = criterion
        self.train_on_gpu = args.train_on_gpu
        self.batch_size = args.batch_size
        self.args = args
        self.root_result_path = root_result_path
        self.model_name = args.model_name + '_Framewise_'
        self.DEVICE = DEVICE

    def result_evaluate(self, data, model):
        test_loader = data.test_data()
        exp_id1 = self.model_name + str(self.DB)
        exp1 = exp_setting(exp_id1, self.best_models, self.root_result_path)
        prob_M1, truth_M = self.score_fusion(exp1, data.n_classes, test_loader, model, self.best_models)
        result = bestM_prob(prob_M1, truth_M)
        if not os.path.exists(self.root_result_path + 'trials/'):
            os.makedirs(self.root_result_path + 'trials/')
        np.save(self.root_result_path + 'trials/' + exp_id1 + '.npy', result)

    def score_fusion(self, exp, num_classes, test_loader, model, bestM=20):
        batch_size = self.batch_size
        n_classes = num_classes
        label_len = test_loader.dataset.samples.shape[0]
        prob_M = np.zeros((bestM, label_len // batch_size * batch_size, n_classes))
        truth_M = np.zeros((bestM, label_len // batch_size * batch_size, n_classes))

        # init for diversity measurement
        model_list = []

        # the order of models: current order is by weighted F1 score.
        models = exp[0]
        # Load one model calculate the cumulative model score
        for i in range(bestM):
            # After ranking and choosing, use index to get the model
            idx = models[i]
            model_path = self.root_result_path + 'saved_models/' + exp[1] + '_' + str(idx)
            checkpoint = torch.load(model_path, map_location=self.DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            model_list.append(model)

            y_pred_total = []
            y_total = []
            for index, (sample, target) in enumerate(test_loader):
                inputs, targets = sample.to(self.DEVICE).float(), target.to(self.DEVICE).long()
                y_pred = model(inputs)
                y_total.append(targets.detach().cpu())
                y_pred_total.append(y_pred.detach().cpu())
            # y_pred_total, y_total = data_loader_run(y_pred_total, y_total, model, data_loader, num_classes)
            y_pred_total = torch.cat(y_pred_total, 0)
            y_total = torch.cat(y_total, 0)
            y_pred_result = y_pred_total.cpu().max(dim=-1)[1]
            y_total_result = y_total.cpu()

            prob_M[i, :, :] = F.softmax(y_pred_total, dim=1).detach().cpu().numpy()
            truth_M[i, :, :] = F.one_hot(y_total_result.to(torch.int64), num_classes)

            # top n mean f1
            # Calculate the mean of cumulative models score
            curr_prob_avg = np.mean(prob_M[:, :, :], axis=0)
            fused_pred = np.argmax(curr_prob_avg, axis=1)

            show_result(i, y_total_result, y_pred_result, y_total_result, fused_pred)
        return prob_M, truth_M


def model_ranking(results, shown_TopN=1, valid_col=2):
    idx_set = np.argsort(results[:, valid_col])[
              ::-1]  # https://docs.scipy.org/doc/numpy/reference/generated/numpy.argsort.html
    idx_set = idx_set[:shown_TopN]

    idx_result = np.zeros(shown_TopN, dtype=int)
    for i in range(shown_TopN):
        idx_result[i] = results[idx_set[i]][0]

    return idx_result


def exp_setting(exp_id, bestM, root_result_path):
    results = np.load(root_result_path + 'results/' + exp_id + '.npy')
    idx_set = model_ranking(results, bestM)
    exp = [idx_set[:bestM], exp_id]

    return exp
