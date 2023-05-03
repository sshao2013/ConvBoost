import numpy as np
import torch
from sklearn.metrics import pairwise

def Q_statistic(M, y_true):
    Qs = []
    for i in range(M.shape[0]):
        for j in range(M.shape[0]):
            N_1_1 = np.sum(np.logical_and(y_true == np.argmax(M[i, :], axis=1), y_true == np.argmax(M[j, :], axis=1)))  # number of both correct
            N_0_0 = np.sum(np.logical_and(y_true != np.argmax(M[i, :], axis=1), y_true != np.argmax(M[j, :], axis=1)))  # number of both incorrect
            N_0_1 = np.sum(np.logical_and(y_true != np.argmax(M[i, :], axis=1), y_true == np.argmax(M[j, :], axis=1)))  # number of j correct but not i
            N_1_0 = np.sum(np.logical_and(y_true == np.argmax(M[i, :], axis=1), y_true != np.argmax(M[j, :], axis=1)))  # number of i correct but not j
            Qs.append((N_1_1*N_0_0 - N_0_1*N_1_0)*1./(N_1_1*N_0_0+N_0_1*N_1_0+np.finfo(float).eps))
    return np.mean(Qs)


def generalized_diversity(M, y_true):
    N = M.shape[1]
    L = M.shape[0]
    pi = np.zeros(N)
    for i in range(N):
        pIdx = 0
        for j in range(L):
            if np.argmax(M[j][i]) != y_true[i]:
                pIdx += 1
        pi[pIdx] += 1

    pi = [x * 1.0 / N for x in pi]

    P1 = 0
    P2 = 0
    for i in range(N):
        P1 += i * 1.0 * pi[i] / L
        P2 += i * (i - 1) * 1.0 * pi[i] / (L * (L - 1))
    return 1.0 - P2 / P1


def cos_similarity(models_list):
    model_mat = []
    for i in models_list:
        each_model_all = []
        for name, param in i.named_parameters():
            param = param.detach().cpu().numpy()
            if 'out' in name:
                continue
            else:
                if int(len(param.shape)) == 1:
                    pass
                else:
                    param = param.flatten()
                    each_model_all = each_model_all + param.tolist()
        model_mat.append(each_model_all)
    cs_result = np.mean(pairwise.cosine_similarity(model_mat, model_mat), axis=0).tolist()
    return cs_result