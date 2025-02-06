# -*- coding: utf-8 -*- 
# @Time : 2023/11/28 8:48
# @File : utils.py
import os
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix

import numpy as np


def _check_probablities(p, q=None):
    assert 0. <= np.all(p) <= 1.
    if q is not None:
        assert len(p) == len(q), \
            'Probabilies and ground truth must have the same number of elements.'


def max_max2(end_list):
    max2 = np.sort(end_list)[-2]
    max = np.max(end_list)
    return max - max2


def max_min(end_list):
    min = np.min(end_list)
    max = np.max(end_list)
    return max - min


def mean_med(end_list):
    mean = np.mean(end_list)
    med = np.median(end_list)
    return med - mean


def predictive_entropy(p, base=2, eps=1e-10, number=10):
    """
    calculate entropy in element-wise
    :param p: probabilities
    :param base: default exp
    :return: average entropy value
    """
    p_arr = np.asarray(p)
    _check_probablities(p)
    enc = -(p_arr * np.log(p_arr + eps) + (1. - p_arr) * np.log(1. - p_arr + eps))
    if base is not None:
        enc = np.clip(enc / np.log(base), a_min=0., a_max=1000)
    enc_ = []
    for item in enc:
        enc_.append([np.sum(item) / number])
    return np.mean(enc_)


def entropy(p, number=None, base=2, eps=1e-10):
    """
    calculate entropy in element-wise
    :param p: probabilities
    :param base: default exp
    :return: average entropy value
    """
    p_arr = np.asarray(p)
    _check_probablities(p)
    enc = -(p_arr * np.log(p_arr + eps) + (1. - p_arr) * np.log(1. - p_arr + eps))
    enc = np.sum(enc)
    return enc


def predictive_kld(p, number=10, w=None, base=2, eps=1e-10):
    """
    The Kullback-Leibler (KL) divergence measures the difference between two probability distributions by quantifying the information lost
    when one distribution is approximated by another. When comparing a probability vector to its mean vector, the KL divergence assesses
    the information difference between the original probabilities and the uniform distribution of their mean.


    calculate Kullback-Leibler divergence in element-wise
    :param p: probabilities
    :param number: the number of likelihood values for each sample
    :param w: weights for probabilities
    :param base: default exp
    :return: average entropy value
    """
    if number <= 1:
        return np.zeros_like(p)

    p_arr = np.asarray(p).reshape((-1, number))
    _check_probablities(p)
    q_arr = np.tile(np.mean(p_arr, axis=-1, keepdims=True), [1, number])
    if w is None:
        w_arr = np.ones(shape=(number, 1), dtype=float) / number
    else:
        w_arr = np.asarray(w).reshape((number, 1))

    kld_elem = p_arr * np.log((p_arr + eps) / (q_arr + eps)) + (1. - p_arr) * np.log(
        (1. - p_arr + eps) / (1. - q_arr + eps))
    if base is not None:
        kld_elem = kld_elem / np.log(base)
    kld = np.matmul(kld_elem, w_arr)
    return kld


def predictive_std(p, number=10, w=None):
    """
    calculate the probabilities deviation
    :param p: probabilities
    :param number: the number of probabilities applied to each sample
    :param w: weights for probabilities
    :param axis: the axis along which the calculation is conducted
    :return:
    """
    if number <= 1:
        return np.zeros_like(p)

    ps_arr = np.asarray(p).reshape((-1, number))
    _check_probablities(ps_arr)
    if w is None:
        w = np.ones(shape=(number, 1), dtype=float) / number
    else:
        w = np.asarray(w).reshape((number, 1))
    assert 0 <= np.all(w) <= 1.
    mean = np.matmul(ps_arr, w)
    var = np.sqrt(np.matmul(np.square(ps_arr - mean), w) * (float(number) / float(number - 1)))
    return var


def nll(p, label, eps=1e-10, base=2):
    """
    negative log likelihood (NLL)
    :param p: predictive labels
    :param eps: a small value prevents the overflow
    :param base: the base of log function
    :return: the mean of NLL
    """
    p = np.array(p)
    q = np.full(len(p), label)
    nll = -(q * np.log(p + eps) + (1. - q) * np.log(1. - p + eps))
    if base is not None:
        nll = np.clip(nll / np.log(base), a_min=0., a_max=1000)
    return np.mean(nll)


def prob_label_kld(p, label, number=10, w=None, base=2, eps=1e-10):
    if number <= 1:
        return np.zeros_like(p)

    p_arr = np.asarray(p).reshape((-1, number))
    _check_probablities(p)
    q_arr = np.full(number, label)
    if w is None:
        w_arr = np.ones(shape=(number, 1), dtype=float) / number
    else:
        w_arr = np.asarray(w).reshape((number, 1))

    kld_elem = p_arr * np.log((p_arr + eps) / (q_arr + eps)) + (1. - p_arr) * np.log(
        (1. - p_arr + eps) / (1. - q_arr + eps))
    if base is not None:
        kld_elem = kld_elem / np.log(base)
    kld = np.matmul(kld_elem, w_arr)
    return kld / number


def Wasserstein_distance(p, label):
    from scipy.stats import wasserstein_distance
    p = np.array(p)
    q = np.full(len(p), label)
    emd = wasserstein_distance(p, q)

    return emd


def Euclidean_distance(p, label):
    p = np.array(p)
    q = np.full(len(p), label)

    v1 = np.array(p)
    v2 = np.array(q)

    distance = np.linalg.norm(v1 - v2)

    return distance


def Manhattan_distance(p, label):
    p = np.array(p)
    q = np.full(len(p), label)
    v1 = np.array(p)
    v2 = np.array(q)
    distance = np.sum(np.abs(v1 - v2)) / len(p)

    return distance


def Chebyshev_distance(p, label):
    p = np.array(p)
    q = np.full(len(p), label)
    v1 = np.array(p)
    v2 = np.array(q)
    distance = np.max(np.abs(v1 - v2))

    return distance


def save_to_txt(goal, txt_path):
    f = open(txt_path, "w")
    for line in goal:
        f.write(line + '\n')
    f.close()


def txt_to_list(txt_path):
    f = open(txt_path, "r")
    return f.read().splitlines()


def confusion(y_test_true, y_new):
    tn, fp, fn, tp = confusion_matrix(y_test_true, y_new).ravel()
    fpr = fp / float(tn + fp)
    fnr = fn / float(tp + fn)

    recall = tp / float(tp + fn)
    precision = tp / float(tp + fp)
    f1 = f1_score(y_test_true, y_new, average='binary')

    print("recall is " + str(recall * 100) + "%" + " precision  is " + str(precision * 100) + "%, F1 score is " + str(
        f1 * 100) + "%,FPR is " + str(fpr * 100) + "%")
    return tp, fp


def mkdir(target):
    try:
        if os.path.isfile(target):
            target = os.path.dirname(target)

        if not os.path.exists(target):
            os.makedirs(target)
        return 0
    except IOError as e:
        raise Exception("Fail to create directory! Error:" + str(e))


def dump_joblib(data, path):
    if not os.path.exists(os.path.dirname(path)):
        mkdir(os.path.dirname(path))

    try:
        import joblib
        with open(path, 'wb') as wr:
            joblib.dump(data, wr)
        return
    except IOError:
        raise IOError("Dump data failed.")


def read_joblib(path):
    import joblib
    if os.path.isfile(path):
        with open(path, 'rb') as fr:
            return joblib.load(fr)
    else:
        raise IOError("The {0} is not a file.".format(path))


uc_metrics_set = ['prob', 'entropy', 'kld', 'std', 'max_max2', 'max_min', 'mean_med',
                  'label_prob', 'nll', 'emd', 'cbs', 'mat', 'eld', 'kld_label']

no_label_metrics_map = {
    'prob': None,
    'entropy': predictive_entropy,
    'kld': predictive_kld,
    'std': predictive_std,
    'max_max2': max_max2,
    'max_min': max_min,
    'mean_med': mean_med
}

with_label_metrics_map = {
    'label_prob': None,
    'kld_label': prob_label_kld,
    'nll': nll,
    'emd': Wasserstein_distance,
    'cbs': Chebyshev_distance,
    'mat': Manhattan_distance,
    'eld': Euclidean_distance
}


def plot_box(group1, group2):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    plt.boxplot(group1, positions=[1], widths=0.6, patch_artist=True, showfliers=True)
    plt.boxplot(group2, positions=[2], widths=0.6, patch_artist=True, showfliers=True)

    plt.xlabel('Groups')
    plt.ylabel('Values')
    plt.title('Comparison of Two Datasets')
    plt.xticks([1, 2], ['Group 1', 'Group 2'])

    plt.grid(axis='y')
    plt.show()


def upbound(data):
    q3 = np.percentile(data, 75)  # 上四分位数 Q3
    iqr = np.percentile(data, 75) - np.percentile(data, 25)  # 四分位数间距 IQR

    # 计算上限
    upper_bound = q3 + 1.5 * iqr

    return upper_bound



