from math import log2

def _process_Y(Y):
    if not all((isinstance(y, list) or isinstance(y, tuple)) for y in Y):
        raise ValueError("Unsupported data format")

    return Y


def precision_at_k(Y_true, Y_pred, k=5):
    """
    Calculate precision at k
    :param Y_true: true labels
    :param Y_pred: ranking of predicted labels
    :param k: k
    :return: value of precision at k
    """
    Y_true = _process_Y(Y_true)
    Y_pred = _process_Y(Y_pred)

    sum = 0
    for t, p in zip(Y_true, Y_pred):
        p_at_k = 0
        for p_i in p[:k]:
            p_at_k += 1 if p_i in t else 0
        sum += p_at_k / k
    return sum / len(Y_true)


def recall_at_k(Y_true, Y_pred, k=5):
    """
    Calculate recall at k
    :param Y_true: true labels
    :param Y_pred: ranking of predicted labels
    :param k: k
    :return: value of recall at k
    """
    Y_true = _process_Y(Y_true)
    Y_pred = _process_Y(Y_pred)

    sum = 0
    for t, p in zip(Y_true, Y_pred):
        r_at_k = 0
        for p_i in p[:k]:
            r_at_k += 1 if p_i in t else 0
        sum += r_at_k / len(t)
    return sum / len(Y_true)



def coverage_at_k(Y_true, Y_pred, k=5):
    """
    Calculate coverage at k
    :param Y_true: true labels
    :param Y_pred: ranking of predicted labels
    :param k: k
    :return: value of coverage at k
    """

    Y_true = _process_Y(Y_true)
    Y_pred = _process_Y(Y_pred)

    uniq_t = set()
    uniq_tp = set()
    for t, p in zip(Y_true, Y_pred):
        for t_i in t:
            uniq_t.add(t_i)
        for p_i in p[:k]:
            if p_i in t:
                uniq_tp.add(p_i)
    return len(uniq_tp) / len(uniq_t)


def dcg_at_k(Y_true, Y_pred, k=5):
    """
    Calculate DCG at k
    :param Y_true: true labels
    :param Y_pred: ranking of predicted labels
    :param k: k
    :return: value of DCG at k
    """
    Y_true = _process_Y(Y_true)
    Y_pred = _process_Y(Y_pred)

    sum = 0
    for t, p in zip(Y_true, Y_pred):
        for i, p_i in enumerate(p[:k]):
            sum += 1 / log2(i + 2) if p_i in t else 0
    return sum / len(Y_true)


def ndcg_at_k(Y_true, Y_pred, k=5):
    """
    Calculate nDCG at k
    :param Y_true: true labels
    :param Y_pred: ranking of predicted labels
    :param k: k
    :return: value of nDCG at k
    """
    Y_true = _process_Y(Y_true)
    Y_pred = _process_Y(Y_pred)

    sum = 0
    for t, p in zip(Y_true, Y_pred):
        dcg = 0
        n = 0
        for i, p_i in enumerate(p[:k]):
            n += 1 / log2(i + 2)
            dcg += 1 / log2(i + 2) if p_i in t else 0
        sum += dcg / n
    return sum / len(Y_true)


def hamming_loss(Y_true, Y_pred):
    """
    Calculate hamming loss
    :param Y_true: true labels
    :param Y_pred: predicted labels
    :return: value of hamming loss
    """

    Y_true = _process_Y(Y_true)
    Y_pred = _process_Y(Y_pred)

    sum = 0
    for t, p in zip(Y_true, Y_pred):
        sum += len(set(t).intersection(p))

    return sum / len(Y_true)
