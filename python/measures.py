def _process_Y(Y):
    if not all((isinstance(y, list) or isinstance(y, tuple)) for y in Y):
        raise ValueError("Unsupported data format")

    return Y


def true_possitive_at_k(Y_true, Y_pred, k=5):
    Y_true = _process_Y(Y_true)
    Y_pred = _process_Y(Y_pred)

    tp = 0
    for t, p in zip(Y_true, Y_pred):
        for p_i in p[:k]:
            tp += 1 if p_i in t else 0
    return tp


def precision_at_k(Y_true, Y_pred, k=5):
    return true_possitive_at_k(Y_true, Y_pred, k=k) / (len(Y_true) * k)


def recall_at_k(Y_true, Y_pred, k=5):
    tp = true_possitive_at_k(Y_true, Y_pred, k=k)
    tc = 0
    for t in Y_true:
        tc += len(t)
    return tp / tc


def coverage_at_k(Y_true, Y_pred, k=5):
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
    pass


def ndcg_at_k(Y_true, Y_pred, k=5):
    pass


def hamming_loss(Y_true, Y_pred):
    Y_true = _process_Y(Y_true)
    Y_pred = _process_Y(Y_pred)

    hl = 0
    for t, p in zip(Y_true, Y_pred):
        hl += len(set(t).intersection(p))

    return hl / len(Y_true)
