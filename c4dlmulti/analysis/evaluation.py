import concurrent.futures
import multiprocessing
import os

import numpy as np
from scipy.integrate import trapezoid
from scipy.stats import norm

try:
    from ..features import batch
except (ImportError, ModuleNotFoundError):
    pass


def confusion_matrix(model, batch_gen, dataset='valid', thresholds=[0.5]):
    batch_seq = batch.BatchSequence(batch_gen, dataset=dataset)
    tp = np.zeros(len(thresholds), dtype=np.uint64)
    fp = np.zeros(len(thresholds), dtype=np.uint64)
    fn = np.zeros(len(thresholds), dtype=np.uint64)
    num_threads = multiprocessing.cpu_count()

    def acc(Y_pred, Y, i, threshold):
        Y_pred_thresh = (Y_pred >= threshold)
        tp[i] += np.count_nonzero(Y_pred_thresh & Y)
        fp[i] += np.count_nonzero(Y_pred_thresh & ~Y)
        fn[i] += np.count_nonzero(~Y_pred_thresh & Y)

    for i in range(len(batch_seq)):
        print("{}/{}".format(i,len(batch_seq)))
        (X,Y) = batch_seq[i]
        Y_pred = model.predict(X)
        Y = Y[0].astype(bool)
        
        with concurrent.futures.ThreadPoolExecutor(num_threads) as executor:
            for (i,threshold) in enumerate(thresholds):            
                executor.submit(acc, Y_pred, Y, i, threshold)

    N = len(batch_seq) * np.prod(Y_pred.shape)
    tn = N - tp - fp - fn

    return np.array(((tp, fn), (fp, tn))) / N


def conf_matrix_models(model, batch_gen, weight_files, out_dir, dataset='valid'):
    thresholds = np.arange(0, 1.0001, 0.001)
    for fn in weight_files:
        model.load_weights(fn)
        conf_matrix = confusion_matrix(model, batch_gen,
            thresholds=thresholds, dataset=dataset)
        fn_root = fn.split("/")[-1].split(".")[0]
        np.save(
            os.path.join(out_dir, "conf_matrix-{}.npy".format(fn_root)), 
            conf_matrix
        )


def conf_matrix_leadtimes(model, batch_gen, dataset='valid', thresholds=[0.5],
    num_leadtimes=12, target="lightning", rain_thresh=(1,10)):

    batch_seq = batch.BatchSequence(batch_gen, dataset=dataset)
    shape = (len(thresholds), num_leadtimes)
    tp = np.zeros(shape, dtype=np.uint64)
    fp = np.zeros(shape, dtype=np.uint64)
    fn = np.zeros(shape, dtype=np.uint64)
    num_threads = multiprocessing.cpu_count()
    log_rain_thresh = np.log(rain_thresh[1])

    def acc(Y_pred, Y, i, t, threshold):
        Y_pred_thresh = (Y_pred >= threshold)
        tp[i,t] += np.count_nonzero(Y_pred_thresh & Y)
        fp[i,t] += np.count_nonzero(Y_pred_thresh & ~Y)
        fn[i,t] += np.count_nonzero(~Y_pred_thresh & Y)

    for k in range(len(batch_seq)):
        print("{}/{}".format(k,len(batch_seq)))
        (X,Y) = batch_seq[k]
        Y_pred = model.predict(X)
        Y = Y[0]

        if target == "rain":
            rr = 10**Y.astype(np.float32)
            rr = rr.mean(axis=1, keepdims=True)
            sig = np.sqrt(np.log(0.33**2+1))
            mu = np.log(rr) - 0.5*sig**2
            Y = norm.sf(log_rain_thresh, loc=mu, scale=sig)
            Y_pred = Y_pred[...,rain_thresh[0]:].sum(axis=-1, keepdims=True)
            Y_pred = np.expand_dims(Y_pred, axis=1)
            
        Y = (Y >= 0.5)

        with concurrent.futures.ThreadPoolExecutor(num_threads) as executor:            
            for (i,threshold) in enumerate(thresholds):
                for t in range(num_leadtimes):                    
                    executor.submit(acc,
                        Y_pred[:,t,...], Y[:,t,...], i, t, threshold
                    )

    N = len(batch_seq) * Y_pred.shape[0] * np.prod(Y_pred.shape[-3:-1])
    tn = N - tp - fp - fn

    return np.array(((tp, fn), (fp, tn))) / N


def precision(conf_matrix):
    ((tp, fn), (fp, tn)) = conf_matrix
    precision = tp / (tp + fp)
    precision[np.isnan(precision)] = 1.0
    return precision


def recall(conf_matrix):
    ((tp, fn), (fp, tn)) = conf_matrix
    return tp / (tp + fn)


def false_alarm_ratio(conf_matrix):
    return 1.0 - precision(conf_matrix)


def intersection_over_union(conf_matrix):
    ((tp, fn), (fp, tn)) = conf_matrix
    return tp / (tp+fp+fn)


def equitable_threat_score(conf_matrix):
    ((tp, fn), (fp, tn)) = conf_matrix
    tp_rnd = (tp+fn) * (tp+fp) / (tp+fp+tn+fn)
    return (tp-tp_rnd) / (tp+fp+fn-tp_rnd)


def peirce_skill_score(conf_matrix):
    ((tp, fn), (fp, tn)) = conf_matrix
    return (tp*tn - fn*fp) / ((tp+fn) * (fp+tn))


def heidke_skill_score(conf_matrix):
    ((tp, fn), (fp, tn)) = conf_matrix
    return 2 * (tp*tn - fn*fp) / ((tp+fn)*(fn+tn) + (tp+fp)*(fp+tn))


def roc_area_under_curve(conf_matrix):
    if conf_matrix.ndim == 4:
        cm = conf_matrix
        auc = [roc_area_under_curve(cm[...,i]) for i in range(cm.shape[-1])]
        return np.stack(auc, axis=-1)

    ((tp, fn), (fp, tn)) = conf_matrix
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)

    auc = trapezoid(tpr[::-1], x=fpr[::-1])
    return auc


def pr_area_under_curve(conf_matrix):
    if conf_matrix.ndim == 4:
        cm = conf_matrix
        auc = [pr_area_under_curve(cm[...,i]) for i in range(cm.shape[-1])]
        return np.stack(auc, axis=-1)

    prec = precision(conf_matrix)
    rec = recall(conf_matrix)

    if (rec[-1] != 0) or (prec[-1] != 1):
        rec = np.hstack((rec, 0.0))
        prec = np.hstack((prec, 1.0))

    auc = trapezoid(prec[::-1], x=rec[::-1])
    return auc
