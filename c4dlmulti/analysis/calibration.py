import os
import gc

import tensorflow as tf
import tensorflow_probability as tfp
from numba import njit
import numpy as np

from ..features import batch


def calibration_curve(model, batch_gen, dataset='valid', nbins=100):
    batch_seq = batch.BatchSequence(batch_gen, dataset=dataset)
    bin_counts = np.zeros(nbins, dtype=np.uint64)
    bin_occurrences = np.zeros(nbins, dtype=np.uint64)

    for i in range(len(batch_seq)):
        print("{}/{}".format(i,len(batch_seq)))
        (X,Y) = batch_seq[i]
        Y_pred = model.predict(X)
        accumulate_hits(Y[0], Y_pred, bin_counts, bin_occurrences)

    p = np.linspace(0,1,nbins+1)
    p = 0.5 * (p[:-1] + p[1:])
    occurrence_rate = bin_occurrences/bin_counts
    return (p, occurrence_rate)


def calibration_curve_models(model, batch_gen, weight_files, out_dir, dataset='valid'):
    for fn in weight_files:
        model.load_weights(fn)
        (p, occurrence_rate) = calibration_curve(model, batch_gen, dataset=dataset)
        fn_root = fn.split("/")[-1].split(".")[0]
        np.save(
            os.path.join(out_dir, "calibration-{}.npy".format(fn_root)), 
            occurrence_rate
        )
        gc.collect()


@njit
def accumulate_hits(Y, Y_pred, bin_counts, bin_occurrences):
    n = len(bin_counts)
    Y_pred = Y_pred.ravel()
    Y = Y.ravel()
    
    for i in range(Y.shape[0]):
        bin_ind = int(Y_pred[i]*n)
        if bin_ind == n:
            bin_ind = n-1
        
        bin_counts[bin_ind] += 1
        if bool(Y[i]):
            bin_occurrences[bin_ind] += 1


def calibrated_model(model, p, occurrence_rate):
    inputs = model.inputs
    out = model(inputs)
    occurrence_rate = tf.convert_to_tensor(occurrence_rate, dtype=tf.float32)

    out_calib = tfp.math.interp_regular_1d_grid(
        out, 
        x_ref_min=tf.constant(p[0], dtype=tf.float32), 
        x_ref_max=tf.constant(p[-1], dtype=tf.float32), 
        y_ref=occurrence_rate, fill_value='extrapolate'
    )
    out_calib = tf.clip_by_value(out_calib, 0.0, 1.0)
    c_model = tf.keras.models.Model(inputs=inputs, outputs=out_calib)
    return c_model

