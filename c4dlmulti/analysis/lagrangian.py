import numpy as np
from pysteps import motion, nowcasts
from pysteps.utils import conversion, transformation
import tensorflow as tf


oflow_method = motion.get_method("LK")
extrap_method = nowcasts.get_method("extrapolation")
def motion_extrapolate(motion_field, extrap_field, num_frames=12):
    V = oflow_method(motion_field[-3:, :, :])
    extrapolated = extrap_method(extrap_field[-1, :, :], V, num_frames)
    return extrapolated


def motion_extrapolate_batch(motion_field, extrap_field, num_frames=12):
    out_shape = (extrap_field.shape[0], num_frames) + extrap_field.shape[2:]
    extrapolated = np.empty(out_shape, dtype=extrap_field.dtype)
    for i in range(extrap_field.shape[0]):
        extrapolated[i,...,0] = motion_extrapolate(
            motion_field[i,...,0], extrap_field[i,...,0],
            num_frames=12
        )
    return extrapolated


def conf_matrix_extrap(
    batch_gen, motion_var="RZC", 
    extrap_var="occurrence-8-10", nan_threshold=-3.690,
    num_batches=None, dataset='test',
    separate_leadtimes=False,
):

    names = batch_gen.pred_names_past
    motion_index = names.index(motion_var)
    extrap_index = names.index(extrap_var)
    if num_batches is None:
        num_batches = len(batch_gen.time_coords[dataset]) // \
            batch_gen.batch_size

    (X,Y) = batch_gen.batch(0, dataset=dataset)
    num_frames = Y[0].shape[1]
    if separate_leadtimes:
        tp = np.zeros(num_frames, dtype=int)
        fp = np.zeros(num_frames, dtype=int)
        fn = np.zeros(num_frames, dtype=int)
        tn = np.zeros(num_frames, dtype=int)
    else:
        tp = fp = fn = tn = 0

    for i in range(num_batches):
        print(f"{i+1}/{num_batches}")
        (X,Y) = batch_gen.batch(i, dataset=dataset)
        motion_field = X[motion_index].astype(np.float32)
        motion_field[motion_field <= nan_threshold] = np.nan
        extrap_field = X[extrap_index].astype(np.float32)
        Y_pred = motion_extrapolate_batch(motion_field, extrap_field,
            num_frames=num_frames)
        Y_pred = (Y_pred >= 0.5)
        Y = (Y[0] >= 0.5)

        tp_batch = Y_pred & Y
        fp_batch = Y_pred & ~Y
        fn_batch = ~Y_pred & Y
        tn_batch = ~Y_pred & ~Y

        if separate_leadtimes:
            for t in range(num_frames):
                tp[t] += np.count_nonzero(tp_batch[:,t,...])
                fp[t] += np.count_nonzero(fp_batch[:,t,...])
                fn[t] += np.count_nonzero(fn_batch[:,t,...])
                tn[t] += np.count_nonzero(tn_batch[:,t,...])
        else:
            tp += np.count_nonzero(tp_batch)
            fp += np.count_nonzero(fp_batch)
            fn += np.count_nonzero(fn_batch)
            tn += np.count_nonzero(tn_batch)

    N = tp + fp + fn + tn

    conf_matrix = np.array(((tp, fn), (fp, tn))) / N
    if separate_leadtimes:
        conf_matrix = conf_matrix.reshape((2,2,1,num_frames))
    else:
        conf_matrix = conf_matrix.reshape((2,2,1))

    return conf_matrix



def loss_extrap(
    batch_gen, loss_func, smooth=None, motion_var="RZC", 
    extrap_var="occurrence-8-10", nan_threshold=-3.690,
    num_batches=None, dataset='test',
    separate_leadtimes=False,
):

    names = batch_gen.pred_names_past
    motion_index = names.index(motion_var)
    extrap_index = names.index(extrap_var)
    if num_batches is None:
        num_batches = len(batch_gen.time_coords[dataset]) // \
            batch_gen.batch_size

    (X,Y) = batch_gen.batch(0, dataset=dataset)
    num_frames = Y[0].shape[1]
    if separate_leadtimes:
        loss = np.zeros(num_frames, dtype=float)
    else:
        loss = 0

    for i in range(num_batches):
        print(f"{i+1}/{num_batches}")
        (X,Y) = batch_gen.batch(i, dataset=dataset)
        motion_field = X[motion_index].astype(np.float32)
        motion_field[motion_field <= nan_threshold] = np.nan
        extrap_field = X[extrap_index].astype(np.float32)
        Y_pred = motion_extrapolate_batch(motion_field, extrap_field,
            num_frames=num_frames)
        Y_pred[np.isnan(Y_pred)] = 0
        if smooth is not None:
            Y_pred = Y_pred * (1-2*smooth) + smooth
        Y_pred = tf.convert_to_tensor(Y_pred.astype(np.float32))
        Y = tf.convert_to_tensor(Y[0].astype(np.float32))

        if separate_leadtimes:
            loss_batch = np.array([
                loss_func(Y[:,t:t+1,...], Y_pred[:,t:t+1,...]).numpy().mean()
                for t in range(num_frames)
            ])
        else:
            loss_batch = loss_func(Y, Y_pred).numpy().mean()

        loss += loss_batch

    loss /= num_batches

    return loss
