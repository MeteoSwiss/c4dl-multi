import gc
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Concatenate, Layer
from tensorflow.keras.layers import Conv2D, UpSampling2D
from tensorflow.keras.layers import TimeDistributed, Lambda, Add

from .blocks import ConvBlock, ResBlock
from ...features.batch import BatchSequence
from .optimizers import AdaBeliefOptimizer
from .rnn import ResGRU


file_dir = os.path.dirname(os.path.abspath(__file__))


def concat(**kwargs):
    """ Workaround for the behavior in Concatenate 
        that raises an error if the input list is of length 1.
    """
    def concat_func(inputs):
        if len(inputs) > 1:
            return Concatenate(**kwargs)(inputs)
        else:
            return inputs[0]
    return concat_func


def create_inputs(
    input_specs,
    base_shape=(256,256),
    past_timesteps=12,
    future_timesteps=12,
):
    # separate inputs by resolution and timeframe; build input list
    inputs_by_shape = {}
    inputs = []
    def add_input(timeframe, shape_divisor, ip):
        if timeframe not in inputs_by_shape:
            inputs_by_shape[timeframe] = {}
        if shape_divisor not in inputs_by_shape[timeframe]:
            inputs_by_shape[timeframe][shape_divisor] = []
        inputs_by_shape[timeframe][shape_divisor].append(ip)

    for input_spec in input_specs:
        timeframe = input_spec["timeframe"]
        if timeframe == "past":
            timesteps = past_timesteps
        elif timeframe == "future":
            timesteps = future_timesteps
        elif timeframe == "static":
            timesteps = 1
        shape_divisor = input_spec.get("shape_divisor", 1)
        shape = (base_shape[0]//shape_divisor, base_shape[1]//shape_divisor)
        channels = input_spec.get("channels", 1)
        dtype = input_spec.get("dtype", tf.float32)
        
        ip = Input(
            shape=(timesteps,shape[0],shape[1],channels),
            name=input_spec["name"],
            dtype=dtype
        )
        inputs.append(ip)
        if dtype != np.float32:
            ip = tf.cast(ip, tf.float32)
        
        if timeframe == "static": # expand static variable in time dimension
            ip_past = tf.repeat(ip, axis=1, repeats=past_timesteps)
            add_input("past", shape_divisor, ip_past)
            if future_timesteps != past_timesteps:
                ip_future = tf.repeat(ip, axis=1, repeats=future_timesteps)
            else:
                ip_future = ip_past
            add_input("future", shape_divisor, ip_future)
        else:
            add_input(timeframe, shape_divisor, ip)

    return (inputs, inputs_by_shape)


def rnn_model(
    input_specs,
    base_shape=(256,256),
    past_timesteps=12,
    future_timesteps=12,
    num_outputs=1,
    dropout=0,
    norm=None,
    last_only=False,
    final_activation='sigmoid'
):
    (inputs, inputs_by_shape) = create_inputs(input_specs,
        base_shape=base_shape, past_timesteps=past_timesteps,
        future_timesteps=future_timesteps)

    # number of channels by depth
    #block_channels = [32, 64, 128, 256]
    block_channels = [32, 64, 128]    
    #block_channels = [24, 48, 96]

    # recurrent downsampling 
    xt_by_time = {}
    for timeframe in inputs_by_shape:
        xt_by_time[timeframe] = {
            s: concat(axis=-1)(inputs_by_shape[timeframe][s])
            for s in inputs_by_shape[timeframe]
        }
    
    intermediate = []
    for timeframe in inputs_by_shape:
        xt = xt_by_time[timeframe]

        for (i,channels) in enumerate(block_channels):
            # merge different resolutions when possible
            s = 2**i
            if (i > 0) and s in inputs_by_shape[timeframe]:
                if 1 in xt:                
                    xt[1] = Concatenate(axis=-1)([xt[1],xt[s]])
                else:
                    xt[1] = xt[s]
                del xt[s]

            for s in xt:
                stride = 2 if (s == 1) else 1 # do not downsample lores data
                xt[s] = ResBlock(channels, time_dist=True, stride=stride,
                    dropout=dropout, norm=norm)(xt[s])
                
                initial_state = Lambda(lambda y: tf.zeros_like(y[:,0,...]))(xt[s])
                # TODO: future steps should iterate backwards in time?
                
                xt[s] = ResGRU(                
                    channels, return_sequences=True,
                    time_steps=past_timesteps if timeframe=="past" else future_timesteps,
                )([xt[s],initial_state])                

            if timeframe == "past":
                intermediate.append(ConvBlock(channels)(xt[1][:,-1,...]))
            elif (timeframe == "future") and ("past" not in inputs_by_shape):
                zeros = Lambda(lambda y: tf.zeros_like(y[:,-1:,...]))(xt[s])
                intermediate.append(zeros)

        xt_by_time[timeframe] = xt[1]

    # recurrent upsampling
    if "future" in xt_by_time:
        xt = xt_by_time["future"]
    else:
        xt = Lambda(lambda y: tf.zeros_like(
            tf.repeat(y[:,:1,...],future_timesteps,axis=1)
        ))(xt_by_time["past"])

    for (i,channels) in reversed(list(enumerate(block_channels))):
        xt = ResGRU(        
            channels, return_sequences=True, time_steps=future_timesteps
        )([xt,intermediate[i]])        
        xt = TimeDistributed(UpSampling2D(interpolation='bilinear'))(xt)
        xt = ResBlock(block_channels[max(i-1,0)], time_dist=True,
            dropout=dropout, norm=norm)(xt)

    seq_out = TimeDistributed(Conv2D(num_outputs, kernel_size=(1,1),
        activation=final_activation))(xt)

    if last_only:
        seq_out = seq_out[:,-1,...]

    model = Model(inputs=inputs, outputs=[seq_out])

    return model


def persistence_model(
    input_specs,
    persistence_vars=None,
    base_shape=(256,256),
    past_timesteps=12,
    future_timesteps=12,
    num_outputs=1
    ):
    if persistence_vars is None:
        persistence_vars = ["occurrence-8-10"]

    (inputs, inputs_by_shape) = create_inputs(input_specs,
        base_shape=base_shape, past_timesteps=past_timesteps,
        future_timesteps=future_timesteps)

    names = [s["name"] for s in input_specs]
    pvar_indices = [names.index(var) for var in persistence_vars]
    last_timesteps = [inputs[k][:,-1:,...] for k in pvar_indices]
    persistence = [
        tf.repeat(lts, axis=1, repeats=future_timesteps)
        for lts in last_timesteps
    ]

    model = Model(inputs=inputs, outputs=persistence)

    return model


def ensemble_model(models, weights=None):
    N = len(models)
    if weights is None:        
        weights = [1/N]*N
    inputs = models[0].inputs
    weighted_outputs = [w*m(inputs) for (w,m) in zip(weights, models)]
    ensemble_output = weighted_outputs[0]
    for weighted_output in weighted_outputs[1:]:
        ensemble_output = ensemble_output + weighted_output
    model = Model(inputs=inputs, outputs=ensemble_output)

    return model


def iou_metric(y_true, y_pred, smooth=1e-6): # this is the same as critical success index
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.math.round(y_pred)
    intersection = tf.math.reduce_mean(y_true * y_pred)
    total = tf.math.reduce_mean(y_true + y_pred)    
    union = total - intersection
    return intersection / (union + smooth)


@tf.function
def iou_loss(y_true, y_pred, smooth=1e-6): # this is the same as critical success index
    y_true = tf.cast(y_true, tf.float32)
    intersection = tf.math.reduce_mean(y_true * y_pred)
    total = tf.math.reduce_mean(y_true + y_pred)    
    union = total - intersection
    return 1.0 - intersection / (union + smooth)


def dice_metric(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.math.round(y_pred)
    true_pos = y_true * y_pred
    false_pos = (1 - y_true) * y_pred
    false_neg = y_true * (1 - y_pred)

    tp = tf.math.reduce_sum(true_pos, axis=(1,2,3,4))
    fp = tf.math.reduce_sum(false_pos, axis=(1,2,3,4))
    fn = tf.math.reduce_sum(false_neg, axis=(1,2,3,4))
    denom = 2*tp + fp + fn
    return tf.where(denom != 0, 2*tp / denom, 1)


def true_pos(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.math.round(y_pred)
    tp = y_true * y_pred
    return tf.math.reduce_mean(tp, axis=(1,2,3,4))


def true_neg(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.math.round(y_pred)
    tn = (1-y_true) * (1-y_pred)
    return tf.math.reduce_mean(tn, axis=(1,2,3,4))


def false_pos(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.math.round(y_pred)
    fp = (1-y_true) * y_pred
    return tf.math.reduce_mean(fp, axis=(1,2,3,4))


def false_neg(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.math.round(y_pred)
    fn = y_true * (1-y_pred)
    return tf.math.reduce_mean(fn, axis=(1,2,3,4))


@tf.function
def prob_binary_crossentropy(y_true, y_pred):
    # defined because according to the documentation
    # standard cross entropy requires labels to be 0 or 1
    return -(y_true * tf.math.log(y_pred) + (1-y_true) * tf.math.log(1-y_pred))


@tf.function
def rain_mean_std(log10_R):
    m = log10_R
    sigma = 0.3215 # equivalent to std = 0.33*mean
    mu = tf.math.log(10.0)*m - 0.5*tf.math.square(sigma)
    return (mu, sigma)


@tf.function
def normal_cdf(x, mu, sigma):
    sqrt_2 = tf.math.sqrt(2.0)
    mu = tf.expand_dims(mu, -1)
    sigma = tf.expand_dims(sigma, -1)
    return 0.5 * (1.0 + 
        tf.math.erf((x[None,:]-mu)/(sqrt_2*sigma))
    )


@tf.function
def tf_log10(x):
    return tf.math.log(x) / tf.math.log(10.0)


def make_rain_loss_hist(bins, rain_thresh=0.1, samples_per_hour=12.0):
    log_bins = tf.convert_to_tensor(np.log(bins).astype(np.float32))
    rain_thresh = tf.constant(np.log10(rain_thresh).astype(np.float32))
    @tf.function
    def rain_loss_hist(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_true = tf.cond(
            tf.rank(y_true)==4,
            lambda: tf.expand_dims(y_true,-1),
            lambda: y_true
        )
        y_pred = tf.cast(y_pred, tf.float32)

        # sum to accumulated rain in log10(mm/h)
        y_true = tf.math.pow(10.0, y_true)
        y_true = tf.where(y_true < rain_thresh, 0.0, y_true)
        y_true = tf.math.reduce_sum(y_true, axis=1)
        y_true = y_true / samples_per_hour
        dry = (y_true < rain_thresh)
        y_true = tf.where(dry, rain_thresh*0.1, y_true)        
        y_true = tf_log10(y_true)

        # create probability bins of precip
        (mu_true, sigma_true) = rain_mean_std(y_true[...,0])
        cdf = normal_cdf(log_bins, mu_true, sigma_true)
        cdf_diff = cdf[...,1:]-cdf[...,:-1]        
        first_bin = cdf[...,:1]
        first_bin = tf.where(dry, 1.0, first_bin)
        later_bins = tf.concat([cdf_diff, 1.0-cdf[...,-1:]], axis=-1)
        later_bins = tf.where(dry, 0.0, later_bins)
        bins_true = tf.concat([first_bin, later_bins], axis=-1)

        # compute cross entropy loss
        xent = -tf.math.reduce_sum(bins_true * tf.math.log(y_pred), 
            axis=-1, keepdims=True)
        return xent
    return rain_loss_hist


def create_weighted_binary_crossentropy(ones_fraction):
    zeros_fraction = 1-ones_fraction
    weights = (
        1./(2*zeros_fraction),
        1./(2*ones_fraction)
    )

    @tf.function
    def weighted_binary_crossentropy(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        loss = tf.losses.binary_crossentropy(y_true, y_pred)
        # Apply the weights
        w = (1 - y_true) * weights[0] + y_true * weights[1]
        weighted_loss = w[...,0] * loss
        # Return the mean error
        return weighted_loss

    return weighted_binary_crossentropy


def create_weighted_focal_loss(ones_fraction, gamma=2.0):
    wce = create_weighted_binary_crossentropy(tf.constant(ones_fraction))
    
    @tf.function
    def weighted_focal_loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        y_pred = 0.001 + y_pred*0.998 # scale to inhibit exploding gradients
        ce = wce(y_true, y_pred)
        pt = tf.where(y_true==1, y_pred, 1-y_pred)
        return (1-pt[...,0])**gamma * ce

    return weighted_focal_loss


def compile_model(
    model,
    optimizer='adabelief',
    loss='weighted_focal_loss', 
    wfc_gamma=2.0,
    metrics=[
        'binary_accuracy', "iou_metric", "dice_metric",
        "true_pos", "true_neg", "false_pos", "false_neg"
    ],
    event_occurrence=0.0106, # occurrence-8-10
    opt_kwargs={}
):
    metric_names = {
        "weighted_binary_crossentropy": create_weighted_binary_crossentropy(
            event_occurrence),
        "weighted_focal_loss": create_weighted_focal_loss(
            event_occurrence, gamma=wfc_gamma),
        "prob_binary_crossentropy": prob_binary_crossentropy,
        "iou_loss": iou_loss,
        "iou_metric": iou_metric,
        "dice_metric": dice_metric,
        "true_pos": true_pos,
        "true_neg": true_neg,
        "false_pos": false_pos,
        "false_neg": false_neg
    }
    loss = metric_names.get(loss, loss)
    metrics = [metric_names.get(m,m) for m in metrics]
    if optimizer == "adabelief":
        optimizer = AdaBeliefOptimizer(**opt_kwargs)
    model.compile(loss=loss,
        optimizer=optimizer, metrics=metrics)


def init_model(batch_gen, model_func=rnn_model, compile=True, 
    init_strategy=True, compile_kwargs={}, **kwargs):

    (past_timesteps, future_timesteps) = batch_gen.timesteps

    # construct input specs from a sample batch
    input_specs = []
    (X, _) = batch_gen.batch(0)
    max_size = max(x.shape[2] for x in X)
    pred_names = batch_gen.pred_names_past + \
        batch_gen.pred_names_future + \
        batch_gen.pred_names_static

    for (i,x) in enumerate(X):
        shape_divisor = max_size // x.shape[2]
        timesteps = x.shape[1]
        channels = x.shape[-1]
        pred_name = pred_names[i]
        if pred_name in batch_gen.pred_names_past:
            timeframe = "past"
        elif pred_name in batch_gen.pred_names_future:
            timeframe = "future"
        elif pred_name in batch_gen.pred_names_static:
            timeframe = "static"        
        input_spec = {
            "shape_divisor": shape_divisor,
            "channels": channels,
            "timeframe": timeframe,
            "name": pred_name,
            "dtype": x.dtype
        }
        input_specs.append(input_spec)

    if init_strategy and len(tf.config.list_physical_devices('GPU')) > 1:
        # initialize multi-GPU strategy
        strategy = tf.distribute.MirroredStrategy()
    else: # use default strategy
        strategy = tf.distribute.get_strategy()

    with strategy.scope():
        model = model_func(
            past_timesteps=past_timesteps,
            future_timesteps=future_timesteps,
            input_specs=input_specs,
            **kwargs
        )
        if compile:
            compile_model(model, **compile_kwargs)

    gc.collect()
    
    return (model, strategy)


def combined_model(models, output_names):
    past_in = Input(shape=models[0].input_shape[1:],
        name="past_in")
    outputs = [
        Layer(name=name)(model(past_in))
        for (model, name) in zip(models, output_names)
    ]
    comb_model = Model(inputs=[past_in], outputs=outputs)

    return comb_model


def train_model(model, strategy, batch_gen,
    weight_fn="model.h5", monitor="val_loss"):

    fn = os.path.join(file_dir, "../../../models", weight_fn)
    steps_per_epoch = len(batch_gen.time_coords["train"]) // batch_gen.batch_size
    validation_steps = len(batch_gen.time_coords["valid"]) // batch_gen.batch_size

    with strategy.scope():        
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            fn, save_weights_only=True, save_best_only=True, mode="min",
            monitor=monitor
        )
        reducelr = tf.keras.callbacks.ReduceLROnPlateau(
            patience=3, mode="min", factor=0.2, monitor=monitor,
            verbose=1, min_delta=0.0
        )
        earlystop = tf.keras.callbacks.EarlyStopping(
            patience=6, mode="min", restore_best_weights=True,
            monitor=monitor
        )
        callbacks = [checkpoint, reducelr, earlystop]

        batch_seq_train = BatchSequence(batch_gen, dataset='train')
        batch_seq_valid = BatchSequence(batch_gen, dataset='valid')

        model.fit(
            batch_seq_train,
            epochs=100,
            steps_per_epoch=steps_per_epoch,
            validation_data=batch_seq_valid,
            validation_steps=validation_steps,
            callbacks=callbacks
        )
