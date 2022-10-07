import string

from matplotlib import pyplot as plt, gridspec
import numpy as np
from scipy.stats import norm
import tensorflow as tf

from c4dlmulti.analysis import calibration, shapley
from c4dlmulti.visualization import plots

import training


def exclusion_plot(prefix="lightning", out_file=None, fig=None, axes=None):
    scores = shapley.load_scores(
        f"/scratch/jleinone/c4dl/results/{prefix}/test/",
        prefix=prefix
    )
    scores_norm = {s: v/scores[''] for (s,v) in scores.items()}
    loss = "FL2" if prefix == "lightning" else "cross_entropy"
    fig = plots.exclusion_plot({loss: scores_norm}, fig=fig, axes=axes)

    if out_file is not None:
        fig.savefig(out_file, bbox_inches='tight')
        plt.close(fig)


def exclusion_plot_all(prefixes=("lightning", "hail", "rain"), out_file=None):
    scores = {
        p: shapley.load_scores(
            f"/scratch/jleinone/c4dl/results/{p}/test/",
            prefix=p
        )
        for p in prefixes
    }
    scores_norm = {
        p:
            {s: v/scores[p][''] for (s,v) in scores[p].items()}
        for p in prefixes
    }
    losses = [
        "FL2" if p == "lightning" else "cross_entropy"
        for p in prefixes
    ]
    fig = plots.exclusion_plot(scores_norm, losses)

    if out_file is not None:
        fig.savefig(out_file, bbox_inches='tight')
        plt.close(fig)


def shapley_leadtime_plot(prefix="lightning", out_file=None):
    sources = 'rlsnd'
    leadtimes = np.arange(1, 13)
    values = {source: np.zeros(len(leadtimes)) for source in sources}
    for (i,lt) in enumerate(leadtimes):
        scores = shapley.load_scores(
            f"/scratch/jleinone/c4dl/results/{prefix}/test/",
            prefix=prefix,
            file_type="eval_leadtime",
            score_index=lt
        )
        for source in sources:
            values[source][i] = shapley.shapley_value(scores, source)
    
    fig = plots.shapley_by_time(leadtimes, values)

    if out_file is not None:
        fig.savefig(out_file, bbox_inches='tight')
        plt.close(fig)


def shapley_leadtime_plots(out_file=None):
    sources = 'rlsnd'
    leadtimes = np.arange(1, 13)
    fig = plt.figure(figsize=(6,9))
    gs = gridspec.GridSpec(7, 1, height_ratios=(1,5,0.75,1,5,0.75,1),
        hspace=0.3)
    
    # plot Shapley values by lead time
    prefixes = ("lightning", "hail")
    subplots = [1, 4]
    for (k,prefix) in enumerate(prefixes):
        values = {source: np.zeros(len(leadtimes)) for source in sources}
        for (i,lt) in enumerate(leadtimes):
            scores = shapley.load_scores(
                f"/scratch/jleinone/c4dl/results/{prefix}/test/",
                prefix=prefix,
                file_type="eval_leadtime",
                score_index=lt
            )
            for source in sources:
                values[source][i] = shapley.shapley_value(scores, source)
        
        ax = fig.add_subplot(gs[k*3+1,0])
        plots.shapley_by_time(leadtimes, values, 
            fig=fig, ax=ax, legend=False)
        
    prefixes = ("lightning", "hail", "rain")
    subplots = [0, 3, 6]
    for (k,prefix) in enumerate(prefixes):
        # plot legend with full shapley values
        scores_full = shapley.load_scores(
            f"/scratch/jleinone/c4dl/results/{prefix}/test/",
            prefix=prefix
        )
        values_full = {s: shapley.shapley_value(scores_full, s) for s in sources}
        ax = fig.add_subplot(gs[subplots[k],0])
        plots.shapley_values_full_legend(values_full, ax)
        ax.axis("off")
        ax.set_title(f"({string.ascii_lowercase[k]}) {plots.prefix_notation[prefix]}")

    if out_file is not None:
        fig.savefig(out_file, bbox_inches='tight')
        plt.close(fig)


def plot_examples(
    batch_gen, model, batch_number=13,
    batch_member=30, out_file=None, 
    shown_inputs=("RZC", "occurrence-8-10", "HRV", "ctth-alti"),
    input_names=("Rain rate", "Lightning", "HRV", "CTH"),
    shown_future_inputs=("CAPE-MU-future",),
    future_input_names=("CAPE-MU",),
    preprocess_rain=False,
    plot_kwargs=None
):
    if plot_kwargs is None:
        plot_kwargs = {}

    names = batch_gen.pred_names_past + batch_gen.pred_names_future
    shown_inputs = [names.index(ip) for ip in shown_inputs]
    shown_future_inputs = [names.index(ip) for ip in shown_future_inputs]
    future_input_codes = [f"input-future-{ip}" for ip in shown_future_inputs]

    (X,Y) = batch_gen.batch(batch_number, dataset='test')
    if preprocess_rain:
        ip = [tf.keras.Input(shape=x.shape[1:]) for x in X]
        pred = model(ip)
        pred = tf.expand_dims(pred, axis=1)
        pred = tf.reduce_sum(pred[...,1:], axis=-1, keepdims=True)
        model = tf.keras.Model(inputs=ip, outputs=pred)

        rr = 10**Y[0].astype(np.float32)
        sig = np.sqrt(np.log(0.33**2+1))
        mu = np.log(rr) - 0.5*sig**2
        Y[0] = norm.sf(1.0, loc=mu, scale=sig)

    fig = plots.plot_model_examples(X, Y, future_input_codes+["obs", model],
        batch_member=batch_member, shown_inputs=shown_inputs,        
        input_names=input_names, future_input_names=future_input_names,
        **plot_kwargs)

    if preprocess_rain:
        for ax in fig.axes:
            if ax.get_title() == "$+5\\,\\mathrm{min}$":
                ax.set_title("Next $60\\,\\mathrm{min}$")
                break

    if out_file is not None:
        fig.savefig(out_file, bbox_inches='tight', dpi=200)
        plt.close(fig)


def plot_all_examples():
    (_, batch_gen, model, _) = training.model_sources(
        "rlsnd", target="occurrence-8-10"
    )
    model.load_weights("../models/lightning/lightning-rlsnd.h5")
    p = np.linspace(0,1,101)
    occurrence = np.load("../results/lightning/test/calibration-lightning-rlsnd.npy")
    calib_model = calibration.calibrated_model(model, p, occurrence)
    sample=(32,11)
    plot_examples(
        batch_gen, calib_model, 
        batch_number=sample[0], batch_member=sample[1],
        shown_inputs=("RZC", "occurrence-8-10", "HRV", "ctth-alti"),
        input_names=("Rain rate", "Lightning", "HRV", "CTH"),
        out_file="../figures/sample-lightning.pdf"
    )

    (_, batch_gen, model, _) = training.model_sources(
        "rlsnd", target="BZC"
    )
    model.load_weights("../models/hail/hail-rlsnd.h5")
    sample=(67,45)
    plot_examples(
        batch_gen, model, 
        batch_number=sample[0], batch_member=sample[1],
        shown_inputs=("RZC", "BZC", "HRV", "ctth-alti"),
        input_names=("Rain rate", "POH", "HRV", "CTH"),
        plot_kwargs={"min_p": 5e-5},
        out_file="../figures/sample-hail.pdf"
    )
    
    (_, batch_gen, model, _) = training.model_sources(
        "rlsnd", target="CPCH"
    )
    model.load_weights("../models/rain/rain-rlsnd.h5")
    sample=(50,28)
    plot_examples(
        batch_gen, model, 
        batch_number=sample[0], batch_member=sample[1],
        shown_inputs=("HRV", "RZC", "ctth-alti"),
        input_names=("HRV", "Rain rate", "CTH"),
        plot_kwargs={"min_p": 5e-4, "output_timesteps": [0]},
        preprocess_rain=True,
        out_file="../figures/sample-rain.pdf"
    )
