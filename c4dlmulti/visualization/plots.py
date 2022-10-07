from datetime import datetime, timedelta
import os
import string

from matplotlib import colors, gridspec, lines, patches, pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import numpy as np

from ..analysis import evaluation


def plot_frame(ax, frame, norm=None):
    im = ax.imshow(frame.astype(np.float32), norm=norm)
    ax.tick_params(left=False, bottom=False,
        labelleft=False, labelbottom=False)
    return im


transform_TB = lambda x: x*10+250
transform_radiance = lambda x: x*100
transform_T = lambda x: x*7.2+290
input_transforms = {
    "Rain rate": lambda x: 10**(x*0.528-0.051),
    "CZC": lambda x: x*8.71+21.3,
    "EZC-20": lambda x: x*1.97,
    "EZC-45": lambda x: x*1.97,
    "HZC": lambda x: x*1.97,
    "LZC": lambda x: 10**(x*0.135-0.274),
    "Lightning": lambda x: x,
    "Light. dens.": lambda x: 10**(x*0.640-0.593),
    "Current dens.": lambda x: 10**(x*0.731-0.0718),
    "POH": lambda x: x,
    "$R > 10\\mathrm{mm\\,h^{-1}}$": lambda x: x,
    "HRV": lambda x: x*100,
    "CTH": lambda x: x*2.810+5.260,
    "CAPE-MU": lambda x: x*0.2,
    "CIN-MU": lambda x: x*21,
    "LCL": lambda x: x*1000,
    "MCONV": lambda x: x*3.8,
    "HZEROCL": lambda x: x*3300,
    "OMEGA": lambda x: x*4.2,
    "SLI": lambda x: x*3.5,
    "T-SO": transform_T,
    "T-2M": transform_T,
    "VIS006": transform_radiance,
    "VIS008": transform_radiance,
    "HRV": transform_radiance,
    "IR-016": transform_radiance,
    "IR-039": lambda x: x*17.5+274,
    "WV-062": transform_TB,
    "WV-073": transform_TB,
    "IR-087": transform_TB,
    "IR-097": transform_TB,
    "IR-108": transform_TB,
    "IR-120": transform_TB,
    "IR-134": transform_TB,
    "CTT": lambda x: x*19.1+260,
    "Altitude": lambda x: x * 280,
    "El. EW-deriv.": lambda x: x * 200,
    "El. NS-deriv.": lambda x: x * 200,
    "Solar zen. ang.": lambda x: x * 127
}
input_norm = {
    "Rain rate": colors.LogNorm(0.01, 100, clip=True),
    "LZC": colors.LogNorm(0.75, 100, clip=True),
    "Light. dens.": colors.LogNorm(0.01, 100, clip=True),
    "Current dens.": colors.LogNorm(0.01, 100, clip=True),
    "Lightning": colors.Normalize(0, 1),
    "POH": colors.Normalize(0, 1),
    "$R > 10\\mathrm{mm\\,h^{-1}}$": colors.Normalize(0, 1),
    "HRV": colors.Normalize(0,100),
    "CTH": colors.Normalize(0,12),
    "CAPE-MU": colors.Normalize(0,2)
}
input_ticks = {
    "Rain rate": [0.1, 1, 10, 100],
    "Lightning": [0, 0.5, 1],
    "POH": [0, 0.5, 1],
    "$R > 10\\mathrm{mm\\,h^{-1}}$": [0, 0.5, 1],
    "HRV": [0, 25, 50, 75],
    "CTH": [0, 5, 10],
    "CAPE-MU": [0.5, 1, 1.5, 2],
}


def plot_model_examples(X, Y, models, shown_inputs=(0,25,12,9),
    input_timesteps=(-4,-1), output_timesteps=(0,2,5,11),
    batch_member=0, interval_mins=5,
    input_names=("Rain rate", "Lightning", "HRV", "CTH"),
    future_input_names=("CAPE-MU",),
    min_p=0.025, plot_scale=256
):
    num_timesteps = len(input_timesteps)+len(output_timesteps)
    gs_rows = 2 * max(len(models),len(shown_inputs))
    gs_cols = num_timesteps
    width_ratios = (
        [0.1, 0.19] +
        [1]*len(input_timesteps) +
        [0.1] +
        [1]*len(output_timesteps) +
        [0.19, 0.1]
    )
    gs = gridspec.GridSpec(gs_rows, gs_cols+5, wspace=0.02, hspace=0.05,
        width_ratios=width_ratios)
    batch = [x[batch_member:batch_member+1,...] for x in X]
    obs = [y[batch_member:batch_member+1,...] for y in Y]

    fig = plt.figure(figsize=(gs_cols*1.5, gs_rows/2*1.5))

    # plot inputs
    row0 = gs_rows//2 - len(shown_inputs)
    for (i,k) in enumerate(shown_inputs):
        row = row0 + 2*i        
        ip = batch[k][0,input_timesteps,:,:,0]
        ip = input_transforms[input_names[i]](ip)
        norm = input_norm[input_names[i]]
        for m in range(len(input_timesteps)):
            col = m+2
            ax = fig.add_subplot(gs[row:row+2,col])
            im = plot_frame(ax, ip[m,:,:], norm=norm)
            if i == 0:
                iv = (input_timesteps[m]+1) * interval_mins
                ax.set_title(f"${iv}\\,\\mathrm{{min}}$")
            if m == 0:
                ax.set_ylabel(input_names[i])
                cax = fig.add_subplot(gs[row:row+2,0])                
                cb = plt.colorbar(im, cax=cax)
                cb.set_ticks(input_ticks[input_names[i]])
                cax.yaxis.set_ticks_position('left')

    # plot outputs
    row0 = 0
    future_input_ind = 0
    norm_log = colors.LogNorm(min_p,1,clip=True)
    for (i,model) in enumerate(models):
        if model == "obs":
            Y_pred = obs[0]
            norm = norm_log
            label = "Observed"
        elif isinstance(model, str) and model.startswith("input-future"):
            var_ind = int(model.split("-")[-1])
            Y_pred = batch[var_ind]
            input_name = future_input_names[future_input_ind]
            Y_pred = input_transforms[input_name](Y_pred)
            norm = input_norm[input_name]
            future_input_ind += 1
            label = input_name
        else:
            Y_pred = model.predict(batch)
            norm = norm_log
            label = "Forecast"
        row = row0 + 2*i
        op = Y_pred[0,output_timesteps,:,:,0]        
        for m in range(len(output_timesteps)):
            col = m + len(input_timesteps) + 3
            ax = fig.add_subplot(gs[row:row+2,col])
            im = plot_frame(ax, op[m,:,:], norm=norm)
            if i==0:
                iv = (output_timesteps[m]+1) * interval_mins
                ax.set_title(f"$+{iv}\\,\\mathrm{{min}}$")
            if m == len(output_timesteps)-1:
                ax.yaxis.set_label_position("right")
                ax.set_ylabel(label)
                if i == len(models)-1:
                    scalebar = AnchoredSizeBar(ax.transData,
                           op.shape[1],
                           f'{plot_scale} km',
                           'lower center', 
                           pad=0.1,
                           color='black',
                           frameon=False,
                           size_vertical=1,
                           bbox_transform=ax.transAxes,
                           bbox_to_anchor=(0.5,-0.27)
                    )
                    ax.add_artist(scalebar)

        if i==len(models)-1:
            r0 = row0 + 2*len(future_input_names)
            r1 = r0 + 4
            cax = fig.add_subplot(gs[r0:r1,-1])            
            cb = plt.colorbar(im, cax=cax)
            cb.set_ticks([min_p, 0.05, 0.1, 0.2, 0.5, 1])
            cb.set_ticklabels([min_p, 0.05, 0.1, 0.2, 0.5, 1])
            cax.set_xlabel("$p$", fontsize=12)
        elif i<len(future_input_names):
            cax = fig.add_subplot(gs[row:row+2,-1])            
            cb = plt.colorbar(im, cax=cax)
            cb.set_ticks(input_ticks[input_name])

    return fig


source_colors = {
    "r": "tab:blue",
    "n": "tab:orange",
    "s": "tab:green",
    "l": "tab:purple",
    "d": "tab:brown"
}


notation = {
    "r": "Rad",
    "l": "Lig",
    "s": "Sat",        
    "n": "NWP",
    "d": "DEM"
}
prefix_notation = {
    "lightning": "Lightning",
    "hail": "Hail",
    "rain": "Precipitation"
}


def exclusion_plot(metrics, metrics_names, fig=None, axes=None,
    variable_names=None, subplot_index=0, significant_digits=3):

    import seaborn as sns

    metric_notation = {
        "binary": "Error rate",
        "cross_entropy": "CE",
        "mae": "MAE",
        "rmse": "RMSE",
        "FL2": "FL $\\gamma=2$"
    }

    prefixes_names = [prefix_notation[k] for k in metrics]
    metrics_tables = {prefix: np.full((8,4), np.nan) for prefix in metrics}
    metric_pos = {
        frozenset(("n", "l", "d", "r", "s")): (0,0),
        frozenset(("n", "l", "d", "r")): (0,1),
        frozenset(("n", "l", "d", "s")): (0,2),
        frozenset(("n", "l", "d")): (0,3),

        frozenset(("n", "l", "r", "s")): (1,0),
        frozenset(("n", "l", "r")): (1,1),
        frozenset(("n", "l", "s")): (1,2),
        frozenset(("n", "l")): (1,3),

        frozenset(("n", "d", "r", "s")): (2,0),
        frozenset(("n", "d", "r")): (2,1),
        frozenset(("n", "d", "s")): (2,2),
        frozenset(("n", "d")): (2,3),

        frozenset(("l", "d", "r", "s")): (3,0),
        frozenset(("l", "d", "r")): (3,1),
        frozenset(("l", "d", "s")): (3,2),
        frozenset(("l", "d")): (3,3),

        frozenset(("n", "r", "s")): (4,0),
        frozenset(("n", "r")): (4,1),
        frozenset(("n", "s")): (4,2),
        frozenset(("n",)): (4,3),

        frozenset(("l", "r", "s")): (5,0),
        frozenset(("l", "r")): (5,1),
        frozenset(("l", "s")): (5,2),
        frozenset(("l",)): (5,3),

        frozenset(("d", "r", "s")): (6,0),
        frozenset(("d", "r")): (6,1),
        frozenset(("d", "s")): (6,2),
        frozenset(("d",)): (6,3),

        frozenset(("r", "s")): (7,0),
        frozenset(("r",)): (7,1),
        frozenset(("s",)): (7,2),
        frozenset(()): (7,3),
    }
    metric_pos_inv = {v: k for (k, v) in metric_pos.items()}

    for prefix in metrics:
        for subset in metrics[prefix]:
            subset_frozen = frozenset(subset)
            (i,j) = metric_pos[subset_frozen]
            metrics_tables[prefix][i,j] = metrics[prefix][subset]

    xlabels_show = frozenset(("r", "s"))
    ylabels_show = frozenset(("n", "l", "d"))

    with sns.plotting_context("paper"):
        if fig is None:
            fig = plt.figure(figsize=(3.125*len(metrics),7.5))
        
        for (i,prefix) in enumerate(metrics):
            xlabels = [
                "\n".join(sorted(notation[s] for s in metric_pos_inv[0,i] & xlabels_show))
                for i in range(metrics_tables[prefix].shape[1])
            ]
            ylabels = [
                "\n".join(sorted(notation[s] for s in metric_pos_inv[i,0] & ylabels_show))
                for i in range(metrics_tables[prefix].shape[0])
            ]

            ax = axes[i] if (axes is not None) else fig.add_subplot(1,len(metrics),i+1)
            heatmap = sns.heatmap(
                metrics_tables[prefix],
                xticklabels=xlabels,
                yticklabels=ylabels,
                annot=True,
                fmt='#.{}g'.format(significant_digits),
                square=True,
                ax=ax,
                cbar_kws={"orientation": "horizontal"}
            )
            heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=90, ha='right')
            heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0, ha='right')
            ax.set_title("({}) {}{}".format(
                string.ascii_lowercase[i+subplot_index],
                prefixes_names[i]+" " if prefixes_names[i] else "",
                metric_notation[metrics_names[i]]+" " if metrics_names[i] else "",
            ))
            ax.tick_params(axis='both', bottom=False, left=False,
                labelleft=(i+subplot_index==0))

    return fig


def shapley_by_time(
        leadtimes,
        shapley_values,
        interval=timedelta(minutes=5),
        fig=None,
        ax=None,
        legend=True,
    ):

    interval_mins = interval.total_seconds() / 60
    leadtimes = leadtimes * interval_mins
    
    if ax is None:
        fig = plt.figure(figsize=(6,3))
        ax = fig.add_subplot()

    val_sum = None
    for values in shapley_values.values():
        if val_sum is None:
            val_sum = values.copy()
        else:
            val_sum += values

    for (source, values) in shapley_values.items():
        ax.plot(
            leadtimes, values/val_sum, linewidth=1,
            label=notation[source], c=source_colors[source]
        )
    if legend:
        ax.legend()
    ax.set_xlim((0, leadtimes[-1]))
    ax.set_xlabel("Lead time [min]")
    ax.set_ylabel("Normalized Shapley value")

    return fig

def shapley_values_full_legend(shapley_values_full, ax):
    val_sum_full = sum(shapley_values_full.values())
    labels = [
        f"{notation[s]}: {shapley_values_full[s]/val_sum_full:.03f}"
        for s in shapley_values_full
    ]
    custom_lines = [
        lines.Line2D([0], [0], color=source_colors[s], lw=1)
        for s in shapley_values_full
    ]
    ax.legend(custom_lines, labels, ncol=3, mode="expand")
