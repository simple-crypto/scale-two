import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib.colors import LogNorm, NoNorm
from matplotlib.ticker import LogFormatterSciNotation, LogLocator
from matplotlib.tri import Triangulation
from matplotlib.patches import Rectangle

import numpy as np
import math

from utils_scale import utils_eval, utils_ta

def display_snr_SBout(snrs, traces):
    mtraces = np.mean(traces,axis=0)

    f = plt.figure()
    ax0 = f.add_subplot(2,1,1)
    ax1 = f.add_subplot(2,1,2)
    ax0.plot(traces.T, color="k")
    ax0.plot(mtraces, color="r", linestyle="dashed", label="Mean")
    ax0.set_ylabel("Power")
    ax0.legend()
    ax1.plot(snrs.T)
    ax1.set_xlabel("Time")
    ax1.set_ylabel("SNR")

def display_snr_results(snrs, traces):
    display_snr_SBout(snrs,traces)

def make_single_heatmap(ax, npois, pndims, byte, pi, ti, cmap, norm, show_xaxis, show_yaxis):
    # Display the title
    axtitle = "Byte {}".format(byte)
    #ax.set_title(axtitle)
    
    m = pi.shape[1]
    n = pi.shape[0]
    x = np.arange(m + 1)
    y = np.arange(n + 1)

    xs, ys = np.meshgrid(x, y)
    squares = [
        (
            i + j * (m + 1),
            i + 1 + j * (m + 1),
            i + 1 + (j + 1) * (m + 1),
            i + (j + 1) * (m + 1),
        )
        for j in range(n)
        for i in range(m)
    ]
    tri_up = [(bxby, bxty, txty) for (bxby, txby, txty, bxty) in squares]
    tri_down = [(bxby, txty, txby) for (bxby, txby, txty, bxty) in squares]
    
    imgs = []
    for tri, z in [(tri_up, ti), (tri_down, pi)]:
        tri = Triangulation(ys.ravel(), xs.ravel(), tri)
        imgs.append(ax.tripcolor(tri, z.ravel(), cmap=cmap, norm=norm))
    
    # Draw square around max pi
    (max_x, max_y) = np.unravel_index(np.argmax(pi), pi.shape)
    ax.add_patch(Rectangle((max_x, max_y), 1, 1, edgecolor="r", facecolor="none"))
    
    # Axis configuration
    ax.invert_yaxis()
    ax.margins(0)
    if show_xaxis:
        xlabels = [str(e) for e in npois]
        ax.set_xticks(0.5 + np.arange(n), labels=xlabels, rotation=45)
        # ax.set_xlabel("POIs amount")
    else:
        ax.set_xticks([], labels=[])
    if show_yaxis:
        ylabels = [str(e) for e in pndims]
        ax.set_yticks(0.5 + np.arange(m), labels=ylabels)
        # ax.set_ylabel(r"$p$")
    else:
        ax.set_yticks([], labels=[])
    return imgs


def make_heatmap(res_explo):
    (npois, ndims, pi, ti) = res_explo
    lvars = len(pi)

    # Create the colormap
    cmap_im = mpl.colormaps.get_cmap("viridis")
    cmap_im.set_bad(color="red")

    # Skip all-nan results.
    if np.isnan(pi).all() and np.isnan(ti).all():
        print('ok')
    # Recover bounds for valid data
    maxv = np.nanmax([pi, ti])
    minv = np.nanmin([pi, ti])
    minv = np.nanmax(
        [minv, maxv / 100]
    )  # Below some threshold, PI might as well just be 0.
    norm_im = LogNorm(vmin=minv, vmax=maxv)

    # Create figure
    f = plt.figure(figsize=(7, 5))

    if lvars == 1:
        axes = [[f.add_subplot(1,1,1)]]
    else:
        axes = f.subplots(4, 4)
    print(npois, ndims)


    # Some global configuration
    imgs = []

    # Enumerate over all the vaiables
    for byte in range(lvars):
        # Create the ax for the variable index
        ax_y = byte % 4
        ax_x = byte // 4
        # ax = f.add_subplot(spec[ax_y, ax_x])
        ax = axes[ax_y][ax_x]
        img = make_single_heatmap(
            ax,
            npois,
            ndims,
            byte,
            pi[byte],
            ti[byte],
            cmap_im,
            norm_im,
            show_xaxis=(byte % 4 == 3) or lvars==1,
            show_yaxis=byte < 4,
        )
        # Append objects for post-processing
        imgs.append(img)

    f.supxlabel("Number of POIs")
    f.supylabel("$p$")

    # Colorbar
    cbar_axes = f.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar_axes.set_label(r"$\text{log}_{2}[PI]$")
    cbar = f.colorbar(
        imgs[0][0],
        cax=cbar_axes,
        ticks=LogLocator(base=2),
        format=LogFormatterSciNotation(base=2.0),
    )
    #cbar_axes.yaxis.minorticks_off()

    f.subplots_adjust(
        left=0.08, right=0.90, bottom=0.13, top=0.99, wspace=0.04, hspace=0.04
    )


MY_COLORS = [
        "xkcd:blue",
        "xkcd:green",
        "xkcd:red",
        "xkcd:orange",
        "xkcd:pink",
        ]

class ITDisplayEntry:
    def __init__(self, results, modelID):
        assert results["dtype"] in ["PI", "TI"], "Dtype not handled"
        assert "qt_s" in results, "Provided results do not contain 'qt_s' value."
        assert "it" in results, "Provided results do not contain 'it' value"
        self.res = results
        self.mID = modelID
        self.color= None
        self._set_linestyle()
        self._set_label()

    def _set_linestyle(self):
        if self.res["dtype"]=="PI":
            self.linestyle = "solid"
        elif self.res["dtype"]=="TI":
            self.linestyle = "dashed"
    
    def _set_label(self):
        self.label = "{} {}".format(self.res["dtype"], self.mID)

    def _set_color(self, color):
        self.color=color


class ITDisplayConFig:
    def __init__(self, config_list):
        self.map_model = {}
        self.am_id = 0
        self.n_p = None
        self.cfgs = []
        for c in config_list:
            if self.am_id == 0:
                self.n_p= c.res["it"].shape[0]
            else:
                assert c.res["it"].shape[0]==self.n_p, "All the provided results must be generated for the same amount on intermediate state"
            if c.mID in self.map_model:
                c._set_color(self.map_model[c.mID][0].color)
                self.map_model[c.mID].append(c)
                self.cfgs.append(c)
            else:
                c._set_color(MY_COLORS[self.am_id % len(MY_COLORS)])
                self.map_model[c.mID] = [c]
                self.cfgs.append(c)
                self.am_id += 1

    def models2plot(self):
        return self.map_model


def Idx2AxeIdx(i):
    m4 = i % 4
    d4 = i // 4
    return 4*m4 + d4 + 1

def display_IT_results(byte_indexes, list_tuples_cfg, scale=1.0, disable_legend=False, show=False):
    # Create the config
    list_config = [ITDisplayEntry(e,k) for (k,e) in list_tuples_cfg]
    display_config = ITDisplayConFig(list_config)
    # Shape 
    n_p = display_config.n_p
    # Plot the res
    ax_sx_inch = 5
    ax_sy_inch = 3
    figsize=(4*ax_sx_inch*scale, 4*ax_sy_inch*scale)
    f = plt.figure(figsize=figsize)
    axes = []
    for i in byte_indexes:
        axes.append(f.add_subplot(4,4,Idx2AxeIdx(i)))
        # Iterate over the config
        for cfg in display_config.cfgs:
            if cfg.res["mB"] is not None:
                axes[i].fill_between(cfg.res["qt_s"], cfg.res["mB"][i,:], cfg.res["MB"][i,:], color=cfg.color, alpha=0.3)
            axes[i].plot(cfg.res["qt_s"], cfg.res["it"][i,:], color=cfg.color, linestyle=cfg.linestyle, label=cfg.label)
        if not(disable_legend):
            axes[i].legend()
        axes[i].set_ylabel("IT metric [bits]")
        axes[i].set_xlabel("training data complexity")
        axes[i].set_title("byte index {}".format(i))
    # Global 
    f.tight_layout()

def display_ranks_full_key(TA_results, byte_indexes):
    utils_eval.display_rank_esti_full_key_axes(TA_results, byte_indexes, utils_ta.key_rank_approximation_scalib)
