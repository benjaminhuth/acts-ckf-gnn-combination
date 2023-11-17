import numpy as np
import matplotlib.pyplot as plt


def plotTEfficency(tefficency, ax, no_yerr=False, **errorbar_kwargs):
    th1 = tefficency.GetTotalHistogram()

    bins = [i for i in range(th1.GetNbinsX()) if th1.GetBinContent(i) > 0.0]

    x = [th1.GetBinCenter(i) for i in bins]

    x_lo = [th1.GetBinLowEdge(i) for i in bins]
    x_width = [th1.GetBinWidth(i) for i in bins]
    x_hi = np.add(x_lo, x_width)
    x_err_lo = np.subtract(x, x_lo)
    x_err_hi = np.subtract(x_hi, x)

    y = [tefficency.GetEfficiency(i) for i in bins]
    y_err_lo = [tefficency.GetEfficiencyErrorLow(i) for i in bins]
    y_err_hi = [tefficency.GetEfficiencyErrorUp(i) for i in bins]

    if no_yerr:
        y_err_lo = len(bins)*[0]
        y_err_hi = len(bins)*[0]

    ax.errorbar(
        x, y, yerr=(y_err_lo, y_err_hi), xerr=(x_err_lo, x_err_hi), **errorbar_kwargs
    )
    return ax


def subplots(nrow, ncol, snakemake):
    # plt.rcParams.update({"font.size": snakemake.config["plot_fontsize"]})
    figsize = (
        ncol * snakemake.config["plot_width"],
        nrow * snakemake.config["plot_height"],
    )
    return plt.subplots(nrow, ncol, figsize=figsize)
