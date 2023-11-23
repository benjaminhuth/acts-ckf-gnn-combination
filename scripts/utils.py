import numpy as np
import matplotlib.pyplot as plt
import ROOT


class TEfficiency:
    def __init__(self, tefficency):
        th1 = tefficency.GetTotalHistogram()

        bins = [i for i in range(th1.GetNbinsX()) if th1.GetBinContent(i) > 0.0]

        self.x = [th1.GetBinCenter(i) for i in bins]

        self.x_lo = [th1.GetBinLowEdge(i) for i in bins]
        self.x_width = [th1.GetBinWidth(i) for i in bins]
        self.x_hi = np.add(self.x_lo, self.x_width)
        self.x_err_lo = np.subtract(self.x, self.x_lo)
        self.x_err_hi = np.subtract(self.x_hi, self.x)

        self.y = [tefficency.GetEfficiency(i) for i in bins]
        self.y_err_lo = [tefficency.GetEfficiencyErrorLow(i) for i in bins]
        self.y_err_hi = [tefficency.GetEfficiencyErrorUp(i) for i in bins]

    def errorbar(self, ax, **errorbar_kwargs):
        ax.errorbar(
            self.x, self.y, yerr=(self.y_err_lo, self.y_err_hi), xerr=(self.x_err_lo, self.x_err_hi), **errorbar_kwargs
        )
        return ax

    def step(self, ax, **step_kwargs):
        ax.step(self.x_hi,self.y, **step_kwargs)
        return ax


    def bar(self, ax, **bar_kwargs):
        ax.bar(self.x, height=self.y, yerr=(self.y_err_lo, self.y_err_hi), **bar_kwargs)
        return ax


def replace_key_text(key):
    replace_dict = {
        # "eta": "$\eta$",
        # "pT": "$p_T$",
        "vs_eta": "",
        "vs_pT": "",
        "duplicationRate" : "Duplication rate",
        "fakerate" : "Fake rate",
        "trackeff" : "Matching efficiency",
    }

    for a, b in replace_dict.items():
        key = key.replace(a, b)
    key = key.replace("_", " ")
    return key


def plot_perf_root_file(axes, plot_keys, performance_file, label, color, eff_min=0.0, fake_max=1.0):
    # performance_file = ROOT.TFile.Open(input_file)

    # name = Path(input_file).name

    for ax, key in zip(axes, plot_keys):
        teff = TEfficiency(performance_file.Get(key))

        teff.errorbar(
            ax,
            fmt="none",
            color=color,
            label=label,
        )
        teff.step(
            ax, color=color,
        )

        if "_pT" in key:
            ax.set_xscale("log")
            ax.set_xlim(0.9e0, 1.1e2)
            ax.set_xticks([1.0, 3.0, 10.0, 30.0, 100.0])
            ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            ax.set_xlabel("$p_T$ [GeV]")
        elif "_eta" in key:
            ax.set_xlabel("$\eta$")
            ax.set_xlim(-3.5, 3.5)

        ax.set_title(replace_key_text(key))
        if "eff" in key:
            ax.set_ylim(eff_min, 1)
        elif "fake" in key:
            ax.set_ylim(0,fake_max)
        else:
            ax.set_ylim(0,1)

        # if "fake" in key:
        #     ax.legend(loc="upper right")
        # else:
        #     ax.legend(loc="lower left")




def subplots(nrow, ncol, snakemake):
    # plt.rcParams.update({"font.size": snakemake.config["plot_fontsize"]})
    figsize = (
        ncol * snakemake.config["plot_width"],
        nrow * snakemake.config["plot_height"],
    )
    return plt.subplots(nrow, ncol, figsize=figsize)
