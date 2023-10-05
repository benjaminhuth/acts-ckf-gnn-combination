import numpy as np

def plotTEfficency(tefficency, ax, **errorbar_kwargs):
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

    ax.errorbar(
        x, y, yerr=(y_err_lo, y_err_hi), xerr=(x_err_lo, x_err_hi), **errorbar_kwargs
    )
    return ax
