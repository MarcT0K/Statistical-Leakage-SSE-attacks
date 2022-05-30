#!/usr/bin/python3
# pylint: disable=invalid-name

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, QuantileRegressor

epsilon_sim = lambda coocc_1, coocc_2: np.linalg.norm(coocc_1 - coocc_2)

params = {
    "text.usetex": True,
    "font.family": "sans-serif",
    "text.latex.preamble": [r"\usepackage{amssymb}", r"\usepackage{amsmath}"],
    "font.size": 22,
    "hatch.linewidth": 2,
    "hatch.color": "white",
}
plt.rcParams.update(params)


def fig_subsec_4C(show=True):
    with open("fig_subsec_4C.csv", "r", encoding="utf-8") as csvfile:
        csvfile.readline()
        arr = np.loadtxt(csvfile, delimiter=",")

    x = np.sqrt(1 / arr[:, 0] + 1 / arr[:, 1])
    y = arr[:, 2]
    a, b = np.polyfit(x, y, 1)
    plt.scatter(x, y)
    plt.axline((0, b), slope=a, color="red")
    plt.xlabel(r"$\sqrt{\frac{1}{n_{atk}}+\frac{1}{n_{ind}}}$")
    plt.ylabel(r"$\epsilon$-similarity")
    plt.text(0.0005, 3.7, f"Slope: {a:.2f}\nIntercept: {b:.4f}")
    plt.grid()
    plt.tight_layout()
    plt.savefig("fig_subsec_4C.png", dpi=400)
    if show:
        plt.show()

    mask = arr[:, 0] == 2023  # n_atk fixed
    x = 1 / arr[mask, 1]
    y = arr[mask, 2]
    a, b = np.polyfit(x, y, 1)
    plt.scatter(x, y)
    plt.axline((0, b), slope=a, color="red")
    plt.xlabel(r"$\frac{1}{n_{ind}}$, Fixed $n_{atk}$=2K")
    plt.ylabel(r"$\epsilon$-similarity")
    plt.text(0.00001, 2.8, f"Slope: {a:.2f}\nIntercept: {b:.4f}")
    plt.grid()
    plt.tight_layout()
    plt.savefig("fig_subsec_4C_n_atk_fixed.png", dpi=400)
    if show:
        plt.show()

    mask = arr[:, 1] == 1820  # n_ind fixed
    x = 1 / arr[mask, 0]
    y = arr[mask, 2]
    a, b = np.polyfit(x, y, 1)
    plt.scatter(x, y)
    plt.axline((0, b), slope=a, color="red")
    plt.xlabel(r"$\frac{1}{n_{atk}}$, Fixed $n_{ind}$=1.8K")
    plt.ylabel(r"$\epsilon$-similarity")
    plt.text(0.00001, 3.5, f"Slope: {a:.2f}\nIntercept: {b:.4f}")
    plt.grid()
    plt.tight_layout()
    plt.savefig("fig_subsec_4C_n_ind_fixed.png", dpi=400)
    if show:
        plt.show()


def fig_risk_assessment():
    dataframe = pd.read_csv("risk_assessment.csv")
    logit = lambda p: np.log(p / (1 - p))
    posit = lambda alpha: np.exp(alpha) / (1 + np.exp(alpha))

    # Linear vs. Quantile regression comparison
    x = np.log(1 / dataframe["Nb server docs"] + 1 / dataframe["Nb similar docs"])
    y = logit(dataframe["IHOP Acc"])

    mask = y != np.inf
    assert sum(~mask) < 10  # To avoid removing too many points

    y = y[mask].to_numpy()
    x = x[mask].to_numpy()
    plt.scatter(x, y)

    linear_regression = LinearRegression().fit(x.reshape(-1, 1), y.reshape(-1, 1))
    lin_slope = linear_regression.coef_[0, 0]
    lin_intercept = linear_regression.intercept_[0]
    plt.axline((x[0], x[0] * lin_slope + lin_intercept), slope=lin_slope, color="red")

    quant_regression = QuantileRegressor(quantile=0.9, alpha=0).fit(x.reshape(-1, 1), y)
    quant_slope = quant_regression.coef_[0]
    quant_intercept = quant_regression.intercept_
    plt.axline(
        (x[0], x[0] * quant_slope + quant_intercept),
        slope=quant_slope,
        color="green",
    )
    plt.show()

    x = 1 / dataframe["Nb server docs"] + 1 / dataframe["Nb similar docs"]
    y = dataframe["IHOP Acc"]
    x_pred = np.arange(x.min(), x.max(), (x.max() - x.min()) / 500)
    y_lin = posit(lin_slope * np.log(x_pred) + lin_intercept)
    y_quant = posit(quant_slope * np.log(x_pred) + quant_intercept)
    plt.scatter(x, y)
    plt.plot(x_pred, y_lin, color="red")
    plt.plot(x_pred, y_quant, color="green")
    # Score+RefinedScore+IHOP risk assessment


if __name__ == "__main__":
    if not os.path.exists("results"):
        raise OSError("No result directory found.")
    os.chdir("results")

    # Call all functions defined in this file
    fig_subsec_4C(False)
