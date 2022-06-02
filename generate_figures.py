#!/usr/bin/python3
# pylint: disable=invalid-name

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cycler import cycler
from sklearn.linear_model import LinearRegression, QuantileRegressor

epsilon_sim = lambda coocc_1, coocc_2: np.linalg.norm(coocc_1 - coocc_2)

params = {
    "text.usetex": True,
    "text.latex.preamble": [r"\usepackage{amssymb}", r"\usepackage{amsmath}"],
    "font.size": 15,
    "axes.labelsize": 25,
    "axes.grid": True,
    "grid.linestyle": "dashed",
    "grid.alpha": 0.7,
    "scatter.marker": "x",
}
plt.style.use("seaborn-colorblind")
plt.rc(
    "axes",
    prop_cycle=(
        plt.rcParams["axes.prop_cycle"]
        + cycler("linestyle", ["-", "--", "-.", ":", "-", "-"])
    ),
)
plt.rcParams.update(params)


def fig_epsilon_nb_docs():
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
    plt.savefig("epsilon_nb_docs.png", dpi=400)
    plt.cla()

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
    plt.savefig("epsilon_n_atk_fixed.png", dpi=400)
    plt.cla()

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
    plt.savefig("epsilon_n_ind_fixed.png", dpi=400)
    plt.cla()


logit = lambda p: np.log(p / (1 - p))
posit = lambda alpha: np.exp(alpha) / (1 + np.exp(alpha))
QUANTILE = 0.95


def data_to_xy(dataframe, col_name):
    x = np.log(1 / dataframe["Nb server docs"] + 1 / dataframe["Nb similar docs"])
    y = logit(dataframe[col_name])

    mask = y != np.inf
    assert sum(~mask) < 10  # To avoid removing too many points

    y = y[mask].to_numpy()
    x = x[mask].to_numpy()
    return x, y


def data_to_quant_reg(dataframe, col_name, quantile=QUANTILE):
    x, y = data_to_xy(dataframe, col_name)
    quant_regression = QuantileRegressor(quantile=quantile, alpha=0).fit(
        x.reshape(-1, 1), y
    )
    slope = quant_regression.coef_[0]
    intercept = quant_regression.intercept_
    return slope, intercept


def fig_indiv_risk_assessment(col_name):
    fig, ax = plt.subplots()
    dataframe = pd.read_csv("risk_assessment.csv")

    log_x, log_y = data_to_xy(dataframe, col_name)
    linear_regression = LinearRegression().fit(
        log_x.reshape(-1, 1), log_y.reshape(-1, 1)
    )
    lin_slope = linear_regression.coef_[0, 0]
    lin_intercept = linear_regression.intercept_[0]

    x = 1 / dataframe["Nb server docs"] + 1 / dataframe["Nb similar docs"]
    y = dataframe[col_name]
    step_size = x.max() / 500
    x_pred = np.arange(step_size, x.max(), step_size)

    # Compute the predictions
    log_y_lin = lin_slope * np.log(x_pred) + lin_intercept
    y_lin = posit(log_y_lin)

    quant095_slope, quant095_intercept = data_to_quant_reg(dataframe, col_name)
    log_y_quant095 = quant095_slope * np.log(x_pred) + quant095_intercept
    y_quant095 = posit(log_y_quant095)

    quant075_slope, quant075_intercept = data_to_quant_reg(dataframe, col_name, 0.75)
    log_y_quant075 = quant075_slope * np.log(x_pred) + quant075_intercept
    y_quant075 = posit(log_y_quant075)

    # Visualization in  the log-log space
    ax.scatter(log_x, log_y, color="black", alpha=0.5, label="Observations")
    ax.plot(np.log(x_pred), log_y_lin, label="Linear")
    ax.plot(np.log(x_pred), log_y_quant075, label="Quantile 0.75")
    ax.plot(np.log(x_pred), log_y_quant095, label="Quantile 0.95")
    ax.set(
        xlabel=r"$\log(\frac{1}{n_\mathrm{atk}}+\frac{1}{n_\mathrm{ind}})$",
        ylabel=r"$\mathrm{logit}(\mathrm{Accuracy})$",
    )
    ax.set_xlim((log_x.min() - 0.1, log_x.max() + 0.1))
    ax.set_ylim((log_y.min() - 0.1, log_y.max() + 0.1))
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"risk_assess_{(col_name.replace(' ','_'))}_log.png", dpi=400)
    plt.cla()

    # Visualization in  the standard space
    ax.scatter(x, y, color="black", alpha=0.5, label="Observations")
    ax.plot(x_pred, y_lin, label="Linear")
    ax.plot(x_pred, y_quant075, label="Quantile 0.75")
    ax.plot(x_pred, y_quant095, label="Quantile 0.95")
    ax.set(
        xlabel=r"$\frac{1}{n_\mathrm{atk}}+\frac{1}{n_\mathrm{ind}}$",
        ylabel="Accuracy",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"risk_assess_{(col_name.replace(' ','_'))}.png", dpi=400)


def fig_comp_risk_assessment():
    fig, ax = plt.subplots()
    dataframe = pd.read_csv("risk_assessment.csv")
    # Score+RefinedScore+IHOP risk assessment
    x = 1 / dataframe["Nb server docs"] + 1 / dataframe["Nb similar docs"]
    step_size = x.max() / 500
    x_pred = np.arange(step_size, x.max(), step_size)

    ihop_quant_slope, ihop_quant_intercept = data_to_quant_reg(dataframe, "IHOP Acc")
    y_quant_ihop = posit(ihop_quant_slope * np.log(x_pred) + ihop_quant_intercept)
    ref_quant_slope, ref_quant_intercept = data_to_quant_reg(
        dataframe, "Refined Score Acc"
    )
    score_quant_slope, score_quant_intercept = data_to_quant_reg(dataframe, "Score Acc")
    y_quant_ref = posit(ref_quant_slope * np.log(x_pred) + ref_quant_intercept)
    y_quant_score = posit(score_quant_slope * np.log(x_pred) + score_quant_intercept)
    ax.plot(x_pred, y_quant_ihop, label="IHOP")
    ax.plot(x_pred, y_quant_ref, label="Refined Score")
    ax.plot(x_pred, y_quant_score, label="Score")
    ax.legend()
    ax.set(
        xlabel=r"$\frac{1}{n_\mathrm{atk}}+\frac{1}{n_\mathrm{ind}}$",
        ylabel="Accuracy",
    )
    fig.tight_layout()
    fig.savefig("risk_assess_comparison.png", dpi=400)


if __name__ == "__main__":
    if not os.path.exists("results"):
        raise OSError("No result directory found.")
    os.chdir("results")

    # Call all functions defined in this file
    # fig_epsilon_nb_docs()
    fig_indiv_risk_assessment("IHOP Acc")
    fig_indiv_risk_assessment("Refined Score Acc")
    fig_indiv_risk_assessment("Score Acc")
    fig_comp_risk_assessment()
