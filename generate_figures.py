#!/usr/bin/python3
# pylint: disable=invalid-name

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cycler import cycler
from sklearn.linear_model import LinearRegression, QuantileRegressor
import scipy.interpolate as interpolate

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
    fig, ax = plt.subplots()

    with open("similarity_exploration.csv", "r", encoding="utf-8") as csvfile:
        csvfile.readline()
        arr = np.loadtxt(csvfile, delimiter=",")

    x = np.sqrt(1 / arr[:, 0] + 1 / arr[:, 1])
    y = arr[:, 2]
    slope, intercept = np.polyfit(x, y, 1)
    ax.scatter(x, y, color="black", alpha=0.5, label="Observations")
    ax.axline(
        (0, intercept),
        slope=slope,
        label=r"Linear reg. ($y = bx + a$)",
    )
    ax.plot([], [], " ", label=rf"$b={slope:.2f}$ / $a={intercept:.3f}$")
    ax.set(
        xlabel=r"$\sqrt{\frac{1}{n_{atk}}+\frac{1}{n_{ind}}}$",
        ylabel=(r"$\epsilon$-similarity"),
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig("epsilon_nb_docs.png", dpi=400)
    plt.cla()

    mask = arr[:, 0] == 2022  # n_atk fixed
    x = np.sqrt(1 / arr[mask, 1])
    y = arr[mask, 2]
    slope, intercept = np.polyfit(x, y, 1)
    ax.scatter(x, y, color="black", alpha=0.5, label="Observations")
    ax.axline(
        (0, intercept),
        slope=slope,
        label=r"Linear reg. ($y = bx + a$)",
    )
    ax.set(
        xlabel=r"$\sqrt{\frac{1}{n_{ind}}}$, Fixed $n_{atk}$=2K",
        ylabel=r"$\epsilon$-similarity",
    )
    ax.plot([], [], " ", label=rf"$b={slope:.2f}$ / $a={intercept:.3f}$")
    ax.legend()
    fig.tight_layout()
    fig.savefig("epsilon_n_atk_fixed.png", dpi=400)
    plt.cla()

    mask = arr[:, 1] == 1820  # n_ind fixed
    x = np.sqrt(1 / arr[mask, 0])
    y = arr[mask, 2]
    slope, intercept = np.polyfit(x, y, 1)
    ax.scatter(x, y, color="black", alpha=0.5, label="Observations")
    ax.axline(
        (0, intercept),
        slope=slope,
        label=r"Linear reg. ($y = bx + a$)",
    )
    ax.plot([], [], " ", label=rf"$b={slope:.2f}$ / $a={intercept:.3f}$")
    ax.set(
        xlabel=r"$\sqrt{\frac{1}{n_{atk}}}$, Fixed $n_{ind}$=1.8K",
        ylabel=r"$\epsilon$-similarity",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig("epsilon_n_ind_fixed.png", dpi=400)
    plt.cla()


logit = lambda p: np.log(p / (1 - p))
posit = lambda alpha: np.exp(alpha) / (1 + np.exp(alpha))
QUANTILE = 0.95


def data_to_xy(dataframe, col_name):
    col = dataframe[col_name]
    mask = np.logical_and(col != 0, col != 1.0)  # To avoid a warning on the logit
    assert sum(~mask) < 1 / 3 * col.shape[0]

    x = np.log(1 / dataframe["Nb server docs"] + 1 / dataframe["Nb similar docs"])
    x = x[mask]
    y = logit(col[mask])

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


def fig_attack_analysis(dataset_name):
    fig, ax = plt.subplots()
    dataframe = pd.read_csv(f"{dataset_name}_results.csv")

    x = dataframe["Epsilon"]
    y = dataframe["Refined Score Acc"]

    mask = np.logical_and(y != 0, y != 1.0)
    assert sum(~mask) < 1 / 3 * y.shape[0]
    y = y[mask]
    x = x[mask]

    log_x = np.array(np.log(x))
    log_y = np.array(logit(y))

    linear_regression = LinearRegression().fit(
        log_x.reshape(-1, 1), log_y.reshape(-1, 1)
    )
    lin_slope = linear_regression.coef_[0, 0]
    lin_intercept = linear_regression.intercept_[0]

    step_size = x.max() / 500
    x_pred = np.arange(step_size, x.max(), step_size)

    # Compute the predictions
    log_y_lin = lin_slope * np.log(x_pred) + lin_intercept
    y_lin = posit(log_y_lin)

    # Visualization in  the logit space
    ax.scatter(x, log_y, color="black", alpha=0.5, label="Observations")
    ax.set(
        xlabel=r"$\epsilon$-similarity",
        ylabel=r"$\mathrm{logit}(\mathrm{Accuracy})$",
    )

    ax.set_ylim((log_y.min() - 0.5, log_y.max() + 0.5))
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"atk_analysis_{dataset_name}_logit.png", dpi=400)
    plt.cla()

    # Visualization in  the logit-log space
    ax.scatter(log_x, log_y, color="black", alpha=0.5, label="Observations")
    ax.plot(np.log(x_pred), log_y_lin, label=r"Linear reg. ($y = bx + a$)")
    ax.plot([], [], " ", label=rf"$b={lin_slope:.2f}$ / $a={lin_intercept:.3f}$")
    ax.set(
        xlabel=r"$\log(\epsilon$-similarity$)$",
        ylabel=r"$\mathrm{logit}(\mathrm{Accuracy})$",
    )
    ax.set_xlim((log_x.min() - 0.1, log_x.max() + 0.1))
    ax.set_ylim((log_y.min() - 0.5, log_y.max() + 0.5))
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"atk_analysis_{dataset_name}_logit-log.png", dpi=400)
    plt.cla()

    # Visualization in  the standard space
    ax.scatter(x, y, color="black", alpha=0.5, label="Observations")
    ax.plot(x_pred, y_lin, label="Regression result")
    ax.set(
        xlabel=r"$\epsilon$-similarity",
        ylabel="Accuracy",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"atk_analysis_{dataset_name}.png", dpi=400)


def fig_comparison_atk():
    def data_to_lin_reg(dataframe, col_name):
        x = np.log(dataframe["Epsilon"])
        y = logit(dataframe[col_name])
        mask = abs(y) != np.inf
        assert sum(~mask) < 1 / 3 * y.shape[0]  # To avoid removing too many points
        y = y[mask].to_numpy()
        x = x[mask].to_numpy()

        linear_regression = LinearRegression().fit(x.reshape(-1, 1), y.reshape(-1, 1))
        slope = linear_regression.coef_[0, 0]
        intercept = linear_regression.intercept_[0]
        return slope, intercept

    fig, ax = plt.subplots()
    dataframe = pd.read_csv("atk_comparison.csv")
    # Score+RefinedScore+IHOP risk assessment
    x = dataframe["Epsilon"]
    step_size = x.max() / 500
    x_pred = np.arange(step_size, x.max(), step_size)

    ihop_quant_slope, ihop_quant_intercept = data_to_lin_reg(dataframe, "IHOP Acc")
    y_quant_ihop = posit(ihop_quant_slope * np.log(x_pred) + ihop_quant_intercept)
    ref_quant_slope, ref_quant_intercept = data_to_lin_reg(
        dataframe, "Refined Score Acc"
    )
    score_quant_slope, score_quant_intercept = data_to_lin_reg(dataframe, "Score Acc")
    y_quant_ref = posit(ref_quant_slope * np.log(x_pred) + ref_quant_intercept)
    y_quant_score = posit(score_quant_slope * np.log(x_pred) + score_quant_intercept)
    ax.plot(x_pred, y_quant_ihop, label="IHOP")
    ax.plot(x_pred, y_quant_ref, label="Refined Score")
    ax.plot(x_pred, y_quant_score, label="Score")
    ax.legend()
    ax.set(
        xlabel=r"$\epsilon$-similarity",
        ylabel="Accuracy",
    )
    fig.tight_layout()
    fig.savefig("atk_comparison.png", dpi=400)


def fig_attack_analysis_tail_distribution():
    fig, ax = plt.subplots()
    dataframe = pd.read_csv("enron_extreme_results.csv")
    dataframe = dataframe.sort_values(by="Epsilon")

    x = dataframe["Epsilon"]
    y = dataframe["Refined Score Acc"]

    mask = np.logical_and(y != 0, y != 1.0)
    assert sum(~mask) < 1 / 3 * y.shape[0]
    y = y[mask]
    x = x[mask]

    log_x = np.array(np.log(x))
    log_y = np.array(logit(y))

    linear_regression = LinearRegression().fit(
        log_x.reshape(-1, 1), log_y.reshape(-1, 1)
    )
    lin_slope = linear_regression.coef_[0, 0]
    lin_intercept = linear_regression.intercept_[0]

    # Peut-être que je dois trier cette liste
    spline_t, spline_c, spline_k = interpolate.splrep(
        log_x, log_y, k=1, t=np.array([0.55])
    )
    spline = interpolate.BSpline(spline_t, spline_c, spline_k, extrapolate=True)

    step_size = x.max() / 500
    x_pred = np.arange(step_size, x.max(), step_size)

    # Compute the predictions
    log_y_lin = lin_slope * np.log(x_pred) + lin_intercept
    y_lin = posit(log_y_lin)
    log_y_spline = spline(np.log(x_pred))
    y_spline = posit(log_y_spline)

    # Visualization in  the logit-log space
    ax.scatter(log_x, log_y, color="black", alpha=0.2, label="Observations")
    ax.plot(np.log(x_pred), log_y_lin, label="Linear regression")
    ax.plot(np.log(x_pred), log_y_spline, label="B-spline")
    ax.set(
        xlabel=r"$\log(\epsilon$-similarity$)$",
        ylabel=r"$\mathrm{logit}(\mathrm{Accuracy})$",
    )
    ax.set_xlim((log_x.min() - 0.1, log_x.max() + 0.1))
    ax.set_ylim((log_y.min() - 0.5, log_y.max() + 0.5))
    ax.legend()
    fig.tight_layout()
    fig.savefig("atk_analysis_enron_extreme_logit-log.png", dpi=400)
    plt.cla()

    # Visualization in  the standard space
    ax.scatter(x, y, color="black", alpha=0.2, label="Observations")
    ax.plot(x_pred, y_lin, label="Linear regression")
    ax.plot(x_pred, y_spline, label="B-spline")
    ax.set(
        xlabel=r"$\epsilon$-similarity",
        ylabel="Accuracy",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig("atk_analysis_enron_extreme.png", dpi=400)


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
    ax.plot(np.log(x_pred), log_y_lin, label="Linear regression")
    ax.plot(np.log(x_pred), log_y_quant075, label="Quantile 0.75")
    ax.plot(np.log(x_pred), log_y_quant095, label="Quantile 0.95")
    ax.set(
        xlabel=r"$\log(\frac{1}{n_\mathrm{atk}}+\frac{1}{n_\mathrm{ind}})$",
        ylabel=r"$\mathrm{logit}(\mathrm{Accuracy})$",
    )
    ax.set_xlim((log_x.min() - 0.1, log_x.max() + 0.1))
    ax.set_ylim((log_y.min() - 0.5, log_y.max() + 0.5))
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"risk_assess_{(col_name.replace(' ','_'))}_log.png", dpi=400)
    plt.cla()

    # Visualization in  the standard space
    ax.scatter(x, y, color="black", alpha=0.5, label="Observations")
    ax.plot(x_pred, y_lin, label="Linear regression")
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


def fig_comp_countermeasure_tuning():
    fig, ax = plt.subplots()
    dataframe = pd.read_csv("risk_assessment_countermeasure.csv")

    x = 1 / dataframe["Nb server docs"] + 1 / dataframe["Nb similar docs"]
    step_size = x.max() / 500
    x_pred = np.arange(step_size, x.max(), step_size)

    def generate_y(colname):
        slope, intercept = data_to_quant_reg(dataframe, colname)
        return posit(slope * np.log(x_pred) + intercept)

    y_baseline = generate_y("Baseline accuracy")
    y_padding_50 = generate_y("Accuracy with padding parameter 50")
    y_padding_100 = generate_y("Accuracy with padding parameter 100")
    y_padding_200 = generate_y("Accuracy with padding parameter 200")
    y_padding_500 = generate_y("Accuracy with padding parameter 500")

    ax.plot(x_pred, y_baseline, label="Baseline")
    ax.plot(x_pred, y_padding_50, label="Padding threshold = 50")
    ax.plot(x_pred, y_padding_100, label="Padding threshold = 100")
    ax.plot(x_pred, y_padding_200, label="Padding threshold = 200")
    ax.plot(x_pred, y_padding_500, label="Padding threshold = 500")
    ax.legend()
    ax.set(
        xlabel=r"$\frac{1}{n_\mathrm{atk}}+\frac{1}{n_\mathrm{ind}}$",
        ylabel="Accuracy",
    )
    fig.tight_layout()
    fig.savefig("parameter_countermeasure_comparison.png", dpi=400)


def fig_comp_parameter_tuning():
    fig, ax = plt.subplots()
    dataframe_classic = pd.read_csv("risk_assessment.csv")
    dataframe_truncated = pd.read_csv("risk_assessment_truncated_voc.csv")

    assert (
        dataframe_classic["Nb server docs"] == dataframe_truncated["Nb server docs"]
    ).all() and (
        dataframe_classic["Nb similar docs"] == dataframe_truncated["Nb similar docs"]
    ).all()
    x = (
        1 / dataframe_classic["Nb server docs"]
        + 1 / dataframe_classic["Nb similar docs"]
    )
    step_size = x.max() / 500
    x_pred = np.arange(step_size, x.max(), step_size)

    classic_slope, classic_intercept = data_to_quant_reg(
        dataframe_classic, "Refined Score Acc"
    )
    y_classic = posit(classic_slope * np.log(x_pred) + classic_intercept)
    truncated_slope, truncated_intercept = data_to_quant_reg(
        dataframe_truncated, "Refined Score Acc"
    )
    y_truncated = posit(truncated_slope * np.log(x_pred) + truncated_intercept)
    ax.plot(x_pred, y_classic, label="Baseline")
    ax.plot(x_pred, y_truncated, label="Truncated vocabulary")
    ax.legend()
    ax.set(
        xlabel=r"$\frac{1}{n_\mathrm{atk}}+\frac{1}{n_\mathrm{ind}}$",
        ylabel="Accuracy",
    )
    fig.tight_layout()
    fig.savefig("parameter_tuning_comparison.png", dpi=400)


def lambda_risk_acc_to_document_size(col_name, n_atk_max=None):
    dataframe = pd.read_csv("risk_assessment.csv")

    # Compute the prediction
    quant095_slope, quant095_intercept = data_to_quant_reg(dataframe, col_name)

    # Reminder logit(acc) = intercept * log(1/natk + 1/nind) + slope
    if n_atk_max is None:
        func = lambda acc_threshold: 1 / np.exp(
            (logit(acc_threshold) - quant095_intercept) / quant095_slope
        )
    else:
        func = lambda acc_threshold: 1 / (
            np.exp((logit(acc_threshold) - quant095_intercept) / quant095_slope)
            - 1 / n_atk_max
        )
    return func


def tab_risk_assess_conclusions():
    risk_inversion_func = lambda_risk_acc_to_document_size("IHOP Acc")
    risk_inversion_func_fixed_atk_200 = lambda_risk_acc_to_document_size(
        "IHOP Acc", 200
    )
    risk_inversion_func_fixed_atk_500 = lambda_risk_acc_to_document_size(
        "IHOP Acc", 500
    )
    risk_inversion_func_fixed_atk_1000 = lambda_risk_acc_to_document_size(
        "IHOP Acc", 1000
    )

    res = []
    for acc in [0.05, 0.1, 0.2, 0.5]:
        res.append(
            (
                risk_inversion_func_fixed_atk_200(acc),
                risk_inversion_func_fixed_atk_500(acc),
                risk_inversion_func_fixed_atk_1000(acc),
                risk_inversion_func(acc),
            )
        )
    res = np.floor(np.array(res).T)
    res[res <= 0] = np.inf
    int_to_str = lambda nbr: str(int(nbr)) if nbr != np.inf else r"\infty"

    print(r"\begin{tabular}{|c|c|c|c|c|}\hline")
    print(r"Accuracy max. (\%) & 5 & 10 & 20 & 50 \\\hline")
    for ind, n_atk in enumerate([200, 500, 1000]):
        print(
            r"When $n_{atk} \le "
            + int_to_str(n_atk)
            + r", n_\text{ind} \le$ & $"
            + int_to_str(res[ind, 0])
            + r"$ & $"
            + int_to_str(res[ind, 1])
            + r"$ & $"
            + int_to_str(res[ind, 2])
            + r"$ & $"
            + int_to_str(res[ind, 3])
            + r"$ \\"
        )
    print(
        r"When $n_{atk} \rightarrow \infty, n_\text{ind} \le$ & $"
        + int_to_str(res[3, 0])
        + r"$ & $"
        + int_to_str(res[3, 1])
        + r"$ & $"
        + int_to_str(res[3, 2])
        + r"$ & $"
        + int_to_str(res[3, 3])
        + r"$ \\\hline"
    )
    print(r"\end{tabular}")


def tab_bonferroni_per_year():
    dataframe = pd.read_csv("bonferroni_tests_by_year.csv")
    print(r"\begin{tabular}{|c|c|c|c|c|}\hline")
    print(
        f'Year split & {dataframe["Year split"][0]} & {dataframe["Year split"][1]} & {dataframe["Year split"][2]} & {dataframe["Year split"][3]}'
        + r"\\\hline"
    )
    print(
        r"$\widetilde{pv}$ "
        + f' & {dataframe["p_bc"][0]} & {dataframe["p_bc"][1]} & {dataframe["p_bc"][2]} & {dataframe["p_bc"][3]}'
        + r"\\\hline"
    )
    print(r"\end{tabular}")


def tab_bonferroni_uniform_sampling():
    dataframe = pd.read_csv("bonferroni_tests.csv")

    print(r"\begin{tabular}{|c|c|c|c|}\hline")
    print(
        r"Avg. $\widetilde{pv}$ & Min. $\widetilde{p}$ & $Q_{\widetilde{p}}(0.1)$\tablefootnote{Remainder: $Q_{X}(\alpha)$ is the quantile of level $\alpha$ for the data distribution $X$.} & $Q_{\widetilde{p}}(0.5)$ \\\hline"
    )
    print(
        +f"{dataframe['p_bc'].mean():.2f} & {dataframe['p_bc'].min():.2f} & {dataframe['p_bc'].quantile(0.1):.2f} & {dataframe['p_bc'].quantile(0.5):.2f}"
        + r"\\ \hline"
    )
    print(r"\end{tabular}")


def draw_figure(fig_function, *args):
    print(f"Drawing figure: {fig_function.__name__} (Args: {args})...", end="")
    fig_function(*args)
    print("OK")


def print_tabular(tab_function, *args):
    print(
        f"Preparing table: {tab_function.__name__} (Args: {args})... Copy the LaTeX code below:"
    )
    tab_function(*args)
    input("...OK\nPress enter when you are done.")


if __name__ == "__main__":
    if not os.path.exists("results"):
        raise OSError("No result directory found.")
    os.chdir("results")

    draw_figure(fig_epsilon_nb_docs)
    draw_figure(fig_attack_analysis, "enron")
    draw_figure(fig_attack_analysis, "apache")
    draw_figure(fig_attack_analysis, "blogs")
    draw_figure(fig_attack_analysis_tail_distribution)
    draw_figure(fig_comparison_atk)
    draw_figure(fig_indiv_risk_assessment, "IHOP Acc")
    draw_figure(fig_indiv_risk_assessment, "Refined Score Acc")
    draw_figure(fig_indiv_risk_assessment, "Score Acc")
    draw_figure(fig_comp_risk_assessment)
    draw_figure(fig_comp_countermeasure_tuning)
    draw_figure(fig_comp_parameter_tuning)

    print_tabular(tab_risk_assess_conclusions)
    print_tabular(tab_bonferroni_per_year)
    print_tabular(tab_bonferroni_uniform_sampling)
