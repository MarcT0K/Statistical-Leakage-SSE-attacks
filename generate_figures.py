#!/usr/bin/python3
# pylint: disable=invalid-name

import os

import numpy as np
import matplotlib.pyplot as plt


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


if __name__ == "__main__":
    if not os.path.exists("results"):
        raise OSError("No result directory found.")
    os.chdir("results")

    # Call all functions defined in this file
    fig_subsec_4C(False)
