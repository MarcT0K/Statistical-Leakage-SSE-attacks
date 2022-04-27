#!/usr/bin/python3

import csv
import pickle

import numpy as np
import tqdm
import matplotlib.pyplot as plt

from src.common import KeywordExtractor, generate_known_queries
from src.email_extraction import (
    extract_apache_ml,
)
from src.attacks.score import RefinedScoreAttacker
from src.attacks.sap import SAPAttacker

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

VOC_SIZE = 1000
QUERYSET_SIZE = 300
KNOWN_QUERIES = 15


def res_subsec_4C():
    enron_emails = extract_apache_ml()
    extractor = KeywordExtractor(enron_emails, VOC_SIZE, 1)
    occ_mat = extractor.occ_array
    n_tot = extractor.occ_array.shape[0]

    choice_serv = np.random.choice(
        range(n_tot), size=(int(n_tot * 0.6),), replace=False
    )
    ind_serv = np.zeros(n_tot, dtype=bool)
    ind_serv[choice_serv] = True
    serv_mat = occ_mat[ind_serv, :]
    serv_max_docs = serv_mat.shape[0]
    kw_mat = occ_mat[~ind_serv, :]
    kw_max_docs = kw_mat.shape[0]

    with open("fig_subsec_4C.csv", "w", newline="") as csvfile:
        fieldnames = [
            "Nb similar docs",
            "Nb server docs",
            "Epsilon sim",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i in tqdm.tqdm(iterable=range(50)):
            sub_choice_kw = np.random.choice(
                range(kw_max_docs),
                size=(int(kw_max_docs * (i + 1) * 0.02),),
                replace=False,
            )
            for j in tqdm.tqdm(iterable=range(50)):
                sub_choice_serv = np.random.choice(
                    range(serv_max_docs),
                    size=(int(serv_max_docs * (j + 1) * 0.02),),
                    replace=False,
                )
                coocc_td = (
                    serv_mat[sub_choice_serv, :].T
                    @ serv_mat[sub_choice_serv, :]
                    / serv_mat[sub_choice_serv, :].shape[0]
                )
                coocc_kw = (
                    kw_mat[sub_choice_kw, :].T
                    @ kw_mat[sub_choice_kw, :]
                    / kw_mat[sub_choice_kw, :].shape[0]
                )

                writer.writerow(
                    {
                        "Nb similar docs": len(sub_choice_kw),
                        "Nb server docs": len(sub_choice_serv),
                        "Epsilon sim": epsilon_sim(coocc_kw, coocc_td),
                    }
                )


def fig_subsec_4C(show=True):
    with open("fig_subsec_4C.csv", "r") as csvfile:
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


def enron_metric_sensivity():
    # enron_emails = extract_sent_mail_contents()
    # extractor = KeywordExtractor(enron_emails, VOC_SIZE, 1)
    with open("enron_extractor.pkl", "rb") as f:
        extractor = pickle.load(f)
    occ_mat = extractor.occ_array
    n_tot = extractor.occ_array.shape[0]

    with open("enron_sim_acc.csv", "w", newline="") as csvfile:
        fieldnames = [
            "Nb similar docs",
            "Nb server docs",
            "Voc size",
            "Nb queries",
            "Nb queries known",
            "Epsilon",
            "Refined Score Acc",
        ]
        writer = csv.DictWriter(csvfile, delimiter=";", fieldnames=fieldnames)
        writer.writeheader()
        for i, j in tqdm.tqdm(
            iterable=[
                (i, j) for i in range(1, 10) for j in range(1, 10) for k in range(5)
            ],
            desc="Running the experiments",
        ):
            choice_serv = np.random.choice(
                range(n_tot), size=(int(n_tot * i * 0.05),), replace=False
            )
            ind_serv = np.zeros(n_tot, dtype=bool)
            ind_serv[choice_serv] = True
            serv_mat = occ_mat[ind_serv, :]
            full_atk_mat = occ_mat[~ind_serv, :]
            atk_max_docs = full_atk_mat.shape[0]

            sub_choice = np.random.choice(
                range(atk_max_docs),
                size=(int(atk_max_docs * (j + 1) * 0.05),),
                replace=False,
            )

            voc = list(extractor.get_sorted_voc())
            queries_ind = np.random.choice(len(voc), QUERYSET_SIZE, replace=False)
            queries = [voc[ind] for ind in queries_ind]
            known_queries = generate_known_queries(
                similar_wordlist=voc,
                stored_wordlist=queries,
                nb_queries=KNOWN_QUERIES,
            )

            ref_atk = RefinedScoreAttacker(
                keyword_occ_array=full_atk_mat[sub_choice, :],
                keyword_sorted_voc=voc,
                trapdoor_occ_array=serv_mat[:, queries_ind],
                trapdoor_sorted_voc=queries,
                nb_stored_docs=serv_mat.shape[0],
                known_queries=known_queries,
                ref_speed=10,
            )

            ref_pred = ref_atk.predict()
            ref_acc = np.mean(
                [word == candidate for word, candidate in ref_pred.items()]
            )

            ind_doc_coocc = serv_mat.T @ serv_mat / serv_mat.shape[0]
            atk_full_coocc = (
                full_atk_mat[sub_choice, :].T
                @ full_atk_mat[sub_choice, :]
                / full_atk_mat[sub_choice, :].shape[0]
            )

            writer.writerow(
                {
                    "Nb similar docs": len(sub_choice),
                    "Nb server docs": ind_serv.sum(),
                    "Voc size": len(voc),
                    "Nb queries": len(queries),
                    "Nb queries known": len(known_queries),
                    "Epsilon": epsilon_sim(atk_full_coocc, ind_doc_coocc),
                    "Refined Score Acc": ref_acc,
                }
            )


def test_known_data_atk():
    # enron_emails = extract_sent_mail_contents()
    # extractor = KeywordExtractor(enron_emails, VOC_SIZE, 1)
    with open("enron_extractor.pkl", "rb") as f:
        extractor = pickle.load(f)
    occ_mat = extractor.occ_array

    with open("test_sap.csv", "w", newline="") as csvfile:
        fieldnames = [
            "Nb similar docs",
            "Nb server docs",
            "Voc size",
            "Nb queries",
            "Nb queries known",
            "Epsilon",
            "Refined Score Acc",
            "SAP Acc",
        ]
        writer = csv.DictWriter(csvfile, delimiter=";", fieldnames=fieldnames)
        writer.writeheader()
        for j in tqdm.tqdm(
            iterable=[j for j in range(10, 11) for k in range(5)],
            desc="Running the experiments",
        ):
            serv_mat = occ_mat
            full_atk_mat = occ_mat
            atk_max_docs = full_atk_mat.shape[0]

            sub_choice = np.random.choice(
                range(atk_max_docs),
                size=(int(atk_max_docs * j * 0.1),),
                replace=False,
            )

            voc = list(extractor.get_sorted_voc())
            queries_ind = np.random.choice(len(voc), QUERYSET_SIZE, replace=False)
            queries = [voc[ind] for ind in queries_ind]
            known_queries = generate_known_queries(
                similar_wordlist=voc,
                stored_wordlist=queries,
                nb_queries=KNOWN_QUERIES,
            )

            # ref_atk = RefinedScoreAttacker(
            #     keyword_occ_array=full_atk_mat[sub_choice, :],
            #     keyword_sorted_voc=voc,
            #     trapdoor_occ_array=serv_mat[:, queries_ind],
            #     trapdoor_sorted_voc=queries,
            #     nb_stored_docs=serv_mat.shape[0],
            #     known_queries=known_queries,
            #     ref_speed=10,
            # )

            # ref_pred = ref_atk.predict()
            # ref_acc = np.mean(
            #     [word == candidate for word, candidate in ref_pred.items()]
            # )

            sap_atk = SAPAttacker(
                keyword_occ_array=full_atk_mat[sub_choice, :],
                keyword_sorted_voc=voc,
                trapdoor_occ_array=serv_mat[:, queries_ind],
                trapdoor_sorted_voc=queries,
                nb_stored_docs=serv_mat.shape[0],
            )

            sap_pred = sap_atk.predict()
            sap_acc = np.mean(
                [word == candidate for word, candidate in sap_pred.items()]
            )

            ind_doc_coocc = serv_mat.T @ serv_mat / serv_mat.shape[0]
            atk_full_coocc = (
                full_atk_mat[sub_choice, :].T
                @ full_atk_mat[sub_choice, :]
                / full_atk_mat[sub_choice, :].shape[0]
            )

            writer.writerow(
                {
                    "Nb similar docs": len(sub_choice),
                    "Nb server docs": occ_mat.shape[0],
                    "Voc size": len(voc),
                    "Nb queries": len(queries),
                    "Nb queries known": len(known_queries),
                    "Epsilon": epsilon_sim(atk_full_coocc, ind_doc_coocc),
                    "Refined Score Acc": 0,
                    "SAP Acc": sap_acc,
                }
            )
