#!/usr/bin/python3

# TODO: factorize code
# TODO: split the code into two files => result generation and plot generation

import csv
import pickle
from typing import List

import numpy as np
import tqdm
import matplotlib.pyplot as plt


from src.common import KeywordExtractor, generate_known_queries
from src.email_extraction import extract_apache_ml, extract_sent_mail_contents
from src.attacks.score import RefinedScoreAttacker, ScoreAttacker
from src.attacks.graphm import GraphMatchingAttacker
from src.attacks.ihop import IHOPAttacker

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

VOC_SIZE = 500
QUERYSET_SIZE = 200
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
    ind_mat = occ_mat[ind_serv, :]
    serv_max_docs = ind_mat.shape[0]
    kw_mat = occ_mat[~ind_serv, :]
    kw_max_docs = kw_mat.shape[0]

    with open("fig_subsec_4C.csv", "w", newline="", encoding="utf-8") as csvfile:
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
                    ind_mat[sub_choice_serv, :].T
                    @ ind_mat[sub_choice_serv, :]
                    / ind_mat[sub_choice_serv, :].shape[0]
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


def generate_adv_knowledge(
    occ_mat: np.array,
    atk_prop: float,
    ind_prop: float,
    voc: List[str],
    nb_queries: int = QUERYSET_SIZE,
    nb_known_queries: int = KNOWN_QUERIES,
    sim_data_atk: bool = True,
):
    assert 0 < atk_prop and atk_prop <= 1
    assert 0 < ind_prop and ind_prop <= 1
    if sim_data_atk:
        assert ind_prop + atk_prop <= 1

    n_tot = occ_mat.shape[0]
    choice_ind = np.random.choice(
        range(n_tot), size=(int(n_tot * ind_prop),), replace=False
    )
    ind_docs = np.zeros(n_tot, dtype=bool)
    ind_docs[choice_ind] = True
    ind_mat = occ_mat[ind_docs, :]

    if sim_data_atk:  # Set the biggest adversary knowledge possible
        full_atk_mat = occ_mat[~ind_docs, :]
    else:
        full_atk_mat = ind_mat
    atk_max_docs = full_atk_mat.shape[0]
    atk_choice = np.random.choice(
        range(atk_max_docs),
        size=(int(atk_max_docs * atk_prop),),
        replace=False,
    )
    atk_mat = full_atk_mat[atk_choice, :]

    queries_ind = np.random.choice(len(voc), nb_queries, replace=False)
    queries = [voc[ind] for ind in queries_ind]
    known_queries = generate_known_queries(
        similar_wordlist=voc,
        stored_wordlist=queries,
        nb_queries=nb_known_queries,
    )
    return ind_mat, atk_mat, queries, queries_ind, known_queries


def simulate_attack(attack_class, **kwargs) -> float:
    attacker = attack_class(**kwargs)
    pred = attacker.predict()
    return np.mean([word == candidate for word, candidate in pred.items()])


def enron_metric_sensivity():
    enron_emails = extract_sent_mail_contents()
    extractor = KeywordExtractor(enron_emails, VOC_SIZE, 1)
    # with open("enron_extractor.pkl", "rb") as f:
    #     extractor = pickle.load(f)
    occ_mat = extractor.occ_array

    with open("enron_sim_acc.csv", "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "Nb similar docs",
            "Nb server docs",
            "Voc size",
            "Nb queries",
            "Nb queries known",
            "Epsilon",
            "Score Acc",
            "Refined Score Acc",
            "IHOP Acc",
        ]
        writer = csv.DictWriter(csvfile, delimiter=";", fieldnames=fieldnames)
        writer.writeheader()
        for i, j in tqdm.tqdm(
            iterable=[
                (i, j) for i in range(1, 11) for j in range(1, 11) for k in range(5)
            ],
            desc="Running the experiments",
        ):
            # Auxiliary knowledge generation
            voc = list(extractor.get_sorted_voc())
            (
                ind_mat,
                atk_mat,
                queries,
                queries_ind,
                known_queries,
            ) = generate_adv_knowledge(occ_mat, i * 0.05, j * 0.05, voc)

            # Score attack
            score_acc = simulate_attack(
                ScoreAttacker,
                keyword_occ_array=atk_mat,
                keyword_sorted_voc=voc,
                trapdoor_occ_array=ind_mat[:, queries_ind],
                trapdoor_sorted_voc=queries,
                nb_stored_docs=ind_mat.shape[0],
                known_queries=known_queries,
            )

            # Refined score attack
            ref_acc = simulate_attack(
                RefinedScoreAttacker,
                keyword_occ_array=atk_mat,
                keyword_sorted_voc=voc,
                trapdoor_occ_array=ind_mat[:, queries_ind],
                trapdoor_sorted_voc=queries,
                nb_stored_docs=ind_mat.shape[0],
                known_queries=known_queries,
            )

            # IHOP attack
            ihop_acc = simulate_attack(
                IHOPAttacker,
                keyword_occ_array=atk_mat,
                keyword_sorted_voc=voc,
                trapdoor_occ_array=ind_mat[:, queries_ind],
                trapdoor_sorted_voc=queries,
                nb_stored_docs=ind_mat.shape[0],
                known_queries=known_queries,
            )

            # Compute espilon-similarity
            ind_doc_coocc = ind_mat.T @ ind_mat / ind_mat.shape[0]
            atk_full_coocc = atk_mat.T @ atk_mat / atk_mat.shape[0]

            writer.writerow(
                {
                    "Nb similar docs": atk_mat.shape[0],
                    "Nb server docs": ind_mat.shape[0],
                    "Voc size": len(voc),
                    "Nb queries": len(queries),
                    "Nb queries known": len(known_queries),
                    "Epsilon": epsilon_sim(atk_full_coocc, ind_doc_coocc),
                    "Score Acc": score_acc,
                    "Refined Score Acc": ref_acc,
                    "IHOP Acc": ihop_acc,
                }
            )


############ TO BE REMOVED ##############""""


def test_known_data_atk():
    enron_emails = extract_sent_mail_contents()
    extractor = KeywordExtractor(enron_emails, 100, 1)
    # with open("enron_extractor.pkl", "rb") as f:
    #     extractor = pickle.load(f)
    occ_mat = extractor.occ_array

    with open("test_graphm.csv", "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "Nb similar docs",
            "Nb server docs",
            "Voc size",
            "Nb queries",
            "Nb queries known",
            "Epsilon",
            "Refined Score Acc",
            "Graphm Acc",
        ]
        writer = csv.DictWriter(csvfile, delimiter=";", fieldnames=fieldnames)
        writer.writeheader()
        for j in tqdm.tqdm(
            iterable=[j for j in range(10, 11) for k in range(5)],
            desc="Running the experiments",
        ):
            ind_mat = occ_mat
            full_atk_mat = occ_mat
            atk_max_docs = full_atk_mat.shape[0]

            sub_choice = np.random.choice(
                range(atk_max_docs),
                size=(int(atk_max_docs * j * 0.1),),
                replace=False,
            )

            voc = list(extractor.get_sorted_voc())
            queries_ind = np.random.choice(len(voc), 100, replace=False)
            queries = [voc[ind] for ind in queries_ind]
            known_queries = generate_known_queries(
                similar_wordlist=voc,
                stored_wordlist=queries,
                nb_queries=KNOWN_QUERIES,
            )

            ref_atk = RefinedScoreAttacker(
                keyword_occ_array=full_atk_mat[sub_choice, :],
                keyword_sorted_voc=voc,
                trapdoor_occ_array=ind_mat[:, queries_ind],
                trapdoor_sorted_voc=queries,
                nb_stored_docs=ind_mat.shape[0],
                known_queries=known_queries,
                ref_speed=10,
            )

            ref_pred = ref_atk.predict()
            ref_acc = np.mean(
                [word == candidate for word, candidate in ref_pred.items()]
            )

            graphm_atk = GraphMatchingAttacker(
                keyword_occ_array=full_atk_mat[sub_choice, :],
                keyword_sorted_voc=voc,
                trapdoor_occ_array=ind_mat[:, queries_ind],
                trapdoor_sorted_voc=queries,
                nb_stored_docs=ind_mat.shape[0],
            )
            graphm_atk.set_alpha(1)
            graphm_pred = graphm_atk.predict()
            graphm_acc = np.mean(
                [word == candidate for word, candidate in graphm_pred.items()]
            )

            ind_doc_coocc = ind_mat.T @ ind_mat / ind_mat.shape[0]
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
                    "Refined Score Acc": ref_acc,
                    "Graphm Acc": graphm_acc,
                }
            )


def test_ihop():
    enron_emails = extract_sent_mail_contents()
    extractor = KeywordExtractor(enron_emails, 1000, 1)
    # with open("enron_extractor.pkl", "rb") as f:
    #     extractor = pickle.load(f)
    occ_mat = extractor.occ_array
    n_tot = extractor.occ_array.shape[0]

    with open("test_ihop.csv", "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "Nb similar docs",
            "Nb server docs",
            "Voc size",
            "Nb queries",
            "Nb queries known",
            "Epsilon",
            "Refined Score Acc",
            "IHOP Acc",
        ]
        writer = csv.DictWriter(csvfile, delimiter=";", fieldnames=fieldnames)
        writer.writeheader()
        for i, j in tqdm.tqdm(
            iterable=[
                (i, j) for i in range(5, 10) for j in range(5, 10) for k in range(1)
            ],
            desc="Running the experiments",
        ):
            choice_serv = np.random.choice(
                range(n_tot), size=(int(n_tot * i * 0.05),), replace=False
            )
            ind_serv = np.zeros(n_tot, dtype=bool)
            ind_serv[choice_serv] = True
            ind_mat = occ_mat[ind_serv, :]
            full_atk_mat = occ_mat[~ind_serv, :]
            atk_max_docs = full_atk_mat.shape[0]

            sub_choice = np.random.choice(
                range(atk_max_docs),
                size=(int(atk_max_docs * (j + 1) * 0.05),),
                replace=False,
            )

            voc = list(extractor.get_sorted_voc())
            queries_ind = np.random.choice(len(voc), 100, replace=False)
            queries = [voc[ind] for ind in queries_ind]
            known_queries = generate_known_queries(
                similar_wordlist=voc,
                stored_wordlist=queries,
                nb_queries=KNOWN_QUERIES,
            )

            ref_atk = RefinedScoreAttacker(
                keyword_occ_array=full_atk_mat[sub_choice, :],
                keyword_sorted_voc=voc,
                trapdoor_occ_array=ind_mat[:, queries_ind],
                trapdoor_sorted_voc=queries,
                nb_stored_docs=ind_mat.shape[0],
                known_queries=known_queries,
                ref_speed=10,
            )

            ref_pred = ref_atk.predict()
            ref_acc = np.mean(
                [word == candidate for word, candidate in ref_pred.items()]
            )

            ihop_atk = IHOPAttacker(
                keyword_occ_array=full_atk_mat[sub_choice, :],
                keyword_sorted_voc=voc,
                trapdoor_occ_array=ind_mat[:, queries_ind],
                trapdoor_sorted_voc=queries,
                nb_stored_docs=ind_mat.shape[0],
            )
            ihop_pred = ihop_atk.predict()
            ihop_acc = np.mean(
                [word == candidate for word, candidate in ihop_pred.items()]
            )

            ind_doc_coocc = ind_mat.T @ ind_mat / ind_mat.shape[0]
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
                    "Refined Score Acc": ref_acc,
                    "IHOP Acc": ihop_acc,
                }
            )
