#!/usr/bin/python3
# pylint: disable=invalid-name
import csv
import os

import numpy as np
import tqdm

from src.simulation_utils import simulate_attack, generate_adv_knowledge
from src.email_extraction import enron_extractor, apache_extractor, blogs_extractor
from src.attacks.score import RefinedScoreAttacker, ScoreAttacker
from src.attacks.graphm import GraphMatchingAttacker
from src.attacks.ihop import IHOPAttacker

epsilon_sim = lambda coocc_1, coocc_2: np.linalg.norm(coocc_1 - coocc_2)

VOC_SIZE = 500
QUERYSET_SIZE = 200
KNOWN_QUERIES = 15


def similarity_exploration():
    extractor = apache_extractor(VOC_SIZE)
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

    with open(
        "similarity_exploration.csv", "w", newline="", encoding="utf-8"
    ) as csvfile:
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


def atk_comparison():
    extractor = enron_extractor(VOC_SIZE)
    occ_mat = extractor.occ_array

    with open("atk_comparison.csv", "w", newline="", encoding="utf-8") as csvfile:
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
            ) = generate_adv_knowledge(
                occ_mat, i * 0.05, j * 0.05, voc, QUERYSET_SIZE, KNOWN_QUERIES
            )

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


def generate_ref_score_results(extractor, dataset_name, truncation_size=-1):
    extractor = enron_extractor(VOC_SIZE)
    occ_mat = extractor.occ_array

    if truncation_size != -1:
        n_tot = occ_mat.shape[0]
        assert n_tot > truncation_size
        truncated_dataset = np.random.choice(
            range(n_tot), size=(truncation_size,), replace=False
        )
        occ_mat = occ_mat[truncated_dataset, :]

    with open(
        f"{dataset_name}_results.csv", "w", newline="", encoding="utf-8"
    ) as csvfile:
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
            ) = generate_adv_knowledge(
                occ_mat, i * 0.05, j * 0.05, voc, QUERYSET_SIZE, KNOWN_QUERIES
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
                    "Refined Score Acc": ref_acc,
                }
            )


# TODO: add a bit of logging?
if __name__ == "__main__":
    if not os.path.exists("results"):
        os.makedirs("results")
    os.chdir("results")

    # Call all functions defined in this file
    similarity_exploration()
    atk_comparison()
    generate_ref_score_results(apache_extractor, "apache")
    generate_ref_score_results(apache_extractor, "apache_reduced", 30000)
    generate_ref_score_results(blogs_extractor, "blogs")
