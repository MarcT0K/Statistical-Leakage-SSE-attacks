#!/usr/bin/python3
# pylint: disable=invalid-name
import csv
import os

import numpy as np
import tqdm

from src.simulation_utils import (
    generate_adv_knowledge_fixed_nb_docs,
    simulate_attack,
    generate_adv_knowledge,
    padding_countermeasure,
)
from src.email_extraction import enron_extractor, apache_extractor, blogs_extractor
from src.attacks.score import RefinedScoreAttacker, ScoreAttacker
from src.attacks.ihop import IHOPAttacker

epsilon_sim = lambda coocc_1, coocc_2: np.linalg.norm(coocc_1 - coocc_2)

VOC_SIZE = 500
QUERYSET_SIZE = 300
KNOWN_QUERIES = 15

# TODO: fix the random seeds


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
            "Nb queries observed",
            "Nb queries known",
            "Epsilon",
            "Score Acc",
            "Score Runtime",
            "Refined Score Acc",
            "Refined Score Runtime",
            "IHOP Acc",
            "IHOP Runtime",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
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
            score_acc, score_runtime = simulate_attack(
                ScoreAttacker,
                keyword_occ_array=atk_mat,
                keyword_sorted_voc=voc,
                trapdoor_occ_array=ind_mat[:, queries_ind],
                trapdoor_sorted_voc=queries,
                nb_stored_docs=ind_mat.shape[0],
                known_queries=known_queries,
            )

            # Refined score attack
            ref_acc, ref_score_runtime = simulate_attack(
                RefinedScoreAttacker,
                keyword_occ_array=atk_mat,
                keyword_sorted_voc=voc,
                trapdoor_occ_array=ind_mat[:, queries_ind],
                trapdoor_sorted_voc=queries,
                nb_stored_docs=ind_mat.shape[0],
                known_queries=known_queries,
            )

            # IHOP attack
            ihop_acc, ihop_runtime = simulate_attack(
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
                    "Nb queries observed": len(queries),
                    "Nb queries known": len(known_queries),
                    "Epsilon": epsilon_sim(atk_full_coocc, ind_doc_coocc),
                    "Score Acc": score_acc,
                    "Score Runtime": score_runtime,
                    "Refined Score Acc": ref_acc,
                    "Refined Score Runtime": ref_score_runtime,
                    "IHOP Acc": ihop_acc,
                    "IHOP Runtime": ihop_runtime,
                }
            )


def generate_ref_score_results(extractor_function, dataset_name, truncation_size=-1):
    extractor = extractor_function(VOC_SIZE)
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
            "Nb queries observed",
            "Nb queries known",
            "Epsilon",
            "Refined Score Acc",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
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
            ref_acc, _runtime = simulate_attack(
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
                    "Nb queries observed": len(queries),
                    "Nb queries known": len(known_queries),
                    "Epsilon": epsilon_sim(atk_full_coocc, ind_doc_coocc),
                    "Refined Score Acc": ref_acc,
                }
            )


def risk_assessment():
    extractor = enron_extractor(VOC_SIZE)
    occ_mat = extractor.occ_array

    n_docs = occ_mat.shape[0]
    min_docs = 500
    assert n_docs > min_docs

    # Sum = 1/n_atk + 1/n_ind but we consider n_atk = n_ind for simplicity => Sum = 2/n_atk
    min_sum = 2 / (n_docs * 0.5)
    max_sum = 2 / min_docs

    with open("risk_assessment.csv", "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "Nb similar docs",
            "Nb server docs",
            "Voc size",
            "Nb queries observed",
            "Nb queries known",
            "Epsilon",
            "Score Acc",
            "Refined Score Acc",
            "IHOP Acc",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i in tqdm.tqdm(
            iterable=[i for i in range(51) for k in range(1)],
            desc="Running the experiments",
        ):
            curr_sum = (max_sum - min_sum) * (i * 2) / 100 + min_sum
            curr_n = int(2 / curr_sum)
            # Auxiliary knowledge generation
            voc = list(extractor.get_sorted_voc())
            (
                ind_mat,
                atk_mat,
                queries,
                queries_ind,  # Even if we observe all queries, we want their order
                known_queries,
            ) = generate_adv_knowledge_fixed_nb_docs(
                occ_mat, curr_n, curr_n, voc, VOC_SIZE, KNOWN_QUERIES
            )

            # Score attack
            score_acc, _runtime = simulate_attack(
                ScoreAttacker,
                keyword_occ_array=atk_mat,
                keyword_sorted_voc=voc,
                trapdoor_occ_array=ind_mat[:, queries_ind],
                trapdoor_sorted_voc=queries,
                nb_stored_docs=ind_mat.shape[0],
                known_queries=known_queries,
            )

            # Refined score attack
            ref_acc, _runtime = simulate_attack(
                RefinedScoreAttacker,
                keyword_occ_array=atk_mat,
                keyword_sorted_voc=voc,
                trapdoor_occ_array=ind_mat[:, queries_ind],
                trapdoor_sorted_voc=queries,
                nb_stored_docs=ind_mat.shape[0],
                known_queries=known_queries,
            )

            # IHOP attack
            ihop_acc, _runtime = simulate_attack(
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
                    "Nb queries observed": len(queries),
                    "Nb queries known": len(known_queries),
                    "Epsilon": epsilon_sim(atk_full_coocc, ind_doc_coocc),
                    "Score Acc": score_acc,
                    "Refined Score Acc": ref_acc,
                    "IHOP Acc": ihop_acc,
                }
            )


def risk_assessment_truncated_vocabulary():
    extractor = enron_extractor(VOC_SIZE)
    occ_mat = extractor.occ_array

    n_docs = occ_mat.shape[0]
    min_docs = 500
    assert n_docs > min_docs

    # Sum = 1/n_atk + 1/n_ind but we consider n_atk = n_ind for simplicity => Sum = 2/n_atk
    min_sum = 2 / (n_docs * 0.5)
    max_sum = 2 / min_docs

    with open(
        "risk_assessment_truncated_voc.csv", "w", newline="", encoding="utf-8"
    ) as csvfile:
        fieldnames = [
            "Nb similar docs",
            "Nb server docs",
            "Voc size",
            "Nb queries observed",
            "Nb queries known",
            "Epsilon",
            "Refined Score Acc",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i in tqdm.tqdm(
            iterable=[i for i in range(51) for k in range(1)],
            desc="Running the experiments",
        ):
            curr_sum = (max_sum - min_sum) * (i * 2) / 100 + min_sum
            curr_n = int(2 / curr_sum)
            # Auxiliary knowledge generation
            trunc_occ_mat = occ_mat[100:, :]
            voc = list(extractor.get_sorted_voc())[100:]
            (
                ind_mat,
                atk_mat,
                queries,
                queries_ind,  # Even if we observe all queries, we want their order
                known_queries,
            ) = generate_adv_knowledge_fixed_nb_docs(
                trunc_occ_mat, curr_n, curr_n, voc, VOC_SIZE, KNOWN_QUERIES
            )

            # Refined score attack
            ref_acc, _runtime = simulate_attack(
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
                    "Nb queries observed": len(queries),
                    "Nb queries known": len(known_queries),
                    "Epsilon": epsilon_sim(atk_full_coocc, ind_doc_coocc),
                    "Refined Score Acc": ref_acc,
                }
            )


def risk_assessment_countermeasure_tuning():
    extractor = enron_extractor(VOC_SIZE)
    occ_mat = extractor.occ_array

    n_docs = occ_mat.shape[0]
    min_docs = 500
    assert n_docs > min_docs

    # Sum = 1/n_atk + 1/n_ind but we consider n_atk = n_ind for simplicity => Sum = 2/n_atk
    min_sum = 2 / (n_docs * 0.5)
    max_sum = 2 / min_docs

    with open(
        "risk_assessment_countermeasure.csv", "w", newline="", encoding="utf-8"
    ) as csvfile:
        fieldnames = [
            "Nb similar docs",
            "Nb server docs",
            "Voc size",
            "Nb queries observed",
            "Nb queries known",
            "Epsilon",
            "Baseline accuracy",
            "Accuracy with padding parameter 50",
            "Overhead with padding parameter 50",
            "Accuracy with padding parameter 100",
            "Overhead with padding parameter 100",
            "Accuracy with padding parameter 200",
            "Overhead with padding parameter 200",
            "Accuracy with padding parameter 500",
            "Overhead with padding parameter 500",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i in tqdm.tqdm(
            iterable=[i for i in range(51) for k in range(1)],
            desc="Running the experiments",
        ):
            curr_sum = (max_sum - min_sum) * (i * 2) / 100 + min_sum
            curr_n = int(2 / curr_sum)
            # Auxiliary knowledge generation
            voc = list(extractor.get_sorted_voc())
            (
                ind_mat,
                atk_mat,
                queries,
                queries_ind,  # Even if we observe all queries, we want their order
                known_queries,
            ) = generate_adv_knowledge_fixed_nb_docs(
                occ_mat, curr_n, curr_n, voc, VOC_SIZE, KNOWN_QUERIES
            )

            # Padding param 50
            mitigated_mat, overhead_50 = padding_countermeasure(ind_mat, 50)
            ref_acc_50, _runtime = simulate_attack(
                RefinedScoreAttacker,
                keyword_occ_array=atk_mat,
                keyword_sorted_voc=voc,
                trapdoor_occ_array=mitigated_mat[:, queries_ind],
                trapdoor_sorted_voc=queries,
                nb_stored_docs=ind_mat.shape[0],
                known_queries=known_queries,
            )

            # Padding param 100
            mitigated_mat, overhead_100 = padding_countermeasure(ind_mat, 100)
            ref_acc_100, _runtime = simulate_attack(
                RefinedScoreAttacker,
                keyword_occ_array=atk_mat,
                keyword_sorted_voc=voc,
                trapdoor_occ_array=mitigated_mat[:, queries_ind],
                trapdoor_sorted_voc=queries,
                nb_stored_docs=ind_mat.shape[0],
                known_queries=known_queries,
            )

            # Padding param 100
            mitigated_mat, overhead_200 = padding_countermeasure(ind_mat, 200)
            ref_acc_200, _runtime = simulate_attack(
                RefinedScoreAttacker,
                keyword_occ_array=atk_mat,
                keyword_sorted_voc=voc,
                trapdoor_occ_array=mitigated_mat[:, queries_ind],
                trapdoor_sorted_voc=queries,
                nb_stored_docs=ind_mat.shape[0],
                known_queries=known_queries,
            )

            # Padding param 500
            mitigated_mat, overhead_500 = padding_countermeasure(ind_mat, 500)
            ref_acc_500, _runtime = simulate_attack(
                RefinedScoreAttacker,
                keyword_occ_array=atk_mat,
                keyword_sorted_voc=voc,
                trapdoor_occ_array=mitigated_mat[:, queries_ind],
                trapdoor_sorted_voc=queries,
                nb_stored_docs=ind_mat.shape[0],
                known_queries=known_queries,
            )

            # Nothing
            ref_acc_nothing, _runtime = simulate_attack(
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
                    "Nb queries observed": len(queries),
                    "Nb queries known": len(known_queries),
                    "Epsilon": epsilon_sim(atk_full_coocc, ind_doc_coocc),
                    "Baseline accuracy": ref_acc_nothing,
                    "Accuracy with padding parameter 50": overhead_50,
                    "Overhead with padding parameter 50": ref_acc_50,
                    "Accuracy with padding parameter 100": overhead_100,
                    "Overhead with padding parameter 100": ref_acc_100,
                    "Accuracy with padding parameter 200": overhead_200,
                    "Overhead with padding parameter 200": ref_acc_200,
                    "Accuracy with padding parameter 500": overhead_500,
                    "Overhead with padding parameter 500": ref_acc_500,
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
    generate_ref_score_results(blogs_extractor, "blogs_reduced", 30000)
    risk_assessment()
