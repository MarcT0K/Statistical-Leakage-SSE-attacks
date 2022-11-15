#!/usr/bin/python3
# pylint: disable=invalid-name,logging-fstring-interpolation
import csv
import logging
import os
import random

import numpy as np
import scipy
import tqdm

from codecarbon import OfflineEmissionsTracker

from src.attacks.ihop import IHOPAttacker
from src.attacks.score import RefinedScoreAttacker, ScoreAttacker
from src.keyword_extract import KeywordExtractor
from src.document_extraction import (
    apache_extractor,
    blogs_extractor,
    enron_extractor,
    extract_apache_ml_by_year,
)
from src.simulation_utils import (
    generate_adv_knowledge,
    generate_adv_knowledge_fixed_nb_docs,
    generate_known_queries,
    padding_countermeasure,
    simulate_attack,
)

VOC_SIZE = 500
QUERYSET_SIZE = 300
KNOWN_QUERIES = 15


def epsilon_sim(coocc_1, coocc_2):
    return np.linalg.norm(coocc_1 - coocc_2)


############## ATTACK ANALYSIS EXPERIMENTS ###########


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
            for j in range(50):
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


def atk_comparison(queryset_size=QUERYSET_SIZE, result_file="atk_comparison.csv"):
    extractor = enron_extractor(VOC_SIZE)
    occ_mat = extractor.occ_array

    n_docs = occ_mat.shape[0]
    min_docs = 500
    assert n_docs > min_docs

    # Sum = 1/n_atk + 1/n_ind but we consider n_atk = n_ind for simplicity => Sum = 2/n_atk
    min_sum = 2 / (n_docs * 0.5)
    max_sum = 2 / min_docs

    with open(result_file, "w", newline="", encoding="utf-8") as csvfile:
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
        for i in tqdm.tqdm(
            iterable=[i for i in range(51)],
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
                queries_ind,
                known_queries,
            ) = generate_adv_knowledge_fixed_nb_docs(
                occ_mat, curr_n, curr_n, voc, queryset_size, KNOWN_QUERIES
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


def generate_ref_score_results(
    extractor_function, dataset_name, truncation_size=-1, voc_size=1000
):
    extractor = extractor_function(voc_size)
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


################# RISK ASSESSMENT EXPERIMENTS #############


def risk_assessment():
    atk_comparison(VOC_SIZE, "risk_assessment.csv")


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
            iterable=[i for i in range(51)],
            desc="Running the experiments",
        ):
            curr_sum = (max_sum - min_sum) * (i * 2) / 100 + min_sum
            curr_n = int(2 / curr_sum)
            # Auxiliary knowledge generation
            trunc_occ_mat = occ_mat[:, 100:]
            voc = list(extractor.get_sorted_voc())[100:]
            (
                ind_mat,
                atk_mat,
                queries,
                queries_ind,  # Even if we observe all queries, we want their order
                known_queries,
            ) = generate_adv_knowledge_fixed_nb_docs(
                trunc_occ_mat, curr_n, curr_n, voc, VOC_SIZE - 100, KNOWN_QUERIES
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
            iterable=[i for i in range(51)],
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
                    "Accuracy with padding parameter 50": ref_acc_50,
                    "Overhead with padding parameter 50": overhead_50,
                    "Accuracy with padding parameter 100": ref_acc_100,
                    "Overhead with padding parameter 100": overhead_100,
                    "Accuracy with padding parameter 200": ref_acc_200,
                    "Overhead with padding parameter 200": overhead_200,
                    "Accuracy with padding parameter 500": ref_acc_500,
                    "Overhead with padding parameter 500": overhead_500,
                }
            )


########## UNIFORM SAMPLING EXPERIMENTS ###########""


def compute_p_bc(coocc_1, n_1, coocc_2, n_2):
    assert n_1 > 0 and n_2 > 0
    avg_coocc = (coocc_1 * n_1 + coocc_2 * n_2) / (n_1 + n_2)
    z_stats = (coocc_1 - coocc_2) / np.sqrt(
        avg_coocc * (1 - avg_coocc) * (1 / n_1 + 1 / n_2)
    )
    p_values = 2 * scipy.stats.norm.sf(abs(z_stats))
    p_values[np.isnan(p_values)] = 1
    p_bc = p_values.min() * (p_values.shape[0] * (p_values.shape[0] + 1)) / 2
    return p_bc


def bonferroni_experiments():
    voc_size = 1000
    extractor = apache_extractor(voc_size)
    occ_mat = extractor.occ_array

    with open("bonferroni_tests.csv", "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "Nb similar docs",
            "Nb server docs",
            "Server voc size",
            "Similarity",
            "Ref Score Acc",
            "p_bc",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for _i in tqdm.tqdm(
            iterable=[i for i in range(100)],
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
                occ_mat, 0.5, 0.5, voc, QUERYSET_SIZE, KNOWN_QUERIES
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
            p_bc = compute_p_bc(
                atk_full_coocc,
                atk_mat.shape[0],
                ind_doc_coocc,
                ind_mat.shape[0],
            )

            writer.writerow(
                {
                    "Nb similar docs": atk_mat.shape[0],
                    "Nb server docs": ind_mat.shape[0],
                    "Server voc size": voc_size,
                    "Similarity": epsilon_sim(atk_full_coocc, ind_doc_coocc),
                    "Ref Score Acc": ref_acc,
                    "p_bc": p_bc,
                }
            )


def bonferroni_experiments_by_year(result_file="bonferroni_tests_by_year.csv"):
    voc_size = 1000
    with open(result_file, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "Nb similar docs",
            "Nb server docs",
            "Year split",
            "Server voc size",
            "Similarity",
            "Ref Score Acc",
            "p_bc",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for (i, year_split) in enumerate([2003, 2005, 2007, 2009]):
            print(f"Experiment {i+1} out of 4")
            ind_docs = extract_apache_ml_by_year(to_year=year_split)
            atk_docs = extract_apache_ml_by_year(from_year=year_split)

            real_extractor = KeywordExtractor(ind_docs, voc_size)
            ind_mat = real_extractor.occ_array
            atk_mat = KeywordExtractor(
                atk_docs,
                fixed_sorted_voc=real_extractor.sorted_voc_with_occ,  # Vocabulary reused
            ).occ_array

            voc = list(real_extractor.get_sorted_voc())
            queries_ind = np.random.choice(len(voc), QUERYSET_SIZE, replace=False)
            queries = [voc[ind] for ind in queries_ind]
            known_queries = generate_known_queries(
                similar_wordlist=voc,
                stored_wordlist=queries,
                nb_queries=KNOWN_QUERIES,
            )

            ref_acc, _runtime = simulate_attack(
                RefinedScoreAttacker,
                keyword_occ_array=atk_mat,
                keyword_sorted_voc=voc,
                trapdoor_occ_array=ind_mat[:, queries_ind],
                trapdoor_sorted_voc=queries,
                nb_stored_docs=ind_mat.shape[0],
                known_queries=known_queries,
            )

            coocc_ind = ind_mat.T @ ind_mat / ind_mat.shape[0]
            coocc_atk = atk_mat.T @ atk_mat / atk_mat.shape[0]

            p_bc = compute_p_bc(
                coocc_atk,
                atk_mat.shape[0],
                coocc_ind,
                ind_mat.shape[0],
            )

            writer.writerow(
                {
                    "Nb similar docs": atk_mat.shape[0],
                    "Nb server docs": ind_mat.shape[0],
                    "Year split": year_split,
                    "Server voc size": voc_size,
                    "Similarity": epsilon_sim(coocc_atk, coocc_ind),
                    "Ref Score Acc": ref_acc,
                    "p_bc": p_bc,
                }
            )


############ MISC ###############


def fix_randomness(seed: int):
    """Fix the random seeds of numpy and random.

    This method is called before each experiment so the experiments can be executed individually.

    Args:
        seed (int): random seed
    """
    np.random.seed(seed)
    random.seed(seed)


class Laboratory:
    def __init__(self, seed):
        self.random_seed = seed

        # Setup result directory
        if not os.path.exists("results"):
            os.makedirs("results")
        os.chdir("results")

        # Setup logger
        self.logger = logging.getLogger("experiments_carbon")
        self.logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler("experiments_carbon.log")
        self.logger.addHandler(fh)
        sh = logging.StreamHandler()
        self.logger.addHandler(sh)
        self.logger.setLevel(logging.DEBUG)

        # Setup carbon tracker
        self.tracker = OfflineEmissionsTracker(
            measure_power_secs=5, country_iso_code="FRA", log_level="error"
        )
        self.tracker.start()

        self.logger.info("BEGIN EXPERIMENTS")
        self.logger.info("=================")

    def execute(self, experiment, *args, **kwargs):
        self.logger.info(
            f"BEGIN EXPERIMENT {experiment.__name__} (Args: {args}, Kwargs: {kwargs}, Random seed: {self.random_seed})"
        )
        fix_randomness(self.random_seed)

        # Pre-experiment carbon measure
        begin_emission = self.tracker.flush()
        begin_energy = self.tracker._total_energy.kWh

        experiment(*args, **kwargs)

        # Post-experiment carbon measure
        end_emission = self.tracker.flush()
        end_energy = self.tracker._total_energy.kWh
        self.logger.info(f"END EXPERIMENT {experiment.__name__}")
        self.logger.info(
            f"Carbon footprint: {end_emission - begin_emission} KgCO2e (Total: {end_emission} KgCO2e)"
        )
        self.logger.info(
            f"Energy consumption: {end_energy - begin_energy} KWh (Total: {end_energy} KWh)"
        )
        self.logger.info("----------------")
        self.random_seed += 1

    def end(self):
        self.logger.info("===============")
        self.logger.info("END EXPERIMENTS")
        emissions = self.tracker.stop()
        energy = self.tracker._total_energy.kWh
        self.logger.info(f"Total carbon footprint: {emissions} KgCO2e")
        self.logger.info(f"Energy consumption: {energy} KWh")


if __name__ == "__main__":
    lab = Laboratory(42)

    lab.execute(similarity_exploration)
    lab.execute(atk_comparison)
    lab.execute(generate_ref_score_results, enron_extractor, "enron")
    lab.execute(
        generate_ref_score_results, enron_extractor, "enron_extreme", voc_size=4000
    )
    lab.execute(generate_ref_score_results, apache_extractor, "apache")
    lab.execute(generate_ref_score_results, blogs_extractor, "blogs")
    lab.execute(risk_assessment)
    lab.execute(risk_assessment_countermeasure_tuning)
    lab.execute(risk_assessment_truncated_vocabulary)
    lab.execute(bonferroni_experiments)
    lab.execute(bonferroni_experiments_by_year)
    lab.end()
