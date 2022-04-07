import csv

import numpy as np
import scipy

from src.common import KeywordExtractor, compute_occ_mat
from src.query_generator import QueryResultExtractor
from src.email_extraction import (
    extract_sent_mail_contents,
    extract_apache_ml_by_year,
    extract_apache_ml,
)


epsilon_sim = lambda coocc_1, coocc_2: np.linalg.norm(coocc_1 - coocc_2)


def Z_mat_to_bonferroni(Z_mat):
    z_flat = Z_mat.flatten()
    z_flat = z_flat[~np.isnan(z_flat)]
    p = (
        (scipy.stats.norm.sf(z_flat) * 2).min()
        * (Z_mat.shape[0] * (Z_mat.shape[0] + 1))
        / 2
    )
    return p


def binom_test_p_values(coocc_1, n_1, coocc_2, n_2):
    assert n_1 > 0 and n_2 > 0
    avg_coocc = (coocc_1 * n_1 + coocc_2 * n_2) / (n_1 + n_2)
    z_stats = (coocc_1 - coocc_2) / np.sqrt(
        avg_coocc * (1 - avg_coocc) * (1 / n_1 + 1 / n_2)
    )
    return 2 * scipy.stats.norm.sf(abs(z_stats))


def prop_reject(p_values_mat, threshold):
    m = p_values_mat.shape[0]
    test_mat = np.tril(p_values_mat)  # symmetric matrix in our case
    nb_rejections = (test_mat < threshold).sum() - m * (
        m - 1
    ) / 2  # we remove the elements we set to 0
    return nb_rejections / (m * (m + 1) / 2)


NB_REP = 5


def apache_sim_by_year(result_file="tests_apache_sim_by_year.csv"):
    epsilon_sim = lambda coocc_1, coocc_2: np.linalg.norm(coocc_1 - coocc_2)
    voc_size = 1000
    with open(result_file, "w", newline="") as csvfile:
        fieldnames = [
            "Nb similar docs",
            "Nb server docs",
            "Year split",
            "Server voc size",
            "Similarity",
            "Tests .05",
            "Tests .01",
            "Tests .001",
            "Avg p-value",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for (i, year_split) in enumerate([2003, 2005, 2007, 2009]):
            print(f"Experiment {i+1} out of {len([2003, 2005, 2007, 2009])}")
            stored_docs = extract_apache_ml_by_year(to_year=year_split)
            similar_docs = extract_apache_ml_by_year(from_year=year_split)

            real_extractor = QueryResultExtractor(stored_docs, voc_size)
            sim_occ_mat = compute_occ_mat(
                similar_docs, real_extractor.sorted_voc_with_occ
            )

            coocc_td = (
                real_extractor.occ_array.T
                @ real_extractor.occ_array
                / real_extractor.occ_array.shape[0]
            )
            coocc_kw = sim_occ_mat.T @ sim_occ_mat / sim_occ_mat.shape[0]

            p_values = binom_test_p_values(
                coocc_kw,
                sim_occ_mat.shape[0],
                coocc_td,
                real_extractor.occ_array.shape[0],
            )

            writer.writerow(
                {
                    "Nb similar docs": sim_occ_mat.shape[0],
                    "Nb server docs": real_extractor.occ_array.shape[0],
                    "Year split": year_split,
                    "Server voc size": voc_size,
                    "Similarity": epsilon_sim(coocc_kw, coocc_td),
                    "Tests .05": prop_reject(p_values, 0.05),
                    "Tests .01": prop_reject(p_values, 0.01),
                    "Tests .001": prop_reject(p_values, 0.001),
                    "Avg p-value": np.nansum(np.tril(p_values))
                    / ((voc_size + 1) * voc_size / 2),
                }
            )


def similar_metric(result_file="tests_apache_sim.csv"):
    experiment_params = [
        0.1 * (j + 1) for j in range(5) for _k in range(NB_REP)
    ]  # size of the attacker document set

    epsilon_sim = lambda coocc_1, coocc_2: np.linalg.norm(coocc_1 - coocc_2)
    voc_size = 1000
    with open(result_file, "w", newline="") as csvfile:
        fieldnames = [
            "Nb similar docs",
            "Nb server docs",
            "Server voc size",
            "Similarity",
            "Tests .05",
            "Tests .01",
            "Tests .001",
            "Avg p-value",
            "Bonferroni",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        apache = extract_apache_ml()

        extractor = KeywordExtractor(apache, voc_size, 1)
        occ_mat = extractor.occ_array
        n_tot = extractor.occ_array.shape[0]

        for (i, attacker_size_ratio) in enumerate(experiment_params):
            print(f"Experiment {i+1} out of {len(experiment_params)}")
            choice_serv = np.random.choice(
                range(n_tot), size=(int(n_tot * 0.6),), replace=False
            )
            ind_serv = np.zeros(n_tot, dtype=bool)
            ind_serv[choice_serv] = True
            serv_mat = occ_mat[ind_serv, :]
            kw_mat = occ_mat[~ind_serv, :]
            kw_max_docs = kw_mat.shape[0]

            sub_choice_kw = np.random.choice(
                range(kw_max_docs),
                size=(int(kw_max_docs * attacker_size_ratio),),
                replace=False,
            )
            coocc_td = serv_mat.T @ serv_mat / serv_mat.shape[0]
            coocc_kw = (
                kw_mat[sub_choice_kw, :].T
                @ kw_mat[sub_choice_kw, :]
                / kw_mat[sub_choice_kw, :].shape[0]
            )

            p_values = binom_test_p_values(
                coocc_kw,
                kw_mat[sub_choice_kw, :].shape[0],
                coocc_td,
                serv_mat.shape[0],
            )

            writer.writerow(
                {
                    "Nb similar docs": kw_mat[sub_choice_kw, :].shape[0],
                    "Nb server docs": serv_mat.shape[0],
                    "Server voc size": voc_size,
                    "Similarity": epsilon_sim(coocc_kw, coocc_td),
                    "Tests .05": prop_reject(p_values, 0.05),
                    "Tests .01": prop_reject(p_values, 0.01),
                    "Tests .001": prop_reject(p_values, 0.001),
                    "Avg p-value": np.nansum(np.tril(p_values))
                    / ((voc_size + 1) * voc_size / 2),
                    "Bonferroni": Z_mat_to_bonferroni(p_values),
                }
            )
