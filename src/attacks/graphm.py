import os
import subprocess
import stat
import shutil

from typing import List
from .utils import KeywordAttacker, compute_log_binomial_probability_matrix
from ..conf import GRAPHM_PATH


class GraphMatchingAttacker(KeywordAttacker):
    def __init__(
        self,
        keyword_occ_array,
        trapdoor_occ_array,
        keyword_sorted_voc: List[str],
        trapdoor_sorted_voc: List[str],
        nb_stored_docs: int,
    ):
        super().__init__(
            keyword_occ_array,
            trapdoor_occ_array,
            keyword_sorted_voc,
            trapdoor_sorted_voc,
            nb_stored_docs,
        )
        self._compute_coocc_matrices(keyword_occ_array, trapdoor_occ_array)
        self.sorted_keywords = keyword_sorted_voc
        self.sorted_trapdoors = trapdoor_sorted_voc
        self.alpha = 0.5

        kw_probs_train = list(
            [
                info["word_occ"] / self.nb_similar_docs
                for _word, info in self.kw_voc_info.items()
            ]
        )  # from the similar data set
        kw_counts_test = list(
            [info["word_occ"] for _td, info in self.td_voc_info.items()]
        )  # from the leakage

        # Computing the cost matrix
        log_prob_matrix = compute_log_binomial_probability_matrix(
            self.nb_indexed_docs, kw_probs_train, kw_counts_test
        )

        self.cost_vol = -log_prob_matrix

    def set_alpha(self, new_alpha: float):
        assert new_alpha >= 0 and new_alpha <= 1
        self.alpha = new_alpha

    def predict(self):
        folder_path = "./tmp_results"

        os.makedirs(folder_path)  # We create and destroy the subdir in this function

        try:
            with open(os.path.join(folder_path, "graph_1"), "wb") as file_ptr:
                write_matrix_to_file_ascii(file_ptr, self.kw_coocc)

            with open(os.path.join(folder_path, "graph_2"), "wb") as file_ptr:
                write_matrix_to_file_ascii(file_ptr, self.td_coocc)

            if self.alpha > 0:
                with open(os.path.join(folder_path, "c_matrix"), "wb") as file_ptr:
                    write_matrix_to_file_ascii(file_ptr, self.cost_vol)

            with open(
                os.path.join(folder_path, "config.txt"), "w", encoding="utf-8"
            ) as file_ptr:
                file_ptr.write(
                    return_config_text(
                        ["PATH"],
                        self.alpha,
                        os.path.relpath(folder_path, "."),
                        "graphm_output",
                    )
                )

            test_script_path = os.path.join(folder_path, "run_script")
            with open(test_script_path, "w", encoding="utf-8") as file_ptr:
                file_ptr.write("#!/bin/sh\n")
                file_ptr.write(
                    f"{GRAPHM_PATH} {os.path.relpath(folder_path, '.')}/config.txt\n"
                )
            os_st = os.stat(test_script_path)
            os.chmod(test_script_path, os_st.st_mode | stat.S_IEXEC)

            # RUN THE ATTACK
            subprocess.run(
                [os.path.join(folder_path, "run_script")],
                capture_output=True,
                check=True,
            )

            results = []
            with open(
                os.path.relpath(folder_path, ".") + "/graphm_output",
                "r",
                encoding="utf-8",
            ) as file_ptr:
                while file_ptr.readline() != "Permutations:\n":
                    pass
                file_ptr.readline()  # This is the line with the attack names (only PATH, in theory)
                for line in file_ptr:
                    results.append(int(line) - 1)  # Line should be a single integer now

            query_predictions_for_each_tag = {}
            for i, tag in enumerate(self.sorted_trapdoors):
                query_predictions_for_each_tag[tag] = self.sorted_keywords[
                    results.index(i)
                ]
        except:
            shutil.rmtree(folder_path)
            raise

        shutil.rmtree(folder_path)
        return query_predictions_for_each_tag


# UTILS


def print_matrix(matrix, precision=2):
    for row in matrix:
        for elem in row:
            print("{val:.{pre}f} ".format(pre=precision, val=elem), end="")
        print("")
    print("")


def write_matrix_to_file_ascii(file, matrix):
    for row in matrix:
        row_str = " ".join("{:.6f}".format(val) for val in row) + "\n"
        file.write(row_str.encode("ascii"))


def return_config_text(algorithms_list, alpha, relpath_experiments, out_filename):
    """relpath_experiments: path from where we run graphm to where the graph files are"""

    config_text = """//*********************GRAPHS**********************************
//graph_1,graph_2 are graph adjacency matrices,
//C_matrix is the matrix of local similarities  between vertices of graph_1 and graph_2.
//If graph_1 is NxN and graph_2 is MxM then C_matrix should be NxM
graph_1={relpath:s}/graph_1 s
graph_2={relpath:s}/graph_2 s
C_matrix={relpath:s}/c_matrix s
//*******************ALGORITHMS********************************
//used algorithms and what should be used as initial solution in corresponding algorithms
algo={alg:s} s
algo_init_sol={init:s} s
solution_file=solution_im.txt s
//coeficient of linear combination between (1-alpha_ldh)*||graph_1-P*graph_2*P^T||^2_F +alpha_ldh*C_matrix
alpha_ldh={alpha:.6f} d
cdesc_matrix=A c
cscore_matrix=A c
//**************PARAMETERS SECTION*****************************
hungarian_max=10000 d
algo_fw_xeps=0.01 d
algo_fw_feps=0.01 d
//0 - just add a set of isolated nodes to the smallest graph, 1 - double size
dummy_nodes=0 i
// fill for dummy nodes (0.5 - these nodes will be connected with all other by edges of weight 0.5(min_weight+max_weight))
dummy_nodes_fill=0 d
// fill for linear matrix C, usually that's the minimum (dummy_nodes_c_coef=0),
// but may be the maximum (dummy_nodes_c_coef=1)
dummy_nodes_c_coef=0.01 d
qcvqcc_lambda_M=10 d
qcvqcc_lambda_min=1e-5 d
//0 - all matching are possible, 1-only matching with positive local similarity are possible
blast_match=0 i
blast_match_proj=0 i
//****************OUTPUT***************************************
//output file and its format
exp_out_file={relpath:s}/{out:s} s
exp_out_format=Parameters Compact Permutation s
//other
debugprint=0 i
debugprint_file=debug.txt s
verbose_mode=1 i
//verbose file may be a file or just a screen:cout
verbose_file=cout s
""".format(
        alg=" ".join(alg for alg in algorithms_list),
        init=" ".join("unif" for _ in algorithms_list),
        out=out_filename,
        alpha=alpha,
        relpath=relpath_experiments,
    )
    return config_text
