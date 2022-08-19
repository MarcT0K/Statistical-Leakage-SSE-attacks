from abc import ABC, abstractmethod
from typing import Dict, List

import colorlog
import numpy as np

logger = colorlog.getLogger("RaaC paper")


class KeywordAttacker(ABC):
    def __init__(
        self,
        keyword_occ_array,
        trapdoor_occ_array,
        keyword_sorted_voc: List[str],
        trapdoor_sorted_voc: List[str],
        nb_stored_docs: int,
    ):
        self.kw_coocc = None
        self.td_coocc = None
        self.nb_similar_docs = keyword_occ_array.shape[0]
        self.nb_indexed_docs = nb_stored_docs

        # NB: kw=KeyWord; td=TrapDoor
        self.kw_voc_info = {
            word: {"vector_ind": ind, "word_occ": sum(keyword_occ_array[:, ind])}
            for ind, word in enumerate(keyword_sorted_voc)
        }

        self.td_voc_info = {
            word: {"vector_ind": ind, "word_occ": sum(trapdoor_occ_array[:, ind])}
            for ind, word in enumerate(trapdoor_sorted_voc)
        }

        for word in self.kw_voc_info.keys():
            self.kw_voc_info[word]["word_freq"] = (
                self.kw_voc_info[word]["word_occ"] / self.nb_similar_docs
            )
        for trapd in self.td_voc_info.keys():
            self.td_voc_info[trapd]["word_freq"] = (
                self.td_voc_info[trapd]["word_occ"] / self.nb_indexed_docs
            )

    def _compute_coocc_matrices(
        self, keyword_occ_array: np.array, trapdoor_occ_array: np.array
    ):
        logger.info("Computing cooccurrence matrices")
        # Can be improved using scipy's sparse matrices since coocc is symmetric
        self.kw_coocc = (
            np.dot(keyword_occ_array.T, keyword_occ_array) / self.nb_similar_docs
        )
        np.fill_diagonal(self.kw_coocc, 0)

        self.td_coocc = (
            np.dot(trapdoor_occ_array.T, trapdoor_occ_array) / self.nb_indexed_docs
        )
        np.fill_diagonal(self.td_coocc, 0)

    @abstractmethod
    def predict(self) -> Dict[str, str]:
        """Returns a prediction for each unknown trapdoor.

        Returns:
            Dict[str, str]: dictionary with a prediction per trapdoor
        """
        raise NotImplementedError


def _log_binomial(n_obs, beta):
    """Computes an approximation of log(binom(n, n*alpha)) for alpha < 1"""
    if beta == 0 or beta == 1:
        return 0
    elif beta < 0 or beta > 1:
        raise ValueError(f"Beta cannot be negative or greater than 1 ({beta})")
    else:
        entropy = -beta * np.log(beta) - (1 - beta) * np.log(1 - beta)
        return n_obs * entropy - 0.5 * np.log(2 * np.pi * n_obs * beta * (1 - beta))


def compute_log_binomial_probability_matrix(ntrials, probabilities, observations):
    """
    Computes the logarithm of binomial probabilities of each pair of probabilities and observations.
    :param ntrials: number of binomial trials
    :param probabilities: vector with probabilities
    :param observations: vector with integers (observations)
    :return log_matrix: |probabilities|x|observations| matrix with the log binomial probabilities
    """
    probabilities = np.array(probabilities)
    probabilities[probabilities == 0] = (
        min(probabilities[probabilities > 0]) / 100
    )  # To avoid numerical errors. An error would mean the adversary information is very off.
    log_binom_term = np.array(
        [_log_binomial(ntrials, obs / ntrials) for obs in observations]
    )  # ROW TERM
    column_term = np.array(
        [np.log(probabilities) - np.log(1 - np.array(probabilities))]
    ).T  # COLUMN TERM
    last_term = np.array(
        [ntrials * np.log(1 - np.array(probabilities))]
    ).T  # COLUMN TERM
    log_matrix = log_binom_term + np.array(observations) * column_term + last_term
    return log_matrix
