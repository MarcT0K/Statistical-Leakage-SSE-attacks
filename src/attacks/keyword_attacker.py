from typing import List, Dict
from abc import ABC, abstractmethod

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

        self.number_similar_docs = keyword_occ_array.shape[0]

        # NB: kw=KeyWord; td=TrapDoor
        self.kw_voc_info = {
            word: {"vector_ind": ind, "word_occ": sum(keyword_occ_array[:, ind])}
            for ind, word in enumerate(keyword_sorted_voc)
        }

        self.td_voc_info = {
            word: {"vector_ind": ind, "word_occ": sum(trapdoor_occ_array[:, ind])}
            for ind, word in enumerate(trapdoor_sorted_voc)
        }
        self.number_real_docs = nb_stored_docs
        for kw in self.kw_voc_info.keys():
            self.kw_voc_info[kw]["word_freq"] = (
                self.kw_voc_info[kw]["word_occ"] / self.number_similar_docs
            )
        for td in self.td_voc_info.keys():
            self.td_voc_info[td]["word_freq"] = (
                self.td_voc_info[td]["word_occ"] / self.number_real_docs
            )

        self._compute_coocc_matrices(keyword_occ_array, trapdoor_occ_array)

    def _compute_coocc_matrices(
        self, keyword_occ_array: np.array, trapdoor_occ_array: np.array
    ):
        logger.info("Computing cooccurrence matrices")
        # Can be improved using scipy's sparse matrices since coocc is symmetric
        self.kw_coocc = (
            np.dot(keyword_occ_array.T, keyword_occ_array) / self.number_similar_docs
        )
        np.fill_diagonal(self.kw_coocc, 0)

        self.td_coocc = (
            np.dot(trapdoor_occ_array.T, trapdoor_occ_array) / self.number_real_docs
        )
        np.fill_diagonal(self.td_coocc, 0)

    @abstractmethod
    def predict(self, trapdoor_list: List[str]) -> Dict[str, List[str]]:
        """Returns a prediction for each trapdoor in the list.

        Args:
            trapdoor_list (List[str]): list of trapdoors to attack.

        Returns:
            Dict[str, List[str]]: dictionary with a prediction per trapdoor
        """
        raise NotImplementedError
