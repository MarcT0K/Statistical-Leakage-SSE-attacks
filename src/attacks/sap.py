from typing import List

from scipy.optimize import linear_sum_assignment as hungarian

from .utils import KeywordAttacker, compute_log_binomial_probability_matrix


class SAPAttacker(KeywordAttacker):
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
        self.sorted_keywords = keyword_sorted_voc
        self.sorted_trapdoors =trapdoor_sorted_voc

    def predict(self):
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

        cost_vol = -log_prob_matrix
        row_ind, col_ind = hungarian(cost_vol)

        query_predictions_for_each_td = {
            self.sorted_trapdoors[td_ind]: self.sorted_keywords[kw_ind]
            for td_ind, kw_ind in zip(col_ind, row_ind)
        }
        return query_predictions_for_each_td
