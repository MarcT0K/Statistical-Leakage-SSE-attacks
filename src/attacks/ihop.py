from typing import List

import numpy as np
from scipy.optimize import linear_sum_assignment as hungarian

from .utils import compute_log_binomial_probability_matrix, KeywordAttacker


class IHOPAttacker(KeywordAttacker):
    def __init__(
        self,
        keyword_occ_array,
        trapdoor_occ_array,
        keyword_sorted_voc: List[str],
        trapdoor_sorted_voc: List[str],
        nb_stored_docs: int,
        pfree: float = 0.25,
        niter: int = 1000,
        **_kwargs  # Parameters to be ignored
    ):
        super().__init__(
            keyword_occ_array,
            trapdoor_occ_array,
            keyword_sorted_voc,
            trapdoor_sorted_voc,
            nb_stored_docs,
        )
        self.sorted_keywords = keyword_sorted_voc
        self.sorted_trapdoors = trapdoor_sorted_voc
        self.pfree = pfree
        self.niter = niter
        self._compute_coocc_matrices(keyword_occ_array, trapdoor_occ_array)

    def _build_cost_vol(self, free_keywords, free_tags, fixed_keywords, fixed_tags):
        free_kw_probs = list(
            [
                self.kw_voc_info[self.sorted_keywords[word]]["word_occ"]
                / self.nb_similar_docs
                for word in free_keywords
            ]
        )  # from the similar data set
        free_td_counts = list(
            [
                self.td_voc_info[self.sorted_trapdoors[td]]["word_occ"]
                for td in free_tags
            ]
        )  # from the leakage

        cost_vol = -compute_log_binomial_probability_matrix(
            self.nb_indexed_docs,
            free_kw_probs,
            free_td_counts,
        )
        for tag, kw in zip(fixed_tags, fixed_keywords):
            cost_vol -= compute_log_binomial_probability_matrix(
                self.nb_indexed_docs,
                self.kw_coocc[kw, free_keywords],
                self.td_coocc[tag, free_tags] * self.nb_indexed_docs,
            )
        return cost_vol

    def predict(self):

        nrep = len(self.sorted_keywords)
        ntok = len(self.sorted_trapdoors)

        unknown_toks = [i for i in range(ntok)]
        unknown_reps = [i for i in range(nrep)]

        # First matching:
        c_matrix_original = self._build_cost_vol(unknown_reps, unknown_toks, [], [])
        row_ind, col_ind = hungarian(c_matrix_original)
        replica_predictions_for_each_token = {}
        for j, i in zip(col_ind, row_ind):
            replica_predictions_for_each_token[unknown_toks[j]] = unknown_reps[i]

        # Iterate using co-occurrence:
        n_free = int(self.pfree * len(unknown_toks))
        assert n_free > 1
        for k in range(self.niter):
            random_unknown_tokens = list(np.random.permutation(unknown_toks))
            free_tokens = random_unknown_tokens[:n_free]
            fixed_tokens = random_unknown_tokens[n_free:]
            fixed_reps = [
                replica_predictions_for_each_token[token] for token in fixed_tokens
            ]
            free_replicas = [rep for rep in unknown_reps if rep not in fixed_reps]

            c_matrix = self._build_cost_vol(
                free_replicas, free_tokens, fixed_reps, fixed_tokens
            )

            row_ind, col_ind = hungarian(c_matrix)
            for j, i in zip(col_ind, row_ind):
                replica_predictions_for_each_token[free_tokens[j]] = free_replicas[i]

        return {
            self.sorted_trapdoors[td_ind]: self.sorted_keywords[kw_ind]
            for td_ind, kw_ind in replica_predictions_for_each_token.items()
        }
