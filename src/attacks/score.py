import multiprocessing

from functools import partial, reduce
from typing import Dict, List, Optional, Tuple

import colorlog
import numpy as np
import tqdm

from .keyword_attacker import KeywordAttacker
from ..common import poolcontext

logger = colorlog.getLogger("RaaC paper")



class AbstractScoreAttacker(KeywordAttacker):
    def __init__(
        self,
        keyword_occ_array,
        trapdoor_occ_array,
        keyword_sorted_voc: List[str],
        trapdoor_sorted_voc: List[str],
        nb_stored_docs: int,
        known_queries: Dict[str, str],
    ):
        """Initialization of the matchmaker

        Arguments:
            keyword_occ_array {np.array} -- Keyword occurrence (row: similar documents; columns: keywords)
            trapdoor_occ_array {np.array} -- Trapdoor occurrence (row: stored documents; columns: trapdoors)
                                            the documents are unknown (just the identifier has
                                            been seen by the attacker)
            keyword_sorted_voc {List[str]} -- Keywoord vocabulary extracted from similar documents.
            known_queries {Dict[str, str]} -- Queries known by the attacker
            trapdoor_sorted_voc {List[str]} -- The trapdoor voc can be a sorted list of hashes
                                                            to hide the underlying keywords.
        """
        super().__init__(
            keyword_occ_array,
            trapdoor_occ_array,
            keyword_sorted_voc,
            trapdoor_sorted_voc,
            nb_stored_docs,
        )

        if not known_queries:
            raise ValueError("Known queries are mandatory.")
        if len(known_queries.values()) != len(set(known_queries.values())):
            raise ValueError("Several trapdoors are linked to the same keyword.")

        self._known_queries = known_queries.copy()  # Keys: trapdoor, Values: keyword

        self._refresh_reduced_coocc()

    def _refresh_reduced_coocc(self):
        """Refresh the co-occurrence matrix based on the known queries."""
        ind_known_kw = [
            self.kw_voc_info[kw]["vector_ind"] for kw in self._known_queries.values()
        ]
        self.kw_reduced_coocc = self.kw_coocc[:, ind_known_kw]
        ind_known_td = [
            self.td_voc_info[td]["vector_ind"] for td in self._known_queries.keys()
        ]
        self.td_reduced_coocc = self.td_coocc[:, ind_known_td]

    def _sub_pred(
        self, _ind: int, td_list: List[str], cluster_max_size=1
    ) -> List[Tuple[str, List[str], float]]:
        """
        Sub-function used to parallelize the prediction.

        Returns:
            List[Tuple[str,List[str], float]] -- a list of tuples (trapdoor, [prediction], certainty) or
                                                    (trapdoor, cluster of predictions, certainty)
        """
        if cluster_max_size < 1:
            raise ValueError("The cluster size must be one or more.")

        prediction = []
        for trapdoor in td_list:
            try:
                trapdoor_ind = self.td_voc_info[trapdoor]["vector_ind"]
            except KeyError:
                logger.warning(f"Unknown trapdoor: {trapdoor}")
                prediction.append(
                    ((trapdoor, [], 0) if cluster_max_size > 1 else (trapdoor, [""], 0))
                )
                continue
            trapdoor_vec = self.td_reduced_coocc[trapdoor_ind]

            score_list = []
            for keyword, kw_info in self.kw_voc_info.items():
                # Computes the matching with each keyword of the vocabulary extracted from similar documents
                keyword_vec = self.kw_reduced_coocc[kw_info["vector_ind"]]
                vec_diff = keyword_vec - trapdoor_vec
                # Distance between the keyword point and the trapdoor point in the known-queries sub-vector space
                td_kw_distance = np.linalg.norm(vec_diff)
                if td_kw_distance:
                    score = -np.log(td_kw_distance)
                else:  # If distance==0 => Perfect match
                    score = np.inf
                score_list.append((keyword, score))
            score_list.sort(key=lambda tup: tup[1])

            if cluster_max_size > 1:  # Cluster mode
                kw_cluster, certainty = self.best_candidate_clustering(
                    score_list,
                    cluster_max_size=cluster_max_size,
                )
                prediction.append((trapdoor, kw_cluster, certainty))
            else:  # No clustering
                best_candidate = score_list[-1][0]
                certainty = score_list[-1][1] - score_list[-2][1]
                prediction.append((trapdoor, [best_candidate], certainty))
        return prediction


class ScoreAttacker(AbstractScoreAttacker):
    def __init__(
        self,
        keyword_occ_array,
        trapdoor_occ_array,
        keyword_sorted_voc,
        trapdoor_sorted_voc,
        nb_stored_docs,
        known_queries,
    ):
        super().__init__(
            keyword_occ_array,
            trapdoor_occ_array,
            keyword_sorted_voc,
            trapdoor_sorted_voc,
            nb_stored_docs,
            known_queries,
        )

    def predict(self, trapdoor_list: List[str]) -> Dict[str, List[str]]:
        predictions = {}
        nb_cores = multiprocessing.cpu_count()
        logger.info("Evaluating every possible keyword-trapdoor pair")
        with poolcontext(processes=nb_cores) as pool:
            pred_func = partial(self._sub_pred)
            results = pool.starmap(
                pred_func,
                enumerate([trapdoor_list[i::nb_cores] for i in range(nb_cores)]),
            )
            pred_list = reduce(lambda x, y: x + y, results)
            predictions = {td: kw for td, kw, _certainty in pred_list}
        return predictions


class RefinedScoreAttacker(AbstractScoreAttacker):
    def __init__(
        self,
        keyword_occ_array,
        trapdoor_occ_array,
        keyword_sorted_voc,
        trapdoor_sorted_voc,
        nb_stored_docs,
        known_queries,
        ref_speed=0,
    ):
        super().__init__(
            keyword_occ_array,
            trapdoor_occ_array,
            keyword_sorted_voc,
            trapdoor_sorted_voc,
            nb_stored_docs,
            known_queries,
        )

        if ref_speed < 1:
            # Default refinement speed: 5% of the total number of trapdoors
            self.ref_speed = int(0.05 * len(self.td_voc_info))
        else:
            self.ref_speed = ref_speed

    def predict(self, trapdoor_list: List[str]) -> Dict[str, List[str]]:

        old_known = self._known_queries.copy()
        nb_cores = multiprocessing.cpu_count()
        unknown_td_list = list(trapdoor_list)

        final_results = []
        with poolcontext(processes=nb_cores) as pool, tqdm.tqdm(
            total=len(trapdoor_list), desc="Refining predictions"
        ) as pbar:
            while True:
                prev_td_nb = len(unknown_td_list)
                unknown_td_list = [
                    td for td in unknown_td_list if td not in self._known_queries.keys()
                ]  # Removes the known trapdoors
                pbar.update(prev_td_nb - len(unknown_td_list))
                pred_func = partial(
                    self._sub_pred,
                    cluster_max_size=1,
                )
                results = pool.starmap(  # Launch parallel predictions
                    pred_func,
                    enumerate([unknown_td_list[i::nb_cores] for i in range(nb_cores)]),
                )
                results = reduce(lambda x, y: x + y, results)

                # Extract the best preditions (must be single-point clusters)
                single_point_results = [tup for tup in results if len(tup[1]) == 1]
                single_point_results.sort(key=lambda tup: tup[2])

                if (
                    len(single_point_results) < self.ref_speed
                ):  # Not enough single-point predictions
                    final_results = [
                        (td, candidates) for td, candidates, _sep in results
                    ]
                    break

                # Add the pseudo-known queries.
                new_known = {
                    td: candidates[-1]
                    for td, candidates, _sep in single_point_results[-self.ref_speed :]
                }
                self._known_queries.update(new_known)
                self._refresh_reduced_coocc()

        # Concatenate known queries and last results
        prediction = {
            td: [kw] for td, kw in self._known_queries.items() if td in trapdoor_list
        }
        prediction.update(dict(final_results))

        # Reset the known queries
        self._known_queries = old_known
        self._refresh_reduced_coocc()
        return prediction
