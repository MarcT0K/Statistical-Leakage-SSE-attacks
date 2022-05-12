import random

from typing import List, Dict

import colorlog
import numpy as np


logger = colorlog.getLogger("RaaC paper")


def generate_known_queries(
    similar_wordlist: List[str], stored_wordlist: List[str], nb_queries: int
) -> Dict[str, str]:
    """Extract random keyword which are present in the similar document set
    and in the server. So the pairs (similar_keyword, trapdoor_keyword) will
    be considered as known queries. Since the trapdoor words are not hashed
    the tuples will be like ("word","word"). We could only return the keywords
    but this tuple represents well what an attacer would have, i.e. tuple linking
    one keyword to a trapdoor they has seen.

    NB: the length of the server wordlist is the number of possible queries

    Arguments:
        similar_wordlist {List[str]} -- List of the keywords of the similar vocabulary
        trapdoor_wordlist {List[str]} -- List of the keywords of the server vocabulary
        nb_queries {int} -- Number of queries wanted

    Returns:
        Dict[str,str] -- dictionary containing known queries
    """
    candidates = set(similar_wordlist).intersection(stored_wordlist)
    return {word: word for word in random.sample(candidates, nb_queries)}


def generate_adv_knowledge(
    occ_mat: np.array,
    atk_prop: float,
    ind_prop: float,
    voc: List[str],
    nb_queries: int,
    nb_known_queries: int,
    sim_data_atk: bool = True,
):
    assert 0 < atk_prop and atk_prop <= 1
    assert 0 < ind_prop and ind_prop <= 1
    if sim_data_atk:
        assert ind_prop + atk_prop <= 1
    else:
        assert atk_prop <= ind_prop

    n_tot = occ_mat.shape[0]
    nb_ind_docs = int(n_tot * ind_prop)
    nb_atk_docs = int(n_tot * atk_prop)

    choice_ind = np.random.choice(range(n_tot), size=(nb_ind_docs,), replace=False)
    ind_docs = np.zeros(n_tot, dtype=bool)
    ind_docs[choice_ind] = True
    ind_mat = occ_mat[ind_docs, :]

    if sim_data_atk:  # Set the biggest adversary knowledge possible
        full_atk_mat = occ_mat[~ind_docs, :]
    else:
        full_atk_mat = ind_mat
    atk_max_docs = full_atk_mat.shape[0]
    atk_choice = np.random.choice(
        range(atk_max_docs),
        size=(nb_atk_docs,),
        replace=False,
    )
    atk_mat = full_atk_mat[atk_choice, :]

    queries_ind = np.random.choice(len(voc), nb_queries, replace=False)
    queries = [voc[ind] for ind in queries_ind]
    known_queries = generate_known_queries(
        similar_wordlist=voc,
        stored_wordlist=queries,
        nb_queries=nb_known_queries,
    )
    return ind_mat, atk_mat, queries, queries_ind, known_queries


def simulate_attack(attack_class, **kwargs) -> float:
    attacker = attack_class(**kwargs)
    pred = attacker.predict()
    return np.mean([word == candidate for word, candidate in pred.items()])
