import logging
import math
import multiprocessing
import random
from contextlib import contextmanager
from datetime import datetime
from typing import Dict, List, Tuple

import colorlog
import numpy as np

logger = colorlog.getLogger("RaaC paper")


def setup_logger():
    logger.handlers = []  # Reset handlers
    handler = colorlog.StreamHandler()
    handler.setFormatter(
        colorlog.ColoredFormatter(
            "%(log_color)s[%(asctime)s %(levelname)s]%(reset)s %(module)s: "
            "%(white)s%(message)s",
            datefmt="%H:%M:%S",
            reset=True,
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red",
            },
        )
    )
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)


@contextmanager
def poolcontext(*args, **kwargs):
    """Context manager to standardize the parallelized functions."""
    pool = multiprocessing.Pool(*args, **kwargs)
    yield pool
    pool.terminate()


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
    n_tot = occ_mat.shape[0]
    nb_ind_docs = int(n_tot * ind_prop)
    nb_atk_docs = int(n_tot * atk_prop)

    return generate_adv_knowledge_fixed_nb_docs(
        occ_mat,
        nb_atk_docs,
        nb_ind_docs,
        voc,
        nb_queries,
        nb_known_queries,
        sim_data_atk,
    )


def generate_adv_knowledge_fixed_nb_docs(
    occ_mat: np.array,
    n_atk: int,
    n_ind: int,
    voc: List[str],
    nb_queries: int,
    nb_known_queries: int,
    sim_data_atk: bool = True,
):

    n_tot = occ_mat.shape[0]
    assert 0 < n_atk and n_atk <= n_tot
    assert 0 < n_ind and n_ind <= n_tot
    if sim_data_atk:
        assert n_ind + n_atk <= n_tot
    else:
        assert n_atk <= n_ind

    choice_ind = np.random.choice(range(n_tot), size=(n_ind,), replace=False)
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
        size=(n_atk,),
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


def simulate_attack(attack_class, **kwargs) -> Tuple[float, float]:
    attacker = attack_class(**kwargs)
    start = datetime.now()
    pred = attacker.predict()
    end = datetime.now()

    acc = np.mean([word == candidate for word, candidate in pred.items()])
    runtime = (end - start).total_seconds()
    return acc, runtime


def padding_countermeasure(input_array, padding_threshold=500):
    """Adds an access pattern padding to an index matrix.
    The "padding threshold" defines the number n such as all the keywords have an occurence divisible by n.
    The algorithms creates random entries to meet this requirement.
    Extremely simple mitigation but not optimized.

    Ref: D.Cash, P.Grubbs, J.Perry and T. Ristenpart. Leakage-abuse attacks against searchable encryption. 2015
    """
    occ_array = input_array.copy()
    _, ncol = occ_array.shape
    number_real_entries = np.sum(occ_array)
    for j in range(ncol):
        nb_entries = sum(occ_array[:, j])
        nb_fake_entries_to_add = int(
            math.ceil(nb_entries / padding_threshold) * padding_threshold - nb_entries
        )
        possible_fake_entries = list(np.argwhere(occ_array[:, j] == 0).flatten())
        if len(possible_fake_entries) < nb_fake_entries_to_add:
            # We need more documents to generate enough fake entries
            # So we generate fake document IDs
            fake_documents = np.zeros(
                (nb_fake_entries_to_add - len(possible_fake_entries), ncol)
            )
            occ_array = np.concatenate((occ_array, fake_documents))
            possible_fake_entries = list(np.argwhere(occ_array[:, j] == 0).flatten())
        fake_entries = random.sample(possible_fake_entries, nb_fake_entries_to_add)
        occ_array[fake_entries, j] = 1

    number_observed_entries = np.sum(occ_array)
    overhead = number_observed_entries / number_real_entries
    return occ_array, overhead
