import multiprocessing
from collections import Counter
from functools import reduce
from typing import Dict, List

import colorlog
import nltk
import numpy as np
import pandas as pd
import tqdm
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

from .simulation_utils import poolcontext

nltk.download("stopwords")
nltk.download("punkt")

logger = colorlog.getLogger("RaaC paper")


class OccRowComputer:
    """Callable class used to parallelize occurrence matrix computation"""

    def __init__(self, sorted_voc_with_occ):
        self.voc = [word for word, occ in sorted_voc_with_occ]

    def __call__(self, word_list):
        return [int(voc_word in word_list) for voc_word in self.voc]


class KeywordExtractor:
    """Class to extract keywords from a corpus/email set"""

    def __init__(self, corpus_df, voc_size=100, fixed_sorted_voc=None):
        glob_freq_dict = {}
        freq_dict = {}
        # Word tokenization
        nb_cores = multiprocessing.cpu_count()
        with poolcontext(processes=nb_cores) as pool:
            results = pool.starmap(
                self.extract_email_voc, enumerate(np.array_split(corpus_df, nb_cores))
            )
            freq_dict, glob_freq_dict = reduce(self._merge_results, results)

        del corpus_df
        logger.info(
            "Number of unique words (except the stopwords): %d", len(glob_freq_dict)
        )

        # Creation of the vocabulary
        glob_freq_list = nltk.FreqDist(glob_freq_dict)
        del glob_freq_dict
        glob_freq_list = (
            glob_freq_list.most_common(voc_size)
            if voc_size
            else glob_freq_list.most_common()
        )
        self.sorted_voc_with_occ = sorted(
            [(word, count) for word, count in glob_freq_list],
            key=lambda d: d[1],
            reverse=True,
        )
        logger.info("Vocabulary size: %d", len(self.sorted_voc_with_occ))
        del glob_freq_list

        # Creation of the occurrence matrix
        if fixed_sorted_voc is None:
            self.occ_array = self.build_occurrence_array(
                sorted_voc_with_occ=self.sorted_voc_with_occ, freq_dict=freq_dict
            )
        else:
            self.occ_array = self.build_occurrence_array(
                sorted_voc_with_occ=fixed_sorted_voc, freq_dict=freq_dict
            )

        if not self.occ_array.any():
            raise ValueError("occurrence array is empty")

    def get_sorted_voc(self) -> List[str]:
        """Returns the sorted vocabulary without the occurrences.

        Returns:
            List[str] -- Word list
        """
        return dict(self.sorted_voc_with_occ).keys()

    @staticmethod
    def _merge_results(res1, res2):
        merge_results2 = Counter(res1[1]) + Counter(res2[1])

        merge_results1 = res1[0].copy()
        merge_results1.update(res2[0])
        return (merge_results1, merge_results2)

    @staticmethod
    def get_voc_from_one_email(email_text, freq=False):
        stopwords_list = stopwords.words("english")
        stopwords_list.extend(["subject", "cc", "from", "to", "forward"])
        stemmer = PorterStemmer()

        stemmed_word_list = [
            stemmer.stem(word.lower())
            for sentence in sent_tokenize(email_text)
            for word in word_tokenize(sentence)
            if word.lower() not in stopwords_list and word.isalnum()
        ]
        if freq:  # (word, occurrence) sorted list
            return nltk.FreqDist(stemmed_word_list)
        else:  # Word list
            return stemmed_word_list

    @staticmethod
    def extract_email_voc(ind, dframe, one_occ_per_doc=True):
        freq_dict = {}
        glob_freq_list = {}
        for row_tuple in tqdm.tqdm(
            iterable=dframe.itertuples(),
            desc=f"Extracting corpus vocabulary (Core {ind})",
            total=len(dframe),
            position=ind,
        ):
            temp_freq_dist = KeywordExtractor.get_voc_from_one_email(
                row_tuple.mail_body, freq=True
            )
            freq_dict[row_tuple.filename] = []
            for word, freq in temp_freq_dist.items():
                freq_to_add = 1 if one_occ_per_doc else freq
                freq_dict[row_tuple.filename].append(word)
                try:
                    glob_freq_list[word] += freq_to_add
                except KeyError:
                    glob_freq_list[word] = freq_to_add
        return freq_dict, glob_freq_list

    @staticmethod
    def build_occurrence_array(
        sorted_voc_with_occ: List, freq_dict: Dict
    ) -> pd.DataFrame:
        occ_list = []
        with poolcontext(processes=multiprocessing.cpu_count()) as pool:
            for row in tqdm.tqdm(
                pool.imap_unordered(
                    OccRowComputer(sorted_voc_with_occ), freq_dict.values()
                ),
                desc="Computing the occurrence array",
                total=len(freq_dict.values()),
            ):
                occ_list.append(row)

        return np.array(occ_list, dtype=np.float64)
