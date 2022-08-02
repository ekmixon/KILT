# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import multiprocessing
from multiprocessing.pool import ThreadPool

from tqdm import tqdm
from drqa import retriever

import kilt.kilt_utils as utils
from kilt.retrievers.base_retriever import Retriever


def _get_predictions_thread(arguments):

    id = arguments["id"]
    queries_data = arguments["queries_data"]
    topk = arguments["topk"]
    ranker = arguments["ranker"]
    logger = arguments["logger"]

    iter_ = tqdm(queries_data) if id == 0 else queries_data
    result_doc_ids = []
    result_doc_scores = []
    result_query_id = []

    for query_element in iter_:

        query = (
            query_element["query"]
            .replace(utils.ENT_END, "")
            .replace(utils.ENT_START, "")
            .strip()
        )
        result_query_id.append(query_element["id"])

        doc_ids = []
        doc_scores = []
        try:
            doc_ids, doc_scores = ranker.closest_docs(query, topk)
        except RuntimeError as e:
            if logger:
                logger.warning(f"RuntimeError: {e}")

        result_doc_ids.append(doc_ids)
        result_doc_scores.append(doc_scores)

    return result_doc_ids, result_doc_scores, result_query_id


class DrQA(Retriever):
    def __init__(self, name, retriever_model, num_threads):
        super().__init__(name)

        self.num_threads = min(num_threads, int(multiprocessing.cpu_count()))

        # initialize a ranker per thread
        self.arguments = [
            {
                "id": id,
                "ranker": retriever.get_class("tfidf")(tfidf_path=retriever_model),
            }
            for id in tqdm(range(self.num_threads))
        ]

    def feed_data(self, queries_data, logger=None):

        chunked_queries = utils.chunk_it(queries_data, self.num_threads)

        for idx, arg in enumerate(self.arguments):
            arg["queries_data"] = chunked_queries[idx]
            arg["logger"] = logger

    def run(self):
        pool = ThreadPool(self.num_threads)
        results = pool.map(_get_predictions_thread, self.arguments)

        provenance = {}

        for x in results:
            i, s, q = x
            for query_id, doc_ids in zip(q, i):
                provenance[query_id] = [
                    {"wikipedia_id": str(d_id).strip()} for d_id in doc_ids
                ]

        pool.terminate()
        pool.join()

        return provenance
