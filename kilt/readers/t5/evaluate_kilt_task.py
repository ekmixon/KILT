# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import glob
import os
from pathlib import Path

import torch
from rouge_score import rouge_scorer, scoring
from tqdm import tqdm

from finetune import Seq2seqTransformer


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def generate_answers(lns, output_file_path, model, tokenizer, batch_size, device):
    output_file = Path(output_file_path).open("w")

    model.to(device)

    # update config with specific params
    task_specific_params = model.config.task_specific_params
    if task_specific_params is not None:
        model.config.update(task_specific_params.get("nq", {}))

    for batch in tqdm(list(chunks(lns, batch_size))):
        batch = [model.config.prefix + text for text in batch]

        dct = tokenizer.batch_encode_plus(
            batch, max_length=64, return_tensors="pt", pad_to_max_length=True
        )
        input_ids = dct["input_ids"].to(device)
        attention_mask = dct["attention_mask"].to(device)

        answers = model.generate(input_ids=input_ids, attention_mask=attention_mask)
        dec = [
            tokenizer.decode(
                g, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            for g in answers
        ]

        for hypothesis in dec:
            output_file.write(hypothesis + "\n")
            output_file.flush()


def calculate_rouge(output_lns, reference_lns, score_path):
    score_file = Path(score_path).open("w")
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    aggregator = scoring.BootstrapAggregator()

    for reference_ln, output_ln in zip(reference_lns, output_lns):
        scores = scorer.score(reference_ln, output_ln)
        aggregator.add_scores(scores)

    result = aggregator.aggregate()
    score_file.write(
        f'ROUGE_1: \n{result["rouge1"]} \n\n ROUGE_2: \n{result["rouge2"]} \n\n ROUGE_L: \n{result["rougeL"]} \n\n'
    )


def run_generate():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_size",
        type=str,
        help="T5 model size, either 't5-small', 't5-base', 't5-large', 't5-3b', 't5-11b'. Defaults to 't5-base'.",
        default="t5-base",
    )
    parser.add_argument(
        "input_path",
        type=str,
        help="like nqa/test_articles_questions.txt",
    )
    parser.add_argument(
        "output_path",
        type=str,
        help="where to save summaries",
    )
    parser.add_argument(
        "reference_path", type=str, help="like nqa/test_reference_answers.txt"
    )
    parser.add_argument(
        "score_path",
        type=str,
        help="where to save the rouge score",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        required=False,
        help="batch size: how many to summarize at a time",
    )
    parser.add_argument(
        "--no_cuda",
        default=False,
        type=bool,
        help="Whether to force the execution on CPU.",
    )

    args = parser.parse_args()
    args.device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )

    source_lns = [x.rstrip() for x in open(args.input_path).readlines()]
    sq2sq = Seq2seqTransformer(args)
    checkpoints = list(
        sorted(
            glob.glob(
                os.path.join(args.output_dir, "checkpointepoch=*.ckpt"), recursive=True
            )
        )
    )

    model = sq2sq.load_from_checkpoint(checkpoints[-1]).model
    tokenizer = sq2sq.tokenizer
    generate_answers(
        source_lns, args.output_path, model, tokenizer, args.batch_size, args.device
    )
    output_lns = [x.rstrip() for x in open(args.output_path).readlines()]
    reference_lns = [x.rstrip() for x in open(args.reference_path).readlines()]

    calculate_rouge(output_lns, reference_lns, args.score_path)


if __name__ == "__main__":
    run_generate()
