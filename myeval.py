#!/usr/bin/env python3

from itertools import zip_longest
from tree import Tree
from typing import Counter, Dict, Optional
import argparse


class Calculator:
    def __init__(self, strict: bool = False) -> None:
        self.num_gold_nt: int = 0
        self.num_pred_nt: int = 0
        self.num_matching_nt: int = 0
        self.strict: bool = strict

    def get_metrics(self):
        precision: float = (
            self.num_matching_nt / self.num_pred_nt) if self.num_pred_nt else 0
        recall: float = (
            self.num_matching_nt / self.num_gold_nt) if self.num_gold_nt else 0
        f1: float = (2.0 * precision * recall /
                     (precision + recall)) if precision + recall else 0

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    def add_instance(self, gold_tree: Tree,
                     pred_tree: Optional[Tree] = None) -> None:
        node_info_gold: Counter = self._get_node_info(gold_tree)
        self.num_gold_nt += sum(node_info_gold.values())

        if pred_tree:
            node_info_pred: Counter = self._get_node_info(pred_tree)
            self.num_pred_nt += sum(node_info_pred.values())
            self.num_matching_nt += sum(
                (node_info_gold & node_info_pred).values())

    def _get_node_info(self, tree) -> Counter:
        nodes = tree.root.list_nonterminals()
        node_info: Counter = Counter()
        for node in nodes:
            node_info[(node.label, self._get_span(node))] += 1
        return node_info

    def _get_span(self, node):
        return node.get_flat_str_spans(
        ) if self.strict else node.get_token_span()


def evaluate_predictions(gold_filename: str, pred_filename: str) -> Dict:

    instance_count: int = 0
    exact_matches: int = 0
    invalid_preds: float = 0
    labeled_bracketing_scores = Calculator(strict=False)
    tree_labeled_bracketing_scores = Calculator(strict=True)

    with open(gold_filename) as gold_file, open(pred_filename) as pred_file:
        for gold_line, pred_line in zip_longest(gold_file, pred_file):

            try:
                gold_line = gold_line
                pred_line = pred_line
            except AttributeError:
                print("WARNING: check format and length of files")
                quit()

            try:
                gold_tree = Tree(gold_line)
                instance_count += 1
            except ValueError:
                # print(pred_line
                print("FATAL: found invalid line in gold file:", gold_line)
                quit()

            try:
                pred_tree = Tree(pred_line)
                labeled_bracketing_scores.add_instance(gold_tree, pred_tree)
                tree_labeled_bracketing_scores.add_instance(
                    gold_tree, pred_tree)
            except ValueError:
                print("WARNING: found invalid line in pred file:", pred_line)
                invalid_preds += 1
                labeled_bracketing_scores.add_instance(gold_tree)
                tree_labeled_bracketing_scores.add_instance(gold_tree)
                continue

            if str(gold_tree) == str(pred_tree):
                exact_matches += 1

    exact_match_fraction: float = (
        exact_matches / instance_count) if instance_count else 0
    tree_validity_fraction: float = (
        1 - (invalid_preds / instance_count)) if instance_count else 0

    return {
        "instance_count":
        instance_count,
        "exact_match":
        exact_match_fraction,
        "labeled_bracketing_scores":
        labeled_bracketing_scores.get_metrics(),
        "tree_labeled_bracketing_scores":
        tree_labeled_bracketing_scores.get_metrics(),
        "tree_validity":
        tree_validity_fraction
    }


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(
#         description='Evaluate TOP-representation predictions')
#     parser.add_argument(
#         'gold',
#         type=str,
#         help='file with each row in format: utterance <tab> tokenized_utterance <tab> TOP-representation'
#     )
#     parser.add_argument(
#         'pred',
#         type=str,
#         help='file with each row having a single TOP-representation')
#     args = parser.parse_args()
#
#     print(evaluate_predictions(args.gold, args.pred))
obj =  evaluate_predictions(pred_filename="t5_k2/epoch81/epoch_108_post_process.txt", gold_filename="t5_k2/test_labels_k_2.txt")
metric_out = "t5_k2/epoch81/metrics.txt"
try:
    with open(metric_out,"w") as m:
        print(obj,file=m)
except:
    f = open(metric_out, "x")
    with open(metric_out,"w") as m:
        print(obj, file=m)
