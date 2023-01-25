from base64 import decode
from lib2to3.pgen2 import token
import random
from tkinter.tix import NoteBook
from unittest.util import _MAX_LENGTH
from xml.dom.minicompat import StringTypes
from transformers import pipeline, AutoModel, AutoTokenizer
import torch
import os
import json
import tqdm
import math
import numpy as np
import logging

# from utils import EarlyStopping, LRScheduler
from os.path import exists

#  Load tokenizers
from transformers import GPT2TokenizerFast
from transformers import Adafactor
from transformers import AdamW
import transformers
from transformers import T5Tokenizer, T5ForConditionalGeneration


# Load Model
from mkultra.tuning import GPTNeoPromptTuningLM, GPT2PromptTuningLM, T5PromptTuning
from mkultra.soft_prompt import SoftPrompt
import wandb
import argparse


# from data_loader import load_data

# logging.basicConfig(filename='log', encoding='utf-8', level=logging.DEBUG)
logging.basicConfig(format="%(asctime)s %(message)s", level=logging.DEBUG)


# pip install wandb
# wandb login
def return_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        "-d",
        type=str,
        default="TOP/",
        help="Data root path for training and test files",
    )
    parser.add_argument(
        "--model_type",
        "-mt",
        type=str,
        default="t5-small",
        help="Data root path for training and test files",
    )

    parser.add_argument(
        "--model-dir",
        "-md",
        type=str,
        default="gpt2",
        help="Specify the model directory or huggingface name.",
    )
    parser.add_argument(
        "--run_tag",
        "-rtag",
        type=str,
        default="T5Run1",
        help="RunTag",
    )

    # parser.add_argument(
    #     "--model-name",
    #     "-mn",
    #     type=str,
    #     default="google/t5-v1_1-base",
    #     help="name of the model ex : gpt2, gpt3 etc..",
    # )
    parser.add_argument(
        "--highest_step",
        "-hstep",
        type=str,
        default="89",
        help="Checkpoint with the highest step so far",
    )

    parser.add_argument(
        "--model-name",
        "-mn",
        type=str,
        default="google/t5-v1_1-base",
        help="name of the model ex : gpt2, gpt3 etc..",
    )

    parser.add_argument(
        "--test_output_file_name",
        "-TOut",
        type=str,
        default="run1_output.txt",
        help="Output of the test label file",
    )
    parser.add_argument(
        "--sp-name",
        "-sp",
        type=str,
        default="run1",
        help=" special name of model ,  ex : run1, run2 etc..",
    )
    parser.add_argument(
        "--sp_step",
        "-spSt",
        type=int,
        default=89,
        help="Highest sp step the model has taken",
    )
    # parser.add_argument(
    #     "--run_tag",
    #     "-rt",
    #     type=str,
    #     default="modelEpoch64.txt",
    #     help="Output of the test label file",
    # )
    return parser


class testModel:
    def __init__(self, args, model_path, test_prompt, test_labels, model_name) -> None:
        self.tokenized_data = []
        if os.path.exists(model_path) and "t5" in args.model_type:
            self.model = (
                T5PromptTuning.from_pretrained(args.model_type).half().to("cuda")
            )
            # Load Tokenizers
            logging.info("Loading Tokenizer")
            self.tokenizer = (
                GPT2TokenizerFast.from_pretrained("gpt2")
                if args.model_type == "gpt2"
                else T5Tokenizer.from_pretrained(args.model_type)
            )
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            # self.test_num_blocks = 0  # math.ceil(31279 / args.batch_size)
            # self.eval_num_blocks = 0  # math.ceil(4462 / args.batch_size)
            self.block_len = 1
            self.filename_for_checkpoint = (
                lambda step: f"{args.sp_name}-{args.model_val}-step-{args.sp_step}.json"
            )
            checkpoint = torch.load(model_path)
            # print(
            #     f"type of torch.load :{type(checkpoint)}, keys of checkpoint: {checkpoint.keys()}"
            # )
            self.highest_step = args.highest_step
            self.initial_prompt = (
                "Convert from src utterance to acquire the tgt semantic parse."
            )
            initial_sp = SoftPrompt.from_string(
                self.initial_prompt, self.model, self.tokenizer
            )

            # if exists(os.path.join(self.filename_for_checkpoint(self.highest_step))):
            initial_sp = SoftPrompt.from_file(
                "/home/roshansh/PromptTuning/run1-t5-step-0.json"
            )
            #     logging.info(f"Soft Prompt Object Loaded from previous checpoint")
            # else:
            #     logging.info(f"New Soft Prompt Object Created")
            self.model.set_soft_prompt(initial_sp)
            # Loading from the checkpoints
            self.model.load_state_dict(checkpoint, strict=False)

            print(f"Type of model object returned: {type(self.model)}")
        else:
            print(f"Model_path incorrect, cannot find model at {model_path}")

        self.project_dir = os.path.join("exp", model_name)
        # Look for existing project directory
        try:
            os.makedirs(self.project_dir)
            logging.info(f"Created project directory at {self.project_dir}")
        except FileExistsError:
            logging.info(f"Found project directory at {self.project_dir}")
        max_lines = 0
        with open(os.path.join(self.project_dir, "Info.txt"), "w+") as f:
            f.write(
                f"This folder has test results from model of path: {model_path}.\n of model name: {model_name}\n run using test prompts from file: {test_prompt}\n and using labels from {test_labels}\n"
            )
            max_lines = len(list(f.readlines()))
        self.maxlen = 2000
        # self.generator = pipeline(
        #     "text-generation", model=self.model, tokenizer=self.tokenizer
        # )

    def test(
        self,
        test_prompts_file_path,
        test_output_file_name="TestOutputFile.txt",
    ):
        torch.cuda.empty_cache()
        self.model.eval()
        prompt_len = 20

        with open(test_output_file_name, "w+") as w, open(
            test_prompts_file_path, "r"
        ) as f, torch.no_grad():
            for line in f.readlines():
                prompts = line.strip()
                encoded_input = self.tokenizer(
                    prompts,
                    padding="max_length",
                    max_length=512,
                    truncation=True,
                    return_tensors="pt",
                )
                decoder_input = self.tokenizer(
                    prompts, padding="max_length", truncation=True, return_tensors="pt"
                )
                # input_ids = self.tokenizer(prompts, return_tensors="pt").input_ids
                out = self.model.generate(
                    input_ids=encoded_input["input_ids"],
                    decoder_input_ids=decoder_input["input_ids"].to("cuda"),
                    do_sample=True,
                    min_length=prompt_len + 50,
                    max_length=prompt_len + 60,
                )
                print(out)
                print(self.tokenizer.decode(out[0], skip_special_tokens=True))
                w.write(self.tokenizer.decode(out[0], skip_special_tokens=True) + "\n")

            # Discard tensor that was moved to GPU
            torch.cuda.empty_cache()


if __name__ == "__main__":
    logging.info("Init")
    parser = return_parser()
    args = parser.parse_args()
    logging.info("Starting trainer")
    model_obj = testModel(
        args,
        model_path="/home/roshansh/PromptTuning/exp/NewTrainValSplitk2Run3/epoch89.pth",
        test_prompt="/home/roshansh/PromptTuning/test_prompts_k_2.txt",
        test_labels="/home/roshansh/PromptTuning/test_labels_k_2.txt",
        model_name="t5_k_2",
    )
    model_obj.test(
        test_prompts_file_path="/home/roshansh/PromptTuning/test_prompts_k_2.txt",
        test_output_file_name="TestOutputFile.txt",
    )
    # model.test()


# loss_log_path = os.path.join(self.project_dir, "loss_log.csv")
# # logging.info(self.num_training_steps)
# # self.optimizer.state["step"] = self.sp_step
# best_test_loss = float("inf")
# best_epoch = 0
# self.model.eval()
# # random.shuffle(self.test_blocks)
# test_loss = 0
# for test_block_index in tqdm.tqdm(range(len(self.test_blocks))):
#     prompts, targets, prompt_mask, target_mask = self.test_blocks[
#         test_block_index
#     ]  ## CHANGE THIS TO BATCH MODE PROMPTS SHAPE [bs,seq_len]; target should have [bs,tgt_seq_len]
#     if prompts.shape[-1] > self.maxlen:
#         logging.warning(
#             f"Found sequence of len > self.maxlen, len is {prompts.shape[-1]}"
#         )
#     (
#         test_prompts,
#         test_targets,
#         test_prompt_mask,
#         test_target_mask,
#     ) = self.test_blocks[test_block_index]
#     # prompts, targets = self.eval_blocks[eval_order[eval_step]]
#     test_input_ids = torch.LongTensor(test_prompts).cuda()
#     outputs = self.model.generate(input_ids=test_input_ids)

#     print("Type of prompts : ", type(prompts))
#     print("Shape of prompts : ", prompts.shape)
#     print("Shape of test_input_ids", test_input_ids.shape)
