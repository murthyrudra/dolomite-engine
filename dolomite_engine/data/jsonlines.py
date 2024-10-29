import glob
import os
import logging
from typing import List
from tqdm import tqdm
import jsonlines
from transformers import AutoTokenizer

from datasets import load_dataset, Dataset

from ..enums import DatasetKeys, DatasetSplit, Mode, TuningMethod
from .base import BaseDataset
from ..utils import log_rank_0


class JSONLinesDataset(BaseDataset):
    """A dataset for loading JSON lines files"""

    def __init__(
        self,
        class_args: dict,
        split: DatasetSplit,
        mode: Mode,
        tokenizer: AutoTokenizer,
        is_encoder_decoder: bool,
        tuning_method: TuningMethod,
        data_name: str,
        input_format: str,
        output_format: str,
        max_input_tokens: int,
        max_output_tokens: int,
        num_virtual_tokens: int = None,
    ) -> None:
        super().__init__(
            class_args=class_args,
            split=split,
            mode=mode,
            tokenizer=tokenizer,
            is_encoder_decoder=is_encoder_decoder,
            tuning_method=tuning_method,
            data_name=data_name,
            input_format=input_format,
            output_format=output_format,
            max_input_tokens=max_input_tokens,
            max_output_tokens=max_output_tokens,
            num_virtual_tokens=num_virtual_tokens,
        )

        self.examples = self.prepare_examples()

    def process_input(self, example):
        for index in range(len(example[DatasetKeys.input.value])):
            example[DatasetKeys.input.value][index] = self.construct_input_from_format(
                example[DatasetKeys.input.value][index]
            )
        return example

    def process_output(self, example):
        for index in range(len(example[DatasetKeys.output.value])):
            example[DatasetKeys.output.value][index] = (
                self.construct_output_from_format(
                    example[DatasetKeys.output.value][index]
                )
            )
        return example

    def prepare_examples(self) -> List[dict]:
        assert (
            "data_path" in self.class_args
        ), "JSONLinesDataset requires additional class_args `data_path`"

        examples = []
        data_files = glob.glob(
            os.path.join(self.class_args["data_path"], self.split.value, "*.jsonl")
        )

        for filename in data_files:
            log_rank_0(logging.INFO, f"Loading dataset file {filename}")

            json_dataset = load_dataset("json", data_files=filename, split="train")

            json_dataset = json_dataset.map(self.process_input, batched=True)
            if self.mode == Mode.training:
                json_dataset = json_dataset.map(self.process_output, batched=True)

            example = self.get_input_output_token_ids(json_dataset)
            examples.extend(example)

            log_rank_0(
                logging.INFO,
                f"Number of examples after loading the dataset is {len(examples)}",
            )
        return examples

    def tokenize_function_input(self, examples):
        examples["input_ids"] = self.tokenizer(
            examples["input"], add_special_tokens=False
        )["input_ids"]
        return examples

    def tokenize_function_output(self, examples):
        examples["output_ids"] = self.tokenizer(
            examples["output"], add_special_tokens=False
        )["input_ids"]
        return examples

    def get_input_output_token_ids(self, dataset: Dataset) -> dict:
        """tokenizes the input and output text

        Args:
            dataset Dataset: input text

        Returns:
            dict: an example
        """

        eos_token_id: int = self.tokenizer.eos_token_id

        tokenized_dataset = dataset.map(
            self.tokenize_function_input,
            batched=True,
            num_proc=48,
            desc="Tokenizing input examples..",
            remove_columns=["input"],
        )

        if self.mode == Mode.training:
            tokenized_dataset = tokenized_dataset.map(
                self.tokenize_function_output,
                batched=True,
                num_proc=48,
                desc="Tokenizing output examples..",
                remove_columns=["output"],
            )

        log_rank_0(logging.INFO, tokenized_dataset)

        examples = []
        for each_example in tqdm(tokenized_dataset, desc="Comverting dataset to dict"):
            example = {}
            example["input"] = each_example["input_ids"]
            example["output"] = each_example["output_ids"]

            examples.append(example)

        del tokenized_dataset

        if self.is_encoder_decoder:
            # if the model is encoder decoder
            for index in range(len(examples)):
                if self.max_input_tokens is not None:
                    # trim the input to max_input_tokens
                    examples[index]["input"] = examples[index]["input"][
                        : self.max_input_tokens - 1
                    ]
                # add eos_token_id as the last token of the input
                examples[index]["input"].append(eos_token_id)
        else:
            if self.max_input_tokens is not None:
                for index in range(len(examples)):
                    # trim the input to max_input_tokens
                    examples[index]["input"] = examples[index]["input"][
                        : self.max_input_tokens
                    ]

        if self.mode == Mode.training:
            if self.max_output_tokens is not None:
                for index in range(len(examples)):
                    # trim the output to max_output_tokens
                    examples[index]["output"] = examples[index]["output"][
                        : self.max_output_tokens - 1
                    ]

            for index in range(len(examples)):
                # add eos_token_id as the last token of the output
                examples[index]["output"].append(eos_token_id)

            if not self.is_encoder_decoder:
                # if we are training a decoder model, the input will contain the output too
                for index in range(len(examples)):
                    examples[index]["input"].extend(examples[index]["output"])

        return examples
