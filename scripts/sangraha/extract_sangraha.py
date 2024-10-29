__author__ = "Rudra Murthy"
__version__ = "0.1.0"

import os
import json
import argparse
from datasets import load_dataset
from tqdm import tqdm


def main(args):
    """
    Extract Sangraha data and convert it to jsonl format
    """

    os.makedirs(args.output, exist_ok=True)

    ds = load_dataset("ai4bharat/sangraha", f"verified", streaming=True)

    with open(
        os.path.join(args.output, f"eng_hin_train.jsonl"),
        "w",
        errors="ignore",
        encoding="utf8",
    ) as writer:

        for each_lang in ds:
            if each_lang not in ["hin", "eng"]:
                continue

            for each_instance in tqdm(
                ds[each_lang], desc=f"For each instance in {each_lang}"
            ):
                json.dump(each_instance, writer)
                writer.write("\n")
            writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--output", help="output path")

    args = parser.parse_args()
    main(args)
