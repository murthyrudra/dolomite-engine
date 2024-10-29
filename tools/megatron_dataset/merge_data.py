import os
from argparse import ArgumentParser, Namespace
import logging
import sys

sys.path.append("/dccstor/cssblr/rmurthyv/IBM/dolomite_new/dolomite-engine/")

from dolomite_engine.data.megatron.indexed_dataset import (
    MMapIndexedDataset,
    MMapIndexedDatasetBuilder,
    get_bin_path,
    get_idx_path,
)

from dolomite_engine.utils import log_rank_0, set_logger


set_logger()


def get_args() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument(
        "--input-prefixes",
        type=str,
        nargs="+",
        required=True,
        help="Path to directory containing all document files to merge",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        required=True,
        help="Path to binary output file without suffix",
    )

    args = parser.parse_args()

    for inp in args.input_prefixes:
        bin = get_bin_path(inp)
        idx = get_idx_path(inp)
        assert os.path.exists(bin) and os.path.exists(
            idx
        ), f"{inp} is not a valid prefix and doesn't exist"

    return args


def main() -> None:
    args = get_args()

    dtype = MMapIndexedDataset(args.input_prefixes[0]).index.dtype

    builder = MMapIndexedDatasetBuilder(get_bin_path(args.output_prefix), dtype=dtype)

    for input_prefix in args.input_prefixes:
        log_rank_0(logging.INFO, f"Processing prefix = {input_prefix}")
        builder.add_index(input_prefix)

    log_rank_0(logging.INFO, f"Saving to = {args.output_prefix}")
    builder.finalize(get_idx_path(args.output_prefix))


if __name__ == "__main__":
    main()
