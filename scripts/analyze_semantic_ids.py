"""Report semantic-ID collisions and codebook use for an n x 4 codes file."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("codes", type=Path)
    parser.add_argument("--codebook-size", type=int, default=256)
    args = parser.parse_args()
    codes = np.load(args.codes)
    prefixes, counts = np.unique(codes[:, :-1], axis=0, return_counts=True)
    size = args.codebook_size
    report = {
        "items": int(len(codes)),
        "unique_full_ids": int(len(np.unique(codes, axis=0))),
        "unique_prefixes": int(len(prefixes)),
        "prefix_collision_rate": float(1 - len(prefixes) / len(codes)),
        "largest_prefix_group": int(counts.max()),
        "items_in_shared_prefixes": int(counts[counts > 1].sum()),
        "prefix_groups_over_codebook": int((counts > size).sum()),
        "items_over_codebook": int(counts[counts > size].sum()),
        "max_last_code": int(codes[:, -1].max()),
        "codes_used_per_level": [int(len(np.unique(codes[:, i]))) for i in range(codes.shape[1] - 1)],
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
