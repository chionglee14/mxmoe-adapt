"""Convert saved Top-K expert IDs into anonymized route features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .route_features import extract_route_features


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="JSON list shaped [tokens, top_k]")
    parser.add_argument("--experts", type=int, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    expert_ids = json.loads(args.input.read_text(encoding="utf-8"))
    features = extract_route_features(expert_ids, args.experts)
    payload = {
        "schema_version": 1,
        "source": str(args.input),
        "privacy": "contains aggregate expert counts only",
        "features": features.as_dict(),
        "route_class": features.route_class(),
        "padding_ratios": {
            str(block): features.padding_ratio(block) for block in (8, 16, 32, 64)
        },
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
