#!/usr/bin/env python3
"""Build gold-free stage features for adaptive verification."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from answer_extraction_v2 import attach_extractions  # noqa: E402
try:  # noqa: E402
    from .adaptive_core import (
        build_feature_rows,
        read_jsonl,
        validate_config,
        write_jsonl,
    )
except ImportError:  # Direct script execution.
    from adaptive_core import (
        build_feature_rows,
        read_jsonl,
        validate_config,
        write_jsonl,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--extractions", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=BASE / "config.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    gate = parser.add_mutually_exclusive_group()
    gate.add_argument("--require-reviewed", action="store_true")
    gate.add_argument("--allow-provisional", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    validate_config(config)
    require_reviewed = args.require_reviewed or not args.allow_provisional
    generations, extraction_status = attach_extractions(
        read_jsonl(args.generations),
        read_jsonl(args.extractions),
        require_reviewed=require_reviewed,
    )
    primary = build_feature_rows(generations, config)
    sensitivity = build_feature_rows(generations, config, all_orders=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "risk_features.jsonl", primary)
    write_jsonl(args.output_dir / "order_sensitivity_features.jsonl", sensitivity)
    manifest = {
        "experiment_version": config["experiment_version"],
        "result_status": extraction_status["result_status"],
        "extraction_status": extraction_status,
        "groups": len(primary),
        "order_feature_rows": len(sensitivity),
        "contains_gold": False,
    }
    (args.output_dir / "feature_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
