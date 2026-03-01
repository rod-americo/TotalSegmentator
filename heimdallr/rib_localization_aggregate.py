import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _read_time_file(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text().strip()


def aggregate_reports(input_dir: Path, output_path: Path) -> Dict[str, Any]:
    json_paths = sorted(p for p in input_dir.glob("*.json") if p.name not in {"aggregate.json"})
    reports: List[Dict[str, Any]] = []
    findings: List[Dict[str, Any]] = []
    total_elapsed_values: List[float] = []

    for path in json_paths:
        data = json.loads(path.read_text())
        shard = path.stem
        reports.append(
            {
                "shard": shard,
                "json_path": str(path),
                "time_path": str(path.with_name(f"{shard}_time.txt")),
                "time_raw": _read_time_file(path.with_name(f"{shard}_time.txt")),
                "total_elapsed_s": data.get("total_elapsed_s"),
                "structures": data.get("structures", []),
                "missing_masks": data.get("missing_masks", []),
            }
        )
        if isinstance(data.get("total_elapsed_s"), (int, float)):
            total_elapsed_values.append(float(data["total_elapsed_s"]))
        for item in data.get("findings", []):
            findings.append(
                {
                    **item,
                    "shard": shard,
                }
            )

    findings.sort(key=lambda item: item["structure"])
    suspicious = [item for item in findings if item.get("segments")]

    aggregate = {
        "input_dir": str(input_dir),
        "num_shards": len(reports),
        "num_ribs": len(findings),
        "shards": reports,
        "wall_clock_estimate_s": round(max(total_elapsed_values), 3) if total_elapsed_values else None,
        "sum_elapsed_s": round(sum(total_elapsed_values), 3) if total_elapsed_values else None,
        "num_suspicious_ribs": len(suspicious),
        "suspicious_ribs": [
            {
                "structure": item["structure"],
                "elapsed_s": item.get("elapsed_s"),
                "segments": item.get("segments", []),
                "shard": item["shard"],
            }
            for item in suspicious
        ],
        "findings": findings,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(aggregate, f, indent=2)
    return aggregate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate rib localization shard reports.")
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    aggregate_reports(args.input_dir, args.output)


if __name__ == "__main__":
    main()
