"""
Run the base coding agent (from the codebase, no DGM evolution) on a single
selected task from either the Polyglot or SWE-bench benchmarks.

All logs (chat history, docker logs, predictions, eval results) are written
to the ``test_agent/`` directory.

Usage examples
--------------
# Polyglot – pick by instance_id
python run_test_agent.py --dataset polyglot --instance_id python__dominoes

# Polyglot – pick from a built-in subset
python run_test_agent.py --dataset polyglot --subset small --sample_index 0

# SWE-bench – pick by instance_id
python run_test_agent.py --dataset swe --instance_id django__django-10973

# SWE-bench – pick from a built-in subset
python run_test_agent.py --dataset swe --subset small --sample_index 2
"""

import argparse
import datetime
import json
import os
from pathlib import Path
from typing import Optional

from utils.common_utils import load_json_file


# ---------------------------------------------------------------------------
# Task selection helpers
# ---------------------------------------------------------------------------

def pick_instance_id(
    dataset: str,
    instance_id: Optional[str],
    sample_index: int,
    subset: Optional[str],
    task_list_file: Optional[str],
) -> str:
    """Return the concrete instance_id to run."""
    if instance_id:
        return instance_id

    if task_list_file:
        task_list = load_json_file(task_list_file)
    elif subset:
        subset_path = f"./{dataset}/subsets/{subset}.json"
        if dataset == "polyglot":
            subset_path = f"./polyglot/subsets/{subset}.json"
        if not os.path.exists(subset_path):
            raise FileNotFoundError(
                f"Subset file not found for dataset '{dataset}': {subset_path}"
            )
        task_list = load_json_file(subset_path)
    else:
        raise ValueError("Provide --instance_id or one of --subset / --task_list_file.")

    if not isinstance(task_list, list) or not task_list:
        raise ValueError("Task list is empty or invalid.")
    if sample_index < 0 or sample_index >= len(task_list):
        raise IndexError(
            f"sample_index {sample_index} out of range for task list of size {len(task_list)}"
        )
    return str(task_list[sample_index])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the base coding agent on a single benchmark task "
            "(Polyglot or SWE-bench) and log everything to test_agent/."
        ),
    )
    parser.add_argument("--dataset", type=str, default="polyglot", choices=["swe", "polyglot"], help="Which benchmark to run (default: polyglot).")
    parser.add_argument("--instance_id", type=str, default=None, help="Specific instance_id to run.  If omitted, use --subset + --sample_index.")
    parser.add_argument("--subset", type=str, default=None, help="Built-in subset name.  SWE: small/medium/big.  Polyglot: small/medium.")
    parser.add_argument("--task_list_file", type=str, default=None, help="Path to a JSON list of instance_ids (alternative to --subset).")
    parser.add_argument("--sample_index", type=int, default=0, help="Index into subset / task_list_file when --instance_id is not given.")
    parser.add_argument("--max_workers", type=int, default=1, help="Max parallel workers for the harness (usually 1 for a single task).")
    parser.add_argument("--polyglot_dataset_path", type=str, default="./polyglot/polyglot_benchmark_metadata.json", help="Path to Polyglot metadata JSON (only used with --dataset polyglot).")
    parser.add_argument("--output_dir", type=str, default="./test_agent", help="Root directory for all output and logs (default: ./test_agent).")
    parser.add_argument("--timeout", type=int, default=60, help="Agent timeout in seconds inside Docker (default: 1800 = 30 min).")
    parser.add_argument("--model", type=str, default="claude-haiku-4-5", help="LLM model string for the coding agent.  Default uses the direct Anthropic API (requires ANTHROPIC_API_KEY).  Set to 'bedrock/us.anthropic.claude-3-5-sonnet-20241022-v2:0' for AWS Bedrock.")
    args = parser.parse_args()

    # Export overrides so they propagate into the coding agent inside Docker
    os.environ["DGM_CLAUDE_MODEL"] = args.model
    os.environ["DGM_AGENT_TIMEOUT"] = str(args.timeout)

    # Resolve the target task
    target_instance = pick_instance_id(
        dataset=args.dataset,
        instance_id=args.instance_id,
        sample_index=args.sample_index,
        subset=args.subset,
        task_list_file=args.task_list_file,
    )

    # Prepare output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    model_name_or_path = f"base_agent_{target_instance}_{timestamp}"

    # Predictions and eval output land under test_agent/
    pred_dname = str(output_root / "predictions")
    eval_output_dir = str(output_root / "eval_results")
    Path(pred_dname).mkdir(parents=True, exist_ok=True)
    Path(eval_output_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"  Dataset:       {args.dataset}")
    print(f"  Instance:      {target_instance}")
    print(f"  Model:         {args.model}")
    print(f"  Timeout:       {args.timeout}s ({args.timeout // 60} min)")
    print(f"  Agent:         base (from codebase, no patches)")
    print(f"  Output dir:    {output_root}")
    print(f"  Predictions:   {pred_dname}")
    print(f"  Eval results:  {eval_output_dir}")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # Run the task through the appropriate harness (no model patches)
    # -----------------------------------------------------------------------
    if args.dataset == "swe":
        try:
            from swe_bench.harness import harness
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "SWE mode requires SWE-bench Python package dependencies.  "
                "Install / setup SWE-bench first (see README)."
            ) from exc

        dnames = harness(
            test_task_list=[target_instance],
            num_samples=-1,
            max_workers=args.max_workers,
            model_name_or_path=model_name_or_path,
            model_patch_paths=None,          # <-- base agent, no patches
            num_evals=1,
            num_evals_parallel=1,
            pred_dname=pred_dname,
        )

        # Optional: generate SWE-bench report
        try:
            from swe_bench.report import make_report
            run_ids = [f"{model_name_or_path}_{i}" for i in range(len(dnames))]
            make_report(
                dnames=dnames,
                run_ids=run_ids,
                dataset_name="princeton-nlp/SWE-bench_Verified",
                output_dir=eval_output_dir,
                dnames_workers=1,
            )
        except Exception as e:
            print(f"[warn] Could not generate SWE-bench report: {e}")

    else:  # polyglot
        try:
            from polyglot.harness import harness as polyglot_harness
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Polyglot mode requires polyglot / SWE-bench Python package "
                "dependencies.  Install first (see README)."
            ) from exc

        polyglot_harness(
            dataset_path=args.polyglot_dataset_path,
            test_task_list=[target_instance],
            num_samples=-1,
            max_workers=args.max_workers,
            model_name_or_path=model_name_or_path,
            model_patch_paths=None,          # <-- base agent, no patches
            num_evals=1,
            num_evals_parallel=1,
            pred_dname=pred_dname,
            output_dir=Path(eval_output_dir),
        )

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print()
    print("=" * 60)
    print("  Run finished.")
    print(f"  All logs and outputs are in:  {output_root.resolve()}")
    print("=" * 60)

    # Print listing of produced files for convenience
    for root, dirs, files in os.walk(output_root):
        level = root.replace(str(output_root), "").count(os.sep)
        indent = "  " * (level + 1)
        print(f"{indent}{os.path.basename(root)}/")
        sub_indent = "  " * (level + 2)
        for f in sorted(files):
            fpath = os.path.join(root, f)
            size = os.path.getsize(fpath)
            print(f"{sub_indent}{f}  ({size:,} bytes)")


if __name__ == "__main__":
    main()
