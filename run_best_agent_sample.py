import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from utils.common_utils import load_json_file
from utils.evo_utils import get_all_performance, get_model_patch_paths, load_dgm_metadata


def _resolve_dgm_run_dir(path: str) -> Path:
    run_dir = Path(path).expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"DGM run directory does not exist: {run_dir}")
    return run_dir


def _load_archive_candidates(dgm_run_dir: Path) -> List[str]:
    metadata_path = dgm_run_dir / "dgm_metadata.jsonl"
    if not metadata_path.exists():
        # Fallback: discover by folder structure if the run metadata is absent.
        candidates = []
        for meta_path in dgm_run_dir.glob("*/metadata.json"):
            candidates.append(meta_path.parent.name)
        if (dgm_run_dir / "initial" / "metadata.json").exists():
            candidates.append("initial")
        return sorted(set(candidates))

    final_state = load_dgm_metadata(str(metadata_path), last_only=True)
    archive = final_state.get("archive", [])
    return [str(node_id) for node_id in archive]


def _get_agent_perf(dgm_run_dir: Path, agent_id: str) -> Optional[Dict]:
    meta_path = dgm_run_dir / agent_id / "metadata.json"
    if not meta_path.exists():
        return None
    meta = load_json_file(str(meta_path))
    perf = meta.get("overall_performance")
    if not isinstance(perf, dict):
        return None
    return perf


def select_best_agent(dgm_run_dir: Path, include_initial: bool = True) -> Tuple[str, Dict]:
    candidates = _load_archive_candidates(dgm_run_dir)
    if not include_initial:
        candidates = [c for c in candidates if c != "initial"]
    if not candidates:
        raise RuntimeError("No candidate agents found in the run directory.")

    scored: List[Tuple[float, int, int, str, Dict]] = []
    for agent_id in candidates:
        perf = _get_agent_perf(dgm_run_dir, agent_id)
        if perf is None:
            continue
        accuracy = float(perf.get("accuracy_score", 0.0))
        resolved = int(perf.get("total_resolved_instances", len(perf.get("total_resolved_ids", []))))
        submitted = int(perf.get("total_submitted_instances", 0))
        scored.append((accuracy, resolved, submitted, agent_id, perf))

    if not scored:
        raise RuntimeError("No candidates had valid overall_performance in metadata.")

    scored.sort(key=lambda x: (-x[0], -x[1], -x[2], x[3]))
    best = scored[0]
    return best[3], best[4]


def pick_instance_id(
    dataset: str,
    instance_id: Optional[str],
    sample_index: int,
    subset: Optional[str],
    task_list_file: Optional[str],
) -> str:
    if instance_id:
        return instance_id

    if task_list_file:
        task_list = load_json_file(task_list_file)
    elif subset:
        subset_path = f"./{dataset}/subsets/{subset}.json"
        if not os.path.exists(subset_path):
            raise FileNotFoundError(
                f"Subset file not found for dataset '{dataset}': {subset_path}"
            )
        task_list = load_json_file(subset_path)
    else:
        raise ValueError("Provide --instance_id or one of --subset/--task_list_file.")

    if not isinstance(task_list, list) or not task_list:
        raise ValueError("Task list is empty or invalid.")
    if sample_index < 0 or sample_index >= len(task_list):
        raise IndexError(
            f"sample_index {sample_index} is out of range for task list size {len(task_list)}"
        )
    return str(task_list[sample_index])


def main() -> None:
    parser = argparse.ArgumentParser(description="Select the best DGM agent from a run and execute it on one task.")
    parser.add_argument("--dataset", type=str, default="swe", choices=["swe", "polyglot"], help="Dataset/harness to run.")
    parser.add_argument("--dgm_run_dir", type=str, required=True, help="Path to one DGM run directory, e.g. ./output_dgm/<run_id> or ./results_downloaded/polyglot_results/polyglot_dgm")
    parser.add_argument("--instance_id", type=str, default=None, help="Specific instance_id to run. If omitted, use subset/list + sample_index.")
    parser.add_argument("--subset", type=str, default=None, help="Pick the task from a built-in subset. SWE: small/medium/big, Polyglot: small/medium.")
    parser.add_argument("--task_list_file", type=str, default=None, help="Path to a JSON list of task instance_ids.")
    parser.add_argument("--sample_index", type=int, default=0, help="Index in subset/task_list_file when --instance_id is not provided.")
    parser.add_argument("--exclude_initial", action="store_true", help="Exclude the 'initial' agent when selecting the best agent.")
    parser.add_argument("--num_evals", type=int, default=1, help="Number of repeated harness runs.")
    parser.add_argument("--max_workers", type=int, default=1, help="Workers for harness entries.")
    parser.add_argument("--num_evals_parallel", type=int, default=1, help="Parallel workers across repeated harness runs.")
    parser.add_argument("--pred_dname", type=str, default=None, help="Directory to write harness predictions.")
    parser.add_argument("--eval_output_dir", type=str, default=None, help="Directory where evaluation JSON files are written.")
    parser.add_argument("--polyglot_dataset_path", type=str, default="./polyglot/polyglot_benchmark_metadata.json", help="Polyglot metadata file path (used when --dataset polyglot).")
    args = parser.parse_args()

    if args.pred_dname is None:
        args.pred_dname = (
            "./swe_bench/predictions_best_agent_single"
            if args.dataset == "swe"
            else "./polyglot/predictions_best_agent_single"
        )
    if args.eval_output_dir is None:
        args.eval_output_dir = "./swe_bench" if args.dataset == "swe" else "./polyglot/predictions"
    Path(args.eval_output_dir).mkdir(parents=True, exist_ok=True)

    dgm_run_dir = _resolve_dgm_run_dir(args.dgm_run_dir)
    target_instance = pick_instance_id(
        dataset=args.dataset,
        instance_id=args.instance_id,
        sample_index=args.sample_index,
        subset=args.subset,
        task_list_file=args.task_list_file,
    )

    best_agent_id, best_perf = select_best_agent(
        dgm_run_dir=dgm_run_dir,
        include_initial=not args.exclude_initial,
    )

    root_dir = os.path.abspath("./")
    patch_paths = get_model_patch_paths(root_dir, str(dgm_run_dir), best_agent_id)
    model_name_or_path = f"best_{best_agent_id}_sample_{target_instance}"

    print(f"Selected best agent: {best_agent_id}")
    print(
        "Best agent score: "
        f"accuracy={best_perf.get('accuracy_score', 0.0)} "
        f"resolved={best_perf.get('total_resolved_instances', len(best_perf.get('total_resolved_ids', [])))} "
        f"submitted={best_perf.get('total_submitted_instances', 0)}"
    )
    print(f"Target instance_id: {target_instance}")
    print(f"Applying {len(patch_paths)} ancestor patch(es).")

    if args.dataset == "swe":
        try:
            from swe_bench.harness import harness
            from swe_bench.report import make_report
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "SWE mode requires SWE-bench Python package dependencies. "
                "Install/setup SWE-bench first (see README SWE-bench setup steps). "
                f"Original error: {e}"
            ) from e

        dnames = harness(
            test_task_list=[target_instance],
            num_samples=-1,
            max_workers=args.max_workers,
            model_name_or_path=model_name_or_path,
            model_patch_paths=patch_paths,
            num_evals=args.num_evals,
            num_evals_parallel=args.num_evals_parallel,
            pred_dname=args.pred_dname,
        )
        run_ids = [f"{model_name_or_path}_{i}" for i in range(len(dnames))]
        make_report(
            dnames=dnames,
            run_ids=run_ids,
            dataset_name="princeton-nlp/SWE-bench_Verified",
            output_dir=args.eval_output_dir,
            dnames_workers=min(5, max(1, len(dnames))),
        )
        pred_dirs = [str(d) for d in dnames]
    else:
        try:
            from polyglot.harness import harness as polyglot_harness
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "Polyglot mode requires polyglot/SWE-bench Python package dependencies. "
                "Install/setup SWE-bench + Polyglot first (see README setup steps). "
                f"Original error: {e}"
            ) from e

        if args.num_evals > 1:
            raise ValueError("polyglot harness currently supports only --num_evals 1")
        polyglot_harness(
            dataset_path=args.polyglot_dataset_path,
            test_task_list=[target_instance],
            num_samples=-1,
            max_workers=args.max_workers,
            model_name_or_path=model_name_or_path,
            model_patch_paths=patch_paths,
            num_evals=1,
            num_evals_parallel=1,
            pred_dname=args.pred_dname,
            output_dir=Path(args.eval_output_dir),
        )
        pred_dirs = [args.pred_dname]

    _performances, overall = get_all_performance(model_name_or_path, results_dir=args.eval_output_dir)

    print("\nSingle-task replay completed.")
    print(f"Prediction directories: {pred_dirs}")
    if overall:
        print(
            "Eval summary: "
            f"accuracy={overall.get('accuracy_score', 0.0)} "
            f"resolved={overall.get('total_resolved_instances', 0)} "
            f"submitted={overall.get('total_submitted_instances', 0)}"
        )
    else:
        print("No aggregated eval summary found.")


if __name__ == "__main__":
    main()
