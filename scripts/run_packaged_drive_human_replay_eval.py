#!/usr/bin/env python3
import argparse
import ast
import configparser
import copy
import csv
import gc
import json
import os
import random
import signal
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pufferlib import pufferl
from pufferlib.ocean.benchmark.evaluator import HumanReplayEvaluator


def _parse_value(value: str) -> Any:
    try:
        return ast.literal_eval(value)
    except Exception:
        return value


def _load_packaged_config(config_path: Path) -> dict[str, dict[str, Any]]:
    parser = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    with config_path.open("r", encoding="utf-8") as f:
        parser.read_file(f)

    data: dict[str, dict[str, Any]] = {}
    for section in parser.sections():
        data[section] = {}
        for key, value in parser[section].items():
            data[section][key] = _parse_value(value)
    return data


def _overlay_args(base_args: dict[str, Any], packaged: dict[str, dict[str, Any]]) -> dict[str, Any]:
    args = copy.deepcopy(base_args)

    for key, value in packaged.get("base", {}).items():
        args[key] = value

    for section in ("vec", "env", "policy", "rnn", "train", "eval", "bc", "bc_train", "sweep"):
        if section in packaged:
            args.setdefault(section, {})
            args[section].update(packaged[section])

    # Packaged model configs may include a [wandb] section with an "enabled" key,
    # while pufferl.load_config exposes wandb/neptune as top-level booleans.
    if "wandb" in packaged and isinstance(packaged["wandb"], dict):
        enabled = packaged["wandb"].get("enabled")
        if enabled is not None:
            args["wandb"] = bool(enabled)

    args["train"]["use_rnn"] = args.get("rnn_name") is not None
    return args


def _scenario_sort_key(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        float(record.get("score", 0.0)),
        float(record.get("completion_rate", 0.0)),
        -float(record.get("collisions_per_agent", 0.0)),
        -float(record.get("offroad_per_agent", 0.0)),
        float(record.get("episode_return", 0.0)),
    )


def _mean_numeric(records: list[dict[str, Any]]) -> dict[str, float]:
    if not records:
        return {}

    numeric_keys = []
    for key in records[0].keys():
        if key in {"map_name", "map_path"}:
            continue
        if isinstance(records[0][key], (int, float)):
            numeric_keys.append(key)

    return {
        key: float(sum(float(record.get(key, 0.0)) for record in records) / len(records))
        for key in numeric_keys
    }


def _write_intermediate_summary(
    output_dir: Path,
    checkpoint_path: Path,
    config_path: Path,
    validation_dir: Path,
    scenario_records: list[dict[str, Any]],
    skipped_records: list[dict[str, Any]],
    processed_maps: int,
    total_maps: int,
    top_k: int,
) -> None:
    intermediate_path = output_dir / "human_replay_eval_intermediate_summary.json"
    aggregate_metrics = _mean_numeric(scenario_records)
    ranked = sorted(scenario_records, key=_scenario_sort_key, reverse=True) if scenario_records else []
    payload = {
        "status": "running",
        "checkpoint": str(checkpoint_path),
        "config": str(config_path),
        "map_dir": str(validation_dir),
        "processed_maps": processed_maps,
        "total_maps": total_maps,
        "completed_maps": len(scenario_records),
        "skipped_maps": len(skipped_records),
        "aggregate_metrics_mean_over_completed_maps": aggregate_metrics,
        "current_best_scenarios": ranked[:top_k],
        "current_worst_scenarios": list(reversed(ranked[-top_k:])),
    }
    _write_json(intermediate_path, payload)


def _select_map_paths(
    map_paths: list[Path],
    sample_size: int | None,
    sample_seed: int,
) -> list[Path]:
    if sample_size is None:
        return map_paths

    if sample_size <= 0:
        raise ValueError(f"sample_size must be positive, got {sample_size}")

    if sample_size >= len(map_paths):
        return map_paths

    rng = random.Random(sample_seed)
    selected = rng.sample(map_paths, sample_size)
    return sorted(selected)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _prepare_single_map_dir(map_path: Path, temp_dir: Path) -> Path:
    temp_dir.mkdir(parents=True, exist_ok=True)
    target = temp_dir / "map_000.bin"
    if target.exists() or target.is_symlink():
        target.unlink()
    os.symlink(map_path.resolve(), target)
    return temp_dir


def _run_single_scenario(args: dict[str, Any], policy, map_path: Path, temp_dir: Path) -> dict[str, Any]:
    scenario_args = copy.deepcopy(args)
    single_map_dir = _prepare_single_map_dir(map_path, temp_dir)
    scenario_args["env"]["map_dir"] = str(single_map_dir)
    scenario_args["env"]["num_maps"] = 1
    scenario_args["eval"]["map_dir"] = str(single_map_dir)
    scenario_args["eval"]["wosac_num_maps"] = 1
    scenario_args["vec"] = {
        "backend": scenario_args["eval"].get("backend", "PufferEnv"),
        "num_envs": int(args.get("_requested_num_envs", 1)),
    }

    vecenv = pufferl.load_env("puffer_drive", scenario_args)
    try:
        evaluator = HumanReplayEvaluator(scenario_args)
        results = evaluator.rollout(scenario_args, vecenv, policy)
        if results is None:
            raise RuntimeError(f"Human replay eval returned no results for {map_path.name}")

        record = {"map_name": map_path.name, "map_path": str(map_path.resolve())}
        for key, value in results.items():
            if isinstance(value, (int, float)):
                record[key] = float(value)
            else:
                record[key] = value
        return record
    finally:
        vecenv.close()
        gc.collect()


def _run_single_map_child(
    config_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
    map_path: Path,
    num_envs: int,
    result_path: Path,
) -> None:
    original_argv = sys.argv[:]
    try:
        sys.argv = [sys.argv[0]]
        base_args = pufferl.load_config("puffer_drive")
    finally:
        sys.argv = original_argv

    packaged = _load_packaged_config(config_path)
    args = _overlay_args(base_args, packaged)
    args["load_model_path"] = str(checkpoint_path)
    args["load_id"] = None
    args["wandb"] = False
    args["neptune"] = False
    args["vec"] = {"backend": args["eval"].get("backend", "PufferEnv"), "num_envs": int(num_envs)}
    args["_requested_num_envs"] = int(num_envs)

    temp_root = output_dir / ".single_map_eval" / map_path.stem
    if temp_root.exists():
        shutil.rmtree(temp_root)
    temp_root.mkdir(parents=True, exist_ok=True)

    try:
        bootstrap_args = copy.deepcopy(args)
        bootstrap_dir = _prepare_single_map_dir(map_path, temp_root)
        bootstrap_args["env"]["map_dir"] = str(bootstrap_dir)
        bootstrap_args["env"]["num_maps"] = 1
        bootstrap_args["eval"]["map_dir"] = str(bootstrap_dir)
        bootstrap_args["eval"]["wosac_num_maps"] = 1
        bootstrap_args["vec"] = {
            "backend": bootstrap_args["eval"].get("backend", "PufferEnv"),
            "num_envs": int(num_envs),
        }
        bootstrap_vecenv = pufferl.load_env("puffer_drive", bootstrap_args)
        try:
            policy = pufferl.load_policy(bootstrap_args, bootstrap_vecenv, env_name="puffer_drive")
            policy.eval()
        finally:
            bootstrap_vecenv.close()
            gc.collect()

        record = _run_single_scenario(args, policy, map_path, temp_root)
        _write_json(result_path, {"status": "completed", "record": record})
    except Exception as exc:
        _write_json(
            result_path,
            {
                "status": "skipped",
                "map_name": map_path.name,
                "map_path": str(map_path.resolve()),
                "error": str(exc),
                "error_type": type(exc).__name__,
                "traceback": traceback.format_exc(),
            },
        )
    finally:
        if temp_root.exists():
            shutil.rmtree(temp_root)
        gc.collect()


def _run_single_map_subprocess(
    script_path: Path,
    config_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
    map_path: Path,
    num_envs: int,
    timeout_sec: int,
) -> dict[str, Any]:
    result_dir = output_dir / ".single_map_results"
    result_dir.mkdir(parents=True, exist_ok=True)
    result_path = result_dir / f"{map_path.stem}.json"
    if result_path.exists():
        result_path.unlink()

    cmd = [
        sys.executable,
        str(script_path),
        "--config",
        str(config_path),
        "--checkpoint",
        str(checkpoint_path),
        "--output-dir",
        str(output_dir),
        "--num-envs",
        str(num_envs),
        "--single-map-path",
        str(map_path),
        "--single-map-result-json",
        str(result_path),
    ]

    proc = subprocess.Popen(cmd)
    deadline = time.monotonic() + timeout_sec
    try:
        while True:
            if result_path.exists():
                with result_path.open("r", encoding="utf-8") as f:
                    payload = json.load(f)
                result_path.unlink()
                return payload

            returncode = proc.poll()
            if returncode is not None:
                if returncode == 0 and result_path.exists():
                    continue
                raise RuntimeError(
                    f"Single-map subprocess failed for {map_path.name} with exit code {returncode}"
                )

            if time.monotonic() >= deadline:
                for sig in (signal.SIGTERM, signal.SIGKILL):
                    try:
                        proc.send_signal(sig)
                    except ProcessLookupError:
                        break
                    time.sleep(1)
                    if proc.poll() is not None:
                        break

                return {
                    "status": "skipped",
                    "map_name": map_path.name,
                    "map_path": str(map_path.resolve()),
                    "error_type": "TimeoutExpired",
                    "error": f"Timed out after {timeout_sec} seconds",
                }

            time.sleep(1)
    finally:
        if result_path.exists():
            result_path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run packaged human-replay eval and save aggregate + scenario logs.")
    parser.add_argument("--config", type=Path, required=True, help="Path to the packaged conf1_discrete.ini")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to the checkpoint to evaluate")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for eval outputs")
    parser.add_argument("--top-k", type=int, default=20, help="Number of best/worst scenarios to summarize")
    parser.add_argument("--num-envs", type=int, default=1, help="Number of eval envs to run in parallel")
    parser.add_argument("--single-map-path", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--single-map-result-json", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Evaluate only a deterministic random sample of validation maps",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=42,
        help="Seed used for deterministic validation-map sampling",
    )
    parser.add_argument(
        "--single-map-timeout-sec",
        type=int,
        default=180,
        help="Timeout for each map subprocess to prevent a single scenario from stalling the full eval",
    )
    args_ns = parser.parse_args()

    config_path = args_ns.config.resolve()
    checkpoint_path = args_ns.checkpoint.resolve()
    output_dir = args_ns.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args_ns.single_map_path is not None:
        if args_ns.single_map_result_json is None:
            raise ValueError("--single-map-result-json is required with --single-map-path")
        _run_single_map_child(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            output_dir=output_dir,
            map_path=args_ns.single_map_path.resolve(),
            num_envs=args_ns.num_envs,
            result_path=args_ns.single_map_result_json.resolve(),
        )
        return

    packaged = _load_packaged_config(config_path)
    original_argv = sys.argv[:]
    try:
        sys.argv = [sys.argv[0]]
        base_args = pufferl.load_config("puffer_drive")
    finally:
        sys.argv = original_argv
    args = _overlay_args(base_args, packaged)
    args["load_model_path"] = str(checkpoint_path)
    args["load_id"] = None
    args["wandb"] = False
    args["neptune"] = False
    args["vec"] = {"backend": args["eval"].get("backend", "PufferEnv"), "num_envs": int(args_ns.num_envs)}
    args["_requested_num_envs"] = int(args_ns.num_envs)

    validation_dir = Path(str(args["eval"]["map_dir"])).resolve()
    map_paths = sorted(validation_dir.glob("map_*.bin"))
    max_maps = int(args["eval"].get("wosac_num_maps", -1))
    if max_maps > 0:
        map_paths = map_paths[:max_maps]
    map_paths = _select_map_paths(map_paths, args_ns.sample_size, args_ns.sample_seed)

    if not map_paths:
        raise FileNotFoundError(f"No validation maps found in {validation_dir}")

    temp_root = output_dir / ".single_map_eval"
    if temp_root.exists():
        shutil.rmtree(temp_root)
    temp_root.mkdir(parents=True, exist_ok=True)

    scenario_records: list[dict[str, Any]] = []
    skipped_records: list[dict[str, Any]] = []
    scenario_csv_path = output_dir / "human_replay_eval_per_scenario.csv"
    skipped_path = output_dir / "human_replay_eval_skipped_scenarios.json"
    progress_path = output_dir / "human_replay_eval_progress.json"
    aggregate_path = output_dir / "human_replay_eval_aggregate.json"
    ranking_path = output_dir / "human_replay_eval_best_worst_scenarios.json"
    run_log_path = output_dir / "human_replay_eval_run.log"
    error_path = output_dir / "human_replay_eval_error.txt"
    sample_manifest_path = output_dir / "human_replay_eval_sampled_maps.json"

    _write_json(
        sample_manifest_path,
        {
            "sampling_enabled": args_ns.sample_size is not None,
            "sample_size_requested": args_ns.sample_size,
            "sample_seed": args_ns.sample_seed,
            "maps_selected": len(map_paths),
            "map_names": [path.name for path in map_paths],
            "map_paths": [str(path.resolve()) for path in map_paths],
        },
    )

    run_log_path.write_text(
        "\n".join(
            [
                "Starting packaged human replay eval",
                f"config={config_path}",
                f"checkpoint={checkpoint_path}",
                f"map_dir={validation_dir}",
                f"maps_requested={len(map_paths)}",
                f"sample_size={args_ns.sample_size}",
                f"sample_seed={args_ns.sample_seed}",
                f"sample_manifest={sample_manifest_path}",
                f"vec_backend={args['vec']['backend']}",
                f"vec_num_envs={args['vec']['num_envs']}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _write_json(
        progress_path,
        {
            "status": "starting",
            "completed_maps": 0,
            "skipped_maps": 0,
            "total_maps": len(map_paths),
            "checkpoint": str(checkpoint_path),
            "config": str(config_path),
        },
    )

    try:
        for index, map_path in enumerate(map_paths, start=1):
            try:
                payload = _run_single_map_subprocess(
                    script_path=Path(__file__).resolve(),
                    config_path=config_path,
                    checkpoint_path=checkpoint_path,
                    output_dir=output_dir,
                    map_path=map_path,
                    num_envs=args_ns.num_envs,
                    timeout_sec=args_ns.single_map_timeout_sec,
                )
                latest_status = payload.get("status", "skipped")
                if latest_status == "completed":
                    scenario_records.append(payload["record"])
                else:
                    skipped_records.append(
                        {
                            "map_name": payload.get("map_name", map_path.name),
                            "map_path": payload.get("map_path", str(map_path.resolve())),
                            "error": payload.get("error", "Unknown single-map failure"),
                            "error_type": payload.get("error_type", "UnknownError"),
                        }
                    )
                    with run_log_path.open("a", encoding="utf-8") as f:
                        f.write(
                            f"[eval] skipped {map_path.name} | "
                            f"{payload.get('error_type', 'UnknownError')}: {payload.get('error', 'unknown error')}\n"
                        )
                    _write_json(skipped_path, skipped_records)
            except subprocess.TimeoutExpired:
                latest_status = "skipped"
                skipped_records.append(
                    {
                        "map_name": map_path.name,
                        "map_path": str(map_path.resolve()),
                        "error": f"Timed out after {args_ns.single_map_timeout_sec} seconds",
                        "error_type": "TimeoutExpired",
                    }
                )
                with run_log_path.open("a", encoding="utf-8") as f:
                    f.write(
                        f"[eval] skipped {map_path.name} | TimeoutExpired: "
                        f"Timed out after {args_ns.single_map_timeout_sec} seconds\n"
                    )
                _write_json(skipped_path, skipped_records)
            except Exception as exc:
                skipped_records.append(
                    {
                        "map_name": map_path.name,
                        "map_path": str(map_path.resolve()),
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                    }
                )
                latest_status = "skipped"
                with run_log_path.open("a", encoding="utf-8") as f:
                    f.write(f"[eval] skipped {map_path.name} | {type(exc).__name__}: {exc}\n")
                _write_json(skipped_path, skipped_records)

            if scenario_records:
                fieldnames = sorted({key for row in scenario_records for key in row.keys()})
                with scenario_csv_path.open("w", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(scenario_records)

            _write_json(
                progress_path,
                {
                    "status": "running",
                    "completed_maps": len(scenario_records),
                    "skipped_maps": len(skipped_records),
                    "processed_maps": index,
                    "total_maps": len(map_paths),
                    "latest_map": map_path.name,
                    "latest_status": latest_status,
                    "checkpoint": str(checkpoint_path),
                    "config": str(config_path),
                },
            )

            with run_log_path.open("a", encoding="utf-8") as f:
                f.write(
                    f"[eval] processed {index}/{len(map_paths)} maps | "
                    f"completed={len(scenario_records)} skipped={len(skipped_records)} "
                    f"| latest={map_path.name} | status={latest_status}\n"
                )
            print(
                f"[eval] processed {index}/{len(map_paths)} maps | "
                f"completed={len(scenario_records)} skipped={len(skipped_records)} "
                f"| latest={map_path.name} | status={latest_status}",
                flush=True,
            )

            if index % 100 == 0:
                _write_intermediate_summary(
                    output_dir=output_dir,
                    checkpoint_path=checkpoint_path,
                    config_path=config_path,
                    validation_dir=validation_dir,
                    scenario_records=scenario_records,
                    skipped_records=skipped_records,
                    processed_maps=index,
                    total_maps=len(map_paths),
                    top_k=args_ns.top_k,
                )
                with run_log_path.open("a", encoding="utf-8") as f:
                    f.write(
                        f"[eval] wrote intermediate summary at {index} processed maps "
                        f"(completed={len(scenario_records)} skipped={len(skipped_records)})\n"
                    )

        aggregate_metrics = _mean_numeric(scenario_records)
        aggregate_payload = {
            "mode": "human_replay_eval",
            "checkpoint": str(checkpoint_path),
            "config": str(config_path),
            "map_dir": str(validation_dir),
            "num_maps_evaluated": len(scenario_records),
            "num_maps_skipped": len(skipped_records),
            "aggregate_metrics_mean_over_maps": aggregate_metrics,
        }
        _write_json(aggregate_path, aggregate_payload)

        ranked = sorted(scenario_records, key=_scenario_sort_key, reverse=True)
        ranking_payload = {
            "ranking_basis": {
                "primary": "score desc",
                "tie_breakers": [
                    "completion_rate desc",
                    "collisions_per_agent asc",
                    "offroad_per_agent asc",
                    "episode_return desc",
                ],
            },
            "best_scenarios": ranked[: args_ns.top_k],
            "worst_scenarios": list(reversed(ranked[-args_ns.top_k :])),
        }
        _write_json(ranking_path, ranking_payload)
        _write_json(skipped_path, skipped_records)
        _write_json(
            progress_path,
            {
                "status": "completed",
                "completed_maps": len(scenario_records),
                "skipped_maps": len(skipped_records),
                "processed_maps": len(map_paths),
                "total_maps": len(map_paths),
                "checkpoint": str(checkpoint_path),
                "config": str(config_path),
            },
        )

        print(json.dumps(aggregate_payload, indent=2, sort_keys=True), flush=True)
    except Exception as exc:
        trace = traceback.format_exc()
        error_path.write_text(trace, encoding="utf-8")
        with run_log_path.open("a", encoding="utf-8") as f:
            f.write(f"[eval] FAILED | {type(exc).__name__}: {exc}\n")
            f.write(trace)
            if not trace.endswith("\n"):
                f.write("\n")
        _write_json(
            progress_path,
            {
                "status": "failed",
                "completed_maps": len(scenario_records),
                "skipped_maps": len(skipped_records),
                "total_maps": len(map_paths),
                "checkpoint": str(checkpoint_path),
                "config": str(config_path),
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise
    finally:
        if temp_root.exists():
            shutil.rmtree(temp_root)


if __name__ == "__main__":
    main()
