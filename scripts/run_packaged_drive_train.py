#!/usr/bin/env python3
import argparse
import ast
import configparser
import copy
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pufferlib import pufferl


def _parse_value(value: str) -> Any:
    lowered = value.strip().lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "none":
        return None

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

    for section in (
        "vec",
        "env",
        "policy",
        "rnn",
        "train",
        "eval",
        "bc",
        "bc_train",
        "bc_kl",
        "sweep",
        "preference_reward",
    ):
        if section in packaged:
            args.setdefault(section, {})
            args[section].update(packaged[section])

    if "wandb" in packaged and isinstance(packaged["wandb"], dict):
        enabled = packaged["wandb"].get("enabled")
        if enabled is not None:
            args["wandb"] = bool(enabled)

    args["train"]["use_rnn"] = args.get("rnn_name") is not None
    return args


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch training from a packaged Drive INI config.")
    parser.add_argument("--config", type=Path, required=True, help="Path to packaged training INI")
    parser.add_argument("--disable-wandb", action="store_true", help="Disable wandb even if config enables it")
    parser.add_argument("--enable-wandb", action="store_true", help="Enable wandb logging for this run")
    parser.add_argument("--wandb-project", type=str, default=None, help="Optional wandb project override")
    parser.add_argument("--wandb-group", type=str, default=None, help="Optional wandb group override")
    parser.add_argument("--wandb-name", type=str, default=None, help="Optional wandb run name override")
    parser.add_argument("--tag", type=str, default=None, help="Optional run tag")
    parser.add_argument("--load-model-path", type=str, default=None, help="Optional checkpoint override")
    args_ns = parser.parse_args()

    config_path = args_ns.config.resolve()
    original_argv = sys.argv[:]
    try:
        sys.argv = [sys.argv[0]]
        base_args = pufferl.load_config("puffer_drive")
    finally:
        sys.argv = original_argv

    packaged = _load_packaged_config(config_path)
    args = _overlay_args(base_args, packaged)
    if args_ns.disable_wandb:
        args["wandb"] = False
    if args_ns.enable_wandb:
        args["wandb"] = True
    if args_ns.wandb_project is not None:
        args["wandb_project"] = args_ns.wandb_project
    if args_ns.wandb_group is not None:
        args["wandb_group"] = args_ns.wandb_group
    if args_ns.wandb_name is not None:
        args["wandb_name"] = args_ns.wandb_name
    if args_ns.tag is not None:
        args["tag"] = args_ns.tag
    if args_ns.load_model_path is not None:
        args["load_model_path"] = args_ns.load_model_path

    pufferl.train("puffer_drive", args=args)


if __name__ == "__main__":
    main()
