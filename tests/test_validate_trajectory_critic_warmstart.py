import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import torch


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "validate_trajectory_critic_warmstart.py"
SPEC = spec_from_file_location("validate_trajectory_critic_warmstart", SCRIPT_PATH)
MODULE = module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_compute_value_metrics_prefers_better_predictions():
    realized = [1.0, 2.0, 3.0, 4.0]
    good_pred = [1.1, 1.9, 3.2, 3.8]
    bad_pred = [4.0, 3.0, 2.0, 1.0]

    good = MODULE.compute_value_metrics(good_pred, realized)
    bad = MODULE.compute_value_metrics(bad_pred, realized)

    assert good["mse"] < bad["mse"]
    assert good["pearson"] > bad["pearson"]
    assert good["spearman"] > bad["spearman"]


def test_summarize_weight_changes_isolates_value_head():
    reference = {
        "actor.weight": torch.zeros(2, 2),
        "value_fn.weight": torch.zeros(1, 2),
        "encoder.weight": torch.zeros(2, 2),
    }
    candidate = {
        "actor.weight": torch.zeros(2, 2),
        "value_fn.weight": torch.ones(1, 2),
        "encoder.weight": torch.zeros(2, 2),
    }

    summary = MODULE.summarize_weight_changes(reference, candidate)

    assert summary["actor"]["unchanged"] is True
    assert summary["backbone"]["unchanged"] is True
    assert summary["value_fn"]["unchanged"] is False
    assert summary["value_fn"]["changed_params"] == 2
