"""Shared configuration and forward bridges for Probe-enhanced baselines."""

from __future__ import annotations

import copy

from ts_benchmark.baselines.probenet.probe_adapter import namespaced_config

PROBE_BRANCH_DEFAULTS = {
    "d_model": 64,
    "d_ff": 128,
    "n_heads": 4,
    "factor": 1,
    "activation": "gelu",
    "patch_len": 24,
    "n_templates": 4,
    "dropout": 0.0,
    "beta": 0.01,
}


def namespaced_config_with_defaults(config, namespace, defaults):
    """Apply namespace overrides while retaining missing nested defaults."""
    result = namespaced_config(config, namespace)
    overrides = getattr(config, namespace, {}) or {}
    for name, value in defaults.items():
        if name not in overrides:
            setattr(result, name, copy.deepcopy(value))
    return result


def four_argument_forward(model, history):
    """Call a Time-Series-Library-style model using history only."""
    return model(history, None, None, None)


class ProbeTestPatchOrderMixin:
    """Enable patch misalignment only when final inference starts.

    ``forecast_fit`` and its validation loop do not call these inference entry
    points, so Probe keeps its default shift of zero during model fitting and
    checkpoint selection.
    """

    def _activate_test_patch_order_shift(self):
        shift = int(getattr(self.config, "test_patch_order_shift", 0))
        model = getattr(self, "model", None)
        if model is None:
            return

        adapter = model.module if hasattr(model, "module") else model
        probenet = getattr(adapter, "probenet", None)
        if probenet is None:
            raise RuntimeError(
                f"{self.model_name} does not expose a Probe branch as " "model.probenet"
            )
        probenet.set_patch_order_shift(shift)

    def forecast(self, *args, **kwargs):
        self._activate_test_patch_order_shift()
        return super().forecast(*args, **kwargs)

    def batch_forecast(self, *args, **kwargs):
        self._activate_test_patch_order_shift()
        return super().batch_forecast(*args, **kwargs)


__all__ = [
    "PROBE_BRANCH_DEFAULTS",
    "ProbeTestPatchOrderMixin",
    "four_argument_forward",
    "namespaced_config_with_defaults",
]
