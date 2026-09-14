"""CrossLinear with Probe handling future exogenous variables."""

from __future__ import annotations

import copy

from ts_benchmark.baselines.crosslinear.crosslinear import (
    MODEL_HYPER_PARAMS as CROSSLINEAR_HYPER_PARAMS,
)
from ts_benchmark.baselines.crosslinear.models.crosslinear_model import (
    CrossLinear_model,
)
from ts_benchmark.baselines.deep_forecasting_model_base import (
    DeepForecastingModelBase,
)
from ts_benchmark.baselines.probenet.model import Probe
from ts_benchmark.baselines.probenet.probe_adapter import (
    ProbeAdapter,
    namespaced_config,
)

MODEL_HYPER_PARAMS = copy.deepcopy(CROSSLINEAR_HYPER_PARAMS)
MODEL_HYPER_PARAMS.update(
    {
        # Probe performs future-exogenous fusion inside the main model. Keeping
        # this empty prevents DeepForecastingModelBase from creating an MLP.
        "fusion_method": "",
        "future_exog_mode": "probe",
        "infer_use_future": True,
        # Final fusion: (1 - alpha) * CrossLinear + alpha * Probe.
        "alpha": 0.5,
        # CrossLinear's own alpha/beta have different meanings, so they live in
        # the baseline namespace rather than sharing the fusion parameters.
        "baseline": {
            "alpha": 0.5,
            "beta": 0.5,
        },
        "probe": {
            "d_model": 64,
            "d_ff": 128,
            "n_heads": 4,
            "patch_len": 24,
            "n_templates": 4,
            "dropout": 0.0,
            "beta": 0.01,
        },
    }
)

BASELINE_DEFAULTS = copy.deepcopy(MODEL_HYPER_PARAMS["baseline"])
PROBE_DEFAULTS = copy.deepcopy(MODEL_HYPER_PARAMS["probe"])


def _namespaced_config_with_defaults(config, namespace, defaults):
    """Apply namespace overrides without losing unspecified nested defaults."""
    result = namespaced_config(config, namespace)
    overrides = getattr(config, namespace, {}) or {}
    for name, value in defaults.items():
        if name not in overrides:
            setattr(result, name, copy.deepcopy(value))
    return result


def _crosslinear_forward(model, history):
    """Bridge CrossLinear's benchmark signature to ``ProbeAdapter``."""
    return model(history, None, None, None)


class CrossLinearProbeModel(ProbeAdapter):
    """Fuse historical CrossLinear forecasts with future-exogenous Probe forecasts."""

    def __init__(self, config):
        baseline_config = _namespaced_config_with_defaults(
            config, "baseline", BASELINE_DEFAULTS
        )
        probe_config = _namespaced_config_with_defaults(config, "probe", PROBE_DEFAULTS)

        super().__init__(
            baseline=CrossLinear_model(baseline_config),
            probenet=Probe(probe_config),
            alpha=config.alpha,
            beta=probe_config.beta,
            baseline_forward=_crosslinear_forward,
        )


class CrossLinearProbe(DeepForecastingModelBase):
    """CrossLinear history branch plus a Probe future-exogenous branch."""

    def __init__(self, **kwargs):
        # Explicitly replace the benchmark-level MLP requested by legacy
        # CrossLinear configurations; future exogenous variables are consumed
        # only by Probe.
        kwargs["fusion_method"] = ""
        kwargs["future_exog_mode"] = "probe"
        super().__init__(MODEL_HYPER_PARAMS, **kwargs)

    @property
    def model_name(self):
        return "CrossLinearProbe"

    def _init_model(self):
        return CrossLinearProbeModel(self.config)

    def _process(self, input, target, input_mark, target_mark, exog_future=None):
        del target, input_mark, target_mark
        if exog_future is None:
            raise ValueError(
                "CrossLinearProbe requires future exogenous variables, got None"
            )

        output, probe_loss = self.model(input, exog_future)
        result = {"output": output}
        if self.model.training:
            result["additional_loss"] = probe_loss
        return result


__all__ = ["CrossLinearProbe", "CrossLinearProbeModel"]
