"""TimeKAN with Probe handling future exogenous variables."""

from __future__ import annotations

import copy

from ts_benchmark.baselines.deep_forecasting_model_base import (
    DeepForecastingModelBase,
)
from ts_benchmark.baselines.probenet.model import Probe
from ts_benchmark.baselines.probenet.probe_adapter import ProbeAdapter
from ts_benchmark.baselines.probenet.probe_wrapper_utils import (
    PROBE_BRANCH_DEFAULTS,
    namespaced_config_with_defaults,
)
from ts_benchmark.baselines.timekan.models.timekan_model import TimeKANModeL
from ts_benchmark.baselines.timekan.timekan import (
    MODEL_HYPER_PARAMS as TIMEKAN_HYPER_PARAMS,
)

MODEL_HYPER_PARAMS = copy.deepcopy(TIMEKAN_HYPER_PARAMS)
MODEL_HYPER_PARAMS.update(
    {
        "fusion_method": "",
        "future_exog_mode": "probe",
        "infer_use_future": True,
        "alpha": 0.5,
        "baseline": {
            "begin_order": TIMEKAN_HYPER_PARAMS["begin_order"],
            "d_ff": TIMEKAN_HYPER_PARAMS["d_ff"],
            "d_model": TIMEKAN_HYPER_PARAMS["d_model"],
            "down_sampling_layers": TIMEKAN_HYPER_PARAMS["down_sampling_layers"],
            "down_sampling_window": TIMEKAN_HYPER_PARAMS["down_sampling_window"],
            "e_layers": TIMEKAN_HYPER_PARAMS["e_layers"],
        },
        "probe": copy.deepcopy(PROBE_BRANCH_DEFAULTS),
    }
)

BASELINE_DEFAULTS = copy.deepcopy(MODEL_HYPER_PARAMS["baseline"])
PROBE_DEFAULTS = copy.deepcopy(MODEL_HYPER_PARAMS["probe"])


class TimeKANProbeModel(ProbeAdapter):
    """Fuse TimeKAN history forecasts with Probe future-exogenous forecasts."""

    def __init__(self, config):
        baseline_config = namespaced_config_with_defaults(
            config, "baseline", BASELINE_DEFAULTS
        )
        probe_config = namespaced_config_with_defaults(config, "probe", PROBE_DEFAULTS)

        super().__init__(
            baseline=TimeKANModeL(baseline_config),
            probenet=Probe(probe_config),
            alpha=config.alpha,
            beta=probe_config.beta,
            baseline_output_dim=config.series_dim,
        )


class TimeKANProbe(DeepForecastingModelBase):
    """TimeKAN history branch plus a Probe future-exogenous branch."""

    def __init__(self, **kwargs):
        kwargs["fusion_method"] = ""
        kwargs["future_exog_mode"] = "probe"
        super().__init__(MODEL_HYPER_PARAMS, **kwargs)

    @property
    def model_name(self):
        return "TimeKANProbe"

    def _init_model(self):
        return TimeKANProbeModel(self.config)

    def _process(self, input, target, input_mark, target_mark, exog_future=None):
        del target, input_mark, target_mark
        if exog_future is None:
            raise ValueError(
                "TimeKANProbe requires future exogenous variables, got None"
            )

        output, probe_loss = self.model(input, exog_future)
        result = {"output": output}
        if self.model.training:
            result["additional_loss"] = probe_loss
        return result


__all__ = ["TimeKANProbe", "TimeKANProbeModel"]
