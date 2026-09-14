"""PatchTST with Probe handling future exogenous variables."""

from __future__ import annotations

import copy

from ts_benchmark.baselines.deep_forecasting_model_base import (
    DeepForecastingModelBase,
)
from ts_benchmark.baselines.probenet.model import Probe
from ts_benchmark.baselines.probenet.probe_adapter import ProbeAdapter
from ts_benchmark.baselines.probenet.probe_wrapper_utils import (
    PROBE_BRANCH_DEFAULTS,
    four_argument_forward,
    namespaced_config_with_defaults,
)
from ts_benchmark.baselines.time_series_library.adapters_for_transformers import (
    MODEL_HYPER_PARAMS as PATCHTST_HYPER_PARAMS,
)
from ts_benchmark.baselines.time_series_library.models.PatchTST import PatchTST

MODEL_HYPER_PARAMS = copy.deepcopy(PATCHTST_HYPER_PARAMS)
MODEL_HYPER_PARAMS.update(
    {
        "fusion_method": "",
        "future_exog_mode": "probe",
        "infer_use_future": True,
        "alpha": 0.5,
        "baseline": {
            "d_ff": PATCHTST_HYPER_PARAMS["d_ff"],
            "d_model": PATCHTST_HYPER_PARAMS["d_model"],
            "e_layers": PATCHTST_HYPER_PARAMS["e_layers"],
            "factor": PATCHTST_HYPER_PARAMS["factor"],
            "n_heads": PATCHTST_HYPER_PARAMS["n_heads"],
            "patch_len": PATCHTST_HYPER_PARAMS["patch_len"],
        },
        "probe": copy.deepcopy(PROBE_BRANCH_DEFAULTS),
    }
)

BASELINE_DEFAULTS = copy.deepcopy(MODEL_HYPER_PARAMS["baseline"])
PROBE_DEFAULTS = copy.deepcopy(MODEL_HYPER_PARAMS["probe"])


class PatchTSTProbeModel(ProbeAdapter):
    """Fuse PatchTST history forecasts with Probe future-exogenous forecasts."""

    def __init__(self, config):
        baseline_config = namespaced_config_with_defaults(
            config, "baseline", BASELINE_DEFAULTS
        )
        probe_config = namespaced_config_with_defaults(config, "probe", PROBE_DEFAULTS)

        super().__init__(
            baseline=PatchTST(baseline_config),
            probenet=Probe(probe_config),
            alpha=config.alpha,
            beta=probe_config.beta,
            baseline_forward=four_argument_forward,
            baseline_output_dim=config.series_dim,
        )


class PatchTSTProbe(DeepForecastingModelBase):
    """PatchTST history branch plus a Probe future-exogenous branch."""

    def __init__(self, **kwargs):
        kwargs["fusion_method"] = ""
        kwargs["future_exog_mode"] = "probe"
        super().__init__(MODEL_HYPER_PARAMS, **kwargs)

    @property
    def model_name(self):
        return "PatchTSTProbe"

    def _init_model(self):
        return PatchTSTProbeModel(self.config)

    def _process(self, input, target, input_mark, target_mark, exog_future=None):
        del target, input_mark, target_mark
        if exog_future is None:
            raise ValueError(
                "PatchTSTProbe requires future exogenous variables, got None"
            )

        output, probe_loss = self.model(input, exog_future)
        result = {"output": output}
        if self.model.training:
            result["additional_loss"] = probe_loss
        return result


__all__ = ["PatchTSTProbe", "PatchTSTProbeModel"]
