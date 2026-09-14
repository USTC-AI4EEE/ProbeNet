from ts_benchmark.baselines.deep_forecasting_model_base import (
    DeepForecastingModelBase,
)
from ts_benchmark.baselines.probenet.model import HistoricalCrossForecaster, Probe
from ts_benchmark.baselines.probenet.probe_adapter import (
    ProbeAdapter,
    namespaced_config,
)
from ts_benchmark.baselines.probenet.probe_wrapper_utils import (
    ProbeTestPatchOrderMixin,
)

MODEL_HYPER_PARAMS = {
    "d_model": 64,
    "d_ff": 256,
    "n_heads": 4,
    "factor": 1,
    "activation": "gelu",
    "batch_size": 64,
    "lradj": "type3",
    "lr": 0.001,
    "num_epochs": 50,
    "num_workers": 0,
    "loss": "MAE",
    "patience": 5,
    "infer_use_future": True,
    "future_exog_mode": "probe",
    "fusion_method": "",
    "mlp_hidden_dims": 128,
    "alpha": 0.5,
    "test_patch_order_shift": 0,
    "baseline": {"patch_len": 24},
    "probe": {
        "patch_len": 24,
        "n_templates": 4,
        "beta": 0.01,
    },
}


class ProbeNetModel(ProbeAdapter):
    """Forecast from history alone or augment it with future exogenous data."""

    def __init__(self, config):
        future_exog_mode = getattr(config, "future_exog_mode", "probe")
        if future_exog_mode not in {"probe", "probe_only", "mlp", "history"}:
            raise ValueError(
                "future_exog_mode must be 'probe', 'probe_only', 'mlp', or "
                f"'history', got {future_exog_mode!r}"
            )

        probe_config = namespaced_config(config, "probe")
        # probe_config.patch_len = config.horizon

        if future_exog_mode == "probe_only":
            baseline = None
        else:
            baseline_config = namespaced_config(config, "baseline")
            baseline = HistoricalCrossForecaster(baseline_config)

        super().__init__(
            baseline=baseline,
            probenet=Probe(probe_config),
            alpha=config.alpha,
            beta=probe_config.beta,
        )

        self.future_exog_mode = future_exog_mode
        if self.future_exog_mode == "mlp" and config.fusion_method != "mlp":
            raise ValueError("future_exog_mode='mlp' requires fusion_method='mlp'")
        if self.future_exog_mode in {"history", "probe_only"} and config.fusion_method:
            raise ValueError(
                f"future_exog_mode={self.future_exog_mode!r} requires "
                "fusion_method=''"
            )

    def forward(self, history, exog_future):
        if self.future_exog_mode == "probe_only":
            probe_prediction, probe_loss = self.probenet(history, exog_future)
            return probe_prediction, self.beta * probe_loss

        if self.future_exog_mode in {"mlp", "history"}:
            # In MLP mode the benchmark-level CovariateFusion consumes
            # exog_future later. History mode deliberately ignores it entirely.
            baseline_output = self.baseline(history)
            baseline_prediction = self._prediction(baseline_output)

            # Preserve the adapter's (prediction, additional_loss) contract.
            zero_loss = baseline_prediction.new_zeros(())
            return baseline_prediction, zero_loss

        return super().forward(history, exog_future)


class ProbeNet(ProbeTestPatchOrderMixin, DeepForecastingModelBase):
    """Forecast with a historical model and a configurable future-exogenous path."""

    def __init__(self, **kwargs):
        super().__init__(MODEL_HYPER_PARAMS, **kwargs)

    @property
    def model_name(self):
        return "ProbeNet"

    def _init_model(self):
        return ProbeNetModel(self.config)

    def _init_criterion(self):
        criterion = super()._init_criterion()
        self.config.criterion = criterion
        return criterion

    def _process(self, input, target, input_mark, target_mark, exog_future=None):
        del target, input_mark, target_mark
        output, probe_loss = self.model(input, exog_future)
        result = {"output": output}
        if self.model.training:
            result["additional_loss"] = probe_loss
        return result


__all__ = ["ProbeNet", "ProbeNetModel"]
