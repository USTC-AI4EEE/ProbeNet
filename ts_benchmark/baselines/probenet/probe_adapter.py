"""A thin adapter that gives a baseline and ProbeNet separate namespaces."""

from __future__ import annotations

import copy
from typing import Any, Callable, Mapping, Optional, Tuple

import torch
import torch.nn as nn


def namespaced_config(config: Any, namespace: str) -> Any:
    """Copy ``config`` and apply overrides from ``config.<namespace>``.

    Shared runtime fields such as ``seq_len``, ``pred_len`` and ``enc_in`` are
    copied into both configs. Model-specific values are nested dictionaries::

        {
            "baseline": {"d_model": 128, "n_heads": 4},
            "probe": {"d_model": 64, "n_heads": 2, "n_templates": 6}
        }

    The returned config is independent, so mutations cannot leak between the
    baseline, ProbeNet and the benchmark-level config.
    """

    result = copy.deepcopy(config)
    overrides = getattr(config, namespace, {})
    if overrides is None:
        overrides = {}
    if not isinstance(overrides, Mapping):
        raise TypeError(f"config.{namespace} must be a mapping, got {type(overrides)}")
    for name, value in overrides.items():
        setattr(result, name, copy.deepcopy(value))
    return result


class ProbeAdapter(nn.Module):
    """Fuse a historical baseline with a future-exogenous ProbeNet.

    When a baseline is provided, parameters are named ``baseline.*`` and
    ``probenet.*`` in ``named_parameters()`` and ``state_dict()``. Passing
    ``baseline=None`` supports wrappers whose forward path uses Probe alone.

    By default, the baseline is called as ``baseline(history)``. For a model
    with a different signature, ``baseline_forward`` provides a small bridge.
    """

    def __init__(
        self,
        baseline: Optional[nn.Module],
        probenet: nn.Module,
        *,
        alpha: float = 0.5,
        beta: float = 0.01,
        baseline_forward: Optional[Callable[[nn.Module, torch.Tensor], Any]] = None,
        baseline_output_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")

        if beta < 0:
            raise ValueError("beta must be non-negative")
        if baseline_output_dim is not None and baseline_output_dim <= 0:
            raise ValueError("baseline_output_dim must be positive when provided")
        if baseline is None and baseline_forward is not None:
            raise ValueError("baseline_forward requires a baseline module")

        self.baseline = baseline
        self.probenet = probenet
        self.alpha = alpha
        self.beta = float(beta)
        self._baseline_forward = baseline_forward
        self.baseline_output_dim = baseline_output_dim

    @staticmethod
    def _prediction(output: Any) -> torch.Tensor:
        if torch.is_tensor(output):
            return output
        if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
            return output[0]
        raise TypeError(
            "baseline must return a tensor or a tuple starting with a tensor"
        )

    def forward(
        self, history: torch.Tensor, exog_future: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.baseline is None:
            raise RuntimeError(
                "ProbeAdapter.forward requires a baseline; a Probe-only wrapper "
                "must call probenet directly"
            )
        if self._baseline_forward is None:
            baseline_output = self.baseline(history)
        else:
            baseline_output = self._baseline_forward(self.baseline, history)
        baseline_prediction = self._prediction(baseline_output)
        if self.baseline_output_dim is not None:
            baseline_prediction = baseline_prediction[..., : self.baseline_output_dim]

        probe_prediction, probe_loss = self.probenet(history, exog_future)
        if baseline_prediction.shape != probe_prediction.shape:
            raise ValueError(
                "baseline and ProbeNet output shapes differ: "
                f"{tuple(baseline_prediction.shape)} != {tuple(probe_prediction.shape)}"
            )

        prediction = (
            1.0 - self.alpha
        ) * baseline_prediction + self.alpha * probe_prediction

        # return prediction, 0  # TODO
        return prediction, self.beta * probe_loss


__all__ = ["ProbeAdapter", "namespaced_config"]
