# -*- coding: utf-8 -*-
from ts_benchmark.evaluation.strategy.fixed_forecast import FixedForecast
from ts_benchmark.evaluation.strategy.rolling_forecast import RollingForecast
from ts_benchmark.evaluation.strategy.validation_search import (
    ProbeNetValidationSearch,
)

STRATEGY = {
    "fixed_forecast": FixedForecast,
    "rolling_forecast": RollingForecast,
    "probenet_validation_search": ProbeNetValidationSearch,
}
