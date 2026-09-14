# -*- coding: utf-8 -*-
import time
from typing import List, Optional

import numpy as np
import pandas as pd

from ts_benchmark.evaluation.strategy.constants import FieldNames
from ts_benchmark.evaluation.strategy.forecasting import ForecastingStrategy
from ts_benchmark.models import ModelFactory
from ts_benchmark.utils.data_processing import split_channel, split_time


FINAL_VALIDATION_LOSS = "final_validation_loss"


class ProbeNetValidationSearch(ForecastingStrategy):
    """Train ProbeNet on train data and return only its final validation loss.

    The suffix after ``tv_ratio`` is reserved as test data and is never passed to
    the model. Checkpoint creation is disabled for every search trial.
    """

    REQUIRED_CONFIGS = ["tv_ratio", "train_ratio_in_tv", "target_channel"]

    def _execute(
        self,
        series: pd.DataFrame,
        meta_info: Optional[pd.Series],
        model_factory: ModelFactory,
        series_name: str,
    ) -> List:
        model = model_factory()
        if getattr(model, "model_name", None) != "ProbeNet":
            raise TypeError("probenet_validation_search only supports ProbeNet")

        tv_ratio = self._get_scalar_config_value("tv_ratio", series_name)
        train_ratio_in_tv = self._get_scalar_config_value(
            "train_ratio_in_tv", series_name
        )
        target_channel = self._get_scalar_config_value("target_channel", series_name)

        if not 0 < tv_ratio < 1:
            raise ValueError("tv_ratio must be between 0 and 1")
        if not 0 < train_ratio_in_tv < 1:
            raise ValueError("train_ratio_in_tv must be between 0 and 1")

        data_len = int(self._get_meta_info(meta_info, "length", len(series)))
        train_valid_length = int(tv_ratio * data_len)
        if train_valid_length <= 0 or train_valid_length >= data_len:
            raise ValueError("The train-validation or test split is empty")

        # Deliberately discard the test suffix before constructing model inputs.
        train_valid_data, _ = split_time(series, train_valid_length)
        target_train_valid_data, exog_train_valid_data = split_channel(
            train_valid_data, target_channel
        )

        start_fit_time = time.time()
        model.forecast_fit(
            target_train_valid_data,
            covariates={"exog": exog_train_valid_data},
            train_ratio_in_tv=train_ratio_in_tv,
            validation_search=True,
        )
        fit_time = time.time() - start_fit_time
        validation_loss = float(model.get_final_validation_loss())

        return [
            validation_loss,
            series_name,
            fit_time,
            0.0,
            np.nan,
            np.nan,
            "",
        ]

    @staticmethod
    def accepted_metrics() -> List[str]:
        # The loss comes from ProbeNet's own configured training criterion.
        return []

    @property
    def field_names(self) -> List[str]:
        return [
            FINAL_VALIDATION_LOSS,
            FieldNames.FILE_NAME,
            FieldNames.FIT_TIME,
            FieldNames.INFERENCE_TIME,
            FieldNames.ACTUAL_DATA,
            FieldNames.INFERENCE_DATA,
            FieldNames.LOG_INFO,
        ]
