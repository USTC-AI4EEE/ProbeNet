# -*- coding: utf-8 -*-
import json
import math
import os
from typing import List, Union

import pandas as pd

from ts_benchmark.common.constant import ROOT_PATH
from ts_benchmark.recording import load_record_data


def report(report_config: dict) -> str:
    """Write compact, machine-readable ProbeNet validation-search results."""
    log_files: Union[List[str], pd.DataFrame] = report_config.get("log_files_list")
    if isinstance(log_files, pd.DataFrame):
        log_data = log_files
    else:
        if not log_files:
            raise ValueError("No validation-search logs to report")
        log_data = load_record_data(log_files)

    required_columns = {
        "file_name",
        "model_params",
        "final_validation_loss",
    }
    missing_columns = required_columns - set(log_data.columns)
    if missing_columns:
        raise ValueError(
            "Validation-search records are missing columns: "
            + ", ".join(sorted(missing_columns))
        )

    records = []
    for _, row in log_data.iterrows():
        loss = row["final_validation_loss"]
        records.append(
            {
                "dataset": row["file_name"],
                "final_validation_loss": (
                    None
                    if pd.isna(loss) or not math.isfinite(float(loss))
                    else float(loss)
                ),
                "hyperparameters": json.loads(row["model_params"]),
            }
        )

    records.sort(
        key=lambda item: (
            item["dataset"],
            item["final_validation_loss"] is None,
            item["final_validation_loss"] or 0.0,
            json.dumps(item["hyperparameters"], sort_keys=True),
        )
    )

    save_path = report_config.get("save_path")
    output_dir = (
        save_path
        if save_path and os.path.isabs(save_path)
        else os.path.join(ROOT_PATH, "result", save_path or "")
    )
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, report_config["result_file_name"])
    with open(output_path, "w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            output_file.write("\n")
    return output_path
