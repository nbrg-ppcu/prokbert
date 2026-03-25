#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


SHEET_TO_FILENAME = {
    "Basemodels": "basemodels.json",
    "DefaultTrainingParameters": "default_training_parameters.json",
    "FinetuningTask": "finetuning_task.json",
    "Task": "task.json",
}


def _py_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return int(value)
    return value


def _records_from_dataframe(df: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        records.append({str(col): _py_value(value) for col, value in row.items()})
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert the modelling workbook into HF-ready JSON registry files.")
    parser.add_argument("excel_path", help="Path to Modelling.xlsx")
    parser.add_argument("output_dir", help="Directory where JSON files will be written")
    args = parser.parse_args()

    excel_path = Path(args.excel_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    xl = pd.ExcelFile(excel_path)
    basemodels = pd.read_excel(excel_path, sheet_name="Basemodels").copy()
    basemodels["model_id"] = basemodels["name"].astype(str)

    lookup: dict[str, str] = {}
    for _, row in basemodels.iterrows():
        model_id = str(row["model_id"])
        for key in ("name", "model_id", "hf_name", "hf_path"):
            if key in row and pd.notna(row[key]):
                lookup[str(row[key])] = model_id

    default_training = pd.read_excel(excel_path, sheet_name="DefaultTrainingParameters").copy()
    default_training["model_id"] = default_training["basemodel"].map(
        lambda value: lookup.get(str(value), str(value)) if pd.notna(value) else None
    )

    finetuning_task = pd.read_excel(excel_path, sheet_name="FinetuningTask").copy()
    if "basemodel" in finetuning_task.columns:
        finetuning_task["model_id"] = finetuning_task["basemodel"].map(
            lambda value: lookup.get(str(value), str(value)) if pd.notna(value) else None
        )

    task = pd.read_excel(excel_path, sheet_name="Task").copy()

    sheet_frames = {
        "Basemodels": basemodels,
        "DefaultTrainingParameters": default_training,
        "FinetuningTask": finetuning_task,
        "Task": task,
    }

    for sheet_name, filename in SHEET_TO_FILENAME.items():
        payload = {
            "schema_version": "1.0.0",
            "source_workbook": excel_path.name,
            "source_sheet": sheet_name,
            "records": _records_from_dataframe(sheet_frames[sheet_name]),
        }
        with open(output_dir / filename, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")

    manifest = {
        "schema_version": "1.0.0",
        "source_workbook": excel_path.name,
        "files": {
            "basemodels": "basemodels.json",
            "default_training_parameters": "default_training_parameters.json",
            "finetuning_task": "finetuning_task.json",
            "task": "task.json",
        },
    }
    with open(output_dir / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
