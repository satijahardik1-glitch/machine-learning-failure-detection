import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from app.pages.batch import _validate_csv_header
from app.inference import (
    FEATURES, ArtifactLoadError, condition_state, load_artifacts, predict_batch,
)

ROOT = Path(__file__).resolve().parents[1]


class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models, cls.metadata = load_artifacts(ROOT / "models")
        cls.sample = pd.DataFrame([{
            "Air temperature [K]": 300.1,
            "Process temperature [K]": 310.1,
            "Rotational speed [rpm]": 1500,
            "Torque [Nm]": 40.0,
            "Tool wear [min]": 100,
            "asset_tag": "demo-001",
        }])

    def test_artifacts_load_with_feature_order(self):
        self.assertEqual(self.metadata["feature_order"], FEATURES)
        self.assertEqual(set(self.models), {"hgb", "random_forest", "isolation_forest"})

    def test_single_and_batch_prediction_are_consistent(self):
        single = predict_batch(self.sample.iloc[[0]], self.models, self.metadata).iloc[0]
        batch = predict_batch(pd.concat([self.sample, self.sample], ignore_index=True), self.models, self.metadata)
        self.assertAlmostEqual(float(single["HGB failure probability"]), float(batch.iloc[0]["HGB failure probability"]))
        self.assertEqual(single["Machine condition"], batch.iloc[0]["Machine condition"])
        self.assertEqual(batch.loc[0, "asset_tag"], "demo-001")

    def test_invalid_rows_are_preserved_and_not_scored(self):
        invalid = self.sample.copy()
        invalid.loc[0, "Torque [Nm]"] = -1
        result = predict_batch(invalid, self.models, self.metadata)
        self.assertIn("cannot be negative", result.loc[0, "Input validation error"])
        self.assertTrue(pd.isna(result.loc[0, "HGB failure probability"]))
        self.assertEqual(result.loc[0, "asset_tag"], "demo-001")

    def test_batch_csv_round_trip_keeps_input_records_and_predictions(self):
        result = predict_batch(self.sample, self.models, self.metadata)
        exported = pd.read_csv(pd.io.common.StringIO(result.to_csv(index=False)))
        self.assertEqual(exported.loc[0, "asset_tag"], "demo-001")
        self.assertIn("Machine condition", exported.columns)
        self.assertIn(exported.loc[0, "Machine condition"], {
            "FAILURE RISK", "ANOMALOUS CONDITION", "NORMAL CONDITION"
        })

    def test_missing_sensor_column_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Missing required sensor"):
            predict_batch(self.sample.drop(columns=[FEATURES[0]]), self.models, self.metadata)

    def test_invalid_csv_headers_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Required sensor column"):
            _validate_csv_header(b"Air temperature [K],Process temperature [K]\n300,310\n")
        with self.assertRaisesRegex(ValueError, "duplicate column"):
            _validate_csv_header((",".join(FEATURES + [FEATURES[0]]) + "\n").encode())

    def test_condition_state_rules(self):
        self.assertEqual(condition_state(1, 1), "FAILURE RISK")
        self.assertEqual(condition_state(0, 1), "ANOMALOUS CONDITION")
        self.assertEqual(condition_state(0, 0), "NORMAL CONDITION")

    def test_missing_artifact_message_is_actionable(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ArtifactLoadError, "train_artifacts.py"):
                load_artifacts(Path(tmp))


if __name__ == "__main__":
    unittest.main()
