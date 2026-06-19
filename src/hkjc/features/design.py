"""Model design matrix (PLAN.md §1G, M3).

Turns the ``features_runner`` frame into a single numeric matrix for the model zoo:

* **numeric** columns = the M2 fundamental set (``BASELINE_FEATURES``), used as-is by the
  logit and the tabular NNs.
* **categorical** columns (sire/dam/dam's-sire/import-type/sex/country/venue) are
  integer-encoded and appended; their indices are reported so the GBMs (esp. CatBoost) can
  treat them natively. These are exactly the high-cardinality signals the logit cannot use.

Integer-encoding a category -> code is a fixed label map (not leakage), so it is fit once on
the whole column. The canary is intentionally *excluded* here -- it did its job proving the
M2 pipeline clean; the zoo trains on real features only.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import polars as pl

from hkjc.features.base import BASELINE_FEATURES
from hkjc.features.store import features_path
from hkjc.models.base import FloatArray

NUMERIC_FEATURES: tuple[str, ...] = BASELINE_FEATURES
CATEGORICAL_FEATURES: tuple[str, ...] = (
    "sire",
    "dam",
    "dams_sire",
    "import_type",
    "sex",
    "country_of_origin",
    "venue",
)
_NA = "__NA__"


@dataclass(frozen=True, slots=True)
class Design:
    """A model-ready matrix plus the metadata each learner needs."""

    x: FloatArray  # (n_rows, n_numeric + n_categorical); categoricals integer-encoded
    feature_names: list[str]
    numeric_indices: list[int]
    categorical_indices: list[int]

    def numeric(self) -> FloatArray:
        """Numeric-only view (for the logit and NNs)."""
        return self.x[:, self.numeric_indices]


def build_design(df: pl.DataFrame) -> Design:
    """Build the combined numeric + integer-encoded categorical design matrix."""
    numeric = df.select(NUMERIC_FEATURES).to_numpy().astype(np.float64)
    cat_cols: list[FloatArray] = []
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            codes = df[col].cast(pl.String).fill_null(_NA).cast(pl.Categorical).to_physical()
            cat_cols.append(codes.to_numpy().astype(np.float64))
        else:
            cat_cols.append(np.zeros(df.height, dtype=np.float64))
    n_num = len(NUMERIC_FEATURES)
    x = np.column_stack([numeric, *cat_cols]) if cat_cols else numeric
    return Design(
        x=x,
        feature_names=[*NUMERIC_FEATURES, *CATEGORICAL_FEATURES],
        numeric_indices=list(range(n_num)),
        categorical_indices=list(range(n_num, n_num + len(CATEGORICAL_FEATURES))),
    )


def data_hash(features_file: str | None = None) -> str:
    """Content hash of the feature store Parquet (logged to MLflow for reproducibility)."""
    path = features_path() if features_file is None else None
    target = features_file if features_file is not None else str(path)
    h = hashlib.sha256()
    with open(target, "rb") as handle:  # noqa: PTH123 - hashing raw bytes
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]
