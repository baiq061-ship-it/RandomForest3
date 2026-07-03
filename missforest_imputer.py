# -*- coding: utf-8 -*-
"""
MissForest-style imputer for mixed clinical tabular data.

The implementation uses iterative RandomForestRegressor / RandomForestClassifier
models from scikit-learn. It is intentionally self-contained so the skill does
not depend on the older `missingpy` package.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


class MissForestImputer(BaseEstimator, TransformerMixin):
    """Iterative random-forest imputer for mixed continuous/categorical data.

    Parameters
    ----------
    continuous_columns, categorical_columns:
        Column names. If omitted, numeric columns are treated as continuous and
        the rest as categorical.
    max_iter:
        Number of MissForest iterations.
    n_estimators, max_depth, min_samples_leaf, max_features:
        Hyperparameters passed to the random forests used for imputation.
    fit_all_columns:
        If False, only variables with missing values in the fitting data get a
        final imputation model; variables that are only missing later are filled
        with the training median/mode. If True, final imputation models are fit
        for every variable that has enough observed values. True is more robust
        for deployment but slower.
    """

    def __init__(
        self,
        continuous_columns: Optional[Sequence[str]] = None,
        categorical_columns: Optional[Sequence[str]] = None,
        max_iter: int = 5,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_leaf: int = 1,
        max_features: Any = "sqrt",
        random_state: int = 42,
        n_jobs: int = 1,
        fit_all_columns: bool = False,
        stopping_tolerance: float = 1e-4,
    ):
        self.continuous_columns = continuous_columns
        self.categorical_columns = categorical_columns
        self.max_iter = max_iter
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.fit_all_columns = fit_all_columns
        self.stopping_tolerance = stopping_tolerance

    def _as_dataframe(self, X: Any) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X.copy()
        cols = getattr(self, "columns_", None)
        if cols is None:
            cols = [f"x{i}" for i in range(np.asarray(X).shape[1])]
        return pd.DataFrame(X, columns=list(cols))

    def _infer_columns(self, df: pd.DataFrame) -> None:
        self.columns_ = list(df.columns)
        if self.continuous_columns is None and self.categorical_columns is None:
            cont = [c for c in self.columns_ if pd.api.types.is_numeric_dtype(df[c])]
            cat = [c for c in self.columns_ if c not in cont]
        else:
            cont = [c for c in (self.continuous_columns or []) if c in self.columns_]
            cat = [c for c in (self.categorical_columns or []) if c in self.columns_]
            rest = [c for c in self.columns_ if c not in set(cont + cat)]
            for c in rest:
                if pd.api.types.is_numeric_dtype(df[c]):
                    cont.append(c)
                else:
                    cat.append(c)
        self.continuous_columns_ = list(dict.fromkeys(cont))
        self.categorical_columns_ = list(dict.fromkeys(cat))
        self._col_index_ = {c: i for i, c in enumerate(self.columns_)}

    def _prepare_encoders(self, df: pd.DataFrame) -> None:
        self.numeric_fill_: Dict[str, float] = {}
        self.categorical_fill_: Dict[str, str] = {}
        self.category_maps_: Dict[str, Dict[str, int]] = {}
        self.category_inv_maps_: Dict[str, Dict[int, str]] = {}

        for c in self.continuous_columns_:
            z = pd.to_numeric(df[c], errors="coerce")
            med = float(z.median()) if z.notna().any() else 0.0
            self.numeric_fill_[c] = med

        for c in self.categorical_columns_:
            s = df[c].astype("object")
            observed = s[s.notna()].astype(str)
            fill = str(observed.mode().iloc[0]) if len(observed) else "__MISSING__"
            cats = sorted(set(observed.tolist() + [fill]))
            mapping = {v: i for i, v in enumerate(cats)}
            self.categorical_fill_[c] = fill
            self.category_maps_[c] = mapping
            self.category_inv_maps_[c] = {i: v for v, i in mapping.items()}

    def _encode(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        z = pd.DataFrame(index=df.index, columns=self.columns_, dtype=float)
        missing = pd.DataFrame(False, index=df.index, columns=self.columns_)

        for c in self.continuous_columns_:
            x = pd.to_numeric(df[c], errors="coerce")
            m = x.isna()
            z[c] = x.fillna(self.numeric_fill_.get(c, 0.0)).astype(float)
            missing[c] = m

        for c in self.categorical_columns_:
            s = df[c].astype("object")
            m = s.isna()
            mapping = self.category_maps_[c]
            fill_code = mapping[self.categorical_fill_[c]]
            encoded = []
            for v in s:
                if pd.isna(v):
                    encoded.append(fill_code)
                else:
                    encoded.append(mapping.get(str(v), fill_code))
            z[c] = np.asarray(encoded, dtype=float)
            missing[c] = m

        return z[self.columns_], missing[self.columns_]

    def _make_estimator(self, column: str, salt: int = 0):
        rs = int(self.random_state) + int(salt) + self._col_index_.get(column, 0) * 13
        common = dict(
            n_estimators=int(self.n_estimators),
            max_depth=self.max_depth,
            min_samples_leaf=int(self.min_samples_leaf),
            max_features=self.max_features,
            random_state=rs,
            n_jobs=int(self.n_jobs),
        )
        if column in self.categorical_columns_:
            return RandomForestClassifier(**common)
        return RandomForestRegressor(**common)

    def _can_fit_target(self, y: np.ndarray, is_cat: bool) -> bool:
        if len(y) < 2:
            return False
        if is_cat and len(np.unique(y.astype(int))) < 2:
            return False
        if (not is_cat) and np.nanstd(y.astype(float)) == 0:
            # A constant continuous target can only be imputed by the fill value.
            return False
        return True

    def fit(self, X: Any, y: Any = None):
        df = self._as_dataframe(X)
        self._infer_columns(df)
        self._prepare_encoders(df)
        z, missing = self._encode(df)
        x_imp = z.to_numpy(dtype=float, copy=True)
        missing_np = missing.to_numpy(dtype=bool)
        p = x_imp.shape[1]
        other_idx = {c: [i for i in range(p) if i != self._col_index_[c]] for c in self.columns_}

        miss_cols = [c for c in self.columns_ if bool(missing[c].any())]
        miss_cols = sorted(miss_cols, key=lambda c: int(missing[c].sum()))
        prev_error = np.inf

        for it in range(max(0, int(self.max_iter))):
            if not miss_cols:
                break
            old = x_imp.copy()
            for c in miss_cols:
                j = self._col_index_[c]
                miss_rows = missing_np[:, j]
                obs_rows = ~miss_rows
                if not np.any(miss_rows):
                    continue
                y_obs = z.iloc[obs_rows, j].to_numpy(dtype=float)
                is_cat = c in self.categorical_columns_
                if not self._can_fit_target(y_obs, is_cat):
                    continue
                est = self._make_estimator(c, salt=it * 1000)
                target = y_obs.astype(int) if is_cat else y_obs.astype(float)
                est.fit(x_imp[obs_rows][:, other_idx[c]], target)
                pred = est.predict(x_imp[miss_rows][:, other_idx[c]])
                x_imp[miss_rows, j] = np.asarray(pred, dtype=float)

            changed_mask = missing_np
            if np.any(changed_mask):
                num = np.nanmean((old[changed_mask] - x_imp[changed_mask]) ** 2)
                den = np.nanmean(old[changed_mask] ** 2) + 1e-12
                error = float(num / den)
                if error < float(self.stopping_tolerance) or error > prev_error:
                    break
                prev_error = error

        self.estimators_: Dict[str, Any] = {}
        cols_to_fit = self.columns_ if bool(self.fit_all_columns) else miss_cols
        for c in cols_to_fit:
            j = self._col_index_[c]
            obs_rows = ~missing_np[:, j]
            if int(obs_rows.sum()) < 2:
                continue
            y_obs = z.iloc[obs_rows, j].to_numpy(dtype=float)
            is_cat = c in self.categorical_columns_
            if not self._can_fit_target(y_obs, is_cat):
                continue
            est = self._make_estimator(c, salt=999_000)
            target = y_obs.astype(int) if is_cat else y_obs.astype(float)
            est.fit(x_imp[obs_rows][:, other_idx[c]], target)
            self.estimators_[c] = est

        self._other_idx_ = other_idx
        self.missing_columns_seen_ = miss_cols
        return self

    def transform(self, X: Any) -> pd.DataFrame:
        df = self._as_dataframe(X)
        # Keep exactly the training columns and order. Extra columns are ignored;
        # missing expected columns are created as NaN and then imputed/fallback-filled.
        for c in self.columns_:
            if c not in df.columns:
                df[c] = np.nan
        df = df[self.columns_].copy()

        z, missing = self._encode(df)
        x_imp = z.to_numpy(dtype=float, copy=True)
        missing_np = missing.to_numpy(dtype=bool)
        miss_cols = [c for c in self.columns_ if bool(missing[c].any())]
        miss_cols = sorted(miss_cols, key=lambda c: int(missing[c].sum()))

        for it in range(max(1, int(self.max_iter))):
            if not miss_cols:
                break
            for c in miss_cols:
                est = self.estimators_.get(c)
                if est is None:
                    continue
                j = self._col_index_[c]
                miss_rows = missing_np[:, j]
                if not np.any(miss_rows):
                    continue
                pred = est.predict(x_imp[miss_rows][:, self._other_idx_[c]])
                x_imp[miss_rows, j] = np.asarray(pred, dtype=float)

        out = df.copy()
        for c in self.continuous_columns_:
            j = self._col_index_[c]
            vals = pd.to_numeric(out[c], errors="coerce").astype(float)
            vals.loc[missing[c]] = x_imp[missing_np[:, j], j]
            vals = vals.fillna(self.numeric_fill_.get(c, 0.0))
            out[c] = vals

        for c in self.categorical_columns_:
            j = self._col_index_[c]
            vals = out[c].astype("object")
            inv = self.category_inv_maps_[c]
            decoded = []
            for raw in x_imp[missing_np[:, j], j]:
                code = int(np.round(raw))
                decoded.append(inv.get(code, self.categorical_fill_[c]))
            if np.any(missing_np[:, j]):
                vals.loc[missing[c]] = decoded
            vals = vals.where(vals.notna(), self.categorical_fill_[c])
            out[c] = vals.astype("object")

        return out[self.columns_]

    def get_feature_names_out(self, input_features: Optional[Sequence[str]] = None):
        if input_features is not None:
            return np.asarray(list(input_features), dtype=object)
        return np.asarray(getattr(self, "columns_", []), dtype=object)
