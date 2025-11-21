"""
Feature engineering pipeline for classification.

Provides reusable transformers for numeric, categorical, text, and datetime features.
Uses scikit-learn's ColumnTransformer for composable pipelines.
"""

from typing import List, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from .config import ClassificationConfig


class DatetimeFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract features from datetime columns."""

    def __init__(self, datetime_cols: List[str]):
        self.datetime_cols = datetime_cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """Extract hour, day, month, day_of_week, is_weekend, etc."""
        X = X.copy()
        extracted_features = []

        for col in self.datetime_cols:
            if col in X.columns:
                # Convert to datetime
                dt_series = pd.to_datetime(X[col], errors='coerce')

                # Extract features
                extracted_features.append(pd.DataFrame({
                    f'{col}_hour': dt_series.dt.hour,
                    f'{col}_day': dt_series.dt.day,
                    f'{col}_month': dt_series.dt.month,
                    f'{col}_dayofweek': dt_series.dt.dayofweek,
                    f'{col}_is_weekend': (dt_series.dt.dayofweek >= 5).astype(int),
                    f'{col}_quarter': dt_series.dt.quarter,
                }))

        if extracted_features:
            return pd.concat([X.drop(columns=self.datetime_cols)] + extracted_features, axis=1)
        return X.drop(columns=self.datetime_cols)


class FeatureEngineer:
    """
    Main feature engineering class.

    Handles automatic detection and transformation of different feature types.
    """

    def __init__(self, config: ClassificationConfig):
        self.config = config
        self.numeric_pipeline = None
        self.categorical_pipeline = None
        self.text_vectorizer = None
        self.feature_transformer = None
        self.feature_names_ = None

    def auto_detect_columns(self, df: pd.DataFrame, target_col: str) -> dict:
        """
        Automatically detect column types.

        Returns:
            dict with keys: numeric, categorical, text, datetime
        """
        columns = {
            'numeric': [],
            'categorical': [],
            'text': [],
            'datetime': []
        }

        # Exclude target and specified columns
        feature_cols = [col for col in df.columns
                        if col != target_col and col not in self.config.exclude_cols]

        for col in feature_cols:
            dtype = df[col].dtype

            # Datetime detection
            if pd.api.types.is_datetime64_any_dtype(dtype):
                columns['datetime'].append(col)
            # Numeric detection
            elif pd.api.types.is_numeric_dtype(dtype):
                # Check if it's actually categorical (few unique values)
                if df[col].nunique() < 10 and df[col].nunique() / len(df) < 0.05:
                    columns['categorical'].append(col)
                else:
                    columns['numeric'].append(col)
            # Text detection (strings with high unique count)
            elif pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype):
                unique_ratio = df[col].nunique() / len(df)
                avg_length = df[col].astype(str).str.len().mean()

                if avg_length > 20 or unique_ratio > 0.5:  # Likely text
                    columns['text'].append(col)
                else:  # Likely categorical
                    if df[col].nunique() <= self.config.max_categories:
                        columns['categorical'].append(col)
                    else:
                        # Too many categories, treat as text or drop
                        columns['text'].append(col)
            else:
                # Boolean or other types -> treat as categorical
                columns['categorical'].append(col)

        return columns

    def build_transformers(self, df: pd.DataFrame, target_col: str) -> ColumnTransformer:
        """
        Build feature transformation pipeline.

        Returns:
            ColumnTransformer for all feature types
        """
        # Auto-detect or use configured columns
        if self.config.numeric_cols is None:
            detected = self.auto_detect_columns(df, target_col)
            numeric_cols = detected['numeric']
            categorical_cols = detected['categorical']
            text_cols = detected['text']
            datetime_cols = detected['datetime']
        else:
            numeric_cols = self.config.numeric_cols
            categorical_cols = self.config.categorical_cols or []
            text_cols = self.config.text_cols or []
            datetime_cols = self.config.datetime_cols or []

        transformers = []

        # Numeric pipeline
        if numeric_cols:
            numeric_transformer = Pipeline([
                ('imputer', SimpleImputer(strategy=self.config.numeric_impute_strategy)),
                ('scaler', self._get_scaler())
            ])
            transformers.append(('numeric', numeric_transformer, numeric_cols))

        # Categorical pipeline
        if categorical_cols:
            categorical_transformer = Pipeline([
                ('imputer', SimpleImputer(strategy='constant',
                                          fill_value=self.config.categorical_impute_value)),
                ('encoder', self._get_encoder())
            ])
            transformers.append(('categorical', categorical_transformer, categorical_cols))

        # Datetime features
        if datetime_cols and self.config.extract_time_features:
            datetime_transformer = DatetimeFeatureExtractor(datetime_cols)
            # Note: This transforms the data, so we need to re-process after

        if not transformers:
            raise ValueError("No valid feature columns detected!")

        self.feature_transformer = ColumnTransformer(
            transformers=transformers,
            remainder='drop',  # Drop columns not specified
            sparse_threshold=0  # Return dense array
        )

        return self.feature_transformer

    def _get_scaler(self):
        """Get numeric scaler based on config."""
        if self.config.numeric_scale_method == "standard":
            return StandardScaler()
        elif self.config.numeric_scale_method == "minmax":
            return MinMaxScaler()
        elif self.config.numeric_scale_method == "robust":
            return RobustScaler()
        else:
            return StandardScaler()

    def _get_encoder(self):
        """Get categorical encoder based on config."""
        if self.config.categorical_encoding == "onehot":
            return OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        elif self.config.categorical_encoding == "ordinal":
            return OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        else:
            return OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    def fit_transform_tabular(self, df: pd.DataFrame, target_col: str) -> Tuple[np.ndarray, List[str]]:
        """
        Fit and transform tabular features.

        Returns:
            (transformed_data, feature_names)
        """
        # Build transformers
        self.build_transformers(df, target_col)

        # Fit and transform
        X_transformed = self.feature_transformer.fit_transform(df)

        # Get feature names
        feature_names = self._get_feature_names()
        self.feature_names_ = feature_names

        return X_transformed, feature_names

    def transform_tabular(self, df: pd.DataFrame) -> np.ndarray:
        """Transform new data using fitted transformers."""
        if self.feature_transformer is None:
            raise ValueError("Must call fit_transform_tabular first!")

        return self.feature_transformer.transform(df)

    def _get_feature_names(self) -> List[str]:
        """Extract feature names from ColumnTransformer."""
        feature_names = []

        for name, transformer, columns in self.feature_transformer.transformers_:
            if name == 'remainder':
                continue

            if isinstance(columns, list):
                if hasattr(transformer, 'get_feature_names_out'):
                    names = transformer.get_feature_names_out(columns)
                else:
                    names = columns
                feature_names.extend(names)

        return feature_names

    def prepare_text_features(self, df: pd.DataFrame, text_col: str,
                              fit: bool = True) -> np.ndarray:
        """
        Prepare text features for LSTM or other models.

        Returns:
            Text feature matrix (TF-IDF or sequences)
        """
        if text_col not in df.columns:
            raise ValueError(f"Text column '{text_col}' not found!")

        # Fill NaN with empty string
        text_data = df[text_col].fillna('').astype(str)

        if self.config.text_vectorization == "tfidf":
            if fit:
                self.text_vectorizer = TfidfVectorizer(
                    max_features=self.config.text_max_features,
                    stop_words='english',
                    lowercase=True
                )
                return self.text_vectorizer.fit_transform(text_data).toarray()
            else:
                if self.text_vectorizer is None:
                    raise ValueError("Text vectorizer not fitted!")
                return self.text_vectorizer.transform(text_data).toarray()

        elif self.config.text_vectorization == "count":
            if fit:
                self.text_vectorizer = CountVectorizer(
                    max_features=self.config.text_max_features,
                    stop_words='english',
                    lowercase=True
                )
                return self.text_vectorizer.fit_transform(text_data).toarray()
            else:
                return self.text_vectorizer.transform(text_data).toarray()

        else:
            # For embedding-based (LSTM), return raw text
            return text_data.values

    def get_feature_importance_names(self) -> List[str]:
        """Get interpretable feature names for feature importance plots."""
        return self.feature_names_ if self.feature_names_ else []
