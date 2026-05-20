import pandas as pd
import numpy as np
from typing import Optional, List, Tuple, Union
from scipy import stats


class DataPreprocessor:
    @staticmethod
    def clean_data(
        df: pd.DataFrame,
        remove_na: bool = True,
        remove_duplicates: bool = True,
        fill_value: Optional[Union[str, float]] = None
    ) -> pd.DataFrame:
        result = df.copy()

        if remove_na:
            result = result.dropna()
        elif fill_value is not None:
            result = result.fillna(fill_value)

        if remove_duplicates:
            result = result.drop_duplicates()

        return result

    @staticmethod
    def normalize(
        df: pd.DataFrame,
        method: str = 'log2',
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        result = df.copy()
        cols = columns or result.select_dtypes(include=[np.number]).columns

        if method == 'log2':
            result[cols] = np.log2(result[cols] + 1)
        elif method == 'log10':
            result[cols] = np.log10(result[cols] + 1)
        elif method == 'zscore':
            for col in cols:
                result[col] = stats.zscore(result[col])
        elif method == 'minmax':
            for col in cols:
                min_val = result[col].min()
                max_val = result[col].max()
                result[col] = (result[col] - min_val) / (max_val - min_val) if max_val != min_val else 0

        return result

    @staticmethod
    def filter_genes(
        df: pd.DataFrame,
        gene_column: str,
        min_expression: Optional[float] = None,
        max_expression: Optional[float] = None,
        top_n: Optional[int] = None
    ) -> pd.DataFrame:
        result = df.copy()

        if min_expression is not None:
            numeric_cols = result.select_dtypes(include=[np.number]).columns
            result = result[
                (result[numeric_cols] >= min_expression).any(axis=1)
            ]

        if max_expression is not None:
            numeric_cols = result.select_dtypes(include=[np.number]).columns
            result = result[
                (result[numeric_cols] <= max_expression).any(axis=1)
            ]

        if top_n is not None:
            numeric_cols = result.select_dtypes(include=[np.number]).columns
            result['mean_expression'] = result[numeric_cols].mean(axis=1)
            result = result.nlargest(top_n, 'mean_expression')
            result = result.drop('mean_expression', axis=1)

        return result

    @staticmethod
    def calculate_differential(
        df: pd.DataFrame,
        group_column: str,
        group1: str,
        group2: str,
        numeric_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        cols = numeric_columns or [
            col for col in df.columns if col != group_column
        ]

        group1_data = df[df[group_column] == group1][cols]
        group2_data = df[df[group_column] == group2][cols]

        result = pd.DataFrame({
            'gene': cols,
            f'mean_{group1}': group1_data.mean().values,
            f'mean_{group2}': group2_data.mean().values,
            f'log2FC': np.log2(
                (group2_data.mean().values + 1) / (group1_data.mean().values + 1)
            )
        })

        return result

    @staticmethod
    def transform_for_heatmap(
        df: pd.DataFrame,
        gene_column: str = 'gene',
        z_score: bool = False
    ) -> Tuple[pd.DataFrame, List[str]]:
        result = df.copy()
        genes = result[gene_column].tolist()

        numeric_cols = result.select_dtypes(include=[np.number]).columns
        data = result[numeric_cols]

        if z_score:
            data = data.apply(stats.zscore)

        return data, genes

    @staticmethod
    def impute_missing_values(
        df: pd.DataFrame,
        method: str = 'mean'
    ) -> pd.DataFrame:
        result = df.copy()
        numeric_cols = result.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if result[col].isnull().any():
                if method == 'mean':
                    result[col].fillna(result[col].mean(), inplace=True)
                elif method == 'median':
                    result[col].fillna(result[col].median(), inplace=True)
                elif method == 'zero':
                    result[col].fillna(0, inplace=True)

        return result
