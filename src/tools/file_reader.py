import pandas as pd
import numpy as np
from typing import Union, Optional, List, Dict, Any
import os


class FileReader:
    SUPPORTED_FORMATS = ['csv', 'tsv', 'xlsx', 'xls', 'txt']

    @staticmethod
    def read_file(
        file_path: str,
        format_type: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        if format_type is None:
            ext = os.path.splitext(file_path)[1].lower().strip('.')
            format_type = ext if ext in FileReader.SUPPORTED_FORMATS else 'csv'

        if format_type == 'csv':
            return pd.read_csv(file_path, **kwargs)
        elif format_type in ['tsv', 'txt']:
            return pd.read_csv(file_path, sep='\t', **kwargs)
        elif format_type in ['xlsx', 'xls']:
            return pd.read_excel(file_path, **kwargs)
        else:
            raise ValueError(f"Unsupported format: {format_type}")

    @staticmethod
    def read_csv(file_path: str, **kwargs) -> pd.DataFrame:
        return pd.read_csv(file_path, **kwargs)

    @staticmethod
    def read_tsv(file_path: str, **kwargs) -> pd.DataFrame:
        return pd.read_csv(file_path, sep='\t', **kwargs)

    @staticmethod
    def read_excel(file_path: str, sheet_name: Union[str, int] = 0, **kwargs) -> pd.DataFrame:
        return pd.read_excel(file_path, sheet_name=sheet_name, **kwargs)

    @staticmethod
    def get_data_info(df: pd.DataFrame) -> Dict[str, Any]:
        return {
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "missing_values": df.isnull().sum().to_dict(),
            "numeric_columns": list(df.select_dtypes(include=[np.number]).columns),
            "categorical_columns": list(df.select_dtypes(include=['object', 'category']).columns)
        }

    @staticmethod
    def validate_expression_data(df: pd.DataFrame) -> Dict[str, Any]:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return {"valid": False, "message": "No numeric columns found"}

        return {
            "valid": True,
            "sample_column": numeric_cols[0] if len(numeric_cols) > 0 else None,
            "gene_column": df.columns[0] if df.shape[1] > 0 else None,
            "expression_range": {
                "min": df[numeric_cols].min().min(),
                "max": df[numeric_cols].max().max(),
                "mean": df[numeric_cols].mean().mean()
            }
        }
