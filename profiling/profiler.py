import pandas as pd
from typing import Dict, Any

class DataProfiler:
    """Statistical & rule-based data profiler."""

    def profile(self, df: pd.DataFrame, schema: Dict[str, Any]) -> Dict[str, Any]:
        profile = {}
        for col in df.columns:
            profile[col] = {
                "type": str(df[col].dtype),
                "missing": int(df[col].isnull().sum()),
                "unique": int(df[col].nunique()),
                "mean": float(df[col].mean()) if pd.api.types.is_numeric_dtype(df[col]) else None,
                "std": float(df[col].std()) if pd.api.types.is_numeric_dtype(df[col]) else None,
            }
        return profile