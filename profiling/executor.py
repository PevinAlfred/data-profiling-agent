import pandas as pd
from typing import Any

class ExecutionEngine:
    """Executes cleaning scripts using Pandas only."""

    def load_csv(self, path: str) -> Any:
        return pd.read_csv(path)

    def save_csv(self, df: Any, path: str) -> None:
        df.to_csv(path, index=False)