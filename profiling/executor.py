import pandas as pd
import dask.dataframe as dd
from pyspark.sql import SparkSession
from typing import Any, Dict

class ExecutionEngine:
    """Executes cleaning scripts using Pandas, Dask, or Spark."""

    def __init__(self, engine: str):
        self.engine = engine

    def load_csv(self, path: str) -> Any:
        if self.engine == "pandas":
            return pd.read_csv(path)
        elif self.engine == "dask":
            return dd.read_csv(path)
        elif self.engine == "spark":
            spark = SparkSession.builder.getOrCreate()
            return spark.read.csv(path, header=True, inferSchema=True)
        else:
            raise ValueError("Unsupported engine")

    def save_csv(self, df: Any, path: str) -> None:
        if self.engine == "pandas":
            df.to_csv(path, index=False)
        elif self.engine == "dask":
            df.compute().to_csv(path, index=False)
        elif self.engine == "spark":
            df.write.csv(path, header=True, mode="overwrite")
        else:
            raise ValueError("Unsupported engine")