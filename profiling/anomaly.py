import pandas as pd
from sklearn.ensemble import IsolationForest
from typing import List, Dict, Any

class AnomalyDetector:
    """ML-based anomaly detection using Isolation Forest."""

    def detect(self, df: pd.DataFrame) -> List[int]:
        anomalies = set()
        # Rule-based: missing values
        for idx, row in df.iterrows():
            if pd.isnull(row.get('id')) or pd.isnull(row.get('name')):
                anomalies.add(idx)
            if pd.isnull(row.get('age')) or (isinstance(row.get('age'), (int, float)) and row.get('age') < 0):
                anomalies.add(idx)
            email = str(row.get('email'))
            if not email or '@' not in email or '.' not in email.split('@')[-1]:
                anomalies.add(idx)
        # ML-based: IsolationForest
        numeric_df = df.select_dtypes(include="number").dropna()
        if not numeric_df.empty and len(numeric_df) >= 2:
            try:
                clf = IsolationForest(random_state=42)
                preds = clf.fit_predict(numeric_df)
                ml_anomalies = numeric_df.index[preds == -1].tolist()
                anomalies.update(ml_anomalies)
            except Exception:
                pass
        return sorted(list(anomalies))