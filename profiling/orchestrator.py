
import pandas as pd
from .cleaning import ScriptGenerator
from .utils import load_json, save_json
from typing import Dict, Any


class Orchestrator:
    """LLM Orchestrator for anomaly detection, cleaning, and script generation."""

    def __init__(self, config: Dict[str, Any]):
        self.llm = ScriptGenerator(config["llm_model"])

    def run(self, domd_path: str, prompts_path: str, csv_path: str, output_dir: str) -> None:
        domd = load_json(domd_path)
        with open(prompts_path, "r") as f:
            prompts = f.read()
        df = pd.read_csv(csv_path)

        # 1. Anomaly detection by LLM
        anomalies = self.llm.detect_anomalies(df, domd, prompts)
        save_json({"anomalies": anomalies}, f"{output_dir}/anomalies.json")

        # 2. Data cleaning by LLM
        clean_df, unclean_df = self.llm.clean_data(df, domd, prompts)
        clean_df.to_csv(f"{output_dir}/clean_data.csv", index=False)
        unclean_df.to_csv(f"{output_dir}/unclean_data.csv", index=False)

        # 3. Generate cleaning script by LLM (for backup)
        script = self.llm.generate_cleaning_script(domd, prompts)
        with open(f"{output_dir}/cleaning_script.py", "w") as f:
            f.write(script)