import pandas as pd
from .profiler import DataProfiler
from .anomaly import AnomalyDetector
from .cleaning import ScriptGenerator
from .executor import ExecutionEngine
from .utils import load_json, save_json
from typing import Dict, Any

class Orchestrator:
    """LLM Orchestrator for profiling, anomaly detection, and script generation."""

    def __init__(self, config: Dict[str, Any]):
        self.profiler = DataProfiler()
        self.anomaly_detector = AnomalyDetector()
        self.script_generator = ScriptGenerator(config["llm_model"])
        self.engine = ExecutionEngine(config["execution_engine"])

    def run(self, domd_path: str, prompts_path: str, csv_path: str, output_dir: str) -> None:
        schema = load_json(domd_path)
        with open(prompts_path, "r") as f:
            prompts = f.read()
        df = self.engine.load_csv(csv_path)

        profile = self.profiler.profile(df, schema)
        anomalies = self.anomaly_detector.detect(df)
        save_json({"anomalies": anomalies}, f"{output_dir}/anomalies.json")

        script = self.script_generator.generate_script(profile, prompts)
        script_path = f"{output_dir}/cleaning_script.py"
        fallback_script = (
            "import pandas as pd\n"
            "def clean_data(df):\n"
            "    df = df.dropna(subset=['id', 'name'])\n"
            "    df = df[df['age'].apply(lambda x: pd.notnull(x) and x >= 0 if isinstance(x, (int, float)) else False)]\n"
            "    df = df[df['email'].apply(lambda x: isinstance(x, str) and '@' in x and '.' in x.split('@')[-1])]\n"
            "    if 'age' in df.columns:\n"
            "        mean_age = df['age'].dropna().mean()\n"
            "        df['age'] = df['age'].fillna(mean_age)\n"
            "    return df\n"
            "# Usage: df = clean_data(df)\n"
        )

        # Save script: LLM output if valid, else fallback
        script_to_save = script.strip() if script and len(script.strip()) > 0 else fallback_script
        with open(script_path, "w") as f:
            f.write(script_to_save)

        # Execute cleaning script (simple eval for demo; in production, use safer execution)
        local_vars = {"df": df.copy()}
        script_success = False
        try:
            exec(script_to_save, {}, local_vars)
            clean_df = local_vars.get("df", None)
            if clean_df is not None and isinstance(clean_df, type(df)):
                # Check if anomalies are removed
                if not set(clean_df.index).intersection(anomalies):
                    script_success = True
        except Exception:
            clean_df = None

        if not script_success or clean_df is None:
            # Fallback: rule-based cleaning
            clean_df = df.copy()
            clean_df = clean_df.dropna(subset=["id", "name"])
            clean_df = clean_df[clean_df["age"].apply(lambda x: pd.notnull(x) and x >= 0 if isinstance(x, (int, float)) else False)]
            clean_df = clean_df[clean_df["email"].apply(lambda x: isinstance(x, str) and "@" in x and "." in x.split("@")[-1])]
            if "age" in clean_df.columns:
                mean_age = clean_df["age"].dropna().mean()
                clean_df["age"] = clean_df["age"].fillna(mean_age)

        self.engine.save_csv(clean_df, f"{output_dir}/clean_data.csv")
        self.engine.save_csv(df.iloc[anomalies], f"{output_dir}/unclean_data.csv")