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
            # Fallback: rule-based cleaning and correction
            clean_df = df.copy()
            corrections = []
            unclean_rows = []
            for idx, row in clean_df.iterrows():
                fixed = row.copy()
                issues = []
                # Date: fix format
                if pd.isnull(fixed.get('date')) or not str(fixed['date']).isdigit() or len(str(fixed['date'])) != 8:
                    issues.append('date')
                    # Try to fix: pad/truncate to 8 digits
                    val = str(fixed.get('date', ''))
                    val = val.zfill(8)[:8] if val else '00000000'
                    fixed['date'] = val if val.isdigit() else '00000000'
                # Site: fix length
                if pd.isnull(fixed.get('site')) or len(str(fixed['site'])) != 4:
                    issues.append('site')
                    val = str(fixed.get('site', ''))
                    fixed['site'] = val.zfill(4)[:4] if val else '0000'
                # Article: fix length and leading zeros
                if pd.isnull(fixed.get('article')) or len(str(fixed['article'])) != 18 or not str(fixed['article']).startswith('000000'):
                    issues.append('article')
                    val = str(fixed.get('article', ''))
                    val = val.zfill(18)[:18]
                    if not val.startswith('000000'):
                        val = '000000' + val[6:]
                    fixed['article'] = val
                # Stock units: must be integer
                try:
                    fixed['stock_units'] = int(float(fixed['stock_units']))
                except Exception:
                    issues.append('stock_units')
                    fixed['stock_units'] = 0
                # Retail value: must be float
                try:
                    fixed['retail_value'] = float(fixed['retail_value'])
                except Exception:
                    issues.append('retail_value')
                    fixed['retail_value'] = 0.0
                # Cost value: must be float
                try:
                    fixed['cost_value'] = float(fixed['cost_value'])
                except Exception:
                    issues.append('cost_value')
                    fixed['cost_value'] = 0.0
                # Currency: must be GBP
                if str(fixed.get('currency', '')).upper() != 'GBP':
                    issues.append('currency')
                    fixed['currency'] = 'GBP'
                # If any required field is still missing or invalid after fix, mark as unclean
                required_fields = ['date', 'site', 'article', 'stock_units', 'retail_value', 'cost_value', 'currency']
                still_invalid = False
                for field in required_fields:
                    val = fixed.get(field)
                    if pd.isnull(val) or val == '' or (field == 'date' and (not str(val).isdigit() or len(str(val)) != 8)):
                        still_invalid = True
                if still_invalid:
                    unclean_rows.append(fixed)
                corrections.append(fixed)

            clean_df = pd.DataFrame(corrections)
            unclean_df = pd.DataFrame(unclean_rows)
        else:
            unclean_df = df.iloc[anomalies]

        self.engine.save_csv(clean_df, f"{output_dir}/clean_data.csv")
        self.engine.save_csv(unclean_df, f"{output_dir}/unclean_data.csv")