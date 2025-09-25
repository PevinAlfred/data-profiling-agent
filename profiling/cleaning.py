import os
import json
import pandas as pd
import requests
from mistralai import Mistral
from dotenv import load_dotenv

load_dotenv()

class ScriptGenerator:
    def __init__(self):
        self.mistral_api_key = os.getenv("MISTRAL_API_KEY")
        self.mistral_model = "mistral-small-latest"
        self.mistral_small_model = "mistral-small-latest"
        self.codestral_model = "codestral-latest"
        self.mistral_client = Mistral(api_key=self.mistral_api_key)


    def detect_anomalies(self, df, domd):
        input_text = (
            f"DOMD: {json.dumps(domd)}\n"
            f"Input Data Sample: {df.to_json(orient='records')}\n"
            "Task: Analyze the input data and DOMD. Detect and list all anomalies, strictly enforcing every constraint in the DOMD for every row and column. Output MUST be a JSON object with a single key 'anomalies' containing a list of anomaly objects. Each anomaly object MUST contain 'row', 'column', 'anomaly_type', and 'details'. Do not include any markdown formatting or explanation."
        )
        messages = [
            {"role": "system", "content": "You are a data profiling and anomaly detection expert. You must strictly enforce every constraint in the DOMD for every row and column. Output only valid JSON as specified, with no extra text."},
            {"role": "user", "content": input_text}
        ]
        try:
            chat_response = self.mistral_client.chat.complete(
                model=self.mistral_small_model,
                messages=messages
            )
            result = chat_response.choices[0].message.content
        except Exception as e:
            print(f"Mistral SDK call failed for anomaly detection: {e}")
            return []
        if not result:
            print("Mistral returned empty response for anomaly detection.")
            return []
        try:
            import re
            # Remove code block markers if present
            code_match = re.search(r"```(?:json|JSON)?\n?(.*)```", result, re.DOTALL)
            json_str = code_match.group(1).strip() if code_match else result.strip()
            # Fix invalid escape sequences (e.g., \/ -> /)
            json_str = json_str.replace(r'\/', '/')
            # Remove any stray backslashes before quotes
            json_str = re.sub(r'\\(["\'])', r'\1', json_str)
            # Remove any invalid control characters
            json_str = re.sub(r'[\x00-\x1F]+', '', json_str)
            try:
                anomalies_obj = json.loads(json_str)
            except Exception as e2:
                print(f"Second attempt to decode JSON failed. Output: {json_str} Error: {e2}")
                anomalies_obj = {}
            anomalies = anomalies_obj.get("anomalies", [])
            if not isinstance(anomalies, list):
                print(f"Mistral output for anomalies was not a list. Output: {result}")
                anomalies = []
        except Exception as e:
            print(f"Failed to decode JSON from Mistral output for anomaly detection. Output was: {result} Error: {e}")
            anomalies = []
        return anomalies

    def clean_data(self, df, domd, batch_size=5):
        cleaned_rows = []
        uncleaned_rows = []
        num_rows = len(df)
        for start in range(0, num_rows, batch_size):
            batch_df = df.iloc[start:start+batch_size]
            input_data_str = batch_df.to_json(orient="records")
            input_text = (
                f"DOMD: {json.dumps(domd)}\n"
                f"Input Data Sample: {input_data_str}\n"
                "Task: Clean the input data strictly according to the DOMD. For every field, strictly enforce all constraints in the DOMD, including length, padding, type, allowed values, and required/nullable status. If a field requires padding (e.g., leading zeros), pad to the required length as specified in the DOMD. If a field is required and cannot be fixed, move the row to uncleaned. Do not hardcode any field-specific logic; follow only the DOMD. Your response MUST be a JSON object with two keys: 'cleaned' and 'uncleaned'. Each key should contain a list of data records (as JSON objects). No row or column in 'cleaned' may violate any DOMD constraint. Do not include any markdown formatting or explanation."
            )
            messages = [
                {"role": "system", "content": "You are a data cleaning expert. You must strictly enforce every constraint in the DOMD for every row and column. For every field, follow the DOMD exactly: enforce length, padding, type, allowed values, and required/nullable status. If a field requires padding (e.g., leading zeros), pad to the required length as specified in the DOMD. Do not hardcode any field-specific logic; follow only the DOMD. Output only valid JSON as specified, with no extra text."},
                {"role": "user", "content": input_text}
            ]
            try:
                chat_response = self.mistral_client.chat.complete(
                    model=self.mistral_model,
                    messages=messages
                )
                result = chat_response.choices[0].message.content
            except Exception as e:
                print(f"Mistral SDK call failed for data cleaning: {e}")
                cleaned_rows.extend(batch_df.to_dict(orient="records"))
                continue
            if not result:
                print("Mistral returned empty response for data cleaning.")
                cleaned_rows.extend(batch_df.to_dict(orient="records"))
                continue
            try:
                import re
                # Remove code block markers if present
                code_match = re.search(r"```(?:json|JSON)?\n?(.*)```", result, re.DOTALL)
                json_str = code_match.group(1).strip() if code_match else result.strip()
                output = json.loads(json_str)
                cleaned_rows.extend(output.get("cleaned", []))
                uncleaned_rows.extend(output.get("uncleaned", []))
            except Exception as e:
                print(f"Failed to decode JSON from Mistral output for data cleaning. Output was: {result} Error: {e}")
                cleaned_rows.extend(batch_df.to_dict(orient="records"))
        # Post-processing: strictly enforce DOMD constraints on clean data
        clean_df = pd.DataFrame(cleaned_rows)
        unclean_df = pd.DataFrame(uncleaned_rows)
        if not clean_df.empty:
            clean_df, moved = self._enforce_domd_constraints_generic(clean_df, domd)
            if not moved.empty:
                unclean_df = pd.concat([unclean_df, moved], ignore_index=True)
        return clean_df, unclean_df

    def _enforce_domd_constraints_generic(self, df, domd):
        import re
        from datetime import datetime
        moved = pd.DataFrame()
        keep_mask = pd.Series([True] * len(df), index=df.index)
        def handle_date(col, s):
            fmt = "%Y%m%d" if "yyyy-mm-dd" not in col.get("constraints", "").lower() else "%Y-%m-%d"
            def norm(v):
                if pd.isnull(v): return None
                v = str(v)
                for f in ["%Y%m%d","%Y-%m-%d","%d%m%Y","%d/%m/%Y","%m/%d/%Y","%Y/%m/%d","%Y.%m.%d","%d.%m.%Y"]:
                    try: return datetime.strptime(v, f).strftime(fmt)
                    except: pass
                d = re.sub(r"[^0-9]", "", v)
                return d.zfill(8) if len(d)<=8 and fmt=="%Y%m%d" else (d[:8] if len(d)>8 and fmt=="%Y%m%d" else None)
            normed = s.apply(norm)
            def is_valid(v):
                if pd.isnull(v): return False
                vstr = str(v)
                if not re.fullmatch(r"\d{8}", vstr): return False
                try:
                    return datetime.strptime(vstr, fmt).date() <= datetime.now().date()
                except Exception:
                    return False
            valid = normed.apply(is_valid)
            return normed, valid
        def handle_currency(col, s):
            canon = (col.get("allowed",[None])[0] or col.get("sample") or "GBP").upper()
            normed = s.apply(lambda v: canon)
            return normed, normed.notnull()
        def handle_pad(col, s):
            l, t, c = col.get("length"), col.get("type","string"), col.get("constraints","")
            if not (l and isinstance(l,int)): return s, pd.Series([True]*len(s), index=s.index)
            def pad(v):
                if pd.isnull(v): return None
                v = str(v)
                if t in ["string","integer"]:
                    if "leading zero" in c.lower() or "pad" in c.lower(): v = v.zfill(l)
                    if len(v)>l: v = v[-l:]
                    if len(v)<l: v = v.zfill(l)
                    return v
                return v
            normed = s.apply(pad)
            return normed, normed.notnull()
        def handle_type(col, s):
            t = col.get("type","string")
            if t=="integer":
                normed = s.apply(lambda v: int(v) if not pd.isnull(v) and str(v).isdigit() else None)
                return normed, normed.notnull()
            if t=="float":
                normed = s.apply(lambda v: float(v) if not pd.isnull(v) and re.match(r"^-?\d+(\.\d+)?$",str(v)) else None)
                return normed, normed.notnull()
            return s, pd.Series([True]*len(s), index=s.index)
        def handle_allowed(col, s):
            a = col.get("allowed",None)
            if not a: return s, pd.Series([True]*len(s), index=s.index)
            normed = s.where(s.isin(a), None)
            return normed, normed.notnull()
        handlers = [
            (lambda c: c.get("type") == "date" or ("yyyy" in c.get("constraints","")), handle_date),
            (lambda c: c.get("type") == "string" and ("currency" in c.get("constraints","") or (c.get("allowed") and all(isinstance(a,str) and len(a)==3 for a in c.get("allowed"))) or (c.get("sample") and isinstance(c.get("sample"),str) and len(c.get("sample"))==3)), handle_currency),
            (lambda c: c.get("length") is not None and isinstance(c.get("length"),int), handle_pad),
            (lambda c: c.get("type") in ["integer","float"], handle_type),
            (lambda c: c.get("allowed") is not None, handle_allowed)
        ]
        for col in domd["columns"]:
            name = col["name"]
            required = col.get("required", False)
            nullable = col.get("nullable", True)
            s = df[name] if name in df.columns else pd.Series([None]*len(df))
            col_mask = pd.Series([True]*len(s), index=s.index)
            for pred, h in handlers:
                if pred(col):
                    s, valid = h(col, s)
                    col_mask &= valid
            df[name] = s
            if required and not nullable:
                col_mask &= df[name].notnull()
            keep_mask &= col_mask
        moved = df[~keep_mask]
        df = df[keep_mask]
        return df.reset_index(drop=True), moved.reset_index(drop=True) if not moved.empty else pd.DataFrame()

    def generate_cleaning_script(self, domd):
        input_text = (
            f"DOMD: {json.dumps(domd)}\n"
            "You are a Python expert. Generate a pandas data cleaning script that strictly enforces every constraint in the DOMD for every row and column. The script should read input.csv, clean the data, and save clean_data.csv. Output only valid Python code. Do not include markdown formatting, the prompt, or any explanation text."
        )
        messages = [
            {"role": "system", "content": "You are a Python expert. You must strictly enforce every constraint in the DOMD for every row and column. Output only valid Python code, with no extra text."},
            {"role": "user", "content": input_text}
        ]
        # Use mistralai SDK for Codestral
        chat_response = self.mistral_client.chat.complete(
            model=self.codestral_model,
            messages=messages
        )
        result = chat_response.choices[0].message.content
        import re
        match = re.search(r"```(?:python\n)?(.*?)```", result, re.DOTALL)
        if match:
            return match.group(1).strip()
        return result.strip()  # Fallback to returning the full response if no code block is found