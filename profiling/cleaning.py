import pandas as pd
import json
import re
import logging
import os
from mistralai import Mistral

logger = logging.getLogger(__name__)

class ScriptGenerator:
    """Uses LLM to analyze, clean, and generate scripts for tabular data."""

    def __init__(self, model_name: str = "mistral-large-latest"):
        api_key = os.environ.get("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("MISTRAL_API_KEY environment variable not set. Please get a key from https://mistral.ai/")
        self.client = Mistral(api_key=api_key)
        self.model = model_name

    def detect_anomalies(self, df, domd, prompts):
        domd_str = str(domd)[:2000]
        prompts_str = str(prompts)[:2000]
        input_data_str = df.to_json(orient="records")
        input_text = (
            f"DOMD: {domd_str}\n"
            f"Prompts: {prompts_str}\n"
            f"Input Data Sample: {input_data_str}\n"
            "Task: Analyze the input data and DOMD to identify data anomalies. Your response MUST be a JSON object with a single key 'anomalies' that contains a list of anomaly objects. Each anomaly object MUST contain 'column_name', 'anomaly_type', and 'details' keys."
        )
        messages = [
            {"role": "system", "content": "You are a data profiling expert. You must respond with a valid JSON object."},
            {"role": "user", "content": input_text}
        ]
        try:
            chat_response = self.client.chat.complete(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"}
            )
            result = chat_response.choices[0].message.content
            anomalies_obj = json.loads(result)
            anomalies = anomalies_obj.get("anomalies", [])
            if not isinstance(anomalies, list):
                logger.error(f"LLM output for anomalies was not a list. Output: {result}")
                anomalies = []
        except Exception as e:
            logger.error(f"Failed to decode JSON from LLM output for anomaly detection. Output was: {e}")
            anomalies = []
        return anomalies

    def clean_data(self, df, domd, prompts, batch_size=5):
        domd_str = str(domd)[:2000]
        prompts_str = str(prompts)[:2000]
        cleaned_rows = []
        uncleaned_rows = []
        num_rows = len(df)
        for start in range(0, num_rows, batch_size):
            batch_df = df.iloc[start:start+batch_size]
            input_data_str = batch_df.to_json(orient="records")
            input_text = (
                f"DOMD: {domd_str}\n"
                f"Prompts: {prompts_str}\n"
                f"Input Data Sample: {input_data_str}\n"
                "Task: Clean the input data according to the DOMD and prompts. Your response MUST be a JSON object with two keys: 'cleaned' and 'uncleaned'. Each key should contain a list of data records (as JSON objects)."
            )
            messages = [
                {"role": "system", "content": "You are a data cleaning expert. You must respond with a valid JSON object containing 'cleaned' and 'uncleaned' keys."},
                {"role": "user", "content": input_text}
            ]
            try:
                chat_response = self.client.chat.complete(
                    model=self.model,
                    messages=messages,
                    response_format={"type": "json_object"}
                )
                result = chat_response.choices[0].message.content
                output = json.loads(result)
                cleaned_rows.extend(output.get("cleaned", []))
                uncleaned_rows.extend(output.get("uncleaned", []))
            except Exception as e:
                logger.error(f"Failed to decode JSON from LLM output for data cleaning. Output was: {e}")
                cleaned_rows.extend(batch_df.to_dict(orient="records"))
        clean_df = pd.DataFrame(cleaned_rows)
        unclean_df = pd.DataFrame(uncleaned_rows)
        return clean_df, unclean_df

    def generate_cleaning_script(self, domd, prompts):
        domd_str = str(domd)[:2000]
        prompts_str = str(prompts)[:2000]
        input_text = (
            f"DOMD: {domd_str}\n"
            f"Prompts: {prompts_str}\n"
            "You are a Python expert. Generate a pandas data cleaning script that cleans the input data according to the DOMD and prompts. The script should read input.csv, clean the data, and save clean_data.csv. Output only valid Python code, wrapped in triple backticks, and do not include the prompt or any explanation text."
        )
        messages = [
            {"role": "system", "content": "You are a Python expert who writes pandas data cleaning scripts. You only output Python code inside a markdown block."},
            {"role": "user", "content": input_text}
        ]
        try:
            chat_response = self.client.chat.complete(
                model=self.model,
                messages=messages,
            )
            result = chat_response.choices[0].message.content
            match = re.search(r"```(?:python\n)?(.*?)```", result, re.DOTALL)
            if match:
                return match.group(1).strip()
            return result # Fallback to returning the full response if no code block is found
        except Exception as e:
            logger.error(f"Failed to generate cleaning script from LLM. Error: {e}")
            return f"# Failed to generate script: {e}"