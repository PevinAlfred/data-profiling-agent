from transformers import pipeline
from typing import Dict, Any

class ScriptGenerator:
    """Uses LLM to generate Python cleaning scripts."""

    def __init__(self, model_name: str):
        self.generator = pipeline("text-generation", model=model_name)

    def extract_code(self, text: str) -> str:
        import re
        # Extract code between triple backticks if present
        match = re.search(r"```(.*?)```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        # Fallback: keep only lines that look like Python code
        code_lines = []
        code_started = False
        for line in text.splitlines():
            stripped = line.strip()
            if not code_started:
                if stripped.startswith((
                    "import ", "from ", "def ", "class ", "#", "df", "for ", "if ", "with ", "try:", "except ", "return ", "print(", "else:", "elif ", "while ", "pass", "raise ", "global ", "nonlocal ", "assert ", "yield ", "lambda ")):
                    code_started = True
            if code_started and stripped and not stripped.startswith("Prompts:"):
                code_lines.append(line)
        return "\n".join(code_lines)

    def __init__(self, model_name: str):
        self.generator = pipeline("text-generation", model=model_name)

    def generate_script(self, profile: Dict[str, Any], prompts: str) -> str:
        input_text = (
            f"Profile: {profile}\n"
            f"Prompts: {prompts}\n"
            "Generate a pandas data cleaning script for the following tasks: "
            "1. Fill missing ages with the mean age. "
            "2. Remove rows with negative ages. "
            "3. Remove rows with invalid email formats (missing '@' or domain). "
            "4. Drop rows with missing required fields (id, name). "
            "Only output valid Python code, wrapped in triple backticks."
        )
        result = self.generator(input_text, max_length=512, truncation=True)[0]["generated_text"]
        return self.extract_code(result)