# AI Data Profiling Agent

## Overview
This project implements an end-to-end AI-powered data profiling agent using a Hugging Face LLM for orchestration, anomaly detection, and Python cleaning script generation. The architecture follows the attached HLD diagrams.

## Features
- Upload DOMD (schema), profiling prompts, and raw CSV data
- Statistical profiling and ML-based anomaly detection
- LLM-generated Python cleaning scripts
- Configurable execution engine (Pandas, Dask, Spark)
- Interactive Streamlit UI for workflow management

## Setup

1. Clone the repo and install dependencies:
`pip install -r requirements.txt`

2. Edit `config/config.yaml` to select LLM model and execution engine.

3. Run the app:
`streamlit run app.py`

4. Upload your files and trigger profiling via the UI.

## Example
Sample files are provided in `sample_data/`. Outputs are saved in `outputs/`.

## Notes
- All code is modular and extensible.
- No deprecated packages or methods are used.
- Streamlit is chosen for its interactive data profiling workflow.

## License
MIT