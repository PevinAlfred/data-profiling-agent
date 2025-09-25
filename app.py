import streamlit as st
import yaml
import os
from profiling.orchestrator import Orchestrator

# Load config
with open("config/config.yaml") as f:
    config = yaml.safe_load(f)

st.title("AI Data Profiling Agent")

st.sidebar.header("Upload Files")
domd_file = st.sidebar.file_uploader("DOMD (schema/metadata)", type=["json"])
csv_file = st.sidebar.file_uploader("Input CSV", type=["csv"])

output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

if st.sidebar.button("Run Profiling") and domd_file and csv_file:
    domd_path = os.path.join(output_dir, "domd.json")
    csv_path = os.path.join(output_dir, "input.csv")

    with open(domd_path, "wb") as f:
        f.write(domd_file.read())
    with open(csv_path, "wb") as f:
        f.write(csv_file.read())

    orchestrator = Orchestrator(config)
    orchestrator.run(domd_path, csv_path, output_dir)
    st.success("Profiling complete!")

    import json
    import pandas as pd

    st.header("Profiling Results")
    st.markdown("---")

    # Summary Section
    st.subheader("Summary")
    st.markdown("- Data profiling and cleaning completed successfully.")
    st.markdown("- See below for detected anomalies, generated cleaning script, and output data tables.")

    st.markdown("---")

    # Anomalies Section
    st.subheader("Detected Anomalies")
    try:
        with open(f"{output_dir}/anomalies.json") as f:
            anomalies = json.load(f)
        if isinstance(anomalies, dict) and "anomalies" in anomalies:
            st.write(f"Total anomalies detected: {len(anomalies['anomalies'])}")
            st.table(pd.DataFrame({"Row Index": anomalies["anomalies"]}))
        else:
            st.json(anomalies)
    except Exception as e:
        st.error(f"Could not load anomalies: {e}")

    st.markdown("---")

    # Cleaning Script Section
    st.subheader("Generated Python Cleaning Script")
    try:
        with open(f"{output_dir}/cleaning_script.py") as f:
            script = f.read()
        st.code(script, language="python")
    except Exception as e:
        st.error(f"Could not load cleaning script: {e}")

    st.markdown("---")

    # Clean Data Table
    st.subheader("Clean Data Table")
    try:
        clean_df = pd.read_csv(f"{output_dir}/clean_data.csv")
        st.dataframe(clean_df, width='stretch')
    except Exception as e:
        st.error(f"Could not load clean data: {e}")

    st.markdown("---")

    # Unclean Data Table
    st.subheader("Unclean Data Table (Anomalies)")
    try:
        unclean_df = pd.read_csv(f"{output_dir}/unclean_data.csv")
        st.dataframe(unclean_df, width='stretch')
    except Exception as e:
        st.error(f"Could not load unclean data: {e}")

# Streamlit is chosen for its interactive workflow, allowing easy upload, profiling, and review.