# Universal Bank - Streamlit Dashboard

This repository provides a Streamlit dashboard for exploratory analysis and model training to predict **Personal Loan** acceptance.
Files in this package are at repository root (no directories).

**How to use**
1. Create a new GitHub repo and upload the files from this zip (all files are at root).
2. Connect the repo to Streamlit Cloud (https://streamlit.io/cloud) and set the main file to `app.py`.
3. On first run, upload your `UniversalBank.csv` using the sidebar uploader.

**Notes**
- The app trains Decision Tree, Random Forest and Gradient Boosting classifiers and displays metrics, ROC, confusion matrices and feature importances.
- Use the "Predict New Data" tab to upload a new CSV for inference (models are kept in session after training).
