import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, confusion_matrix)
import base64

st.set_page_config(page_title="Universal Bank - Marketing Dashboard", layout="wide")

st.title("Universal Bank — Marketing Intelligence & Loan Conversion Dashboard")

# ---------------------- Helper functions ----------------------

def normalize_columns(df):
    cols_map = {}
    for c in df.columns:
        low = c.lower()
        if low == 'id' or low == 'id ':
            cols_map[c] = 'ID'
        elif 'personal' in low and 'loan' in low:
            cols_map[c] = 'Personal Loan'
        elif low.startswith('age'):
            cols_map[c] = 'Age'
        elif 'experience' in low:
            cols_map[c] = 'Experience'
        elif 'income' in low:
            cols_map[c] = 'Income'
        elif 'zip' in low:
            cols_map[c] = 'Zip code'
        elif 'family' in low:
            cols_map[c] = 'Family'
        elif 'ccavg' in low or 'cc avg' in low:
            cols_map[c] = 'CCAvg'
        elif 'education' in low:
            cols_map[c] = 'Education'
        elif 'mortgage' in low:
            cols_map[c] = 'Mortgage'
        elif 'secur' in low:
            cols_map[c] = 'Securities'
        elif 'cd' in low and 'account' in low:
            cols_map[c] = 'CDAccount'
        elif 'online' in low:
            cols_map[c] = 'Online'
        elif 'credit' in low and 'card' in low:
            cols_map[c] = 'CreditCard'
    return df.rename(columns=cols_map)

def prepare_model_data(df):
    # standard set of model columns
    model_cols = ['Age','Experience','Income','Family','CCAvg','Education','Mortgage',
                  'Securities','CDAccount','Online','CreditCard']
    present = [c for c in model_cols if c in df.columns]
    X = df[present].copy().fillna(df[present].median())
    y = df['Personal Loan'].astype(int).copy()
    return X, y, present

def compute_metrics_and_cv(model, X_train, y_train, X_test, y_test, cv_splits=5):
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_test_proba = model.predict_proba(X_test)[:,1]
    else:
        y_test_proba = model.decision_function(X_test)
    metrics = {
        'Training Accuracy': accuracy_score(y_train, y_train_pred),
        'Testing Accuracy': accuracy_score(y_test, y_test_pred),
        'Precision': precision_score(y_test, y_test_pred, zero_division=0),
        'Recall': recall_score(y_test, y_test_pred, zero_division=0),
        'F1 Score': f1_score(y_test, y_test_pred, zero_division=0),
        'AUC': roc_auc_score(y_test, y_test_proba)
    }
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    cv_acc = cross_val_score(model, X_train, y_train, cv=skf, scoring='accuracy')
    cv_auc = cross_val_score(model, X_train, y_train, cv=skf, scoring='roc_auc')
    metrics['CV Accuracy Mean'] = cv_acc.mean()
    metrics['CV AUC Mean'] = cv_auc.mean()
    return metrics, model, (y_test_proba if hasattr(model,"predict_proba") else None)

def plot_roc_all(models_dict, X_test, y_test):
    plt.figure(figsize=(7,6))
    for name, model in models_dict.items():
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:,1]
        else:
            y_proba = model.decision_function(X_test)
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc_val = roc_auc_score(y_test, y_proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.3f})")
    plt.plot([0,1],[0,1],'--', label='Random')
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title('ROC Curves — Comparison')
    plt.legend(loc='lower right')
    st.pyplot(plt.gcf())
    plt.close()

def plot_confusion(cm, title):
    plt.figure(figsize=(4,3))
    plt.imshow(cm, interpolation='nearest')
    plt.title(title)
    plt.colorbar()
    plt.xticks([0,1], ['No','Yes']); plt.yticks([0,1], ['No','Yes'])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i,j], ha='center', va='center', color='white' if cm[i,j] > cm.max()/2 else 'black')
    st.pyplot(plt.gcf())
    plt.close()

def get_table_download_link_csv(df, filename="data_with_predictions.csv"):
    towrite = BytesIO()
    df.to_csv(towrite, index=False)
    towrite.seek(0)
    b64 = base64.b64encode(towrite.read()).decode()
    href = f"data:file/csv;base64,{b64}"
    return href

# ---------------------- Data load / Upload ----------------------
st.sidebar.header("Data")
uploaded = st.sidebar.file_uploader("Upload UniversalBank CSV (or use sample below)", type=['csv'])

use_sample = False
if uploaded is None:
    st.sidebar.write("No file uploaded. You can upload your file or use the sample if you add it to Streamlit app folder.")
else:
    df = pd.read_csv(uploaded)
    df = normalize_columns(df)

if 'df' not in globals():
    st.info("Please upload your `UniversalBank.csv` using the sidebar uploader to enable full functionality.")
    # provide minimal placeholder dataframe for layout, but disable model tabs until data loaded
    df = pd.DataFrame()

# ---------------------- Main UI: Tabs ----------------------
tabs = st.tabs(["Overview & Insights", "Model Training & Metrics", "Predict New Data"])

# --------- Tab 1: Overview & Insights ---------
with tabs[0]:
    st.header("Exploratory Insights — 5 Actionable Charts")
    if df.empty:
        st.warning("Upload dataset to view charts.")
    else:
        # Make sure target present
        if 'Personal Loan' not in df.columns:
            st.error("Uploaded dataset must contain 'Personal Loan' column (1=Yes, 0=No).")
        else:
            # Clean and ensure numeric types
            df['Personal Loan'] = df['Personal Loan'].astype(int)
            # Chart 1: Conversion rate by Age bins (actionable: target age groups)
            st.subheader("1) Conversion Rate by Age Group (age bins)")
            df['age_bin'] = pd.cut(df['Age'], bins=[20,30,40,50,60,80], right=False)
            cr_by_age = df.groupby('age_bin')['Personal Loan'].mean().reset_index()
            fig = plt.figure(); plt.bar(cr_by_age['age_bin'].astype(str), cr_by_age['Personal Loan'])
            plt.xticks(rotation=45); plt.ylabel('Conversion Rate')
            st.pyplot(fig)
            plt.close()

            # Chart 2: Income deciles vs conversion (actionable: income segments)
            st.subheader("2) Conversion Rate by Income Decile")
            df['income_decile'] = pd.qcut(df['Income'].rank(method='first'), 10, labels=False) + 1
            cr_by_inc = df.groupby('income_decile')['Personal Loan'].mean().reset_index()
            fig = plt.figure(); plt.plot(cr_by_inc['income_decile'], cr_by_inc['Personal Loan'], marker='o')
            plt.xlabel('Income Decile (1=lowest)'); plt.ylabel('Conversion Rate'); plt.grid(False)
            st.pyplot(fig); plt.close()

            # Chart 3: CCAvg vs Income scatter colored by Personal Loan (actionable: identify high-spend low-income prospects)
            st.subheader("3) CCAvg vs Income (highlighted by loan acceptance)")
            fig = plt.figure(figsize=(6,4))
            accepted = df[df['Personal Loan']==1]
            rejected = df[df['Personal Loan']==0]
            plt.scatter(rejected['Income'], rejected['CCAvg'], alpha=0.3, label='No')
            plt.scatter(accepted['Income'], accepted['CCAvg'], alpha=0.7, marker='x', label='Yes')
            plt.xlabel('Income ($000)'); plt.ylabel('CCAvg ($000)'); plt.legend()
            st.pyplot(fig); plt.close()

            # Chart 4: Feature interaction — Family size & Education pivot (actionable: family+education segmentation)
            st.subheader("4) Conversion rate by Family size and Education level (segmentation)")
            pivot = df.pivot_table(index='Family', columns='Education', values='Personal Loan', aggfunc='mean')
            fig = plt.figure(); plt.imshow(pivot.fillna(0), interpolation='nearest'); plt.colorbar()
            plt.title('Conversion rate matrix (Family x Education)')
            plt.xticks(range(len(pivot.columns)), pivot.columns); plt.yticks(range(len(pivot.index)), pivot.index)
            st.pyplot(fig); plt.close()

            # Chart 5: Account features adoption vs conversion (actionable: cross-sell opportunities)
            st.subheader("5) Account features adoption vs Conversion (Securities/CD/Online/CreditCard)")
            features = [c for c in ['Securities','CDAccount','Online','CreditCard'] if c in df.columns]
            adoption = df[features].mean().reset_index()
            adoption.columns = ['feature','proportion']
            conversion = {f: df[df[f]==1]['Personal Loan'].mean() if f in df.columns else 0 for f in features}
            fig = plt.figure(); plt.bar(features, [conversion[f] for f in features])
            plt.ylabel('Conversion rate among users with the feature')
            st.pyplot(fig); plt.close()

            st.markdown("**Actionable suggestions (quick):**")
            st.write("- Target customers in age/income segments with higher conversion rate with tailored offers.")
            st.write("- Prioritize cross-sell to customers with high CCAvg but moderate income.")
            st.write("- Create messaging bundles for specific Family+Education segments.")

# --------- Tab 2: Model Training & Metrics ---------
with tabs[1]:
    st.header("Train models (Decision Tree, Random Forest, Gradient Boosting)")

    if df.empty or 'Personal Loan' not in df.columns:
        st.warning("Upload data with 'Personal Loan' column to enable training.")
    else:
        st.write("We'll use the standard features (Age, Experience, Income, Family, CCAvg, Education, Mortgage, Securities, CDAccount, Online, CreditCard).")
        if 'trained_models' not in st.session_state:
            st.session_state['trained_models'] = None
            st.session_state['model_cols'] = None
            st.session_state['metrics_table'] = None

        col1, col2 = st.columns([1,3])
        with col1:
            cv_splits = st.number_input("CV folds (Stratified)", min_value=3, max_value=10, value=5)
            test_size = st.slider("Test set proportion", min_value=0.1, max_value=0.5, value=0.3)
            run_button = st.button("Train all 3 models and generate metrics")
        with col2:
            st.write("After training you'll see: metrics table, ROC overlay, confusion matrices (train/test), feature importances.")

        if run_button:
            # normalize columns and prepare data
            df2 = normalize_columns(df.copy())
            X, y, model_cols = prepare_model_data(df2)
            st.session_state['model_cols'] = model_cols
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)

            models = {
                'Decision Tree': DecisionTreeClassifier(random_state=42),
                'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
                'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
            }
            metrics_rows = []
            trained = {}
            proba_store = {}

            progress = st.progress(0)
            total = len(models)
            i = 0
            for name, model in models.items():
                st.write(f"Training: {name} ...")
                metrics, fitted, y_test_proba = compute_metrics_and_cv(model, X_train, y_train, X_test, y_test, cv_splits=cv_splits)
                metrics['Algorithm'] = name
                metrics_rows.append(metrics)
                trained[name] = fitted
                if y_test_proba is not None:
                    proba_store[name] = y_test_proba
                i += 1
                progress.progress(int(100 * i / total))

            metrics_df = pd.DataFrame(metrics_rows).set_index('Algorithm')
            st.session_state['trained_models'] = trained
            st.session_state['metrics_table'] = metrics_df

            st.success("Training complete. Metrics table below:")
            st.dataframe(metrics_df.style.format("{:.4f}"))

            # ROC plot
            st.subheader("ROC curves (all models)")
            plot_roc_all(trained, X_test, y_test)

            # Confusion matrices (train & test)
            st.subheader("Confusion Matrices (Train & Test)")
            for split_name, Xs, ys in [('Train', X_train, y_train), ('Test', X_test, y_test)]:
                st.write(f"**{split_name} set**")
                cols = st.columns(len(trained))
                for idx, (name, model) in enumerate(trained.items()):
                    cm = confusion_matrix(ys, model.predict(Xs))
                    with cols[idx]:
                        st.write(name)
                        plot_confusion(cm, f"{name} - {split_name}")

            # Feature importances
            st.subheader("Feature importances")
            for name, model in trained.items():
                if hasattr(model, "feature_importances_"):
                    imp = model.feature_importances_
                    inds = np.argsort(imp)[::-1]
                    feat_names = [model_cols[i] for i in inds]
                    fig = plt.figure(figsize=(6,3))
                    plt.bar(feat_names, imp[inds])
                    plt.xticks(rotation=45, ha='right')
                    plt.title(f"Feature importances - {name}")
                    st.pyplot(fig)
                    plt.close()
            st.info("Models saved in session. Use 'Predict New Data' tab to upload new dataset and get predictions.")

        # show previously trained metrics if available
        if st.session_state.get('metrics_table') is not None and not run_button:
            st.write("Previously trained metrics (session):")
            st.dataframe(st.session_state['metrics_table'].style.format("{:.4f}"))

# --------- Tab 3: Predict New Data ---------
with tabs[2]:
    st.header("Upload new data to predict 'Personal Loan' label and download results")
    st.write("Upload a CSV with the same feature columns as the training data (Age, Experience, Income, Family, CCAvg, Education, Mortgage, Securities, CDAccount, Online, CreditCard).")
    uploaded_new = st.file_uploader("Upload new CSV for prediction", type=['csv'], key="pred_uploader")
    if uploaded_new is not None:
        df_new = pd.read_csv(uploaded_new)
        df_new = normalize_columns(df_new)
        if st.session_state.get('trained_models') is None:
            st.warning("No trained models found in session. Please train models first in 'Model Training & Metrics' tab.")
        else:
            # use Random Forest by default for predictions (you can change)
            model_choice = st.selectbox("Choose model for prediction", options=list(st.session_state['trained_models'].keys()))
            chosen_model = st.session_state['trained_models'][model_choice]
            model_cols = st.session_state.get('model_cols', None)
            if model_cols is None:
                st.error("Model feature list not found. Re-train models to populate feature list.")
            else:
                # ensure columns present
                missing = [c for c in model_cols if c not in df_new.columns]
                if missing:
                    st.error(f"The uploaded file is missing these required columns: {missing}")
                else:
                    X_new = df_new[model_cols].fillna(df_new[model_cols].median())
                    preds = chosen_model.predict(X_new)
                    proba = chosen_model.predict_proba(X_new)[:,1] if hasattr(chosen_model, "predict_proba") else None
                    df_new_out = df_new.copy()
                    df_new_out['Predicted Personal Loan'] = preds
                    if proba is not None:
                        df_new_out['Prediction Probability'] = proba
                    st.write(df_new_out.head(50))
                    csv_link = get_table_download_link_csv(df_new_out)
                    st.markdown(f"[Download predictions CSV]({csv_link})", unsafe_allow_html=True)
