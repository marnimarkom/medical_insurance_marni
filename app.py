"""Medical insurance teaching dashboard. Run: streamlit run app.py.

Matches AI_Regression_Medical_Insurance.py: drop ID and Charges, then
pd.get_dummies(drop_first=True, dtype=int); no scaling or refitting.
"""
from datetime import datetime, timezone, timedelta
from pathlib import Path
import math

import joblib
import pandas as pd
import streamlit as st
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

# Add ONLY verified original test scores, with their source noted here.
TEST_SCORES = {"R²": None, "RMSE": None, "MAE": None}
CURRENCY = "dataset currency units"  # Exact currency is not established.
RAW_COLUMNS = ["Age", "Sex", "BMI", "Children", "Smoker", "Region"]
CATEGORIES = {
    "Sex": ["Female", "Male"],
    "Smoker": ["No", "Yes"],
    "Region": ["Northeast", "Northwest", "Southeast", "Southwest"],
}
DISCLAIMER = (
    "This dashboard is developed for teaching and learning. Predictions are "
    "estimates from a machine-learning model and are not official insurance "
    "quotations, medical advice or guarantees of actual charges."
)


def resolve_file(name):
    """Prefer files beside app.py; /content is a Colab fallback."""
    for folder in (Path(__file__).resolve().parent, Path("/content")):
        path = folder / name
        if path.is_file():
            return path
    return None


@st.cache_resource
def load_model(path, modified_ns):
    # modified_ns invalidates the cache when the file is replaced.
    model = joblib.load(path)
    if not callable(getattr(model, "predict", None)):
        raise ValueError("The saved object is not a prediction model.")
    check_is_fitted(model)
    return model


@st.cache_data
def load_dataset(path, modified_ns):
    return pd.read_excel(path)


def expected_columns(model):
    names = getattr(model, "feature_names_in_", None)
    if names is None:
        raise ValueError(
            "The model has no feature_names_in_. Supply the saved feature "
            "order or re-export the model from the supplied DataFrame-based "
            "training script; the dashboard will not guess feature order."
        )
    return list(names)


def prepare_input(raw, model):
    """Use documented mappings; align to the model's exact spelling/order."""
    names = expected_columns(model)
    if isinstance(model, Pipeline):
        # The supplied trainer saves an estimator, not a Pipeline. A different
        # Pipeline needs its preprocessing confirmed before it can be used.
        raise ValueError(
            "This artifact is a Pipeline, unlike the supplied training script. "
            "Provide its training code to confirm its raw-input contract."
        )
    encoded = pd.DataFrame({
        "age": raw["Age"].astype(int),
        "bmi": raw["BMI"].astype(float),
        "children": raw["Children"].astype(int),
        "sex_male": (raw["Sex"] == "Male").astype(int),
        "smoker_yes": (raw["Smoker"] == "Yes").astype(int),
        "region_northwest": (raw["Region"] == "Northwest").astype(int),
        "region_southeast": (raw["Region"] == "Southeast").astype(int),
        "region_southwest": (raw["Region"] == "Southwest").astype(int),
    })
    keys = [str(name).casefold() for name in names]
    if len(keys) != len(set(keys)) or set(keys) != set(encoded.columns):
        raise ValueError(
            f"Model features do not match the supplied training description: {names}. "
            "Check for numeric category codes, extra columns, or different labels."
        )
    result = encoded.loc[:, keys].copy()
    result.columns = names
    if result.isna().any().any():
        raise ValueError("Model inputs contain missing values.")
    return result


def validate_dataset_encoding(data, model):
    """Check actual dummy names/reference levels, without fitting anything."""
    features = data.drop(columns=["ID", "Charges"], errors="ignore")
    for column, levels in CATEGORIES.items():
        if column not in features:
            raise ValueError(f"Dataset is missing {column}.")
        if not pd.api.types.is_object_dtype(features[column].dtype) and not isinstance(
            features[column].dtype, (pd.StringDtype, pd.CategoricalDtype)
        ):
            raise ValueError(f"{column} is not categorical text as documented.")
        values = features[column].dropna().unique().tolist()
        if {str(v).casefold() for v in values} != {v.casefold() for v in levels}:
            raise ValueError(f"Unexpected or incomplete categories in {column}.")
        # Verify that pandas dropped the documented reference, including case.
        dummies = pd.get_dummies(features[[column]], drop_first=True, dtype=int)
        wanted = {f"{column}_{v}".casefold() for v in levels[1:]}
        if {str(c).casefold() for c in dummies.columns} != wanted:
            raise ValueError(f"The reference category for {column} differs from the documentation.")
    actual = list(pd.get_dummies(features, drop_first=True, dtype=int).columns)
    if actual != expected_columns(model):
        raise ValueError("Dataset dummy columns/order differ from the saved model.")


def predict_records(raw, model):
    prepared = prepare_input(raw, model)
    predictions = [float(value) for value in model.predict(prepared)]
    if len(predictions) != len(raw):
        raise ValueError("The model did not return one estimate per applicant.")
    if any(not math.isfinite(v) or v < 0 for v in predictions):
        raise ValueError("The model returned an invalid cost estimate (negative or non-finite).")
    return predictions, prepared


def cost_thresholds(data):
    if data is None or "Charges" not in data:
        return None
    values = pd.to_numeric(data["Charges"], errors="coerce")
    if values.isna().any() or values.empty or not values.map(math.isfinite).all() or (values < 0).any():
        return None
    return values.quantile([0.25, 0.50, 0.75]).tolist()


def cost_category(value, thresholds):
    if thresholds is None:
        return "Unavailable"
    for limit, label in zip(thresholds, ["Low", "Moderate", "High"]):
        if value <= limit:
            return label
    return "Very High"


def main():
    st.set_page_config(page_title="Medical Insurance Cost Prediction", page_icon="🏥", layout="wide")
    st.markdown("""<style>
    .stApp {background:#f5f9fb; color:#142c45;}
    h1,h2,h3 {color:#123451;}
    div[data-testid="stMetric"] {background:#fff; padding:1rem;
        border:1px solid #dce9ee; border-radius:14px;}
    div[data-testid="stForm"] {background:white; border-radius:16px; padding:1.3rem;}
    .stButton>button, .stFormSubmitButton>button {background:#087e83;
        color:white; border-radius:10px; min-height:3rem;}
    .result {background:#123451; color:white; padding:1.7rem;
        border-radius:16px; margin:1rem 0;}
    .result .amount {font-size:clamp(2rem,5vw,3rem); font-weight:700;}
    </style>""", unsafe_allow_html=True)
    st.title("🏥 Medical Insurance Cost Prediction")
    st.write("AI-Based Regression Dashboard for Estimating Medical Insurance Charges")
    st.caption("Enter the applicant’s details and select Predict Insurance Cost.")

    path = resolve_file("medical_insurance_model.pkl")
    if path is None:
        st.error("Missing medical_insurance_model.pkl. Place it beside app.py or in /content/.")
        st.stop()
    try:
        stamp = path.stat().st_mtime_ns
        model = load_model(str(path), stamp)
        # Validate schema before accepting a prediction request.
        example = pd.DataFrame([[30, "Female", 25.0, 0, "No", "Northeast"]], columns=RAW_COLUMNS)
        prepare_input(example, model)
    except ModuleNotFoundError as error:
        st.error(f"Model dependency missing: {error.name}. If the best model is XGBoost, add xgboost to requirements.txt.")
        st.stop()
    except Exception as error:
        st.error(f"Cannot use the saved model: {error}")
        st.caption("Use the same library versions as the training environment.")
        st.stop()

    data = None
    dataset_path = resolve_file("medical_insurance_dataset.xlsx")
    if dataset_path:
        try:
            data = load_dataset(str(dataset_path), dataset_path.stat().st_mtime_ns)
        except Exception as error:
            st.warning(f"Dataset could not be read: {error}. Dataset-dependent features are disabled.")
    else:
        st.info("Dataset unavailable. Prediction remains available; dataset guidance and cost categories are disabled.")
    if data is not None:
        try:
            validate_dataset_encoding(data, model)
        except Exception as error:
            st.error(f"Dataset/model preprocessing mismatch: {error}")
            st.stop()
    thresholds = cost_thresholds(data)
    model_name = type(model).__name__
    artifact_key = (str(path), stamp, path.stat().st_size)
    if st.session_state.get("artifact_key") != artifact_key:
        st.session_state.pop("insurance_result", None)
        st.session_state["artifact_key"] = artifact_key

    with st.sidebar:
        st.header("About the System")
        st.write("Estimate insurance charges from six applicant details using a saved regression model.")
        st.caption("Regression predicts a numerical value from input features.")
        st.subheader("Model performance")
        st.write(f"Model: {model_name}")
        for label, value in TEST_SCORES.items():
            st.write(f"{label}: {'Not provided' if value is None else f'{value:,.4f}'}")
        st.caption("R² describes model fit and can be negative; it is not accuracy. RMSE and MAE measure prediction error in target units; lower is better on the same evaluation data.")
        st.subheader("Model inputs")
        st.write("Age · BMI · Children · Sex · Smoking status · Region")
        st.caption(DISCLAIMER)

    if data is not None:
        ranges = []
        for column in ("Age", "BMI"):
            if column in data:
                values = pd.to_numeric(data[column], errors="coerce").dropna()
                values = values[values.map(math.isfinite)]
                if not values.empty:
                    ranges.append(f"{column}: {values.min():g}–{values.max():g}")
        if ranges:
            st.caption("Observed dataset ranges — " + "; ".join(ranges) + ". These are not the interface limits or a verified training-only range.")
    st.caption("Interface limits: age 0–120; BMI 1–100; children 0–30. Values outside the observed dataset range may produce unreliable estimates.")
    with st.form("applicant"):
        st.subheader("Applicant information")
        a, b, c = st.columns(3)
        age = a.number_input("Age", min_value=0, max_value=120, value=30, step=1)
        sex = a.selectbox("Sex", CATEGORIES["Sex"])
        bmi = b.number_input("BMI", min_value=1.0, max_value=100.0, value=25.0, step=0.1)
        smoker = b.selectbox("Smoking status", CATEGORIES["Smoker"])
        children = c.number_input("Number of children", min_value=0, max_value=30, value=0, step=1)
        region = c.selectbox("Region", CATEGORIES["Region"])
        submitted = st.form_submit_button("Predict Insurance Cost")

    if submitted:
        st.session_state.pop("insurance_result", None)
        raw = pd.DataFrame([[age, sex, bmi, children, smoker, region]], columns=RAW_COLUMNS)
        try:
            values, encoded = predict_records(raw, model)
            comparison = raw.loc[raw.index.repeat(2)].reset_index(drop=True)
            comparison["Smoker"] = ["No", "Yes"]
            try:
                pair, _ = predict_records(comparison, model)
                comparison_error = None
            except Exception as error:
                pair, comparison_error = None, str(error)
            st.session_state["insurance_result"] = {
                "raw": raw, "encoded": encoded, "cost": values[0], "pair": pair,
                "comparison_error": comparison_error,
                "time": datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S UTC+08:00"),
            }
        except Exception as error:
            st.error(f"Prediction failed: {error}")

    result = st.session_state.get("insurance_result")
    if result:
        value = result["cost"]
        st.markdown(f'<div class="result">Estimated Medical Insurance Cost<div class="amount">{value:,.2f}</div>{CURRENCY}<br>Estimate generated using the trained regression model.</div>', unsafe_allow_html=True)
        a, b, c = st.columns(3)
        a.metric("Cost category", cost_category(value, thresholds))
        b.metric("BMI entered", f'{result["raw"].iloc[0]["BMI"]:.1f}')
        c.metric("Model used", model_name)
        st.subheader("Applicant summary")
        summary = result["raw"].iloc[0].to_dict()
        summary.update({"Estimated cost": f"{value:,.2f} {CURRENCY}", "Prediction timestamp": result["time"]})
        st.table(pd.DataFrame({"Item": list(summary), "Value": [str(v) for v in summary.values()]}))
        st.subheader("Smoker versus non-smoker comparison")
        if result["pair"] is None:
            st.warning("Comparison unavailable: " + result["comparison_error"])
        else:
            no, yes = result["pair"]
            a, b, c = st.columns(3)
            a.metric("Non-smoker estimate", f"{no:,.2f}")
            b.metric("Smoker estimate", f"{yes:,.2f}")
            c.metric("Difference (smoker − non-smoker)", f"{yes - no:,.2f}")
            st.caption(f"Values in {CURRENCY}.")
            if no > 0:
                st.write(f"Percentage difference: {(yes - no) / no * 100:,.1f}%")
            st.bar_chart(pd.DataFrame({"Estimated cost": [no, yes]}, index=["Non-smoker", "Smoker"]))
        st.caption("This comparison shows how the trained model’s estimate changes when smoking status changes while the other entered values remain fixed. It does not establish a causal effect or provide an insurer quotation.")

    with st.expander("How the inputs are prepared"):
        st.write("Confirmed training method: pd.get_dummies(drop_first=True, dtype=int). No scaling. ID and Charges are excluded from inputs. The saved model determines feature order.")
        st.table(pd.DataFrame({
            "Input": ["Sex", "Smoker", "Region"],
            "Encoding": ["Female = 0; Male = 1", "No = 0; Yes = 1", "Northeast = (0,0,0); Northwest = (1,0,0); Southeast = (0,1,0); Southwest = (0,0,1)"],
        }))
        st.caption("Region tuple order: Northwest, Southeast, Southwest. Actual model-column spelling is preserved.")
        if result:
            st.write("Raw submitted inputs")
            st.dataframe(result["raw"])
            st.write("Actual encoded model inputs, in prediction order")
            st.dataframe(result["encoded"])
    with st.expander("How cost categories are defined"):
        if thresholds is None:
            st.write("Unavailable: a dataset with a valid, non-negative Charges column is required.")
        else:
            q1, q2, q3 = thresholds
            st.write(f"Low: ≤ {q1:,.2f}; Moderate: > {q1:,.2f} to {q2:,.2f}; High: > {q2:,.2f} to {q3:,.2f}; Very High: > {q3:,.2f} ({CURRENCY}).")
        st.caption("These are dataset-relative cost groups, not official insurer pricing bands.")
    st.info(DISCLAIMER)
    st.divider()
    st.markdown("**Developed by**  \nIr. Dr. Marni Azira Markom  \nFaculty of Intelligent Computing  \nUniversiti Malaysia Perlis (UniMAP)")


if __name__ == "__main__":
    main()
