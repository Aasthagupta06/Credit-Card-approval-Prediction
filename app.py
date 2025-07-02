import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, OrdinalEncoder
from imblearn.over_sampling import SMOTE
import joblib
import streamlit as st
import requests
from streamlit_lottie import st_lottie_spinner
import base64

# --- Animated Heading and Definition ---
st.markdown(
    """
<style>
@keyframes colorPulse {
    0% { color: #1976D2; }
    50% { color: #64b5f6; }
    100% { color: #1976D2; }
}
@keyframes fadeInOut {
    0% { opacity: 1; }
    50% { opacity: 0.6; }
    100% { opacity: 1; }
}
.animated-heading {
    animation: colorPulse 2s infinite, fadeInOut 2.5s infinite;
    font-size: 2.8em;
    font-weight: bold;
    margin-bottom: 0.05em;
    margin-top: -1.2em;
}
.model-def {
    color: #e3f2fd;
    font-size: 1.35em;
    margin-bottom: 1.2em;
}
.main .block-container {
    padding-top: 0.5rem;
}
</style>
<div style='text-align: center;'>
    <div class='animated-heading'>💳 Credit Card Approval Prediction</div>
    <div class='model-def'>Discover your credit card approval chances in seconds!</div>
</div>
    """,
    unsafe_allow_html=True
)
# Add a little vertical space after the heading/definition
st.markdown("<div style='margin-bottom: 1.2em;'></div>", unsafe_allow_html=True)

# --- Sidebar: Improved app info and credits ---
st.sidebar.markdown("""
# 💳 Credit Card Approval App

Instantly check your credit card approval chances!

---

**Model:** Gradient Boosting Classifier

**Dataset:** Credit Card Fraud Detection (Kaggle)

---

<div style='margin-top:2.5em; text-align:center;'>
    <button style='background-color:#1976D2; color:#fff; font-weight:bold; font-size:1.05em; border:none; border-radius:7px; padding:0.7em 1.2em; cursor:pointer;'>
        Made by Aastha Gupta
    </button>
</div>
""", unsafe_allow_html=True)

# Load data
train_original = pd.read_csv("dataset/train.csv")
test_original = pd.read_csv("dataset/test.csv")
full_data = pd.concat([train_original, test_original], axis=0)
full_data = full_data.sample(frac=1).reset_index(drop=True)

def data_split(df, test_size):
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

train_original, test_original = data_split(full_data, 0.2)
train_copy = train_original.copy()

def value_cnt_norm_cal(df, feature):
    ftr_value_cnt = df[feature].value_counts()
    ftr_value_cnt_norm = df[feature].value_counts(normalize=True) * 100
    ftr_value_cnt_concat = pd.concat([ftr_value_cnt, ftr_value_cnt_norm], axis=1)
    ftr_value_cnt_concat.columns = ["Count", "Frequency (%)"]
    return ftr_value_cnt_concat

# --- Custom Transformers ---
class OutlierRemover(BaseEstimator, TransformerMixin):
    def __init__(self, feat_with_outliers=["Family member count", "Income", "Employment length"]):
        self.feat_with_outliers = feat_with_outliers
    def fit(self, df):
        return self
    def transform(self, df):
        if set(self.feat_with_outliers).issubset(df.columns):
            Q1 = df[self.feat_with_outliers].quantile(0.25)
            Q3 = df[self.feat_with_outliers].quantile(0.75)
            IQR = Q3 - Q1
            df = df[
                ~(
                    (df[self.feat_with_outliers] < (Q1 - 3 * IQR))
                    | (df[self.feat_with_outliers] > (Q3 + 3 * IQR))
                ).any(axis=1)
            ]
            return df
        else:
            print("One or more features are not in the dataframe")
            return df

class DropFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, feature_to_drop=["Has a mobile phone", "Children count", "Job title", "Account age"]):
        self.feature_to_drop = feature_to_drop
    def fit(self, df):
        return self
    def transform(self, df):
        if set(self.feature_to_drop).issubset(df.columns):
            df.drop(self.feature_to_drop, axis=1, inplace=True)
            return df
        else:
            print("One or more features are not in the dataframe")
            return df

class TimeConversionHandler(BaseEstimator, TransformerMixin):
    def __init__(self, feat_with_days=["Employment length", "Age"]):
        self.feat_with_days = feat_with_days
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        if set(self.feat_with_days).issubset(X.columns):
            X[["Employment length", "Age"]] = np.abs(X[["Employment length", "Age"]])
            return X
        else:
            print("One or more features are not in the dataframe")
            return X

class RetireeHandler(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, df):
        return self
    def transform(self, df):
        if "Employment length" in df.columns:
            df_ret_idx = df["Employment length"][df["Employment length"] == 365243].index
            df.loc[df_ret_idx, "Employment length"] = 0
            return df
        else:
            print("Employment length is not in the dataframe")
            return df

class SkewnessHandler(BaseEstimator, TransformerMixin):
    def __init__(self, feat_with_skewness=["Income", "Age"]):
        self.feat_with_skewness = feat_with_skewness
    def fit(self, df):
        return self
    def transform(self, df):
        if set(self.feat_with_skewness).issubset(df.columns):
            df[self.feat_with_skewness] = np.cbrt(df[self.feat_with_skewness])
            return df
        else:
            print("One or more features are not in the dataframe")
            return df

class BinningNumToYN(BaseEstimator, TransformerMixin):
    def __init__(self, feat_with_num_enc=["Has a work phone", "Has a phone", "Has an email"]):
        self.feat_with_num_enc = feat_with_num_enc
    def fit(self, df):
        return self
    def transform(self, df):
        if set(self.feat_with_num_enc).issubset(df.columns):
            for ft in self.feat_with_num_enc:
                df[ft] = df[ft].map({1: "Y", 0: "N"})
            return df
        else:
            print("One or more features are not in the dataframe")
            return df

class OneHotWithFeatNames(BaseEstimator, TransformerMixin):
    def __init__(self, one_hot_enc_ft=[
        "Gender",
        "Marital status",
        "Dwelling",
        "Employment status",
        "Has a car",
        "Has a property",
        "Has a work phone",
        "Has a phone",
        "Has an email",
    ]):
        self.one_hot_enc_ft = one_hot_enc_ft
    def fit(self, df):
        return self
    def transform(self, df):
        if set(self.one_hot_enc_ft).issubset(df.columns):
            def one_hot_enc(df, one_hot_enc_ft):
                one_hot_enc = OneHotEncoder()
                one_hot_enc.fit(df[one_hot_enc_ft])
                feat_names_one_hot_enc = one_hot_enc.get_feature_names_out(one_hot_enc_ft)
                df = pd.DataFrame(
                    one_hot_enc.transform(df[self.one_hot_enc_ft]).toarray(),
                    columns=feat_names_one_hot_enc,
                    index=df.index,
                )
                return df
            def concat_with_rest(df, one_hot_enc_df, one_hot_enc_ft):
                rest_of_features = [ft for ft in df.columns if ft not in one_hot_enc_ft]
                df_concat = pd.concat([one_hot_enc_df, df[rest_of_features]], axis=1)
                return df_concat
            one_hot_enc_df = one_hot_enc(df, self.one_hot_enc_ft)
            full_df_one_hot_enc = concat_with_rest(df, one_hot_enc_df, self.one_hot_enc_ft)
            return full_df_one_hot_enc
        else:
            print("One or more features are not in the dataframe")
            return df

class OrdinalFeatNames(BaseEstimator, TransformerMixin):
    def __init__(self, ordinal_enc_ft=["Education level"]):
        self.ordinal_enc_ft = ordinal_enc_ft
    def fit(self, df):
        return self
    def transform(self, df):
        if "Education level" in df.columns:
            ordinal_enc = OrdinalEncoder()
            df[self.ordinal_enc_ft] = ordinal_enc.fit_transform(df[self.ordinal_enc_ft])
            return df
        else:
            print("Education level is not in the dataframe")
            return df

class MinMaxWithFeatNames(BaseEstimator, TransformerMixin):
    def __init__(self, min_max_scaler_ft=["Age", "Income", "Employment length"]):
        self.min_max_scaler_ft = min_max_scaler_ft
    def fit(self, df):
        return self
    def transform(self, df):
        if set(self.min_max_scaler_ft).issubset(df.columns):
            min_max_enc = MinMaxScaler()
            df[self.min_max_scaler_ft] = min_max_enc.fit_transform(df[self.min_max_scaler_ft])
            return df
        else:
            print("One or more features are not in the dataframe")
            return df

class ChangeToNumTarget(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, df):
        return self
    def transform(self, df):
        if "Is high risk" in df.columns:
            df["Is high risk"] = pd.to_numeric(df["Is high risk"])
            return df
        else:
            print("Is high risk is not in the dataframe")
            return df

class OversampleSMOTE(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, df):
        return self
    def transform(self, df):
        if "Is high risk" in df.columns:
            smote = SMOTE()
            X_bal, y_bal = smote.fit_resample(df.iloc[:, :-1], df.iloc[:, -1])
            X_y_bal = pd.concat([pd.DataFrame(X_bal), pd.DataFrame(y_bal)], axis=1)
            return X_y_bal
        else:
            print("Is high risk is not in the dataframe")
            return df

def full_pipeline(df):
    pipeline = Pipeline(
        [
            ("outlier_remover", OutlierRemover()),
            ("feature_dropper", DropFeatures()),
            ("time_conversion_handler", TimeConversionHandler()),
            ("retiree_handler", RetireeHandler()),
            ("skewness_handler", SkewnessHandler()),
            ("binning_num_to_yn", BinningNumToYN()),
            ("one_hot_with_feat_names", OneHotWithFeatNames()),
            ("ordinal_feat_names", OrdinalFeatNames()),
            ("min_max_with_feat_names", MinMaxWithFeatNames()),
            ("change_to_num_target", ChangeToNumTarget()),
            ("oversample_smote", OversampleSMOTE()),
        ]
    )
    df_pipe_prep = pipeline.fit_transform(df)
    return df_pipe_prep

# --- Prepare selectbox/radio options and mappings before UI ---
marital_status_values = list(value_cnt_norm_cal(full_data, "Marital status").index)
marital_status_key = [
    "Married",
    "Single/not married",
    "Civil marriage",
    "Separated",
    "Widowed",
]
marital_status_dict = dict(zip(marital_status_key, marital_status_values))

dwelling_type_values = list(value_cnt_norm_cal(full_data, "Dwelling").index)
dwelling_type_key = [
    "House / apartment",
    "Live with parents",
    "Municipal apartment ",
    "Rented apartment",
    "Office apartment",
    "Co-op apartment",
]
dwelling_type_dict = dict(zip(dwelling_type_key, dwelling_type_values))

employment_status_values = list(value_cnt_norm_cal(full_data, "Employment status").index)
employment_status_key = [
    "Working",
    "Commercial associate",
    "Pensioner",
    "State servant",
    "Student",
]
employment_status_dict = dict(zip(employment_status_key, employment_status_values))

edu_level_values = list(value_cnt_norm_cal(full_data, "Education level").index)
edu_level_key = [
    "Secondary school",
    "Higher education",
    "Incomplete higher",
    "Lower secondary",
    "Academic degree",
]
edu_level_dict = dict(zip(edu_level_key, edu_level_values))

# --- Streamlit UI ---

# Use two main columns with a spacer for a balanced layout
left_col, spacer, right_col = st.columns([1, 0.1, 1])

with left_col:
    input_gender = st.radio(
        "Select your gender",
        ["Male", "Female"],
        index=0,
        help="Select your gender as per your official documents."
    )
    input_marital_status_key = st.selectbox(
        "Select your marital status",
        marital_status_key,
        help="Choose your current marital status."
    )
    input_marital_status_val = marital_status_dict.get(input_marital_status_key)
    input_dwelling_type_key = st.selectbox(
        "Select the type of dwelling you reside in",
        dwelling_type_key,
        help="Type of residence you currently live in."
    )
    input_dwelling_type_val = dwelling_type_dict.get(input_dwelling_type_key)
    input_employment_status_key = st.selectbox(
        "Select your employment status",
        employment_status_key,
        help="Your current employment type."
    )
    input_employment_status_val = employment_status_dict.get(input_employment_status_key)
    input_edu_level_key = st.selectbox(
        "Select your education status",
        edu_level_key,
        help="Your highest completed education level."
    )
    input_edu_level_val = edu_level_dict.get(input_edu_level_key)
    input_prop_ownship = st.radio(
        "Do you own a property?",
        ["Yes", "No"],
        index=0,
        help="Select 'Yes' if you own any property (house, land, etc.)."
    )
    input_email = st.radio(
        "Do you have an email?",
        ["Yes", "No"],
        index=0,
        help="Select 'Yes' if you have a personal email address."
    )
    email_dict = {"Yes": 1, "No": 0}
    email_val = email_dict.get(input_email)

with right_col:
    input_age = np.negative(
        st.slider(
            "Select your age",
            value=42,
            min_value=18,
            max_value=70,
            step=1,
            help="Enter your current age in years."
        ) * 365.25
    )
    fam_member_count = float(
        st.selectbox(
            "Select your family member count",
            [1, 2, 3, 4, 5, 6],
            help="Total number of family members living with you."
        )
    )
    input_income = int(
        st.text_input(
            "Enter your income (in USD)",
            0,
            help="Enter your monthly income in US Dollars."
        )
    )
    input_employment_length = np.negative(
        st.slider(
            "Select your employment length",
            value=6,
            min_value=0,
            max_value=30,
            step=1,
            help="Number of years you have been employed at your current job."
        ) * 365.25
    )
    input_car_ownship = st.radio(
        "Do you own a car?",
        ["Yes", "No"],
        index=0,
        help="Select 'Yes' if you own a car."
    )
    input_phone = st.radio(
        "Do you have a phone?",
        ["Yes", "No"],
        index=0,
        help="Select 'Yes' if you have a personal phone."
    )
    work_dict = {"Yes": 1, "No": 0}
    phone_val = work_dict.get(input_phone)
    input_work_phone = st.radio(
        "Do you have a work phone?",
        ["Yes", "No"],
        index=0,
        help="Select 'Yes' if you have a phone provided by your employer."
    )
    work_phone_dict = {"Yes": 1, "No": 0}
    work_phone_val = work_phone_dict.get(input_work_phone)

predict_bt = st.button("Am I Approved?")

profile_to_predict = [
    0,  # ID
    input_gender[:1],  # gender
    input_car_ownship[:1],  # car ownership
    input_prop_ownship[:1],  # property ownership
    0,  # Children count (which will be dropped in the pipeline)
    input_income,  # Income
    input_employment_status_val,  # Employment status
    input_edu_level_val,  # Education level
    input_marital_status_val,  # Marital status
    input_dwelling_type_val,  # Dwelling type
    input_age,  # Age
    input_employment_length,  # Employment length
    1,  # Has a mobile phone (which will be dropped in the pipeline)
    work_phone_val,  # Work phone
    phone_val,  # Phone
    email_val,  # Email
    "to_be_droped",  # Job title (which will be dropped in the pipeline)
    fam_member_count,  # Family member count
    0.00,  # Account age (which will be dropped in the pipeline)
    0,  # target set to 0 as a placeholder
]

profile_to_predict_df = pd.DataFrame([profile_to_predict], columns=train_copy.columns)
train_copy_with_profile_to_pred = pd.concat(
    [train_copy, profile_to_predict_df], ignore_index=True
)
train_copy_with_profile_to_pred_prep = full_pipeline(train_copy_with_profile_to_pred)
profile_to_pred_prep = train_copy_with_profile_to_pred_prep[
    train_copy_with_profile_to_pred_prep["ID"] == 0
].drop(columns=["ID", "Is high risk"])

@st.cache_data
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_loading_an = load_lottieurl(
    "https://assets3.lottiefiles.com/packages/lf20_szlepvdh.json"
)

def make_prediction():
    try:
        model = joblib.load('final_model/gradient_boosting_model.sav')
        return model.predict(profile_to_pred_prep)
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return None

if predict_bt:
    with st_lottie_spinner(
        lottie_loading_an, quality="high", height="200px", width="200px"
    ):
        final_pred = make_prediction()
    if final_pred is not None:
        if final_pred[0] == 0:
            st.success("## You have been approved for a credit card")
            st.balloons()
        elif final_pred[0] == 1:
            st.error("## Unfortunately, you have not been approved for a credit card")
        # Show summary table after result message
        summary = {
            "Gender": input_gender,
            "Age": int(abs(input_age // 365.25)),
            "Marital Status": input_marital_status_key,
            "Family Members": int(fam_member_count),
            "Dwelling": input_dwelling_type_key,
            "Income (USD)": input_income,
            "Employment Status": input_employment_status_key,
            "Employment Length (years)": int(abs(input_employment_length // 365.25)),
            "Education Level": input_edu_level_key,
            "Owns Car": input_car_ownship,
            "Owns Property": input_prop_ownship,
            "Work Phone": input_work_phone,
            "Phone": input_phone,
            "Email": input_email,
        }
        st.markdown('### 📝 Your Input Summary')
        st.write(pd.DataFrame(summary.items(), columns=["Field", "Value"]))
    else:
        st.error(
            "Unable to make a prediction due to an error. Please check the logs and try again."
        )