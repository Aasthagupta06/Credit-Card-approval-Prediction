# Credit Card Approval Prediction

A machine learning web app to predict whether a credit card application will be approved, built with Streamlit. The app uses a Gradient Boosting model trained on real-world data and provides an interactive UI for users to input applicant details and get instant predictions.

---

## 🚀 Features
- Predicts credit card approval based on user input
- Interactive Streamlit web interface
- Data preprocessing pipeline with outlier removal, encoding, scaling, and SMOTE balancing
- Loads model and data locally (no cloud dependencies)
- Ready for deployment on GitHub

---

## 📁 Project Structure
```
Credit_card_Approval/
├── dataset/
│   ├── application_record.csv
│   ├── credit_record.csv
│   ├── test.csv
│   └── train.csv
├── final_model/
│   └── gradient_boosting_model.sav
├── notebook/
│   └── Credit_Card_Approval.ipynb
├── pandas_profile_file/
│   └── income_class_profile.html
├── app.py
├── requirements.txt
└── README.md
```

---

## 🛠️ Setup Instructions

### 1. Clone the Repository
```sh
git clone <your-repo-url>
cd Credit_card_Approval
```

### 2. Create and Activate a Virtual Environment
```sh
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```sh
pip install -r requirements.txt
```

### 4. Ensure Data and Model Files Are Present
- Place your data files in the `dataset/` folder (`train.csv`, `test.csv`, etc.)
- Place your trained model in `final_model/gradient_boosting_model.sav`

---

## ▶️ Running the Streamlit App
```sh
streamlit run app.py
```
- The app will open in your browser at `http://localhost:8501`
- Fill in the form and click **Predict** to see the result

---

## 📒 Notebooks & Profiling
- The `notebook/` folder contains the original Jupyter notebook used for data exploration, feature engineering, and model training. You can review the full workflow and experiments there.
- The `pandas_profile_file/` folder contains a pandas profiling HTML report for quick data analysis and insights.

---

## 🤖 Machine Learning Model
- The trained machine learning model is a **Gradient Boosting Classifier**.
- The model is saved as a `.sav` file in the `final_model/` directory: `final_model/gradient_boosting_model.sav`.
- This model is loaded by the Streamlit app (`app.py`) to make real-time predictions based on user input.
- The model was trained using the features and preprocessing pipeline described in the notebook.

---

## 📦 Summary
This project demonstrates a complete machine learning workflow: from data analysis and feature engineering in Jupyter notebooks, to model training and evaluation, and finally to deployment as an interactive web app using Streamlit. All data and models are handled locally for privacy and ease of use. The app is ready for further extension or deployment as needed.

Thank you for checking out this project! 