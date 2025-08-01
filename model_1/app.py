import pickle
import pandas as pd
import numpy as np
from flask import Flask, request, render_template, jsonify

# Initialize the Flask application
app = Flask(__name__)

# --- Load the Trained Model Pipeline ---
# This single .pkl file contains your preprocessor and the best trained model.
try:
    with open('champion_churn_model_pipeline_with_smote.pkl', 'rb') as f:
        pipeline = pickle.load(f)
    print("✅ Model pipeline loaded successfully.")
except FileNotFoundError:
    print("❌ Model file not found. Ensure 'champion_churn_model_pipeline_with_smote.pkl' is in the root directory.")
    pipeline = None

# --- Main Route: Render the HTML Dashboard ---
@app.route('/')
def home():
    # This function simply serves the main HTML page.
    return render_template('index.html')

# --- Analysis Route: Handle Predictions ---
@app.route('/analyze', methods=['POST'])
def analyze():
    if pipeline is None:
        return jsonify({'error': 'Model not loaded.'}), 500

    try:
        # --- 1. Get Data from the Frontend ---
        data = request.get_json()
        input_df = pd.DataFrame([data])
        
        # Ensure data types are correct for the model pipeline
        input_df['tenure'] = pd.to_numeric(input_df['tenure'])
        input_df['MonthlyCharges'] = pd.to_numeric(input_df['MonthlyCharges'])
        input_df['TotalCharges'] = pd.to_numeric(input_df['TotalCharges'])

        # --- 2. Make Prediction using the Pipeline ---
        # The pipeline handles all preprocessing (scaling, encoding) automatically.
        churn_probability = pipeline.predict_proba(input_df)[0][1]
        
        # --- 3. Generate Business Insights based on the prediction ---
        risk_level, recommendations = get_business_insights(churn_probability)
        segment, segment_color = get_customer_segment(churn_probability, input_df.iloc[0]['MonthlyCharges'], input_df.iloc[0]['tenure'])
        financial_impact = churn_probability * input_df.iloc[0]['MonthlyCharges'] * 12
        key_risk_factors = get_feature_importances()

        # --- 4. Prepare the JSON Response for the Frontend ---
        response_data = {
            'risk': churn_probability,
            'riskLevel': risk_level,
            'importance': key_risk_factors,
            'recommendations': recommendations,
            'segment': {'name': segment, 'color': segment_color},
            'financialImpact': financial_impact
        }
        
        return jsonify(response_data)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

# --- Helper Functions for generating dynamic insights ---
def get_feature_importances():
    """Extracts and formats the top feature importances from the pipeline's classifier."""
    try:
        preprocessor = pipeline.named_steps['preprocessor']
        classifier = pipeline.named_steps['classifier']
        
        cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out()
        num_features = preprocessor.named_transformers_['num'].get_feature_names_out()
        all_feature_names = np.concatenate([num_features, cat_features])
        
        importances = pd.Series(classifier.feature_importances_, index=all_feature_names).sort_values(ascending=False)
        
        top_features = []
        for feature, importance in importances.head(5).items():
            clean_name = feature.replace('cat__', '').replace('num__', '').replace('_', ' ').title()
            top_features.append({'label': clean_name, 'value': round(importance * 100, 1)})
        return top_features
    except Exception:
        # Fallback if feature importances aren't available
        return [
            {'label': 'Contract Type', 'value': 35}, {'label': 'Customer Tenure', 'value': 25},
            {'label': 'Online Security', 'value': 15}, {'label': 'Tech Support', 'value': 10}
        ]

def get_business_insights(risk):
    """Generates a risk level and actionable recommendations."""
    if risk > 0.7:
        level = 'High Risk'
        recs = [
            {'icon': '🏆', 'text': 'Offer high-value 2-year contract.', 'action': 'upgrade-two-year'},
            {'icon': '📦', 'text': 'Bundle all security & support services.', 'action': 'bundle-services'}
        ]
    elif risk > 0.4:
        level = 'Medium Risk'
        recs = [
            {'icon': '📄', 'text': 'Propose a 1-year contract discount.', 'action': 'upgrade-one-year'},
            {'icon': '🛡️', 'text': 'Add Online Security as a trial.', 'action': 'add-security'}
        ]
    else:
        level = 'Low Risk'
        recs = [{'icon': '👍', 'text': 'Maintain service and monitor satisfaction.', 'action': None}]
    return level, recs

def get_customer_segment(risk, monthly_charges, tenure):
    """Categorizes the customer into a meaningful segment."""
    if risk > 0.5 and monthly_charges > 85:
        return 'High-Value, At-Risk', '#e11d48' # Red
    elif risk > 0.5 and tenure < 12:
        return 'New & Vulnerable', '#f59e0b' # Orange
    elif risk < 0.3 and monthly_charges > 70:
        return 'Loyal & High-Value', '#16a34a' # Green
    else:
        return 'Standard Customer', '#6b7280' # Muted text color

# --- Run the Flask App ---
if __name__ == "__main__":
    app.run(debug=True)