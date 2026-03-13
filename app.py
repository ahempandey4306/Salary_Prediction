from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import os

app = Flask(__name__)

class SalaryPredictionService:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.feature_info = None
        self.load_model()
    
    def load_model(self):
        """Load the saved model and preprocessor"""
        try:
            self.model = joblib.load('best_salary_model_voting_regressor.pkl')
            self.preprocessor = joblib.load('best_salary_model_voting_regressor_preprocessor.pkl')
            self.feature_info = joblib.load('best_salary_model_voting_regressor_features.pkl')
            print("✓ Model loaded successfully!")
            return True
        except FileNotFoundError as e:
            print(f"❌ Model files not found: {e}")
            print("Please run salary_prediction_model.py first to train and save the model.")
            return False
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def predict_salary(self, age, years_experience, gender, education_level, job_title):
        """
        Predict salary based on input features
        """
        if self.model is None:
            return None, "Model not loaded"
        
        try:
            # Create input DataFrame
            input_data = pd.DataFrame({
                'Age': [float(age)],
                'Years of Experience': [float(years_experience)],
                'Gender': [gender],
                'Education Level': [education_level],
                'Job Title': [job_title]
            })
            
            # Preprocess the data
            input_processed = self.preprocessor.transform(input_data)
            
            # Make prediction
            prediction = self.model.predict(input_processed)[0]
            
            return prediction, None
            
        except Exception as e:
            return None, str(e)

# Initialize the prediction service
prediction_service = SalaryPredictionService()

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle salary prediction requests"""
    try:
        # Get data from the request
        data = request.get_json()
        
        age = data.get('age')
        years_experience = data.get('years_experience')
        gender = data.get('gender')
        education_level = data.get('education_level')
        job_title = data.get('job_title')
        
        # Validate input
        if not all([age, years_experience, gender, education_level, job_title]):
            return jsonify({
                'success': False,
                'error': 'All fields are required'
            }), 400
        
        # Make prediction
        predicted_salary, error = prediction_service.predict_salary(
            age, years_experience, gender, education_level, job_title
        )
        
        if error:
            return jsonify({
                'success': False,
                'error': error
            }), 500
        
        return jsonify({
            'success': True,
            'predicted_salary': round(predicted_salary, 2),
            'formatted_salary': f"${predicted_salary:,.2f}"
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    model_status = "loaded" if prediction_service.model is not None else "not loaded"
    return jsonify({
        'status': 'healthy',
        'model_status': model_status
    })

if __name__ == '__main__':
    if prediction_service.model is None:
        print("\n" + "="*60)
        print("WARNING: Model not loaded!")
        print("Please run 'python salary_prediction_model.py' first")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("SALARY PREDICTION WEB APP")
        print("Model loaded successfully!")
        print("Starting Flask server...")
        print("="*60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
