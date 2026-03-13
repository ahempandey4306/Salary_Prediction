Salary Prediction Tool
A machine learning-powered web application that predicts salaries based on employee characteristics using ensemble learning models.

Features
Advanced ML Models: Uses Random Forest, Gradient Boosting, and Voting Regressor ensemble methods
Web Interface: Beautiful, responsive HTML/CSS/JS interface
Real-time Predictions: Instant salary predictions via Flask API
Data Preprocessing: Handles categorical encoding and feature scaling automatically
Example Profiles: Quick-fill examples for different job roles
Project Structure
salary_predict/
├── app.py                              # Flask web application
├── salary_prediction_model.py          # ML model training script
├── load_salary_data.py                 # Data loading utilities
├── predict_salary.py                   # Console prediction app
├── Salary Data.csv                     # Dataset
├── requirements.txt                     # Python dependencies
├── templates/
│   └── index.html                      # Web interface
└── venv/                               # Virtual environment
Setup Instructions
1. Activate Virtual Environment
# Windows
.\venv\Scripts\Activate.ps1

# Check activation
(venv) should appear in your prompt
2. Install Dependencies
pip install -r requirements.txt
3. Train the Model (First Time Only)
python salary_prediction_model.py
This will:

Load and preprocess the salary dataset
Train ensemble models (Random Forest, Gradient Boosting, Voting Regressor)
Evaluate model performance
Save the best model (Voting Regressor) to disk
4. Run the Web Application
python app.py
The application will start on http://localhost:5000

Usage
Web Interface
Open your browser and go to http://localhost:5000
Fill in the employee details:
Age: 18-70 years
Years of Experience: 0-50 years
Gender: Male/Female
Education Level: Bachelor's, Master's, or PhD
Job Title: Any job title (e.g., Software Engineer, Data Analyst)
Click "Predict Salary" to get the prediction
Use the quick examples for testing different profiles
Console Application
python predict_salary.py
Model Performance
The trained ensemble model achieves:

Test R² Score: ~0.89 (89% variance explained)
Test MAE: ~$10,261 (Mean Absolute Error)
Test RMSE: ~$15,881 (Root Mean Square Error)
Technical Details
Data Preprocessing
Missing Values: Handled by dropping rows (< 10% missing data)
Feature Encoding: One-hot encoding for categorical variables
Feature Scaling: StandardScaler for numerical features
Train/Test Split: 80/20 split with stratification
Machine Learning Models
Linear Regression

100 estimators, max_depth=10
Handles non-linear relationships well
Gradient Boosting Regressor

100 estimators, learning_rate=0.1
Sequential improvement approach
Voting Regressor ⭐ (Best Model)

Combines RF and GB predictions
Averages predictions for better accuracy
Features Used
Age (numerical)
Years of Experience (numerical)
Gender (categorical)
Education Level (categorical)
Job Title (categorical)
API Endpoints
GET /
Returns the main web interface

POST /predict
Predicts salary based on input features

Request Body:

{
    "age": 30,
    "years_experience": 5,
    "gender": "Male",
    "education_level": "Master's",
    "job_title": "Software Engineer"
}
Response:

{
    "success": true,
    "predicted_salary": 95000.50,
    "formatted_salary": "$95,000.50"
}
GET /health
Health check endpoint

Example Predictions
Profile	Predicted Salary
Junior Developer (24, 1 year, Bachelor's)	~$55,000
Senior Engineer (32, 8 years, Master's)	~$120,000
Product Manager (35, 10 years, Master's)	~$135,000
Director (45, 18 years, PhD)	~$180,000
Troubleshooting
Model Not Found Error
If you see "Model files not found", run the training script first:

python salary_prediction_model.py
Port Already in Use
Change the port in app.py:

app.run(debug=True, host='0.0.0.0', port=5001)  # Use different port
Virtual Environment Issues
Ensure you're in the correct directory and the virtual environment is activated:

cd c:\Users\Lenovo\PycharmProjects\salary_predict
.\venv\Scripts\Activate.ps1
Future Enhancements
 Add more features (location, company size, industry)
 Implement model retraining capabilities
 Add data visualization dashboard
 Include confidence intervals for predictions
 Deploy to cloud platform (AWS, Azure, Heroku)
License
This project is for educational purposes.
