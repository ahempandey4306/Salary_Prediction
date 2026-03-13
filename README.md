💰 Salary Prediction Tool

A Machine Learning powered web application that predicts employee salaries based on various professional and demographic characteristics.
The system uses ensemble learning models and provides predictions through a Flask-based web interface.

🚀 Features

🤖 Machine Learning Models

Linear Regression

Random Forest Regressor

Gradient Boosting Regressor

Voting Regressor (Best performing)

🌐 Web Interface

Responsive UI built with HTML, CSS, and JavaScript

Simple input form for employee details

⚡ Real-time Predictions

Instant predictions through a Flask API

🧹 Automated Data Preprocessing

Handles missing values

Categorical feature encoding

Numerical feature scaling

📊 Example Profiles

Quick test profiles for common job roles

📂 Project Structure
salary_predict/
│
├── app.py                        # Flask web application
├── salary_prediction_model.py    # ML model training script
├── load_salary_data.py           # Data loading utilities
├── predict_salary.py             # Console prediction tool
├── Salary Data.csv               # Dataset
├── requirements.txt              # Python dependencies
│
├── templates/
│   └── index.html                # Web interface
│
└── venv/                         # Virtual environment
⚙️ Installation & Setup
1️⃣ Clone the Repository
git clone https://github.com/yourusername/salary-prediction-tool.git
cd salary-prediction-tool
2️⃣ Create / Activate Virtual Environment
Windows
python -m venv venv
.\venv\Scripts\Activate.ps1
Linux / Mac
python3 -m venv venv
source venv/bin/activate

After activation, (venv) should appear in your terminal.

3️⃣ Install Dependencies
pip install -r requirements.txt
🧠 Train the Model (First Time Only)

Run the training script:

python salary_prediction_model.py

This will:

Load and preprocess the dataset

Train multiple ML models

Evaluate performance

Save the best performing model (Voting Regressor)

▶️ Run the Web Application

Start the Flask server:

python app.py

Open your browser and visit:

http://localhost:5000
🖥️ Web Interface Usage

Enter employee details:

Feature	Description
Age	18 – 70
Years of Experience	0 – 50
Gender	Male / Female
Education Level	Bachelor's, Master's, PhD
Job Title	Any job title

Click Predict Salary to get the predicted value.

💻 Console Prediction

You can also run predictions directly in the terminal:

python predict_salary.py
📊 Model Performance

The ensemble model achieved the following metrics:

Metric	Score
R² Score	~0.89
MAE	~$10,261
RMSE	~$15,881

This means the model explains ~89% of the variance in salary data.

🔬 Machine Learning Models
1️⃣ Linear Regression

Basic regression baseline model.

2️⃣ Random Forest Regressor

100 estimators

max_depth = 10

Handles non-linear relationships well.

3️⃣ Gradient Boosting Regressor

100 estimators

learning_rate = 0.1

Sequential model improvement.

⭐ 4️⃣ Voting Regressor (Best Model)

Combines predictions from:

Random Forest

Gradient Boosting

Final prediction is average of both models, improving accuracy.

📈 Features Used
Feature	Type
Age	Numerical
Years of Experience	Numerical
Gender	Categorical
Education Level	Categorical
Job Title	Categorical
🔌 API Endpoints
GET /

Returns the main web interface.

POST /predict

Predicts salary.

Request
{
  "age": 30,
  "years_experience": 5,
  "gender": "Male",
  "education_level": "Master's",
  "job_title": "Software Engineer"
}
Response
{
  "success": true,
  "predicted_salary": 95000.50,
  "formatted_salary": "$95,000.50"
}
GET /health

Health check endpoint.

🧪 Example Predictions
Profile	Predicted Salary
Junior Developer (24, 1 year, Bachelor's)	~$55,000
Senior Engineer (32, 8 years, Master's)	~$120,000
Product Manager (35, 10 years, Master's)	~$135,000
Director (45, 18 years, PhD)	~$180,000
🛠️ Troubleshooting
❌ Model Not Found

Run the training script first:

python salary_prediction_model.py
❌ Port Already in Use

Change port in app.py:

app.run(debug=True, host='0.0.0.0', port=5001)
❌ Virtual Environment Issues

Make sure you're inside the project directory and activate the environment:

cd salary_predict
.\venv\Scripts\Activate.ps1
🔮 Future Enhancements

Add location, company size, and industry

Build data visualization dashboard

Implement automatic model retraining

Add prediction confidence intervals

Deploy on AWS / Azure / Heroku

Add user authentication system

📚 Tech Stack

Python

Flask

Scikit-learn

Pandas

NumPy

HTML / CSS / JavaScript

📜 License

This project is developed for educational and learning purposes.
