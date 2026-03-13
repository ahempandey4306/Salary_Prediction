import pandas as pd
import joblib
import numpy as np

class SalaryPredictionApp:
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
        except FileNotFoundError:
            print("❌ Model files not found. Please run salary_prediction_model.py first.")
            return False
    
    def predict_salary(self, age, years_experience, gender, education_level, job_title):
        """
        Predict salary based on input features
        """
        if self.model is None:
            print("❌ Model not loaded!")
            return None
        
        # Create input DataFrame
        input_data = pd.DataFrame({
            'Age': [age],
            'Years of Experience': [years_experience],
            'Gender': [gender],
            'Education Level': [education_level],
            'Job Title': [job_title]
        })
        
        # Preprocess the data
        input_processed = self.preprocessor.transform(input_data)
        
        # Make prediction
        prediction = self.model.predict(input_processed)[0]
        
        return prediction
    
    def interactive_prediction(self):
        """Interactive interface for salary prediction"""
        print("\n" + "="*60)
        print("SALARY PREDICTION TOOL")
        print("="*60)
        
        while True:
            try:
                print("\nEnter employee details:")
                
                # Get input from user
                age = float(input("Age: "))
                years_exp = float(input("Years of Experience: "))
                
                print("\nGender options: Male, Female")
                gender = input("Gender: ").strip()
                
                print("\nEducation Level options: Bachelor's, Master's, PhD")
                education = input("Education Level: ").strip()
                
                print("\nJob Title (e.g., Software Engineer, Data Analyst, Manager): ")
                job_title = input("Job Title: ").strip()
                
                # Make prediction
                predicted_salary = self.predict_salary(age, years_exp, gender, education, job_title)
                
                if predicted_salary is not None:
                    print(f"\n🎯 PREDICTED SALARY: ${predicted_salary:,.2f}")
                    
                    # Show prediction confidence
                    print(f"\nEmployee Profile:")
                    print(f"  Age: {age}")
                    print(f"  Experience: {years_exp} years")
                    print(f"  Gender: {gender}")
                    print(f"  Education: {education}")
                    print(f"  Job Title: {job_title}")
                
                # Ask if user wants to make another prediction
                another = input("\nMake another prediction? (y/n): ").lower().strip()
                if another != 'y':
                    break
                    
            except ValueError:
                print("❌ Please enter valid numeric values for age and experience.")
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

def batch_prediction_example():
    """Example of batch prediction with multiple employees"""
    predictor = SalaryPredictionApp()
    
    if predictor.model is None:
        return
    
    print("\n" + "="*60)
    print("BATCH PREDICTION EXAMPLE")
    print("="*60)
    
    # Example employee data
    employees = [
        {'Age': 28, 'Years of Experience': 3, 'Gender': 'Female', 'Education Level': "Master's", 'Job Title': 'Data Scientist'},
        {'Age': 35, 'Years of Experience': 8, 'Gender': 'Male', 'Education Level': "Bachelor's", 'Job Title': 'Product Manager'},
        {'Age': 42, 'Years of Experience': 15, 'Gender': 'Female', 'Education Level': 'PhD', 'Job Title': 'Senior Manager'},
        {'Age': 25, 'Years of Experience': 1, 'Gender': 'Male', 'Education Level': "Bachelor's", 'Job Title': 'Junior Developer'},
        {'Age': 50, 'Years of Experience': 20, 'Gender': 'Male', 'Education Level': "Master's", 'Job Title': 'Director'}
    ]
    
    print("Predicting salaries for sample employees:\n")
    
    for i, emp in enumerate(employees, 1):
        salary = predictor.predict_salary(
            emp['Age'], 
            emp['Years of Experience'], 
            emp['Gender'], 
            emp['Education Level'], 
            emp['Job Title']
        )
        
        print(f"Employee {i}:")
        print(f"  Profile: {emp['Age']} years old, {emp['Years of Experience']} years exp, {emp['Gender']}")
        print(f"  Education: {emp['Education Level']}, Job: {emp['Job Title']}")
        print(f"  💰 Predicted Salary: ${salary:,.2f}\n")

def main():
    """Main function to run the salary prediction app"""
    predictor = SalaryPredictionApp()
    
    if predictor.model is None:
        print("Please run 'python salary_prediction_model.py' first to train and save the model.")
        return
    
    print("Welcome to the Salary Prediction Tool!")
    print("\nOptions:")
    print("1. Interactive Prediction")
    print("2. Batch Prediction Example")
    print("3. Exit")
    
    while True:
        try:
            choice = input("\nSelect an option (1-3): ").strip()
            
            if choice == '1':
                predictor.interactive_prediction()
            elif choice == '2':
                batch_prediction_example()
            elif choice == '3':
                print("Goodbye!")
                break
            else:
                print("Invalid choice. Please select 1, 2, or 3.")
                
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break

if __name__ == "__main__":
    main()
