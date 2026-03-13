import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

class SalaryPredictor:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.feature_columns = None
        
    def load_and_prepare_data(self, csv_path="Salary Data.csv"):
        """
        Load and prepare the salary dataset for machine learning
        """
        print("="*60)
        print("STEP 1: LOADING AND PREPARING DATA")
        print("="*60)
        
        # Load the dataset
        df = pd.read_csv(csv_path)
        print(f"Dataset loaded successfully! Shape: {df.shape}")
        print(f"\nOriginal columns: {list(df.columns)}")
        
        # Display initial data info
        print(f"\nInitial data info:")
        print(df.info())
        
        # Check for missing values
        print(f"\nMissing values per column:")
        missing_values = df.isnull().sum()
        print(missing_values)
        
        # Handle missing values
        print(f"\nHandling missing values...")
        initial_rows = len(df)
        
        # Option 1: Drop rows with missing values (if few missing values)
        if missing_values.sum() < len(df) * 0.1:  # If less than 10% missing
            df = df.dropna()
            print(f"Dropped rows with missing values. Rows removed: {initial_rows - len(df)}")
        else:
            # Option 2: Fill missing values with median/mode
            for col in df.columns:
                if df[col].isnull().sum() > 0:
                    if df[col].dtype in ['int64', 'float64']:
                        df[col].fillna(df[col].median(), inplace=True)
                        print(f"Filled missing values in {col} with median")
                    else:
                        df[col].fillna(df[col].mode()[0], inplace=True)
                        print(f"Filled missing values in {col} with mode")
        
        print(f"Final dataset shape after handling missing values: {df.shape}")
        
        # Feature Selection
        print(f"\nSTEP 2: FEATURE SELECTION")
        print("-" * 40)
        
        # Define features based on available columns
        available_features = []
        target_column = 'Salary'
        
        # Check which features are available in the dataset
        feature_mapping = {
            'Age': 'Age',
            'Years of Experience': 'Years of Experience', 
            'Gender': 'Gender',
            'Education Level': 'Education Level',
            'Job Title': 'Job Title'
        }
        
        for feature_name, column_name in feature_mapping.items():
            if column_name in df.columns:
                available_features.append(column_name)
                print(f"✓ Using feature: {column_name}")
            else:
                print(f"✗ Feature not available: {column_name}")
        
        # Prepare feature matrix X and target vector y
        X = df[available_features].copy()
        y = df[target_column].copy()
        
        print(f"\nSelected features: {available_features}")
        print(f"Target variable: {target_column}")
        print(f"Feature matrix shape: {X.shape}")
        print(f"Target vector shape: {y.shape}")
        
        return X, y, df
    
    def encode_categorical_features(self, X):
        """
        Encode categorical features using One-Hot Encoding
        """
        print(f"\nSTEP 3: ENCODING CATEGORICAL FEATURES")
        print("-" * 40)
        
        # Identify categorical and numerical columns
        categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
        numerical_columns = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        print(f"Categorical columns: {categorical_columns}")
        print(f"Numerical columns: {numerical_columns}")
        
        # Create preprocessor
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_columns),
                ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_columns)
            ]
        )
        
        self.preprocessor = preprocessor
        self.feature_columns = {
            'numerical': numerical_columns,
            'categorical': categorical_columns
        }
        
        return preprocessor
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """
        Split the data into training and testing sets
        """
        print(f"\nSTEP 4: SPLITTING THE DATA")
        print("-" * 40)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"Training set size: {X_train.shape[0]} samples")
        print(f"Testing set size: {X_test.shape[0]} samples")
        print(f"Training set percentage: {(1-test_size)*100:.1f}%")
        print(f"Testing set percentage: {test_size*100:.1f}%")
        
        return X_train, X_test, y_train, y_test
    
    def create_ensemble_models(self):
        """
        Create ensemble learning models
        """
        print(f"\nSTEP 5: CREATING ENSEMBLE MODELS")
        print("-" * 40)
        
        # Random Forest Regressor
        rf_model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2
        )
        
        # Gradient Boosting Regressor
        gb_model = GradientBoostingRegressor(
            n_estimators=100,
            random_state=42,
            max_depth=6,
            learning_rate=0.1,
            min_samples_split=5,
            min_samples_leaf=2
        )
        
        # Voting Regressor (combines RF and GB)
        voting_model = VotingRegressor(
            estimators=[
                ('random_forest', rf_model),
                ('gradient_boosting', gb_model)
            ]
        )
        
        models = {
            'Random Forest': rf_model,
            'Gradient Boosting': gb_model,
            'Voting Regressor': voting_model
        }
        
        print("Created ensemble models:")
        for name in models.keys():
            print(f"✓ {name}")
        
        return models
    
    def train_and_evaluate_models(self, models, X_train, X_test, y_train, y_test):
        """
        Train and evaluate all models
        """
        print(f"\nSTEP 6: TRAINING AND EVALUATING MODELS")
        print("=" * 60)
        
        # Fit the preprocessor and transform the data
        X_train_processed = self.preprocessor.fit_transform(X_train)
        X_test_processed = self.preprocessor.transform(X_test)
        
        results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Train the model
            model.fit(X_train_processed, y_train)
            
            # Make predictions
            y_train_pred = model.predict(X_train_processed)
            y_test_pred = model.predict(X_test_processed)
            
            # Calculate metrics
            train_mae = mean_absolute_error(y_train, y_train_pred)
            test_mae = mean_absolute_error(y_test, y_test_pred)
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
            
            # Store results
            results[name] = {
                'model': model,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse
            }
            
            # Print results
            print(f"{name} Results:")
            print(f"  Training MAE: ${train_mae:,.2f}")
            print(f"  Testing MAE:  ${test_mae:,.2f}")
            print(f"  Training R²:  {train_r2:.4f}")
            print(f"  Testing R²:   {test_r2:.4f}")
            print(f"  Training RMSE: ${train_rmse:,.2f}")
            print(f"  Testing RMSE:  ${test_rmse:,.2f}")
        
        return results
    
    def find_best_model(self, results):
        """
        Find the best performing model based on test R² score
        """
        print(f"\nSTEP 7: MODEL COMPARISON")
        print("=" * 60)
        
        # Create comparison dataframe
        comparison_data = []
        for name, metrics in results.items():
            comparison_data.append({
                'Model': name,
                'Test MAE': f"${metrics['test_mae']:,.2f}",
                'Test R²': f"{metrics['test_r2']:.4f}",
                'Test RMSE': f"${metrics['test_rmse']:,.2f}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        print("Model Performance Comparison:")
        print(comparison_df.to_string(index=False))
        
        # Find best model based on test R² score
        best_model_name = max(results.keys(), key=lambda k: results[k]['test_r2'])
        best_model = results[best_model_name]['model']
        
        print(f"\n🏆 Best Model: {best_model_name}")
        print(f"   Test R² Score: {results[best_model_name]['test_r2']:.4f}")
        print(f"   Test MAE: ${results[best_model_name]['test_mae']:,.2f}")
        
        return best_model_name, best_model
    
    def save_model(self, model, model_name, filename=None):
        """
        Save the trained model and preprocessor to files
        """
        print(f"\nSTEP 8: SAVING THE MODEL")
        print("-" * 40)
        
        if filename is None:
            filename = f"best_salary_model_{model_name.lower().replace(' ', '_')}"
        
        # Save the model
        model_path = f"{filename}.pkl"
        joblib.dump(model, model_path)
        print(f"✓ Model saved as: {model_path}")
        
        # Save the preprocessor
        preprocessor_path = f"{filename}_preprocessor.pkl"
        joblib.dump(self.preprocessor, preprocessor_path)
        print(f"✓ Preprocessor saved as: {preprocessor_path}")
        
        # Save feature information
        feature_info = {
            'feature_columns': self.feature_columns,
            'model_name': model_name
        }
        feature_path = f"{filename}_features.pkl"
        joblib.dump(feature_info, feature_path)
        print(f"✓ Feature info saved as: {feature_path}")
        
        return model_path, preprocessor_path, feature_path
    
    def load_model(self, model_path, preprocessor_path, feature_path):
        """
        Load a saved model for making predictions
        """
        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocessor_path)
        feature_info = joblib.load(feature_path)
        self.feature_columns = feature_info['feature_columns']
        
        print(f"Model loaded successfully!")
        return True
    
    def predict_salary(self, input_data):
        """
        Make salary predictions for new data
        """
        if self.model is None or self.preprocessor is None:
            raise ValueError("Model not loaded. Please load a trained model first.")
        
        # Preprocess the input data
        input_processed = self.preprocessor.transform(input_data)
        
        # Make prediction
        prediction = self.model.predict(input_processed)
        
        return prediction

def main():
    """
    Main function to run the complete salary prediction pipeline
    """
    print("SALARY PREDICTION WITH ENSEMBLE LEARNING")
    print("=" * 60)
    
    # Initialize the predictor
    predictor = SalaryPredictor()
    
    # Step 1: Load and prepare data
    X, y, df = predictor.load_and_prepare_data()
    
    # Step 2: Encode categorical features
    preprocessor = predictor.encode_categorical_features(X)
    
    # Step 3: Split the data
    X_train, X_test, y_train, y_test = predictor.split_data(X, y)
    
    # Step 4: Create ensemble models
    models = predictor.create_ensemble_models()
    
    # Step 5: Train and evaluate models
    results = predictor.train_and_evaluate_models(models, X_train, X_test, y_train, y_test)
    
    # Step 6: Find best model
    best_model_name, best_model = predictor.find_best_model(results)
    
    # Step 7: Save the best model
    predictor.model = best_model
    model_paths = predictor.save_model(best_model, best_model_name)
    
    print(f"\n🎉 SALARY PREDICTION MODEL TRAINING COMPLETED!")
    print(f"Best model ({best_model_name}) has been saved and is ready for predictions.")
    
    # Demonstrate prediction on a sample
    print(f"\nSTEP 9: SAMPLE PREDICTION")
    print("-" * 40)
    
    # Create a sample prediction
    sample_data = X.iloc[:1].copy()  # Take first row as sample
    prediction = predictor.predict_salary(sample_data)
    actual_salary = y.iloc[0]
    
    print(f"Sample Input Data:")
    for col in sample_data.columns:
        print(f"  {col}: {sample_data.iloc[0][col]}")
    
    print(f"\nPredicted Salary: ${prediction[0]:,.2f}")
    print(f"Actual Salary: ${actual_salary:,.2f}")
    print(f"Prediction Error: ${abs(prediction[0] - actual_salary):,.2f}")

if __name__ == "__main__":
    main()
