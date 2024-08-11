import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from src.exception import CustomException
from src.utils import save_object
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

class TrainPipeline:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.model = None
        self.preprocessor = None

    def load_data(self):
        try:
            # Load the dataset
            data = pd.read_csv(self.data_path)
            return data
        except Exception as e:
            raise CustomException(e, sys)

    def preprocess_data(self, data: pd.DataFrame):
        try:
            # Define features and target
            features = data.drop("math_score", axis=1) 
            target = data["math_score"]  
            
            # Identify categorical and numerical features
            categorical_features = features.select_dtypes(include=['object']).columns
            numerical_features = features.select_dtypes(include=['number']).columns
            
            # Define transformers
            numerical_transformer = StandardScaler()
            categorical_transformer = OneHotEncoder(handle_unknown='ignore')
            
            # Create preprocessing pipeline
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numerical_transformer, numerical_features),
                    ('cat', categorical_transformer, categorical_features)
                ]
            )
            
            # Apply transformations
            X_transformed = preprocessor.fit_transform(features)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X_transformed, target, test_size=0.2, random_state=42)
            
            return X_train, X_test, y_train, y_test
        except Exception as e:
            raise CustomException(e, sys)

    def train_model(self, X_train, y_train):
        try:
            self.model = RandomForestClassifier()
            self.model.fit(X_train, y_train)
        except Exception as e:
            raise CustomException(e, sys)

    def save_artifacts(self):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            
            if not os.path.exists("artifacts"):
                os.makedirs("artifacts")
            
            save_object(file_path=model_path, obj=self.model)
            save_object(file_path=preprocessor_path, obj=self.preprocessor)
        except Exception as e:
            raise CustomException(e, sys)

    def run(self):
        data = self.load_data()
        X_train, X_test, y_train, y_test = self.preprocess_data(data)
        self.train_model(X_train, y_train)
        self.save_artifacts()

if __name__ == "__main__":
    data_path = "notebook/data/stud.csv"  
    pipeline = TrainPipeline(data_path=data_path)
    pipeline.run()
