import os
import pickle
import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List, Union
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

class MycolModel:
    """Model for mycology bioactivity prediction."""
    
    def __init__(self, model_type: str = 'regressor'):
        """
        Initialize the model.
        
        Args:
            model_type: Type of model to use ('regressor' or 'classifier')
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.version = "0.1.0"
        
        # Initialize model based on type
        if model_type == 'regressor':
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        elif model_type == 'classifier':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def fit(self, X: Union[pd.DataFrame, Dict[str, List[float]]], y: Union[List[float], np.ndarray]) -> 'MycolModel':
        """
        Fit the model to the training data.
        
        Args:
            X: Training features
            y: Target values
            
        Returns:
            The fitted model instance
        """
        # Convert dictionary to DataFrame if necessary
        if isinstance(X, dict):
            X = pd.DataFrame(X)
        
        # Store feature names
        self.feature_names = list(X.columns)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit the model
        self.model.fit(X_scaled, y)
        
        return self
    
    def predict(self, X: Union[pd.DataFrame, Dict[str, List[float]]]) -> Dict[str, Any]:
        """
        Make predictions with the model.
        
        Args:
            X: Input features
            
        Returns:
            Dictionary with prediction results
        """
        # Handle case where model is not fit yet (simulated model)
        if self.model is None:
            logger.warning("Model not trained. Using simulated predictions.")
            
            # Create simulated model for demonstration
            if isinstance(X, dict):
                n_samples = len(list(X.values())[0])
                
                if self.model_type == 'regressor':
                    # Simulate regression predictions (bioactivity scores)
                    predictions = np.random.uniform(0, 1, n_samples)
                    
                    return {
                        'bioactivity_scores': predictions.tolist(),
                        'confidence_intervals': [
                            (max(0, p - 0.1), min(1, p + 0.1)) for p in predictions
                        ],
                        'feature_importance': {
                            f"feature_{i}": np.random.uniform(0, 1) 
                            for i in range(len(X))
                        }
                    }
                else:
                    # Simulate classification predictions (compound categories)
                    categories = ['active', 'inactive', 'moderate']
                    predictions = np.random.choice(categories, n_samples)
                    probabilities = np.random.uniform(0.5, 0.9, n_samples)
                    
                    return {
                        'categories': predictions.tolist(),
                        'probabilities': probabilities.tolist(),
                        'feature_importance': {
                            f"feature_{i}": np.random.uniform(0, 1) 
                            for i in range(len(X))
                        }
                    }
            else:
                # For DataFrames or other types
                n_samples = len(X)
                
                if self.model_type == 'regressor':
                    predictions = np.random.uniform(0, 1, n_samples)
                    return {
                        'bioactivity_scores': predictions.tolist(),
                        'confidence_intervals': [
                            (max(0, p - 0.1), min(1, p + 0.1)) for p in predictions
                        ],
                        'feature_importance': {
                            col: np.random.uniform(0, 1) 
                            for col in (X.columns if hasattr(X, 'columns') else range(X.shape[1]))
                        }
                    }
                else:
                    categories = ['active', 'inactive', 'moderate']
                    predictions = np.random.choice(categories, n_samples)
                    probabilities = np.random.uniform(0.5, 0.9, n_samples)
                    
                    return {
                        'categories': predictions.tolist(),
                        'probabilities': probabilities.tolist(),
                        'feature_importance': {
                            col: np.random.uniform(0, 1) 
                            for col in (X.columns if hasattr(X, 'columns') else range(X.shape[1]))
                        }
                    }
        
        # Convert dictionary to DataFrame if necessary
        if isinstance(X, dict):
            X = pd.DataFrame(X)
        
        # Ensure all expected features are present
        if hasattr(X, 'columns') and set(self.feature_names) != set(X.columns):
            logger.warning(f"Feature mismatch. Expected {self.feature_names}, got {X.columns}")
            
            # Align features, filling missing ones with 0
            missing_features = set(self.feature_names) - set(X.columns)
            for feature in missing_features:
                X[feature] = 0
            
            # Reorder columns to match training data
            X = X[self.feature_names]
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        if self.model_type == 'regressor':
            predictions = self.model.predict(X_scaled)
            
            # Generate confidence intervals (simplified approach)
            std_dev = np.std(predictions) if len(predictions) > 1 else 0.1
            confidence_intervals = [
                (max(0, p - 1.96 * std_dev), min(1, p + 1.96 * std_dev))
                for p in predictions
            ]
            
            # Get feature importance
            feature_importance = dict(zip(
                self.feature_names, 
                self.model.feature_importances_
            ))
            
            return {
                'bioactivity_scores': predictions.tolist(),
                'confidence_intervals': confidence_intervals,
                'feature_importance': feature_importance
            }
        else:
            # For classifier
            predictions = self.model.predict(X_scaled)
            probabilities = np.max(self.model.predict_proba(X_scaled), axis=1)
            
            # Get feature importance
            feature_importance = dict(zip(
                self.feature_names, 
                self.model.feature_importances_
            ))
            
            return {
                'categories': predictions.tolist(),
                'probabilities': probabilities.tolist(),
                'feature_importance': feature_importance
            }
    
    def save(self, filepath: str) -> None:
        """
        Save the model to a file.
        
        Args:
            filepath: Path to save the model
        """
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_type': self.model_type,
            'version': self.version
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    @classmethod
    def load(cls, filepath: str) -> 'MycolModel':
        """
        Load a model from a file.
        
        Args:
            filepath: Path to the model file
            
        Returns:
            Loaded model instance
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Create a new instance
        instance = cls(model_type=model_data['model_type'])
        
        # Restore model components
        instance.model = model_data['model']
        instance.scaler = model_data['scaler']
        instance.feature_names = model_data['feature_names']
        instance.version = model_data.get('version', '0.1.0')
        
        return instance


def load_model(model_path: str = None, model_type: str = 'regressor') -> MycolModel:
    """
    Load or create a model instance.
    
    Args:
        model_path: Path to a saved model file (optional)
        model_type: Type of model to create if no path is provided
        
    Returns:
        Model instance
    """
    if model_path and os.path.exists(model_path):
        try:
            return MycolModel.load(model_path)
        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {str(e)}")
            logger.info("Creating a new model instance instead.")
    
    # Create a new model instance if no path provided or loading fails
    return MycolModel(model_type=model_type)


if __name__ == "__main__":
    # Example usage
    model = load_model()
    
    # Example features
    features = {
        'feature1': [1.0, 2.0, 3.0],
        'feature2': [4.0, 5.0, 6.0]
    }
    
    # Make predictions
    predictions = model.predict(features)
    print(f"Predictions: {predictions}")
