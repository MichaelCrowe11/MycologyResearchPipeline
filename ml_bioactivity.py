"""
Machine Learning Models for Compound Bioactivity Prediction.

This module contains the implementation of advanced machine learning models
for predicting bioactive compounds and their properties in medicinal mushrooms.
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Any, Optional
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score
import joblib

# Configure logging
logger = logging.getLogger(__name__)

# Paths for model storage
MODEL_DIR = os.path.join(os.getcwd(), 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

# Define model types
BIOACTIVITY_MODEL = os.path.join(MODEL_DIR, 'bioactivity_model.joblib')
COMPOUND_CLASSIFIER = os.path.join(MODEL_DIR, 'compound_classifier.joblib')
POTENCY_PREDICTOR = os.path.join(MODEL_DIR, 'potency_predictor.joblib')

class ModelFeatureExtractor:
    """Extract features from sample data for model input."""
    
    def __init__(self):
        """Initialize the feature extractor."""
        self.feature_columns = [
            # Chemical profile features
            'ph_value', 'moisture_content', 'protein_content', 'polysaccharide_content',
            'peak_1_intensity', 'peak_2_intensity', 'peak_3_intensity', 'peak_4_intensity',
            'uv_absorbance_280nm', 'uv_absorbance_320nm', 'uv_absorbance_360nm',
            
            # Morphological features from computer vision
            'spine_density', 'color_intensity_r', 'color_intensity_g', 'color_intensity_b',
            'texture_coarseness', 'texture_contrast', 'edge_density', 'shape_circularity',
            
            # Growth condition features
            'substrate_type_encoded', 'growth_stage_encoded', 'temperature', 'humidity',
            'light_exposure', 'growing_time_days',
            
            # Metadata features
            'species_encoded', 'geographic_origin_encoded', 'processing_method_encoded',
            'sample_age_days'
        ]
    
    def extract_features(self, sample_data: Dict[str, Any], 
                         include_vision: bool = True,
                         include_spectral: bool = True) -> np.ndarray:
        """
        Extract features from sample data.
        
        Args:
            sample_data: Dictionary containing sample attributes
            include_vision: Whether to include computer vision features
            include_spectral: Whether to include spectral analysis features
            
        Returns:
            Feature vector as numpy array
        """
        # In a real implementation, this would process the actual sample data
        # For demonstration, we'll create synthetic features
        
        features = []
        
        # Add basic chemical profile features
        if include_spectral:
            features.extend([
                sample_data.get('ph_value', 6.5),
                sample_data.get('moisture_content', 8.2),
                sample_data.get('protein_content', 22.3),
                sample_data.get('polysaccharide_content', 35.7),
                sample_data.get('peak_1_intensity', 0.83),
                sample_data.get('peak_2_intensity', 0.57),
                sample_data.get('peak_3_intensity', 0.29),
                sample_data.get('peak_4_intensity', 0.12),
                sample_data.get('uv_absorbance_280nm', 0.92),
                sample_data.get('uv_absorbance_320nm', 0.78),
                sample_data.get('uv_absorbance_360nm', 0.45)
            ])
        else:
            # Add placeholder values if spectral analysis not included
            features.extend([0.0] * 11)
        
        # Add computer vision features
        if include_vision:
            features.extend([
                sample_data.get('spine_density', 0.65),
                sample_data.get('color_intensity_r', 0.72),
                sample_data.get('color_intensity_g', 0.58),
                sample_data.get('color_intensity_b', 0.49),
                sample_data.get('texture_coarseness', 0.83),
                sample_data.get('texture_contrast', 0.67),
                sample_data.get('edge_density', 0.54),
                sample_data.get('shape_circularity', 0.91)
            ])
        else:
            # Add placeholder values if vision features not included
            features.extend([0.0] * 8)
        
        # Add growth condition features
        features.extend([
            sample_data.get('substrate_type_encoded', 1),
            sample_data.get('growth_stage_encoded', 2),
            sample_data.get('temperature', 22.5),
            sample_data.get('humidity', 85.0),
            sample_data.get('light_exposure', 6.5),
            sample_data.get('growing_time_days', 28)
        ])
        
        # Add metadata features
        features.extend([
            sample_data.get('species_encoded', 1),
            sample_data.get('geographic_origin_encoded', 3),
            sample_data.get('processing_method_encoded', 2),
            sample_data.get('sample_age_days', 45)
        ])
        
        return np.array(features).reshape(1, -1)

class BioactivityModel:
    """
    Model for predicting bioactive properties of mushroom compounds.
    """
    
    def __init__(self):
        """Initialize the bioactivity prediction model."""
        self.feature_extractor = ModelFeatureExtractor()
        self.compound_classifier = self._create_compound_classifier()
        self.bioactivity_predictor = self._create_bioactivity_predictor()
        self.potency_predictor = self._create_potency_predictor()
        
        # Load pre-trained models if available
        self._load_models()
    
    def _create_compound_classifier(self) -> Pipeline:
        """
        Create a pipeline for compound classification.
        
        Returns:
            Scikit-learn pipeline for compound classification
        """
        return Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            ))
        ])
    
    def _create_bioactivity_predictor(self) -> Pipeline:
        """
        Create a pipeline for bioactivity prediction.
        
        Returns:
            Scikit-learn pipeline for bioactivity prediction
        """
        return Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', MLPClassifier(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                alpha=0.0001,
                max_iter=1000,
                random_state=42
            ))
        ])
    
    def _create_potency_predictor(self) -> Pipeline:
        """
        Create a pipeline for potency prediction.
        
        Returns:
            Scikit-learn pipeline for potency prediction
        """
        return Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            ))
        ])
    
    def _load_models(self):
        """Load pre-trained models if available."""
        try:
            if os.path.exists(COMPOUND_CLASSIFIER):
                self.compound_classifier = joblib.load(COMPOUND_CLASSIFIER)
                logger.info("Loaded compound classifier model")
            
            if os.path.exists(BIOACTIVITY_MODEL):
                self.bioactivity_predictor = joblib.load(BIOACTIVITY_MODEL)
                logger.info("Loaded bioactivity predictor model")
            
            if os.path.exists(POTENCY_PREDICTOR):
                self.potency_predictor = joblib.load(POTENCY_PREDICTOR)
                logger.info("Loaded potency predictor model")
        
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            logger.info("Using default models")
    
    def train_models(self, training_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Train all models using the provided training data.
        
        Args:
            training_data: DataFrame containing training samples
            
        Returns:
            Dictionary with training metrics
        """
        # In a real implementation, this would use actual training data
        # For demonstration, we'll use synthetic data
        
        # Create synthetic training data if not provided
        if training_data is None or len(training_data) == 0:
            training_data = self._create_synthetic_training_data()
        
        # Extract features and targets
        X = training_data.drop(['compound_class', 'bioactivity', 'potency'], axis=1)
        y_compound = training_data['compound_class']
        y_bioactivity = training_data['bioactivity']
        y_potency = training_data['potency']
        
        # Split data
        X_train, X_test, y_compound_train, y_compound_test = train_test_split(
            X, y_compound, test_size=0.2, random_state=42
        )
        _, _, y_bioactivity_train, y_bioactivity_test = train_test_split(
            X, y_bioactivity, test_size=0.2, random_state=42
        )
        _, _, y_potency_train, y_potency_test = train_test_split(
            X, y_potency, test_size=0.2, random_state=42
        )
        
        # Train compound classifier
        self.compound_classifier.fit(X_train, y_compound_train)
        compound_pred = self.compound_classifier.predict(X_test)
        compound_metrics = {
            'accuracy': accuracy_score(y_compound_test, compound_pred),
            'precision': precision_score(y_compound_test, compound_pred, average='weighted'),
            'recall': recall_score(y_compound_test, compound_pred, average='weighted'),
            'f1': f1_score(y_compound_test, compound_pred, average='weighted')
        }
        
        # Train bioactivity predictor
        self.bioactivity_predictor.fit(X_train, y_bioactivity_train)
        bioactivity_pred = self.bioactivity_predictor.predict(X_test)
        bioactivity_metrics = {
            'accuracy': accuracy_score(y_bioactivity_test, bioactivity_pred),
            'precision': precision_score(y_bioactivity_test, bioactivity_pred, average='weighted'),
            'recall': recall_score(y_bioactivity_test, bioactivity_pred, average='weighted'),
            'f1': f1_score(y_bioactivity_test, bioactivity_pred, average='weighted')
        }
        
        # Train potency predictor
        self.potency_predictor.fit(X_train, y_potency_train)
        potency_pred = self.potency_predictor.predict(X_test)
        potency_metrics = {
            'r2': r2_score(y_potency_test, potency_pred),
            'mse': np.mean((y_potency_test - potency_pred) ** 2)
        }
        
        # Save models
        joblib.dump(self.compound_classifier, COMPOUND_CLASSIFIER)
        joblib.dump(self.bioactivity_predictor, BIOACTIVITY_MODEL)
        joblib.dump(self.potency_predictor, POTENCY_PREDICTOR)
        
        return {
            'compound_classifier': compound_metrics,
            'bioactivity_predictor': bioactivity_metrics,
            'potency_predictor': potency_metrics
        }
    
    def _create_synthetic_training_data(self) -> pd.DataFrame:
        """
        Create synthetic training data for demonstration.
        
        Returns:
            DataFrame with synthetic training data
        """
        np.random.seed(42)
        n_samples = 1000
        
        # Create feature columns
        feature_extractor = ModelFeatureExtractor()
        data = {}
        
        for col in feature_extractor.feature_columns:
            if 'encoded' in col:
                data[col] = np.random.randint(0, 5, n_samples)
            elif col in ['temperature', 'humidity', 'growing_time_days', 'sample_age_days']:
                data[col] = np.random.uniform(15, 30, n_samples)
            else:
                data[col] = np.random.uniform(0, 1, n_samples)
        
        # Create target columns
        data['compound_class'] = np.random.randint(0, 5, n_samples)  # 5 compound classes
        data['bioactivity'] = np.random.randint(0, 3, n_samples)  # 3 bioactivity classes
        data['potency'] = np.random.uniform(0, 100, n_samples)  # Potency score
        
        return pd.DataFrame(data)
    
    def predict(self, sample_data: Dict[str, Any], 
                include_vision: bool = True,
                include_spectral: bool = True) -> Dict[str, Any]:
        """
        Predict compounds and their bioactive properties.
        
        Args:
            sample_data: Dictionary containing sample attributes
            include_vision: Whether to include computer vision features
            include_spectral: Whether to include spectral analysis features
            
        Returns:
            Dictionary with prediction results
        """
        # Extract features
        features = self.feature_extractor.extract_features(
            sample_data, 
            include_vision=include_vision,
            include_spectral=include_spectral
        )
        
        # Predict compound class
        compound_class = self.compound_classifier.predict(features)[0]
        compound_probas = self.compound_classifier.predict_proba(features)[0]
        compound_confidence = np.max(compound_probas) * 100
        
        # Predict bioactivity
        bioactivity = self.bioactivity_predictor.predict(features)[0]
        bioactivity_probas = self.bioactivity_predictor.predict_proba(features)[0]
        bioactivity_confidence = np.max(bioactivity_probas) * 100
        
        # Predict potency
        potency = self.potency_predictor.predict(features)[0]
        
        # Map predictions to compound information
        compound_info = self._get_compound_info(compound_class, potency)
        
        # Prepare result
        result = {
            'compounds': compound_info,
            'bioactivity_type': self._get_bioactivity_type(bioactivity),
            'bioactivity_confidence': bioactivity_confidence,
            'model_metrics': {
                'compound_confidence': compound_confidence,
                'feature_importance': self._get_feature_importance()
            }
        }
        
        return result
    
    def _get_compound_info(self, compound_class: int, potency: float) -> List[Dict[str, Any]]:
        """
        Get information about predicted compounds.
        
        Args:
            compound_class: Predicted compound class
            potency: Predicted potency score
            
        Returns:
            List of compound information dictionaries
        """
        # Map compound classes to actual compounds
        # In a real implementation, this would use a database lookup
        compounds_by_class = {
            0: [
                {'name': 'Hericenone B', 'structure_id': 'HER-B', 'type': 'terpenoid'},
                {'name': 'Erinacine A', 'structure_id': 'ERI-A', 'type': 'terpenoid'},
                {'name': 'Erinacine E', 'structure_id': 'ERI-E', 'type': 'terpenoid'}
            ],
            1: [
                {'name': 'Ganoderic Acid A', 'structure_id': 'GAN-A', 'type': 'triterpene'},
                {'name': 'Ganoderic Acid B', 'structure_id': 'GAN-B', 'type': 'triterpene'},
                {'name': 'Ganoderol B', 'structure_id': 'GOL-B', 'type': 'triterpene'}
            ],
            2: [
                {'name': 'Polysaccharide Krestin', 'structure_id': 'PSK', 'type': 'polysaccharide'},
                {'name': 'Polysaccharide Peptide', 'structure_id': 'PSP', 'type': 'polysaccharide'},
                {'name': 'Coriolan', 'structure_id': 'COR', 'type': 'polysaccharide'}
            ],
            3: [
                {'name': 'Cordycepin', 'structure_id': 'COR-P', 'type': 'nucleoside'},
                {'name': 'Cordycepic Acid', 'structure_id': 'COR-A', 'type': 'organic acid'},
                {'name': 'Ergosterol', 'structure_id': 'ERG', 'type': 'sterol'}
            ],
            4: [
                {'name': 'Lentinan', 'structure_id': 'LEN', 'type': 'polysaccharide'},
                {'name': 'Eritadenine', 'structure_id': 'ERI', 'type': 'alkaloid'},
                {'name': 'Lenthionine', 'structure_id': 'LTH', 'type': 'organosulfur'}
            ]
        }
        
        # Get compounds for the predicted class
        compounds = compounds_by_class.get(compound_class, compounds_by_class[0])
        
        # Add predicted properties
        result = []
        base_confidence = max(85, min(95, potency))
        base_bioactivity = max(70, min(90, potency))
        
        for i, compound in enumerate(compounds):
            # Adjust confidence and bioactivity for each compound
            confidence_adjustment = -5 * i
            bioactivity_adjustment = -7 * i
            
            result.append({
                'name': compound['name'],
                'type': compound['type'],
                'structure_id': compound['structure_id'],
                'confidence': round(base_confidence + confidence_adjustment, 1),
                'bioactivity_score': round(base_bioactivity + bioactivity_adjustment, 1),
                'structure_url': f"/static/images/structures/{compound['structure_id'].lower()}.svg"
            })
        
        return result
    
    def _get_bioactivity_type(self, bioactivity_class: int) -> str:
        """
        Map bioactivity class to description.
        
        Args:
            bioactivity_class: Predicted bioactivity class
            
        Returns:
            String description of bioactivity
        """
        bioactivity_types = {
            0: "Neuroprotective",
            1: "Immunomodulatory",
            2: "Antioxidant"
        }
        return bioactivity_types.get(bioactivity_class, "Unknown")
    
    def _get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance from the models.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        # In a real implementation, this would extract feature importance
        # from the trained models
        
        return {
            'Chemical Profile': 92.0,
            'Growth Conditions': 85.0,
            'Molecular Weight': 78.0,
            'Visual Morphology': 72.0,
            'Spectral Data': 68.0,
            'Substrate Type': 65.0,
            'Growth Stage': 61.0
        }


def create_sample_data(species: str = 'Hericium erinaceus',
                      location: str = 'Pacific Northwest',
                      substrate: str = 'Hardwood') -> Dict[str, Any]:
    """
    Create sample data for prediction.
    
    Args:
        species: Mushroom species
        location: Geographic location
        substrate: Growth substrate
        
    Returns:
        Dictionary with sample data
    """
    species_mapping = {
        'Hericium erinaceus': 1,  # Lion's Mane
        'Ganoderma lucidum': 2,   # Reishi
        'Trametes versicolor': 3, # Turkey Tail
        'Cordyceps militaris': 4  # Cordyceps
    }
    
    location_mapping = {
        'Pacific Northwest': 1,
        'Northeast Asia': 2,
        'Central Europe': 3,
        'East Asia': 4
    }
    
    substrate_mapping = {
        'Hardwood': 1,
        'Softwood': 2,
        'Rice': 3,
        'Grain': 4
    }
    
    # Create sample data with default values for Lion's Mane
    if species == 'Hericium erinaceus':
        sample_data = {
            'species_encoded': species_mapping.get(species, 1),
            'geographic_origin_encoded': location_mapping.get(location, 1),
            'substrate_type_encoded': substrate_mapping.get(substrate, 1),
            'growth_stage_encoded': 2,  # Mature
            'processing_method_encoded': 2,  # Freeze-dried
            'ph_value': 6.8,
            'moisture_content': 8.5,
            'protein_content': 22.3,
            'polysaccharide_content': 41.2,
            'peak_1_intensity': 0.87,
            'peak_2_intensity': 0.64,
            'peak_3_intensity': 0.32,
            'peak_4_intensity': 0.18,
            'uv_absorbance_280nm': 0.94,
            'uv_absorbance_320nm': 0.76,
            'uv_absorbance_360nm': 0.41,
            'spine_density': 0.72,
            'color_intensity_r': 0.76,
            'color_intensity_g': 0.62,
            'color_intensity_b': 0.58,
            'texture_coarseness': 0.85,
            'texture_contrast': 0.69,
            'edge_density': 0.58,
            'shape_circularity': 0.88,
            'temperature': 22.5,
            'humidity': 85.0,
            'light_exposure': 6.5,
            'growing_time_days': 28,
            'sample_age_days': 45
        }
    # For Reishi, adjust values
    elif species == 'Ganoderma lucidum':
        sample_data = {
            'species_encoded': species_mapping.get(species, 2),
            'geographic_origin_encoded': location_mapping.get(location, 4),
            'substrate_type_encoded': substrate_mapping.get(substrate, 1),
            'growth_stage_encoded': 3,  # Fully mature
            'processing_method_encoded': 1,  # Air-dried
            'ph_value': 6.2,
            'moisture_content': 9.1,
            'protein_content': 17.8,
            'polysaccharide_content': 38.5,
            'peak_1_intensity': 0.92,
            'peak_2_intensity': 0.78,
            'peak_3_intensity': 0.45,
            'peak_4_intensity': 0.23,
            'uv_absorbance_280nm': 0.88,
            'uv_absorbance_320nm': 0.72,
            'uv_absorbance_360nm': 0.38,
            'spine_density': 0.12,  # Reishi has different morphology
            'color_intensity_r': 0.82,
            'color_intensity_g': 0.54,
            'color_intensity_b': 0.42,
            'texture_coarseness': 0.76,
            'texture_contrast': 0.82,
            'edge_density': 0.34,
            'shape_circularity': 0.65,
            'temperature': 24.0,
            'humidity': 80.0,
            'light_exposure': 5.5,
            'growing_time_days': 35,
            'sample_age_days': 60
        }
    # For Turkey Tail, adjust values
    elif species == 'Trametes versicolor':
        sample_data = {
            'species_encoded': species_mapping.get(species, 3),
            'geographic_origin_encoded': location_mapping.get(location, 1),
            'substrate_type_encoded': substrate_mapping.get(substrate, 1),
            'growth_stage_encoded': 2,  # Mature
            'processing_method_encoded': 2,  # Freeze-dried
            'ph_value': 6.5,
            'moisture_content': 7.8,
            'protein_content': 19.4,
            'polysaccharide_content': 45.6,
            'peak_1_intensity': 0.81,
            'peak_2_intensity': 0.75,
            'peak_3_intensity': 0.42,
            'peak_4_intensity': 0.24,
            'uv_absorbance_280nm': 0.91,
            'uv_absorbance_320nm': 0.84,
            'uv_absorbance_360nm': 0.51,
            'spine_density': 0.15,
            'color_intensity_r': 0.65,
            'color_intensity_g': 0.68,
            'color_intensity_b': 0.71,
            'texture_coarseness': 0.72,
            'texture_contrast': 0.88,
            'edge_density': 0.76,
            'shape_circularity': 0.42,
            'temperature': 21.5,
            'humidity': 75.0,
            'light_exposure': 7.0,
            'growing_time_days': 24,
            'sample_age_days': 38
        }
    # For Cordyceps, adjust values
    else:
        sample_data = {
            'species_encoded': species_mapping.get(species, 4),
            'geographic_origin_encoded': location_mapping.get(location, 4),
            'substrate_type_encoded': substrate_mapping.get(substrate, 3),  # Rice
            'growth_stage_encoded': 2,  # Mature
            'processing_method_encoded': 2,  # Freeze-dried
            'ph_value': 6.1,
            'moisture_content': 6.5,
            'protein_content': 25.6,
            'polysaccharide_content': 32.4,
            'peak_1_intensity': 0.94,
            'peak_2_intensity': 0.81,
            'peak_3_intensity': 0.62,
            'peak_4_intensity': 0.35,
            'uv_absorbance_280nm': 0.89,
            'uv_absorbance_320nm': 0.81,
            'uv_absorbance_360nm': 0.47,
            'spine_density': 0.08,
            'color_intensity_r': 0.72,
            'color_intensity_g': 0.52,
            'color_intensity_b': 0.32,
            'texture_coarseness': 0.65,
            'texture_contrast': 0.74,
            'edge_density': 0.48,
            'shape_circularity': 0.76,
            'temperature': 23.0,
            'humidity': 85.0,
            'light_exposure': 4.5,
            'growing_time_days': 42,
            'sample_age_days': 65
        }
    
    return sample_data


def predict_compounds(sample_id: str,
                     species: str = 'Hericium erinaceus',
                     include_vision: bool = True,
                     include_spectral: bool = True) -> Dict[str, Any]:
    """
    Predict compounds and their bioactivity for a sample.
    
    Args:
        sample_id: ID of the sample
        species: Species name
        include_vision: Whether to include computer vision features
        include_spectral: Whether to include spectral analysis features
        
    Returns:
        Dictionary with prediction results
    """
    # Create sample data based on species
    sample_data = create_sample_data(species=species)
    
    # Initialize model
    model = BioactivityModel()
    
    # Make prediction
    prediction_result = model.predict(
        sample_data, 
        include_vision=include_vision,
        include_spectral=include_spectral
    )
    
    # Add metadata
    prediction_result['sample_id'] = sample_id
    prediction_result['timestamp'] = pd.Timestamp.now().isoformat()
    prediction_result['species'] = species
    prediction_result['analysis_methods'] = {
        'computer_vision': include_vision,
        'spectral_analysis': include_spectral
    }
    
    return prediction_result


def train_bioactivity_models(training_data_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Train bioactivity prediction models.
    
    Args:
        training_data_path: Path to training data CSV (optional)
        
    Returns:
        Dictionary with training metrics
    """
    # Load training data if provided
    training_data = None
    if training_data_path and os.path.exists(training_data_path):
        try:
            training_data = pd.read_csv(training_data_path)
            logger.info(f"Loaded training data from {training_data_path}")
        except Exception as e:
            logger.error(f"Error loading training data: {str(e)}")
    
    # Initialize model
    model = BioactivityModel()
    
    # Train models
    metrics = model.train_models(training_data)
    
    return metrics


if __name__ == "__main__":
    # For testing
    logging.basicConfig(level=logging.INFO)
    
    # Train models
    print("Training models...")
    metrics = train_bioactivity_models()
    print(f"Training metrics: {metrics}")
    
    # Make a prediction
    print("\nMaking prediction for Lion's Mane sample...")
    result = predict_compounds('HE-2025-042', species='Hericium erinaceus')
    print(f"Prediction result: {result}")