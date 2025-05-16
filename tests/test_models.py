import pytest
import numpy as np
from model import MycolModel, load_model


def test_model_initialization():
    """Test model initialization."""
    # Test regressor initialization
    model_reg = MycolModel(model_type='regressor')
    assert model_reg.model_type == 'regressor'
    assert model_reg.model is not None
    
    # Test classifier initialization
    model_cls = MycolModel(model_type='classifier')
    assert model_cls.model_type == 'classifier'
    assert model_cls.model is not None
    
    # Test invalid model type
    with pytest.raises(ValueError):
        MycolModel(model_type='invalid')


def test_model_fit_predict_regressor():
    """Test model fitting and prediction for regressor."""
    # Create a simple dataset
    X = {
        'feature1': [1.0, 2.0, 3.0, 4.0, 5.0],
        'feature2': [5.0, 4.0, 3.0, 2.0, 1.0]
    }
    y = [0.2, 0.4, 0.6, 0.8, 1.0]
    
    # Create and fit model
    model = MycolModel(model_type='regressor')
    model.fit(X, y)
    
    # Test prediction
    X_test = {
        'feature1': [2.5, 3.5],
        'feature2': [3.5, 2.5]
    }
    predictions = model.predict(X_test)
    
    # Verify prediction structure
    assert 'bioactivity_scores' in predictions
    assert 'confidence_intervals' in predictions
    assert 'feature_importance' in predictions
    
    # Verify prediction dimensions
    assert len(predictions['bioactivity_scores']) == 2
    assert len(predictions['confidence_intervals']) == 2
    assert len(predictions['feature_importance']) == 2
    
    # Verify prediction values are within expected range
    assert all(0 <= score <= 1 for score in predictions['bioactivity_scores'])
    
    # Verify confidence intervals make sense
    for (low, high) in predictions['confidence_intervals']:
        assert low <= high
        assert 0 <= low <= 1
        assert 0 <= high <= 1


def test_model_fit_predict_classifier():
    """Test model fitting and prediction for classifier."""
    # Create a simple dataset
    X = {
        'feature1': [1.0, 2.0, 3.0, 4.0, 5.0],
        'feature2': [5.0, 4.0, 3.0, 2.0, 1.0]
    }
    y = ['inactive', 'inactive', 'moderate', 'active', 'active']
    
    # Create and fit model
    model = MycolModel(model_type='classifier')
    model.fit(X, y)
    
    # Test prediction
    X_test = {
        'feature1': [2.5, 4.5],
        'feature2': [3.5, 1.5]
    }
    predictions = model.predict(X_test)
    
    # Verify prediction structure
    assert 'categories' in predictions
    assert 'probabilities' in predictions
    assert 'feature_importance' in predictions
    
    # Verify prediction dimensions
    assert len(predictions['categories']) == 2
    assert len(predictions['probabilities']) == 2
    assert len(predictions['feature_importance']) == 2
    
    # Verify prediction values
    assert all(category in ['inactive', 'moderate', 'active'] for category in predictions['categories'])
    assert all(0 <= prob <= 1 for prob in predictions['probabilities'])


def test_model_save_load(tmp_path):
    """Test model saving and loading."""
    # Create a simple model
    model = MycolModel(model_type='regressor')
    X = {
        'feature1': [1.0, 2.0, 3.0],
        'feature2': [4.0, 5.0, 6.0]
    }
    y = [0.1, 0.5, 0.9]
    model.fit(X, y)
    
    # Save the model
    model_path = tmp_path / "test_model.pkl"
    model.save(str(model_path))
    
    # Load the model
    loaded_model = MycolModel.load(str(model_path))
    
    # Verify loaded model properties
    assert loaded_model.model_type == 'regressor'
    assert loaded_model.feature_names == ['feature1', 'feature2']
    assert loaded_model.version == model.version
    
    # Test prediction with loaded model
    X_test = {
        'feature1': [2.0],
        'feature2': [5.0]
    }
    original_pred = model.predict(X_test)
    loaded_pred = loaded_model.predict(X_test)
    
    # Check that predictions are identical
    assert original_pred['bioactivity_scores'] == loaded_pred['bioactivity_scores']


def test_load_model_function():
    """Test the load_model convenience function."""
    # Test with non-existent path (should create new model)
    model = load_model(model_path="nonexistent_path.pkl")
    assert isinstance(model, MycolModel)
    assert model.model_type == 'regressor'
    
    # Test with specified model type
    model = load_model(model_type='classifier')
    assert isinstance(model, MycolModel)
    assert model.model_type == 'classifier'


def test_model_with_different_feature_sets():
    """Test model handling different feature sets."""
    # Train with two features
    model = MycolModel(model_type='regressor')
    X_train = {
        'feature1': [1.0, 2.0, 3.0],
        'feature2': [4.0, 5.0, 6.0]
    }
    y_train = [0.1, 0.5, 0.9]
    model.fit(X_train, y_train)
    
    # Test with missing feature (should use default value)
    X_test_missing = {
        'feature1': [2.0, 3.0]
    }
    
    # This should not raise an error when using the simulated model
    # In a real model, it would handle missing features
    predictions = model.predict(X_test_missing)
    assert 'bioactivity_scores' in predictions
    
    # Test with extra feature (should ignore extra)
    X_test_extra = {
        'feature1': [2.0, 3.0],
        'feature2': [5.0, 6.0],
        'feature3': [7.0, 8.0]
    }
    predictions = model.predict(X_test_extra)
    assert 'bioactivity_scores' in predictions
