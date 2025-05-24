import json
import pytest
from app import create_app, db
from models import Sample, Compound, Analysis


@pytest.fixture
def app():
    """Create and configure a Flask app for testing."""
    app = create_app('testing')
    
    # Create tables
    with app.app_context():
        db.create_all()
    
    yield app
    
    # Clean up
    with app.app_context():
        db.drop_all()


@pytest.fixture
def client(app):
    """A test client for the app."""
    return app.test_client()


@pytest.fixture
def sample(app):
    """Create a test sample."""
    with app.app_context():
        sample = Sample(
            name="Test Sample",
            description="A test sample for unit tests",
            species="Test Species",
        )
        db.session.add(sample)
        db.session.commit()
        
        return sample


@pytest.fixture
def compound(app, sample):
    """Create a test compound."""
    with app.app_context():
        compound = Compound(
            sample_id=sample.id,
            name="Test Compound",
            formula="C10H15N5O10P2",
            molecular_weight=507.18,
            concentration=0.25,
            bioactivity_index=0.75
        )
        db.session.add(compound)
        db.session.commit()
        
        return compound


@pytest.fixture
def analysis(app, sample):
    """Create a test analysis."""
    with app.app_context():
        analysis = Analysis(
            sample_id=sample.id,
            analysis_type="bioactivity_analysis",
            parameters={"param1": "value1"},
            status="completed",
            results={"result1": 0.5}
        )
        db.session.add(analysis)
        db.session.commit()
        
        return analysis


def test_health_check(client):
    """Test health check endpoint."""
    response = client.get('/api/health')
    assert response.status_code == 200
    
    data = json.loads(response.data)
    assert data['status'] == 'ok'
    assert 'version' in data
    assert 'timestamp' in data
    assert 'components' in data
    assert 'database' in data['components']


def test_get_samples(client, sample):
    """Test get samples endpoint."""
    response = client.get('/api/samples')
    assert response.status_code == 200
    
    data = json.loads(response.data)
    assert data['status'] == 'success'
    assert data['count'] == 1
    assert len(data['samples']) == 1
    assert data['samples'][0]['name'] == "Test Sample"


def test_get_sample(client, sample, compound, analysis):
    """Test get specific sample endpoint."""
    response = client.get(f'/api/samples/{sample.id}')
    assert response.status_code == 200
    
    data = json.loads(response.data)
    assert data['status'] == 'success'
    assert data['sample']['name'] == "Test Sample"
    assert len(data['sample']['compounds']) == 1
    assert data['sample']['compounds'][0]['name'] == "Test Compound"
    assert len(data['sample']['analyses']) == 1
    assert data['sample']['analyses'][0]['analysis_type'] == "bioactivity_analysis"


def test_get_nonexistent_sample(client):
    """Test get nonexistent sample."""
    response = client.get('/api/samples/999')
    assert response.status_code == 404
    
    data = json.loads(response.data)
    assert data['status'] == 'error'
    assert 'not found' in data['message']


def test_create_sample(client):
    """Test create sample endpoint."""
    sample_data = {
        "name": "New Sample",
        "description": "A newly created sample",
        "species": "New Species",
        "location": "Lab A"
    }
    
    response = client.post(
        '/api/samples',
        data=json.dumps(sample_data),
        content_type='application/json'
    )
    assert response.status_code == 201
    
    data = json.loads(response.data)
    assert data['status'] == 'success'
    assert 'sample_id' in data


def test_create_sample_missing_name(client):
    """Test create sample with missing name."""
    sample_data = {
        "description": "A sample with missing name"
    }
    
    response = client.post(
        '/api/samples',
        data=json.dumps(sample_data),
        content_type='application/json'
    )
    assert response.status_code == 400
    
    data = json.loads(response.data)
    assert data['status'] == 'error'
    assert 'Name is required' in data['message']


def test_get_analysis(client, analysis):
    """Test get analysis endpoint."""
    response = client.get(f'/api/analyses/{analysis.id}')
    assert response.status_code == 200
    
    data = json.loads(response.data)
    assert data['status'] == 'success'
    assert data['analysis']['analysis_type'] == "bioactivity_analysis"
    assert data['analysis']['status'] == "completed"
    assert data['analysis']['parameters']['param1'] == "value1"
    assert data['analysis']['results']['result1'] == 0.5


def test_process_data(client):
    """Test process data endpoint."""
    process_data = {
        "input_data": "feature1,feature2\n1.0,4.0\n2.0,5.0\n3.0,6.0",
        "sample_name": "API Test Sample",
        "parameters": {
            "normalization": True
        }
    }
    
    response = client.post(
        '/api/process',
        data=json.dumps(process_data),
        content_type='application/json'
    )
    assert response.status_code == 200
    
    data = json.loads(response.data)
    assert data['status'] == 'success'
    assert 'analysis_id' in data
    assert 'results' in data
    
    # Verify the results structure
    assert 'bioactivity_scores' in data['results'] or 'categories' in data['results']
    assert 'feature_importance' in data['results']


def test_process_data_missing_input(client):
    """Test process data with missing input."""
    process_data = {
        "parameters": {
            "normalization": True
        }
    }
    
    response = client.post(
        '/api/process',
        data=json.dumps(process_data),
        content_type='application/json'
    )
    assert response.status_code == 400
    
    data = json.loads(response.data)
    assert data['status'] == 'error'
    assert 'Missing required input_data field' in data['message']
