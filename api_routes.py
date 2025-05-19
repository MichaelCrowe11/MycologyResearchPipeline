import json
import time
import logging
from datetime import datetime
from flask import Blueprint, request, jsonify, current_app
from flasgger import swag_from
from sqlalchemy.exc import SQLAlchemyError

from app import db
from models import Sample, Compound, Analysis, BatchJob
from model import load_model
from batch_processor import process_batch
from monitoring import record_request_duration

logger = logging.getLogger(__name__)
api_bp = Blueprint('api', __name__)


@api_bp.route('/health', methods=['GET'])
@record_request_duration
def health_check():
    """
    Health check endpoint
    ---
    responses:
      200:
        description: Service is healthy
        schema:
          type: object
          properties:
            status:
              type: string
              example: ok
            version:
              type: string
              example: 1.0.0
            timestamp:
              type: string
              example: "2023-10-21T12:34:56Z"
    """
    try:
        # Check database connection using SQLAlchemy text()
        from sqlalchemy import text
        db.session.execute(text("SELECT 1"))
        db_status = "connected"
    except SQLAlchemyError as e:
        logger.error(f"Database connection error: {str(e)}")
        db_status = "error"
    
    return jsonify({
        'status': 'ok' if db_status == 'connected' else 'degraded',
        'version': '0.1.0',  # Version from configuration
        'timestamp': datetime.utcnow().isoformat(),
        'components': {
            'database': db_status,
            'model': 'loaded'  # Could check model availability here
        }
    })


@api_bp.route('/process', methods=['POST'])
@record_request_duration
def process_data():
    """
    Process data for analysis
    ---
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          properties:
            input_data:
              type: string
              description: Input data or file reference
            parameters:
              type: object
              description: Processing parameters
    responses:
      200:
        description: Processing results
        schema:
          type: object
          properties:
            analysis_id:
              type: integer
              example: 123
            status:
              type: string
              example: success
            results:
              type: object
    """
    data = request.json
    
    if not data or 'input_data' not in data:
        return jsonify({
            'status': 'error',
            'message': 'Missing required input_data field'
        }), 400
    
    try:
        # Load the model
        model = load_model()
        
        # Create a new analysis record
        sample = Sample(
            name=data.get('sample_name', 'API Sample'),
            description=data.get('description', 'Sample submitted via API'),
            metadata=data.get('metadata', {})
        )
        db.session.add(sample)
        db.session.flush()  # Get ID without committing
        
        # Create an analysis record
        analysis = Analysis(
            sample_id=sample.id,
            analysis_type=data.get('analysis_type', 'bioactivity_analysis'),
            parameters=data.get('parameters', {}),
            status='processing',
            start_time=datetime.utcnow()
        )
        db.session.add(analysis)
        db.session.commit()
        
        # Process the data with the model
        input_features = prepare_features(data['input_data'], data.get('parameters', {}))
        prediction_results = model.predict(input_features)
        
        # Extract compounds if present
        if 'compounds' in data:
            for compound_data in data['compounds']:
                compound = Compound(
                    sample_id=sample.id,
                    name=compound_data.get('name', 'Unknown'),
                    formula=compound_data.get('formula'),
                    molecular_weight=compound_data.get('molecular_weight'),
                    concentration=compound_data.get('concentration'),
                    bioactivity_index=compound_data.get('bioactivity_index'),
                    metadata=compound_data.get('metadata', {})
                )
                db.session.add(compound)
        
        # Update the analysis with results
        analysis.status = 'completed'
        analysis.results = prediction_results
        analysis.end_time = datetime.utcnow()
        db.session.commit()
        
        return jsonify({
            'status': 'success',
            'analysis_id': analysis.id,
            'results': prediction_results
        })
        
    except Exception as e:
        logger.error(f"Error processing data: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@api_bp.route('/samples', methods=['GET'])
@record_request_duration
def get_samples():
    """
    Get a list of samples
    ---
    parameters:
      - name: limit
        in: query
        type: integer
        default: 50
      - name: offset
        in: query
        type: integer
        default: 0
    responses:
      200:
        description: List of samples
    """
    limit = request.args.get('limit', 50, type=int)
    offset = request.args.get('offset', 0, type=int)
    
    try:
        samples = Sample.query.limit(limit).offset(offset).all()
        result = [{
            'id': sample.id,
            'name': sample.name,
            'species': sample.species,
            'collection_date': sample.collection_date.isoformat() if sample.collection_date else None,
            'created_at': sample.created_at.isoformat()
        } for sample in samples]
        
        return jsonify({
            'status': 'success',
            'count': len(result),
            'samples': result
        })
    
    except Exception as e:
        logger.error(f"Error retrieving samples: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@api_bp.route('/samples/<int:sample_id>', methods=['GET'])
@record_request_duration
def get_sample(sample_id):
    """
    Get a specific sample by ID
    ---
    parameters:
      - name: sample_id
        in: path
        type: integer
        required: true
    responses:
      200:
        description: Sample details
      404:
        description: Sample not found
    """
    try:
        sample = Sample.query.get(sample_id)
        
        if not sample:
            return jsonify({
                'status': 'error',
                'message': f'Sample with ID {sample_id} not found'
            }), 404
        
        # Get related compounds
        compounds = [{
            'id': compound.id,
            'name': compound.name,
            'formula': compound.formula,
            'molecular_weight': compound.molecular_weight,
            'concentration': compound.concentration,
            'bioactivity_index': compound.bioactivity_index
        } for compound in sample.compounds]
        
        # Get related analyses
        analyses = [{
            'id': analysis.id,
            'analysis_type': analysis.analysis_type,
            'status': analysis.status,
            'start_time': analysis.start_time.isoformat() if analysis.start_time else None,
            'end_time': analysis.end_time.isoformat() if analysis.end_time else None
        } for analysis in sample.analyses]
        
        return jsonify({
            'status': 'success',
            'sample': {
                'id': sample.id,
                'name': sample.name,
                'description': sample.description,
                'species': sample.species,
                'collection_date': sample.collection_date.isoformat() if sample.collection_date else None,
                'location': sample.location,
                'metadata': sample.metadata,
                'created_at': sample.created_at.isoformat(),
                'updated_at': sample.updated_at.isoformat(),
                'compounds': compounds,
                'analyses': analyses
            }
        })
    
    except Exception as e:
        logger.error(f"Error retrieving sample: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@api_bp.route('/samples', methods=['POST'])
@record_request_duration
def create_sample():
    """
    Create a new sample
    ---
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          properties:
            name:
              type: string
              required: true
            description:
              type: string
            species:
              type: string
            location:
              type: string
            metadata:
              type: object
    responses:
      201:
        description: Sample created
      400:
        description: Invalid input
    """
    data = request.json
    
    if not data or 'name' not in data:
        return jsonify({
            'status': 'error',
            'message': 'Name is required for creating a sample'
        }), 400
    
    try:
        sample = Sample(
            name=data['name'],
            description=data.get('description'),
            species=data.get('species'),
            collection_date=datetime.fromisoformat(data['collection_date']) if 'collection_date' in data else datetime.utcnow(),
            location=data.get('location'),
            metadata=data.get('metadata', {})
        )
        
        db.session.add(sample)
        db.session.commit()
        
        return jsonify({
            'status': 'success',
            'message': 'Sample created successfully',
            'sample_id': sample.id
        }), 201
    
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error creating sample: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@api_bp.route('/analyses/<int:analysis_id>', methods=['GET'])
@record_request_duration
def get_analysis(analysis_id):
    """
    Get a specific analysis by ID
    ---
    parameters:
      - name: analysis_id
        in: path
        type: integer
        required: true
    responses:
      200:
        description: Analysis details
      404:
        description: Analysis not found
    """
    try:
        analysis = Analysis.query.get(analysis_id)
        
        if not analysis:
            return jsonify({
                'status': 'error',
                'message': f'Analysis with ID {analysis_id} not found'
            }), 404
        
        return jsonify({
            'status': 'success',
            'analysis': {
                'id': analysis.id,
                'sample_id': analysis.sample_id,
                'analysis_type': analysis.analysis_type,
                'parameters': analysis.parameters,
                'results': analysis.results,
                'status': analysis.status,
                'start_time': analysis.start_time.isoformat() if analysis.start_time else None,
                'end_time': analysis.end_time.isoformat() if analysis.end_time else None,
                'created_at': analysis.created_at.isoformat(),
                'updated_at': analysis.updated_at.isoformat()
            }
        })
    
    except Exception as e:
        logger.error(f"Error retrieving analysis: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@api_bp.route('/batch', methods=['POST'])
@record_request_duration
def create_batch_job():
    """
    Create a new batch processing job
    ---
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          properties:
            name:
              type: string
            description:
              type: string
            input_file:
              type: string
              required: true
            parameters:
              type: object
    responses:
      201:
        description: Batch job created
      400:
        description: Invalid input
    """
    data = request.json
    
    if not data or 'input_file' not in data:
        return jsonify({
            'status': 'error',
            'message': 'input_file is required for creating a batch job'
        }), 400
    
    try:
        # Create a new batch job record
        batch_job = BatchJob(
            name=data.get('name', f'Batch Job {datetime.utcnow().isoformat()}'),
            description=data.get('description'),
            input_file=data['input_file'],
            parameters=data.get('parameters', {}),
            status='queued'
        )
        
        db.session.add(batch_job)
        db.session.commit()
        
        # In a production environment, you would queue this job for async processing
        # For now, we'll simulate this with simple processing
        
        # Start the batch job
        batch_job.status = 'processing'
        batch_job.start_time = datetime.utcnow()
        db.session.commit()
        
        # Process the batch
        try:
            result = process_batch(data['input_file'], batch_job.id, data.get('parameters', {}))
            output_file = f"results/batch_{batch_job.id}_results.csv"
            
            # Save the results to a file
            with open(output_file, 'w') as f:
                f.write(result.to_csv())
            
            # Update the batch job record
            batch_job.status = 'completed'
            batch_job.end_time = datetime.utcnow()
            batch_job.output_file = output_file
            batch_job.total_records = len(result)
            batch_job.processed_records = len(result)
            db.session.commit()
            
        except Exception as e:
            batch_job.status = 'failed'
            batch_job.end_time = datetime.utcnow()
            batch_job.error_message = str(e)
            db.session.commit()
            raise
        
        return jsonify({
            'status': 'success',
            'message': 'Batch job created and processed',
            'batch_job_id': batch_job.id,
            'status': batch_job.status
        }), 201
    
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error creating batch job: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@api_bp.route('/batch/<int:job_id>', methods=['GET'])
@record_request_duration
def get_batch_job(job_id):
    """
    Get a specific batch job by ID
    ---
    parameters:
      - name: job_id
        in: path
        type: integer
        required: true
    responses:
      200:
        description: Batch job details
      404:
        description: Batch job not found
    """
    try:
        job = BatchJob.query.get(job_id)
        
        if not job:
            return jsonify({
                'status': 'error',
                'message': f'Batch job with ID {job_id} not found'
            }), 404
        
        return jsonify({
            'status': 'success',
            'batch_job': {
                'id': job.id,
                'name': job.name,
                'description': job.description,
                'input_file': job.input_file,
                'output_file': job.output_file,
                'parameters': job.parameters,
                'status': job.status,
                'start_time': job.start_time.isoformat() if job.start_time else None,
                'end_time': job.end_time.isoformat() if job.end_time else None,
                'error_message': job.error_message,
                'total_records': job.total_records,
                'processed_records': job.processed_records,
                'created_at': job.created_at.isoformat()
            }
        })
    
    except Exception as e:
        logger.error(f"Error retrieving batch job: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


# Helper function to prepare features for the model
def prepare_features(input_data, parameters):
    """Convert input data to the format expected by the model."""
    # Here you would implement the actual feature preparation based on your data
    # This is just a simple placeholder
    features = {}
    
    if isinstance(input_data, str):
        # If it's a string, it might be a file path or raw text data
        try:
            with open(input_data, 'r') as f:
                content = f.read()
                # Parse file content to extract features
                features = {
                    'feature1': [1.0, 2.0, 3.0],  # Example features
                    'feature2': [4.0, 5.0, 6.0]
                }
        except FileNotFoundError:
            # Treat as raw data
            features = {
                'feature1': [float(val) for val in input_data.split(',')],
                'feature2': [0.0, 0.0, 0.0]  # Default values
            }
    elif isinstance(input_data, dict):
        # If it's already a dict, use it directly
        features = input_data
    
    # Apply any parameters if needed
    if 'normalization' in parameters and parameters['normalization']:
        # Apply normalization to features (placeholder)
        pass
    
    return features
