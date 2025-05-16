import os
import logging
from datetime import datetime
from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify, send_file, current_app
from werkzeug.utils import secure_filename

from app import db
from models import Sample, Compound, Analysis, BatchJob, Version, ResearchLog
from model import load_model
from batch_processor import process_batch

logger = logging.getLogger(__name__)
web_bp = Blueprint('web', __name__)


@web_bp.route('/')
def index():
    """Render the homepage."""
    # Get latest samples and analyses for the dashboard
    recent_samples = Sample.query.order_by(Sample.created_at.desc()).limit(5).all()
    recent_analyses = Analysis.query.order_by(Analysis.created_at.desc()).limit(5).all()
    
    # Get current version
    current_version = Version.query.filter_by(is_current=True).first()
    if not current_version:
        current_version = Version(
            version="0.1.0",
            description="Initial version",
            is_current=True,
            changelog="First release of the Mycology Research Pipeline"
        )
        db.session.add(current_version)
        db.session.commit()
    
    return render_template(
        'index.html',
        recent_samples=recent_samples,
        recent_analyses=recent_analyses,
        current_version=current_version
    )


@web_bp.route('/dashboard')
def dashboard():
    """Render the main dashboard."""
    # Get statistics for the dashboard
    sample_count = Sample.query.count()
    analysis_count = Analysis.query.count()
    compound_count = Compound.query.count()
    batch_job_count = BatchJob.query.count()
    
    # Get latest completed analyses for visualization
    recent_analyses = Analysis.query.filter_by(status='completed').order_by(Analysis.created_at.desc()).limit(10).all()
    
    # Get success/failure rates
    completed_analyses = Analysis.query.filter_by(status='completed').count()
    failed_analyses = Analysis.query.filter_by(status='failed').count()
    success_rate = (completed_analyses / analysis_count * 100) if analysis_count > 0 else 0
    
    return render_template(
        'dashboard.html',
        sample_count=sample_count,
        analysis_count=analysis_count,
        compound_count=compound_count,
        batch_job_count=batch_job_count,
        recent_analyses=recent_analyses,
        success_rate=success_rate
    )


@web_bp.route('/samples')
def list_samples():
    """List all samples."""
    samples = Sample.query.order_by(Sample.created_at.desc()).all()
    return render_template('samples.html', samples=samples)


@web_bp.route('/samples/<int:sample_id>')
def view_sample(sample_id):
    """View a specific sample."""
    sample = Sample.query.get_or_404(sample_id)
    return render_template('sample_details.html', sample=sample)


@web_bp.route('/analysis/new', methods=['GET', 'POST'])
def new_analysis():
    """Create a new analysis."""
    if request.method == 'POST':
        # Get form data
        sample_id = request.form.get('sample_id')
        analysis_type = request.form.get('analysis_type')
        parameters = request.form.get('parameters', '{}')
        
        try:
            # Convert parameters from JSON string
            parameters_dict = json.loads(parameters)
            
            # Get the sample
            sample = Sample.query.get(sample_id)
            if not sample:
                flash('Sample not found', 'error')
                return redirect(url_for('web.new_analysis'))
            
            # Create a new analysis
            analysis = Analysis(
                sample_id=sample_id,
                analysis_type=analysis_type,
                parameters=parameters_dict,
                status='processing',
                start_time=datetime.utcnow()
            )
            db.session.add(analysis)
            db.session.commit()
            
            # Perform the analysis
            # In a production app, this would be a background job
            # For simplicity, we'll do it inline here
            model = load_model()
            
            # Prepare data for the model
            features = {
                'feature1': [1.0, 2.0, 3.0],  # Example features
                'feature2': [4.0, 5.0, 6.0]
            }
            
            # Get predictions
            results = model.predict(features)
            
            # Update the analysis with results
            analysis.status = 'completed'
            analysis.results = results
            analysis.end_time = datetime.utcnow()
            db.session.commit()
            
            flash('Analysis completed successfully', 'success')
            return redirect(url_for('web.view_analysis', analysis_id=analysis.id))
            
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error creating analysis: {str(e)}")
            flash(f'Error creating analysis: {str(e)}', 'error')
    
    # For GET request, render the form
    samples = Sample.query.all()
    analysis_types = ['bioactivity_analysis', 'compound_identification', 'potency_prediction']
    
    return render_template(
        'analysis.html',
        samples=samples,
        analysis_types=analysis_types
    )


@web_bp.route('/analysis/<int:analysis_id>')
def view_analysis(analysis_id):
    """View a specific analysis."""
    analysis = Analysis.query.get_or_404(analysis_id)
    return render_template('analysis_details.html', analysis=analysis)


@web_bp.route('/batch/new', methods=['GET', 'POST'])
def new_batch_job():
    """Create a new batch processing job."""
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part', 'error')
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            flash('No selected file', 'error')
            return redirect(request.url)
        
        if file:
            filename = secure_filename(file.filename)
            upload_folder = current_app.config['UPLOAD_FOLDER']
            file_path = os.path.join(upload_folder, filename)
            file.save(file_path)
            
            # Get form data
            name = request.form.get('name')
            description = request.form.get('description')
            parameters = request.form.get('parameters', '{}')
            
            try:
                # Convert parameters from JSON string
                parameters_dict = json.loads(parameters)
                
                # Create a new batch job
                batch_job = BatchJob(
                    name=name or f'Batch Job {datetime.utcnow().isoformat()}',
                    description=description,
                    input_file=file_path,
                    parameters=parameters_dict,
                    status='queued'
                )
                db.session.add(batch_job)
                db.session.commit()
                
                # Start processing (for simplicity we'll do it inline)
                batch_job.status = 'processing'
                batch_job.start_time = datetime.utcnow()
                db.session.commit()
                
                # Process the batch
                try:
                    result = process_batch(file_path, batch_job.id, parameters_dict)
                    output_file = os.path.join(
                        current_app.config['RESULTS_FOLDER'],
                        f"batch_{batch_job.id}_results.csv"
                    )
                    
                    # Save the results to a file
                    result.to_csv(output_file, index=False)
                    
                    # Update the batch job record
                    batch_job.status = 'completed'
                    batch_job.end_time = datetime.utcnow()
                    batch_job.output_file = output_file
                    batch_job.total_records = len(result)
                    batch_job.processed_records = len(result)
                    db.session.commit()
                    
                    flash('Batch job processed successfully', 'success')
                    
                except Exception as e:
                    batch_job.status = 'failed'
                    batch_job.end_time = datetime.utcnow()
                    batch_job.error_message = str(e)
                    db.session.commit()
                    flash(f'Error processing batch job: {str(e)}', 'error')
                
                return redirect(url_for('web.view_batch_job', job_id=batch_job.id))
                
            except Exception as e:
                db.session.rollback()
                logger.error(f"Error creating batch job: {str(e)}")
                flash(f'Error creating batch job: {str(e)}', 'error')
    
    # For GET request, render the form
    return render_template('batch_upload.html')


@web_bp.route('/batch/<int:job_id>')
def view_batch_job(job_id):
    """View a specific batch job."""
    job = BatchJob.query.get_or_404(job_id)
    return render_template('batch_details.html', job=job)


@web_bp.route('/batch/<int:job_id>/download')
def download_batch_results(job_id):
    """Download the results of a batch job."""
    job = BatchJob.query.get_or_404(job_id)
    
    if not job.output_file or job.status != 'completed':
        flash('Results not available for download', 'error')
        return redirect(url_for('web.view_batch_job', job_id=job_id))
    
    return send_file(job.output_file, as_attachment=True)


@web_bp.route('/results')
def view_results():
    """View analysis results."""
    # Get completed analyses
    analyses = Analysis.query.filter_by(status='completed').order_by(Analysis.created_at.desc()).all()
    return render_template('results.html', analyses=analyses)


@web_bp.route('/logs/new', methods=['GET', 'POST'])
def new_research_log():
    """Create a new research log entry."""
    if request.method == 'POST':
        title = request.form.get('title')
        content = request.form.get('content')
        sample_id = request.form.get('sample_id')
        analysis_id = request.form.get('analysis_id')
        
        if not title:
            flash('Title is required', 'error')
            return redirect(request.url)
        
        try:
            log = ResearchLog(
                title=title,
                content=content,
                sample_id=sample_id or None,
                analysis_id=analysis_id or None
            )
            db.session.add(log)
            db.session.commit()
            
            flash('Research log created successfully', 'success')
            return redirect(url_for('web.list_research_logs'))
            
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error creating research log: {str(e)}")
            flash(f'Error creating research log: {str(e)}', 'error')
    
    # For GET request, render the form
    samples = Sample.query.all()
    analyses = Analysis.query.all()
    
    return render_template(
        'log_form.html',
        samples=samples,
        analyses=analyses
    )


@web_bp.route('/logs')
def list_research_logs():
    """List research logs."""
    logs = ResearchLog.query.order_by(ResearchLog.created_at.desc()).all()
    return render_template('logs.html', logs=logs)


@web_bp.route('/logs/<int:log_id>')
def view_research_log(log_id):
    """View a specific research log."""
    log = ResearchLog.query.get_or_404(log_id)
    return render_template('log_details.html', log=log)


@web_bp.route('/about')
def about():
    """About page with version information."""
    current_version = Version.query.filter_by(is_current=True).first()
    versions = Version.query.order_by(Version.release_date.desc()).all()
    
    return render_template(
        'about.html',
        current_version=current_version,
        versions=versions
    )


@web_bp.route('/documentation')
def documentation():
    """Documentation page."""
    return render_template('documentation.html')
