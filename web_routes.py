"""
Web routes for the Mycology Research Pipeline.

This module contains the web routes for the user interface.
"""
from datetime import datetime
import os
import logging
from flask import Blueprint, render_template, request, redirect, url_for, flash, current_app, send_from_directory
from flask_login import current_user, login_required
from werkzeug.utils import secure_filename
from app import db
from models import User, Sample, Analysis, BatchJob, LiteratureReference, Version, ResearchLog

# Configure logging
logger = logging.getLogger(__name__)

# Create Blueprint
web_bp = Blueprint('web', __name__)

@web_bp.route('/')
def index():
    """Homepage."""
    # Get the current version
    current_version = Version.query.filter_by(is_current=True).first()
    if not current_version:
        current_version = Version(
            version="1.0.0",
            release_date=datetime.now(),
            description="Initial release of the Mycology Research Pipeline.",
            is_current=True
        )
        db.session.add(current_version)
        db.session.commit()
    
    # Get recent samples if user is logged in
    recent_samples = []
    if current_user.is_authenticated:
        recent_samples = Sample.query.filter_by(user_id=current_user.id).order_by(Sample.created_at.desc()).limit(5).all()
    
    return render_template('index.html', 
                          current_version=current_version,
                          recent_samples=recent_samples)

@web_bp.route('/dashboard')
@login_required
def dashboard():
    """User dashboard."""
    # Get user samples
    user_samples = Sample.query.filter_by(user_id=current_user.id).order_by(Sample.created_at.desc()).all()
    
    # Get recent analyses
    recent_analyses = Analysis.query.filter_by(user_id=current_user.id).order_by(Analysis.created_at.desc()).limit(5).all()
    
    # Get batch jobs
    batch_jobs = BatchJob.query.filter_by(user_id=current_user.id).order_by(BatchJob.created_at.desc()).all()
    
    # Get literature references
    literature_refs = LiteratureReference.query.filter_by(user_id=current_user.id, is_favorite=True).limit(5).all()
    
    return render_template('dashboard.html',
                          user_samples=user_samples,
                          recent_analyses=recent_analyses,
                          batch_jobs=batch_jobs,
                          literature_refs=literature_refs)

@web_bp.route('/samples')
@login_required
def samples():
    """User samples list."""
    user_samples = Sample.query.filter_by(user_id=current_user.id).order_by(Sample.created_at.desc()).all()
    return render_template('samples.html', samples=user_samples)

@web_bp.route('/sample/<int:sample_id>')
@login_required
def view_sample(sample_id):
    """View a specific sample."""
    sample = Sample.query.get_or_404(sample_id)
    
    # Check if user has access to this sample
    if sample.user_id != current_user.id and not sample.is_public:
        flash('You do not have permission to view this sample.', 'danger')
        return redirect(url_for('web.samples'))
    
    # Get analyses for this sample
    analyses = Analysis.query.filter_by(sample_id=sample_id).order_by(Analysis.created_at.desc()).all()
    
    return render_template('sample_detail.html', sample=sample, analyses=analyses)

@web_bp.route('/image_analysis', methods=['GET', 'POST'])
@login_required
def image_analysis():
    """Image analysis page."""
    if request.method == 'POST':
        # Check if an image was uploaded
        if 'image' not in request.files:
            flash('No image selected', 'danger')
            return redirect(request.url)
        
        image_file = request.files['image']
        
        # If no file was selected
        if image_file.filename == '':
            flash('No image selected', 'danger')
            return redirect(request.url)
        
        # If file has the correct format
        if image_file and allowed_file(image_file.filename, ['jpg', 'jpeg', 'png']):
            # Create a unique filename
            filename = secure_filename(image_file.filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            new_filename = f"{timestamp}_{filename}"
            
            # Ensure upload directory exists
            upload_dir = os.path.join(current_app.root_path, 'uploads')
            if not os.path.exists(upload_dir):
                os.makedirs(upload_dir)
            
            # Save the file
            file_path = os.path.join(upload_dir, new_filename)
            image_file.save(file_path)
            
            # Create a new analysis record
            analysis = Analysis(
                user_id=current_user.id,
                name=f"Image Analysis: {filename}",
                description=request.form.get('notes', ''),
                type="image_analysis",
                parameters={
                    "identify_species": "identify_species" in request.form,
                    "analyze_morphology": "analyze_morphology" in request.form,
                    "analyze_color": "analyze_color" in request.form,
                    "analyze_growth": "analyze_growth" in request.form,
                },
                status="processing",
                start_time=datetime.now()
            )
            db.session.add(analysis)
            db.session.commit()
            
            # In a production app, we would process the image asynchronously
            # For demonstration purposes, we'll just simulate a completed analysis
            analysis.status = "completed"
            analysis.results = {
                "primary_species": "Amanita muscaria",
                "primary_confidence": 95,
                "secondary_species": "Amanita pantherina",
                "secondary_confidence": 72,
                "morphology": {
                    "cap_diameter": "10.2 cm",
                    "stem_height": "12.5 cm",
                    "stem_width": "2.1 cm",
                    "cap_shape": "Convex"
                },
                "colors": {
                    "cap": ["#B22222", "#CD5C5C", "#F5F5F5"],
                    "stem": ["#F5F5F5", "#E8E8E8", "#D3D3D3"]
                },
                "growth_stage": "Mature",
                "growth_progress": 85,
                "days_to_harvest": "Optimal for harvest"
            }
            analysis.end_time = datetime.now()
            db.session.commit()
            
            flash('Image analysis completed successfully!', 'success')
            return redirect(url_for('web.view_analysis', analysis_id=analysis.id))
        
        else:
            flash('Invalid file type. Please upload a JPG or PNG image.', 'danger')
            return redirect(request.url)
    
    return render_template('image_analysis.html')

@web_bp.route('/analysis/<int:analysis_id>')
@login_required
def view_analysis(analysis_id):
    """View a specific analysis."""
    analysis = Analysis.query.get_or_404(analysis_id)
    
    # Check if user has access to this analysis
    if analysis.user_id != current_user.id:
        flash('You do not have permission to view this analysis.', 'danger')
        return redirect(url_for('web.dashboard'))
    
    return render_template('analysis_detail.html', analysis=analysis)

@web_bp.route('/batch_processing', methods=['GET', 'POST'])
@login_required
def batch_processing():
    """Batch processing page."""
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'input_file' not in request.files:
            flash('No file selected', 'danger')
            return redirect(request.url)
        
        input_file = request.files['input_file']
        
        # If no file was selected
        if input_file.filename == '':
            flash('No file selected', 'danger')
            return redirect(request.url)
        
        # If file has the correct format
        if input_file and allowed_file(input_file.filename, ['csv']):
            # Create a unique filename
            filename = secure_filename(input_file.filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            new_filename = f"{timestamp}_{filename}"
            
            # Ensure upload directory exists
            upload_dir = os.path.join(current_app.root_path, 'uploads')
            if not os.path.exists(upload_dir):
                os.makedirs(upload_dir)
            
            # Save the file
            file_path = os.path.join(upload_dir, new_filename)
            input_file.save(file_path)
            
            # Create a new batch job record
            batch_job = BatchJob(
                user_id=current_user.id,
                name=request.form.get('name', f"Batch Job: {filename}"),
                description=request.form.get('description', ''),
                type=request.form.get('type', 'compound_analysis'),
                parameters={
                    "analysis_type": request.form.get('analysis_type', 'default'),
                    "options": request.form.getlist('options')
                },
                input_file=new_filename,
                status="processing",
                start_time=datetime.now()
            )
            db.session.add(batch_job)
            db.session.commit()
            
            try:
                # In a production app, we would process the batch job asynchronously
                # For demonstration purposes, we'll just simulate a completed job
                from batch_processor import process_batch
                
                results = process_batch(file_path, batch_job.id)
                
                # Save results
                output_filename = f"results_{new_filename}"
                output_path = os.path.join(upload_dir, output_filename)
                results.to_csv(output_path, index=False)
                
                # Update batch job
                batch_job.status = "completed"
                batch_job.end_time = datetime.now()
                batch_job.output_file = output_filename
                batch_job.total_records = len(results)
                batch_job.processed_records = len(results)
                db.session.commit()
                
                flash('Batch processing completed successfully!', 'success')
                
            except Exception as e:
                logger.error(f"Batch processing error: {str(e)}")
                batch_job.status = "failed"
                batch_job.end_time = datetime.now()
                batch_job.error_message = str(e)
                db.session.commit()
                
                flash(f'Error processing batch job: {str(e)}', 'danger')
            
            return redirect(url_for('web.view_batch_job', job_id=batch_job.id))
        
        else:
            flash('Invalid file type. Please upload a CSV file.', 'danger')
            return redirect(request.url)
    
    return render_template('batch_processing.html')

@web_bp.route('/batch_job/<int:job_id>')
@login_required
def view_batch_job(job_id):
    """View a specific batch job."""
    batch_job = BatchJob.query.get_or_404(job_id)
    
    # Check if user has access to this batch job
    if batch_job.user_id != current_user.id:
        flash('You do not have permission to view this batch job.', 'danger')
        return redirect(url_for('web.dashboard'))
    
    return render_template('batch_job_detail.html', batch_job=batch_job)

@web_bp.route('/download/<path:filename>')
@login_required
def download_file(filename):
    """Download a file."""
    upload_dir = os.path.join(current_app.root_path, 'uploads')
    return send_from_directory(upload_dir, filename, as_attachment=True)

@web_bp.route('/literature_search', methods=['GET', 'POST'])
@login_required
def literature_search():
    """Literature search page."""
    if request.method == 'POST':
        # Get search parameters
        query = request.form.get('query', '')
        database = request.form.get('database', 'pubmed')
        start_date = request.form.get('start_date', '')
        end_date = request.form.get('end_date', '')
        
        if not query:
            flash('Please enter a search query', 'warning')
            return redirect(request.url)
        
        try:
            # Perform search using the appropriate module
            if database == 'pubmed':
                from literature import search_pubmed as search
            elif database == 'scopus':
                from literature import search_scopus as search
            else:
                from literature import search_science_direct as search
            
            # Search for literature
            results = search(query, start_date=start_date, end_date=end_date)
            
            # Log the search
            log = ResearchLog(
                user_id=current_user.id,
                title=f"Literature Search: {query}",
                content=f"Searched {database} for '{query}' between {start_date} and {end_date}. Found {len(results)} results."
            )
            db.session.add(log)
            db.session.commit()
            
            return render_template('literature_results.html', 
                                  query=query, 
                                  database=database,
                                  results=results)
            
        except Exception as e:
            logger.error(f"Literature search error: {str(e)}")
            flash(f'Error performing literature search: {str(e)}', 'danger')
            return redirect(request.url)
    
    return render_template('literature_search.html')

@web_bp.route('/save_reference', methods=['POST'])
@login_required
def save_reference():
    """Save a literature reference."""
    # Get reference data
    title = request.form.get('title', '')
    authors = request.form.get('authors', '')
    journal = request.form.get('journal', '')
    publication_date = request.form.get('publication_date', '')
    doi = request.form.get('doi', '')
    pubmed_id = request.form.get('pubmed_id', '')
    abstract = request.form.get('abstract', '')
    
    # Validate data
    if not title or not authors:
        flash('Title and authors are required', 'danger')
        return redirect(url_for('web.literature_search'))
    
    # Check if reference already exists
    existing_ref = None
    if doi:
        existing_ref = LiteratureReference.query.filter_by(doi=doi).first()
    elif pubmed_id:
        existing_ref = LiteratureReference.query.filter_by(pubmed_id=pubmed_id).first()
    
    if existing_ref:
        flash('This reference is already saved in your library', 'info')
        return redirect(url_for('web.view_reference', ref_id=existing_ref.id))
    
    # Create new reference
    reference = LiteratureReference(
        user_id=current_user.id,
        title=title,
        authors=authors,
        journal=journal,
        publication_date=datetime.strptime(publication_date, '%Y-%m-%d').date() if publication_date else None,
        doi=doi,
        pubmed_id=pubmed_id,
        abstract=abstract,
        is_favorite=True
    )
    
    db.session.add(reference)
    db.session.commit()
    
    flash('Reference saved to your library', 'success')
    return redirect(url_for('web.view_reference', ref_id=reference.id))

@web_bp.route('/reference/<int:ref_id>')
@login_required
def view_reference(ref_id):
    """View a specific literature reference."""
    reference = LiteratureReference.query.get_or_404(ref_id)
    
    # Check if user has access to this reference
    if reference.user_id != current_user.id:
        flash('You do not have permission to view this reference.', 'danger')
        return redirect(url_for('web.literature_search'))
    
    return render_template('reference_detail.html', reference=reference)

@web_bp.route('/documentation')
def documentation():
    """Documentation page."""
    return render_template('documentation.html')

@web_bp.route('/about')
def about():
    """About page."""
    return render_template('about.html')

def allowed_file(filename, allowed_extensions):
    """Check if a file has an allowed extension."""
    if '.' not in filename:
        return False
    
    import os
    ext = os.path.splitext(filename)[1].lower().replace('.', '')
    return ext in allowed_extensions