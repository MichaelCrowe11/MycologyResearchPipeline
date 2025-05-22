from datetime import datetime
import os
import logging
from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify, current_app, send_file
from flask_login import current_user, login_required
from werkzeug.utils import secure_filename
import json

from app import db
from models import (
    User, Analysis, BatchJob, ImageAnalysis, 
    LiteratureReference, LiteratureNote, SavedSearch,
    ResearchLog, Version
)

# Configure logging
logger = logging.getLogger(__name__)

# Create Blueprint
web_bp = Blueprint('web', __name__)

# Helper functions
def get_current_version():
    """Get the current version of the application."""
    version = Version.query.order_by(Version.id.desc()).first()
    if not version:
        # Create initial version if none exists
        version = Version(
            version="1.0.0",
            release_date=datetime.now(),
            description="Initial release of the Mycology Research Pipeline",
            changelog=json.dumps([
                "Initial application structure",
                "User authentication system",
                "Image analysis functionality",
                "Literature search capabilities"
            ])
        )
        db.session.add(version)
        db.session.commit()
    return version


# Route handlers
@web_bp.route('/')
def index():
    """Render the homepage."""
    current_version = get_current_version()
    return render_template('index.html', current_version=current_version)


@web_bp.route('/dashboard')
@login_required
def dashboard():
    """Render the user dashboard."""
    # Get user's recent activities
    recent_analyses = Analysis.query.filter_by(user_id=current_user.id) \
        .order_by(Analysis.created_at.desc()).limit(5).all()
    
    recent_literature = LiteratureNote.query.filter_by(user_id=current_user.id) \
        .order_by(LiteratureNote.updated_at.desc()).limit(5).all()
    
    batch_jobs = BatchJob.query.filter_by(user_id=current_user.id) \
        .order_by(BatchJob.created_at.desc()).limit(3).all()
    
    return render_template(
        'dashboard.html',
        recent_analyses=recent_analyses,
        recent_literature=recent_literature,
        batch_jobs=batch_jobs
    )


@web_bp.route('/image-analysis', methods=['GET', 'POST'])
@login_required
def image_analysis():
    """Handle image analysis functionality."""
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'image' not in request.files:
            flash('No file part', 'danger')
            return redirect(request.url)
        
        file = request.files['image']
        
        # If user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file', 'danger')
            return redirect(request.url)
        
        if file:
            # Save the file
            filename = secure_filename(file.filename)
            upload_dir = os.path.join(current_app.root_path, 'uploads')
            os.makedirs(upload_dir, exist_ok=True)
            
            file_path = os.path.join(upload_dir, filename)
            file.save(file_path)
            
            # Create analysis record
            analysis = Analysis(
                user_id=current_user.id,
                name=f"Image Analysis: {filename}",
                description="Automatic image analysis of mushroom specimen",
                analysis_type="image",
                parameters={
                    "identify_species": request.form.get('identify_species') == 'on',
                    "analyze_morphology": request.form.get('analyze_morphology') == 'on',
                    "analyze_color": request.form.get('analyze_color') == 'on',
                    "analyze_growth": request.form.get('analyze_growth') == 'on',
                    "notes": request.form.get('notes', '')
                },
                input_file=file_path,
                status="processing",
                start_time=datetime.now()
            )
            
            db.session.add(analysis)
            db.session.commit()
            
            # Process the image (this would typically be a background job)
            # For now, we'll simulate success
            
            # Update with "results"
            analysis.status = "completed"
            analysis.results = {
                "primary_species": "Amanita muscaria",
                "primary_confidence": 0.95,
                "secondary_species": "Amanita pantherina",
                "secondary_confidence": 0.72,
                "morphology": {
                    "cap_diameter": 10.2,
                    "stem_height": 12.5,
                    "stem_width": 2.1,
                    "cap_shape": "Convex"
                },
                "color_analysis": {
                    "cap_colors": ["#B22222", "#CD5C5C", "#F5F5F5"],
                    "stem_colors": ["#F5F5F5", "#E8E8E8", "#D3D3D3"]
                },
                "growth_stage": {
                    "stage": "Mature",
                    "progress": 0.85,
                    "days_to_harvest": 0
                }
            }
            analysis.end_time = datetime.now()
            
            # Create image analysis details
            image_analysis = ImageAnalysis(
                analysis_id=analysis.id,
                image_path=file_path,
                processed_image_path=file_path,  # In real app, this would be different
                analyze_species=request.form.get('identify_species') == 'on',
                analyze_morphology=request.form.get('analyze_morphology') == 'on',
                analyze_color=request.form.get('analyze_color') == 'on',
                analyze_growth=request.form.get('analyze_growth') == 'on',
                primary_species="Amanita muscaria",
                primary_confidence=0.95,
                secondary_species="Amanita pantherina",
                secondary_confidence=0.72,
                cap_diameter=10.2,
                stem_height=12.5,
                stem_width=2.1,
                cap_shape="Convex",
                growth_stage="Mature",
                growth_progress=0.85,
                days_to_harvest=0,
                detailed_results=analysis.results,
                notes=request.form.get('notes', '')
            )
            
            db.session.add(image_analysis)
            db.session.commit()
            
            flash('Image analysis completed successfully!', 'success')
            return redirect(url_for('web.view_analysis', analysis_id=analysis.id))
    
    return render_template('image_analysis.html')


@web_bp.route('/analysis/<int:analysis_id>')
@login_required
def view_analysis(analysis_id):
    """View a specific analysis."""
    analysis = Analysis.query.get_or_404(analysis_id)
    
    # Make sure the user can only see their own analyses
    if analysis.user_id != current_user.id:
        flash('You do not have permission to view this analysis.', 'danger')
        return redirect(url_for('web.dashboard'))
    
    # Get associated image analysis if applicable
    image_analysis = None
    if analysis.analysis_type == 'image':
        image_analysis = ImageAnalysis.query.filter_by(analysis_id=analysis.id).first()
    
    return render_template(
        'view_analysis.html',
        analysis=analysis,
        image_analysis=image_analysis
    )


@web_bp.route('/batch-jobs')
@login_required
def batch_jobs():
    """Show user's batch jobs."""
    jobs = BatchJob.query.filter_by(user_id=current_user.id) \
        .order_by(BatchJob.created_at.desc()).all()
    
    return render_template('batch_jobs.html', jobs=jobs)


@web_bp.route('/batch-jobs/<int:job_id>')
@login_required
def view_batch_job(job_id):
    """View a specific batch job."""
    job = BatchJob.query.get_or_404(job_id)
    
    # Make sure the user can only see their own jobs
    if job.user_id != current_user.id:
        flash('You do not have permission to view this job.', 'danger')
        return redirect(url_for('web.batch_jobs'))
    
    return render_template('view_batch_job.html', job=job)


@web_bp.route('/literature-search')
@login_required
def literature_search():
    """Literature search functionality."""
    # Get any saved searches
    saved_searches = SavedSearch.query.filter_by(
        user_id=current_user.id,
        search_type='literature'
    ).all()
    
    return render_template('literature_search.html', saved_searches=saved_searches)


@web_bp.route('/literature-reference/<int:reference_id>')
@login_required
def view_literature_reference(reference_id):
    """View a specific literature reference."""
    reference = LiteratureReference.query.get_or_404(reference_id)
    
    # Get user's notes for this reference
    notes = LiteratureNote.query.filter_by(
        user_id=current_user.id,
        reference_id=reference.id
    ).all()
    
    return render_template(
        'view_literature.html',
        reference=reference,
        notes=notes
    )


@web_bp.route('/saved-searches')
@login_required
def saved_searches():
    """Show user's saved searches."""
    searches = SavedSearch.query.filter_by(user_id=current_user.id) \
        .order_by(SavedSearch.created_at.desc()).all()
    
    return render_template('saved_searches.html', searches=searches)


@web_bp.route('/research-log')
@login_required
def research_log():
    """Show user's research log."""
    logs = ResearchLog.query.filter_by(user_id=current_user.id) \
        .order_by(ResearchLog.created_at.desc()).all()
    
    return render_template('research_log.html', logs=logs)


@web_bp.route('/export-references')
@login_required
def export_references():
    """Export literature references."""
    # Get user's saved references
    references = LiteratureReference.query.join(LiteratureNote).filter(
        LiteratureNote.user_id == current_user.id
    ).distinct().all()
    
    # Format options for template rendering
    return render_template('export_references.html', references=references)


@web_bp.route('/documentation')
def documentation():
    """Application documentation."""
    current_version = get_current_version()
    return render_template('documentation.html', current_version=current_version)