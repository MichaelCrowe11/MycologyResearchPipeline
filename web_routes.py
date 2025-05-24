import os
import json
import logging
from datetime import datetime, timedelta
from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify, send_file, current_app, session
from werkzeug.utils import secure_filename

from app import db
from models import Sample, Compound, Analysis, BatchJob, Version, ResearchLog, LiteratureReference
from literature import initialize_entrez, fetch_pubmed_articles, fetch_species_literature, update_sample_literature
from model import load_model
from batch_processor import process_batch
from enhanced_identification import identify_dried_specimen
from data_validation import (
    validate_all, validate_all_samples, validate_all_compounds,
    validate_all_literature_references, validate_database_integrity,
    export_validation_report
)

logger = logging.getLogger(__name__)
web_bp = Blueprint('web', __name__)


@web_bp.route('/')
def splash():
    """Render the splash page."""
    return render_template('splash.html')


@web_bp.route('/home')
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


@web_bp.route('/samples/new', methods=['GET', 'POST'])
def new_sample():
    """Create a new sample."""
    if request.method == 'POST':
        name = request.form.get('name')
        species = request.form.get('species')
        location = request.form.get('location')
        description = request.form.get('description')
        
        if not name:
            flash('Sample name is required', 'error')
            return render_template('sample_form.html')
        
        try:
            sample = Sample(
                name=name,
                species=species,
                location=location,
                description=description,
                created_at=datetime.now()
            )
            db.session.add(sample)
            db.session.commit()
            
            flash('Sample created successfully', 'success')
            return redirect(url_for('web.view_sample', sample_id=sample.id))
            
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error creating sample: {str(e)}")
            flash(f'Error creating sample: {str(e)}', 'error')
    
    return render_template('sample_form.html')


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


@web_bp.route('/api-testing')
def api_testing():
    """Render the API testing guide."""
    return render_template('api_testing.html')


@web_bp.route('/logs/<int:log_id>')
def view_research_log(log_id):
    """View a specific research log."""
    log = ResearchLog.query.get_or_404(log_id)
    return render_template('log_details.html', log=log)


@web_bp.route('/literature')
def literature():
    """View scientific literature references."""
    # Get filter parameters
    species_filter = request.args.get('species')
    year_filter = request.args.get('year')
    search_query = request.args.get('search')
    
    # Build query
    query = LiteratureReference.query
    
    if species_filter:
        # Join with samples to filter by species
        samples = Sample.query.filter_by(species=species_filter).all()
        sample_ids = [s.id for s in samples]
        query = query.filter(LiteratureReference.sample_id.in_(sample_ids))
    
    if year_filter:
        query = query.filter_by(year=int(year_filter))
    
    if search_query:
        search_term = f"%{search_query}%"
        query = query.filter(
            db.or_(
                LiteratureReference.title.ilike(search_term),
                LiteratureReference.authors.ilike(search_term),
                LiteratureReference.journal.ilike(search_term),
                LiteratureReference.abstract.ilike(search_term)
            )
        )
    
    # Get results
    references = query.order_by(LiteratureReference.created_at.desc()).all()
    
    # Get unique species for the filter dropdown
    species_list = db.session.query(Sample.species).distinct().filter(Sample.species.isnot(None)).all()
    species_list = [s[0] for s in species_list]
    
    # Get unique years for the filter dropdown
    year_list = db.session.query(LiteratureReference.year).distinct().filter(
        LiteratureReference.year.isnot(None)
    ).order_by(LiteratureReference.year.desc()).all()
    year_list = [y[0] for y in year_list]
    
    # Get all samples for the custom search form
    samples = Sample.query.all()
    
    return render_template(
        'literature.html',
        references=references,
        species_list=species_list,
        year_list=year_list,
        samples=samples
    )


@web_bp.route('/literature/fetch', methods=['GET'])
def fetch_literature():
    """Fetch recent literature for existing samples."""
    # Initialize Entrez with email
    initialize_entrez()
    
    # Get all samples with species information
    samples_with_species = Sample.query.filter(Sample.species.isnot(None)).all()
    
    references_added = 0
    for sample in samples_with_species:
        # Fetch literature for the sample
        articles = fetch_species_literature(sample.species, max_results=3)
        
        for article in articles:
            # Check if reference already exists
            existing = LiteratureReference.query.filter_by(
                reference_id=article["pmid"],
                reference_type="pubmed"
            ).first()
            
            if not existing:
                # Create new reference
                ref = LiteratureReference(
                    sample_id=sample.id,
                    reference_id=article["pmid"],
                    title=article["title"],
                    authors=article["authors"],
                    journal=article["journal"],
                    year=article["year"],
                    url=article["url"],
                    abstract=article.get("abstract"),
                    reference_type="pubmed",
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow()
                )
                db.session.add(ref)
                references_added += 1
    
    db.session.commit()
    
    flash(f"Successfully fetched {references_added} new literature references", "success")
    return redirect(url_for('web.literature'))


@web_bp.route('/literature/custom-search', methods=['POST'])
def custom_literature_search():
    """Perform a custom literature search."""
    # Get form data
    custom_query = request.form.get('custom_query')
    max_results = int(request.form.get('max_results', 5))
    associate_with = request.form.get('associate_with')
    
    if not custom_query:
        flash("Please provide a search query", "warning")
        return redirect(url_for('web.literature'))
    
    # Initialize Entrez with email
    initialize_entrez()
    
    # Fetch articles
    articles = fetch_pubmed_articles(custom_query, max_results=max_results)
    
    references_added = 0
    for article in articles:
        # Check if reference already exists
        existing = LiteratureReference.query.filter_by(
            reference_id=article["pmid"],
            reference_type="pubmed"
        ).first()
        
        if existing:
            # Update if associating with a sample
            if associate_with and not existing.sample_id:
                existing.sample_id = associate_with
                db.session.add(existing)
                references_added += 1
        else:
            # Create new reference
            ref = LiteratureReference(
                sample_id=associate_with if associate_with else None,
                reference_id=article["pmid"],
                title=article["title"],
                authors=article["authors"],
                journal=article["journal"],
                year=article["year"],
                url=article["url"],
                abstract=article.get("abstract"),
                reference_type="pubmed",
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            db.session.add(ref)
            references_added += 1
    
    db.session.commit()
    
    flash(f"Successfully found {len(articles)} articles and added {references_added} new references", "success")
    return redirect(url_for('web.literature'))


@web_bp.route('/literature/<int:reference_id>')
def view_reference(reference_id):
    """View a specific literature reference."""
    reference = LiteratureReference.query.get_or_404(reference_id)
    
    # Find similar references (same species or same journal)
    similar_references = []
    if reference.sample_id:
        # Get references for the same sample
        similar_references = LiteratureReference.query.filter(
            LiteratureReference.sample_id == reference.sample_id,
            LiteratureReference.id != reference.id
        ).limit(5).all()
    
    # If no similar references by sample, try by journal
    if not similar_references and reference.journal:
        similar_references = LiteratureReference.query.filter(
            LiteratureReference.journal == reference.journal,
            LiteratureReference.id != reference.id
        ).limit(5).all()
    
    return render_template(
        'reference_details.html',
        reference=reference,
        similar_references=similar_references
    )


@web_bp.route('/samples/<int:sample_id>/update-literature')
def update_sample_literature_route(sample_id):
    """Update literature for a specific sample."""
    sample = Sample.query.get_or_404(sample_id)
    
    if not sample.species:
        flash("This sample does not have species information", "warning")
        return redirect(url_for('web.view_sample', sample_id=sample_id))
    
    # Initialize Entrez
    initialize_entrez()
    
    # Update literature
    references = update_sample_literature(db, sample_id)
    
    flash(f"Successfully updated literature references: {len(references)} references found", "success")
    return redirect(url_for('web.view_sample', sample_id=sample_id))


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


@web_bp.route('/documentation/pubmed')
def pubmed_documentation():
    """PubMed integration documentation page."""
    return render_template('documentation/pubmed_integration.html')


@web_bp.route('/documentation/cmid')
def cmid_documentation():
    """CMID Research Kit documentation page."""
    return render_template('documentation/cmid_integration.html')


@web_bp.route('/validation')
def validation():
    """Data validation page."""
    return render_template('validation.html')


@web_bp.route('/validation/run', methods=['POST'])
def run_validation():
    """Run data validation based on form parameters."""
    # Get form parameters
    validate_samples_flag = 'validate_samples' in request.form
    validate_compounds_flag = 'validate_compounds' in request.form
    validate_literature_flag = 'validate_literature' in request.form
    validate_integrity_flag = 'validate_integrity' in request.form
    report_format = request.form.get('report_format', 'html')
    include_suggestions = 'include_suggestions' in request.form
    
    # Initialize validation result
    result = None
    
    # Run selected validations
    if validate_samples_flag and validate_compounds_flag and validate_literature_flag and validate_integrity_flag:
        # Run all validations
        result = validate_all()
    else:
        # Run specific validations
        validation_results = []
        
        if validate_samples_flag:
            validation_results.append(validate_all_samples())
        
        if validate_compounds_flag:
            validation_results.append(validate_all_compounds())
        
        if validate_literature_flag:
            validation_results.append(validate_all_literature_references())
        
        if validate_integrity_flag:
            validation_results.append(validate_database_integrity())
        
        # Combine results
        if validation_results:
            result = validation_results[0]
            for r in validation_results[1:]:
                result.errors.extend(r.errors)
                result.warnings.extend(r.warnings)
                result.suggestions.extend(r.suggestions)
    
    if result:
        # Generate report file
        timestamp = datetime.utcnow().strftime('%Y%m%d%H%M%S')
        report_path = os.path.join(current_app.config['RESULTS_FOLDER'], f"validation_report_{timestamp}.{report_format}")
        export_validation_report(result, format=report_format, output_file=report_path)
        
        # Store report path in session for download
        session['validation_report_path'] = report_path
        
        # Convert to dictionary for template
        result_dict = result.as_dict()
        
        return render_template(
            'validation.html',
            result=result_dict,
            include_suggestions=include_suggestions
        )
    else:
        flash('No validation options were selected', 'warning')
        return redirect(url_for('web.validation'))


@web_bp.route('/validation/download')
def download_validation_report():
    """Download the most recent validation report."""
    report_path = session.get('validation_report_path')
    
    if not report_path or not os.path.exists(report_path):
        flash('Validation report not found', 'error')
        return redirect(url_for('web.validation'))
    
    report_filename = os.path.basename(report_path)
    return send_file(report_path, as_attachment=True, download_name=report_filename)


@web_bp.route('/molecular-viewer')
def molecular_viewer():
    """3D Molecular Visualization."""
    return render_template('molecular_viewer.html')


@web_bp.route('/network-visualization')
def network_visualization():
    """Network Visualization for compound relationships."""
    return render_template('network_visualization.html')


@web_bp.route('/literature-search')
def literature_search():
    """Structured literature search across multiple databases."""
    return render_template('literature_search.html')


@web_bp.route('/literature-search/run', methods=['POST'])
def run_literature_search():
    """Run a structured literature search based on form parameters."""
    from scheduled_literature_search import StructuredLiteratureSearch, ResearchPaper
    
    # Get form parameters
    search_type = request.form.get('search_type', 'quick')
    species = request.form.getlist('species')
    compounds = request.form.getlist('compounds')
    biological_effects = request.form.getlist('biological_effects')
    date_range = request.form.get('date_range', 'last_5_years')
    databases = request.form.getlist('databases')
    max_results = int(request.form.get('max_results', 50))
    output_format = request.form.get('output_format', 'csv')
    additional_terms = request.form.get('additional_terms', '')
    
    # Process additional terms
    if additional_terms:
        additional_terms = [term.strip() for term in additional_terms.split(',')]
    
    # Convert date range to actual dates
    today = datetime.now()
    end_date = today.strftime("%Y/%m/%d")
    
    if date_range == 'last_year':
        start_date = (today - timedelta(days=365)).strftime("%Y/%m/%d")
    elif date_range == 'last_5_years':
        start_date = (today - timedelta(days=5*365)).strftime("%Y/%m/%d")
    elif date_range == 'last_10_years':
        start_date = (today - timedelta(days=10*365)).strftime("%Y/%m/%d")
    elif date_range == 'last_20_years':
        start_date = (today - timedelta(days=20*365)).strftime("%Y/%m/%d")
    else:  # all_time
        start_date = None
    
    # Build search queries
    if search_type == 'quick':
        # Create a focused query combining species, compounds, and effects
        queries = []
        for s in species:
            for c in compounds:
                for e in biological_effects:
                    query = f'"{s}" AND "{c}" AND "{e}"'
                    queries.append(query)
        
        # Limit to a reasonable number of queries for quick search
        queries = queries[:3]
    else:
        # For comprehensive search, let the library handle query generation
        # This will be handled by the backend service
        queries = ["Comprehensive search - queries generated by backend"]
    
    # Initialize searchers based on selected databases
    use_pubmed = 'pubmed' in databases
    use_scopus = 'scopus' in databases
    use_sciencedirect = 'sciencedirect' in databases
    use_webofscience = 'webofscience' in databases
    use_googlescholar = 'googlescholar' in databases
    
    # Create the searcher
    searcher = StructuredLiteratureSearch(
        use_pubmed=use_pubmed,
        use_scopus=use_scopus,
        use_sciencedirect=use_sciencedirect,
        use_webofscience=use_webofscience,
        use_googlescholar=use_googlescholar
    )
    
    # Run searches
    all_papers = []
    
    try:
        for query in queries:
            papers = searcher.search(
                query=query,
                max_results_per_db=max_results,
                start_date=start_date,
                end_date=end_date,
                deduplicate=False  # We'll deduplicate after all queries
            )
            all_papers.extend(papers)
        
        # Deduplicate results
        unique_papers = searcher._deduplicate_papers(all_papers)
        
        # Save results to file
        result_filename = None
        if unique_papers:
            output_file = searcher.save_results(unique_papers, output_format=output_format)
            result_filename = os.path.basename(output_file)
            
            # Flash success message
            flash(f"Search completed successfully! Found {len(unique_papers)} relevant papers.", "success")
        else:
            flash("No papers found matching your search criteria. Try broadening your search.", "warning")
        
        # Convert papers to dictionaries for the template
        papers_dict = [paper.to_dict() for paper in unique_papers]
        
        return render_template(
            'literature_search.html',
            search_results=papers_dict,
            result_filename=result_filename
        )
    
    except Exception as e:
        logger.error(f"Error in literature search: {str(e)}")
        flash(f"An error occurred during the search: {str(e)}", "error")
        return redirect(url_for('web.literature_search'))


@web_bp.route('/literature-search/download/<filename>')
def download_search_results(filename):
    """Download search results file."""
    results_dir = os.path.join(current_app.config['RESULTS_FOLDER'], 'literature_searches')
    return send_file(os.path.join(results_dir, filename), as_attachment=True)


@web_bp.route('/scheduled-searches')
def scheduled_searches():
    """Manage scheduled literature searches."""
    # This would be implemented in a future version
    # For MVP we'll just redirect to the main search page with a message
    flash("Scheduled searches will be available in a future update.", "info")
    return redirect(url_for('web.literature_search'))


@web_bp.route('/image-analysis')
def image_analysis():
    """Computer vision analysis for mushroom samples."""
    return render_template('image_analysis.html')


@web_bp.route('/process-image', methods=['POST'])
def process_image():
    """Process uploaded image with computer vision module."""
    import computer_vision
    import os
    import uuid
    import json
    import numpy as np
    from werkzeug.utils import secure_filename
    
    # Check if file was uploaded
    if 'image_file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['image_file']
    
    # Check if filename is empty
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Check file extension
    allowed_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    file_ext = os.path.splitext(file.filename)[1].lower()
    
    if file_ext not in allowed_extensions:
        return jsonify({'error': f'File type not allowed. Supported formats: {", ".join(allowed_extensions)}'}), 400
    
    # Get analysis options
    analyze_options = request.form.getlist('analyze_options')
    analyze_species = 'species' in analyze_options
    analyze_morphology = 'morphology' in analyze_options
    analyze_color = 'color' in analyze_options
    analyze_growth = 'growth' in analyze_options
    
    # Create unique filename
    unique_filename = str(uuid.uuid4()) + file_ext
    sample_name = request.form.get('sample_name', 'Unnamed sample')
    
    # Create upload directory if it doesn't exist
    upload_folder = current_app.config['UPLOAD_FOLDER']
    os.makedirs(upload_folder, exist_ok=True)
    
    # Create directory for results
    results_folder = os.path.join(current_app.config['RESULTS_FOLDER'], 'vision_analysis')
    os.makedirs(results_folder, exist_ok=True)
    
    # Save the file
    filepath = os.path.join(upload_folder, unique_filename)
    file.save(filepath)
    
    # Create directory for this analysis
    analysis_id = str(uuid.uuid4())
    analysis_dir = os.path.join(results_folder, analysis_id)
    os.makedirs(analysis_dir, exist_ok=True)
    
    try:
        # Process the image
        results = computer_vision.process_sample_image(
            image_path=filepath,
            output_dir=analysis_dir,
            analyze_species=analyze_species,
            analyze_morphology=analyze_morphology,
            analyze_color=analyze_color,
            analyze_growth=analyze_growth
        )
        
        # Add sample metadata
        results['sample_name'] = sample_name
        results['description'] = request.form.get('sample_description', '')
        results['analysis_id'] = analysis_id
        
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy_types(obj):
            """Convert numpy types to Python types for JSON serialization."""
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            return obj
        
        # Convert results before saving
        results_clean = convert_numpy_types(results)
        
        # Save results to a JSON file
        result_file = os.path.join(analysis_dir, 'analysis_results.json')
        with open(result_file, 'w') as f:
            json.dump(results_clean, f, indent=2)
        
        # Convert image paths to URLs
        if 'species_output_image' in results:
            results['species_output_image'] = url_for('web.get_analysis_image', 
                                                      analysis_id=analysis_id, 
                                                      filename=os.path.basename(results['species_output_image']))
        
        if 'morphology_output_image' in results:
            results['morphology_output_image'] = url_for('web.get_analysis_image', 
                                                        analysis_id=analysis_id, 
                                                        filename=os.path.basename(results['morphology_output_image']))
        
        # Convert results for JSON response (apply same conversion)
        results_for_response = convert_numpy_types(results)
        return jsonify(results_for_response)
    
    except Exception as e:
        current_app.logger.error(f"Error processing image: {str(e)}")
        return jsonify({'error': str(e)}), 500


@web_bp.route('/analysis-image/<analysis_id>/<filename>')
def get_analysis_image(analysis_id, filename):
    """Get image from analysis results."""
    results_dir = os.path.join(current_app.config['RESULTS_FOLDER'], 'vision_analysis', analysis_id)
    return send_file(os.path.join(results_dir, filename))


@web_bp.route('/prediction-dashboard')
def prediction_dashboard():
    """Compound bioactivity prediction dashboard."""
    return render_template('prediction_dashboard.html')


@web_bp.route('/ml-prediction', methods=['POST'])
def ml_prediction():
    """Run machine learning prediction for compound bioactivity."""
    from ml_bioactivity import predict_compounds
    
    # Get form data
    sample_id = request.form.get('sample_id', 'SAMPLE-001')
    species = request.form.get('species', 'Hericium erinaceus')
    
    # Parse analysis methods
    include_vision = 'cv_analysis' in request.form
    include_spectral = 'spectral_analysis' in request.form
    
    try:
        # Run the ML prediction
        results = predict_compounds(
            sample_id=sample_id,
            species=species,
            include_vision=include_vision,
            include_spectral=include_spectral
        )
        
        # Log the prediction
        logger.info(f"Bioactivity prediction completed for sample {sample_id}, species {species}")
        
        return jsonify(results)
    
    except Exception as e:
        logger.error(f"Error in bioactivity prediction: {str(e)}")
        return jsonify({
            'error': str(e),
            'status': 'error',
            'message': 'An error occurred during bioactivity prediction'
        }), 500
