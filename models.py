"""
Database models for the Mycology Research Pipeline.
"""
from datetime import datetime
import json
from flask_login import UserMixin
from sqlalchemy.dialects.postgresql import JSONB
from app import db


class User(UserMixin, db.Model):
    """User model for authentication and access control."""
    __tablename__ = 'users'
    
    id = db.Column(db.String, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=True)
    first_name = db.Column(db.String(64), nullable=True)
    last_name = db.Column(db.String(64), nullable=True)
    profile_image_url = db.Column(db.String(255), nullable=True)
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.now)
    updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)
    
    # Relationships
    samples = db.relationship('Sample', back_populates='user', lazy='dynamic')
    analyses = db.relationship('Analysis', back_populates='user', lazy='dynamic')
    batch_jobs = db.relationship('BatchJob', back_populates='user', lazy='dynamic')
    literature_references = db.relationship('LiteratureReference', back_populates='user', lazy='dynamic')
    membership = db.relationship('Membership', back_populates='user', uselist=False)
    
    @property
    def full_name(self):
        """Return the user's full name."""
        if self.first_name and self.last_name:
            return f"{self.first_name} {self.last_name}"
        elif self.first_name:
            return self.first_name
        elif self.last_name:
            return self.last_name
        return "Anonymous User"


class Membership(db.Model):
    """User membership levels and subscription details."""
    __tablename__ = 'memberships'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String, db.ForeignKey('users.id'), nullable=False)
    tier = db.Column(db.String(20), nullable=False, default='free')  # free, basic, premium, enterprise
    
    # Subscription details
    start_date = db.Column(db.DateTime, default=datetime.now)
    end_date = db.Column(db.DateTime, nullable=True)
    is_active = db.Column(db.Boolean, default=True)
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.now)
    updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)
    
    # Relationships
    user = db.relationship('User', back_populates='membership')
    
    @property
    def is_premium(self):
        """Check if the user has premium features."""
        return self.tier in ['premium', 'enterprise'] and self.is_active
    
    @property
    def is_enterprise(self):
        """Check if the user has enterprise features."""
        return self.tier == 'enterprise' and self.is_active
    
    @property
    def max_samples(self):
        """Maximum number of samples based on membership tier."""
        limits = {
            'free': 5,
            'basic': 100,
            'premium': 1000,
            'enterprise': 10000
        }
        return limits.get(self.tier, 5)
    
    @property
    def max_batch_size(self):
        """Maximum batch job size based on membership tier."""
        limits = {
            'free': 5,
            'basic': 100,
            'premium': 1000,
            'enterprise': 10000
        }
        return limits.get(self.tier, 5)


class Sample(db.Model):
    """Sample data model for mycology specimens."""
    __tablename__ = 'samples'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String, db.ForeignKey('users.id'), nullable=False)
    name = db.Column(db.String(128), nullable=False)
    description = db.Column(db.Text, nullable=True)
    species = db.Column(db.String(128), nullable=True)
    collection_date = db.Column(db.DateTime, nullable=True)
    location = db.Column(db.String(255), nullable=True)
    sample_metadata = db.Column(JSONB, nullable=True)
    is_public = db.Column(db.Boolean, default=False)
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.now)
    updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)
    
    # Relationships
    user = db.relationship('User', back_populates='samples')
    analyses = db.relationship('Analysis', back_populates='sample', lazy='dynamic')
    
    def __repr__(self):
        return f"<Sample {self.name}>"
    
    @property
    def formatted_metadata(self):
        """Return formatted metadata as a dictionary."""
        if not self.sample_metadata:
            return {}
        
        if isinstance(self.sample_metadata, str):
            try:
                return json.loads(self.sample_metadata)
            except:
                return {}
        
        return self.sample_metadata


class Analysis(db.Model):
    """Analysis data model for processing results."""
    __tablename__ = 'analyses'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String, db.ForeignKey('users.id'), nullable=False)
    sample_id = db.Column(db.Integer, db.ForeignKey('samples.id'), nullable=True)
    name = db.Column(db.String(128), nullable=False)
    description = db.Column(db.Text, nullable=True)
    type = db.Column(db.String(50), nullable=False)  # image_analysis, bioactivity_prediction, etc.
    parameters = db.Column(JSONB, nullable=True)
    results = db.Column(JSONB, nullable=True)
    status = db.Column(db.String(20), default='pending')  # pending, processing, completed, failed
    error_message = db.Column(db.Text, nullable=True)
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.now)
    start_time = db.Column(db.DateTime, nullable=True)
    end_time = db.Column(db.DateTime, nullable=True)
    updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)
    
    # Relationships
    user = db.relationship('User', back_populates='analyses')
    sample = db.relationship('Sample', back_populates='analyses')
    
    def __repr__(self):
        return f"<Analysis {self.name} ({self.status})>"


class BatchJob(db.Model):
    """Batch processing job for multiple samples."""
    __tablename__ = 'batch_jobs'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String, db.ForeignKey('users.id'), nullable=False)
    name = db.Column(db.String(128), nullable=False)
    description = db.Column(db.Text, nullable=True)
    type = db.Column(db.String(50), nullable=False)  # image_analysis, bioactivity_prediction, etc.
    parameters = db.Column(JSONB, nullable=True)
    input_file = db.Column(db.String(255), nullable=True)
    output_file = db.Column(db.String(255), nullable=True)
    status = db.Column(db.String(20), default='pending')  # pending, processing, completed, failed
    error_message = db.Column(db.Text, nullable=True)
    total_records = db.Column(db.Integer, default=0)
    processed_records = db.Column(db.Integer, default=0)
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.now)
    start_time = db.Column(db.DateTime, nullable=True)
    end_time = db.Column(db.DateTime, nullable=True)
    updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)
    
    # Relationships
    user = db.relationship('User', back_populates='batch_jobs')
    
    def __repr__(self):
        return f"<BatchJob {self.name} ({self.status})>"
    
    @property
    def progress(self):
        """Calculate progress percentage."""
        if not self.total_records or self.total_records == 0:
            return 0
        return int((self.processed_records / self.total_records) * 100)


class LiteratureReference(db.Model):
    """Scientific literature reference data."""
    __tablename__ = 'literature_references'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String, db.ForeignKey('users.id'), nullable=False)
    title = db.Column(db.String(255), nullable=False)
    authors = db.Column(db.Text, nullable=True)
    journal = db.Column(db.String(255), nullable=True)
    publication_date = db.Column(db.Date, nullable=True)
    doi = db.Column(db.String(100), nullable=True, unique=True)
    pubmed_id = db.Column(db.String(20), nullable=True, unique=True)
    abstract = db.Column(db.Text, nullable=True)
    keywords = db.Column(db.Text, nullable=True)
    url = db.Column(db.String(255), nullable=True)
    citation_count = db.Column(db.Integer, default=0)
    is_favorite = db.Column(db.Boolean, default=False)
    notes = db.Column(db.Text, nullable=True)
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.now)
    updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)
    
    # Relationships
    user = db.relationship('User', back_populates='literature_references')
    
    def __repr__(self):
        return f"<LiteratureReference {self.title}>"


class ResearchLog(db.Model):
    """Research activity log for tracking work."""
    __tablename__ = 'research_logs'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String, db.ForeignKey('users.id'), nullable=False)
    title = db.Column(db.String(255), nullable=False)
    content = db.Column(db.Text, nullable=False)
    is_public = db.Column(db.Boolean, default=False)
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.now)
    updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)
    
    # Relationships
    user = db.relationship('User', backref=db.backref('research_logs', lazy='dynamic'))
    
    def __repr__(self):
        return f"<ResearchLog {self.title}>"


class Version(db.Model):
    """System version information."""
    __tablename__ = 'versions'
    
    id = db.Column(db.Integer, primary_key=True)
    version = db.Column(db.String(20), nullable=False)
    release_date = db.Column(db.DateTime, default=datetime.now)
    description = db.Column(db.Text, nullable=True)
    is_current = db.Column(db.Boolean, default=False)
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.now)
    
    def __repr__(self):
        return f"<Version {self.version}>"