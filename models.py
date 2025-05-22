from datetime import datetime
import json
from sqlalchemy import JSON, ForeignKey, Column, Integer, String, Boolean, DateTime, Text, Float
from sqlalchemy.orm import relationship
from flask_login import UserMixin
from app import db

# User and authentication models
class User(UserMixin, db.Model):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    external_id = Column(String(64), unique=True, nullable=True)  # For OAuth providers
    email = Column(String(120), unique=True, nullable=True)
    first_name = Column(String(64), nullable=True)
    last_name = Column(String(64), nullable=True)
    profile_image_url = Column(String(255), nullable=True)
    
    # Basic user information
    username = Column(String(64), unique=True, nullable=True)
    password_hash = Column(String(256), nullable=True)  # For local authentication
    bio = Column(Text, nullable=True)
    organization = Column(String(120), nullable=True)
    
    # User settings
    notification_email = Column(Boolean, default=True)
    dark_mode = Column(Boolean, default=False)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    # Relationships
    membership = relationship('Membership', back_populates='user', uselist=False)
    analyses = relationship('Analysis', back_populates='user')
    saved_searches = relationship('SavedSearch', back_populates='user')
    literature_notes = relationship('LiteratureNote', back_populates='user')
    research_logs = relationship('ResearchLog', back_populates='user')
    
    @property
    def full_name(self):
        """Return user's full name or username if name not available."""
        if self.first_name and self.last_name:
            return f"{self.first_name} {self.last_name}"
        elif self.first_name:
            return self.first_name
        elif self.username:
            return self.username
        else:
            return "Anonymous User"
    
    @property
    def membership_level(self):
        """Return user's membership level or 'Free' if none exists."""
        if self.membership and self.membership.plan_name:
            return self.membership.plan_name
        return "Free"
    
    def __repr__(self):
        return f"<User {self.username or self.id}>"


class Membership(db.Model):
    __tablename__ = 'memberships'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False, unique=True)
    plan_name = Column(String(50), nullable=False)
    
    # Plan limits and features
    analysis_limit = Column(Integer, default=10)
    batch_limit = Column(Integer, default=0)
    advanced_features = Column(Boolean, default=False)
    literature_access = Column(Boolean, default=False)
    api_access = Column(Boolean, default=False)
    
    # Subscription details
    start_date = Column(DateTime, default=datetime.now)
    end_date = Column(DateTime, nullable=True)
    is_active = Column(Boolean, default=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    # Relationships
    user = relationship('User', back_populates='membership')
    
    @property
    def is_valid(self):
        """Check if membership is valid (active and not expired)."""
        if not self.is_active:
            return False
        if self.end_date and self.end_date < datetime.now():
            return False
        return True
    
    @property
    def days_remaining(self):
        """Return days remaining in subscription or None if no end date."""
        if not self.end_date:
            return None
        delta = self.end_date - datetime.now()
        return max(0, delta.days)
    
    def __repr__(self):
        return f"<Membership {self.plan_name} for User {self.user_id}>"


class OAuthToken(db.Model):
    __tablename__ = 'oauth_tokens'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    provider = Column(String(50), nullable=False)
    token = Column(String(255), nullable=False)
    refresh_token = Column(String(255), nullable=True)
    expires_at = Column(DateTime, nullable=True)
    browser_session_key = Column(String(255), nullable=False)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    # Relationships
    user = relationship('User')
    
    __table_args__ = (
        db.UniqueConstraint('user_id', 'browser_session_key', 'provider', name='uq_user_browser_session_key_provider'),
    )
    
    def __repr__(self):
        return f"<OAuthToken {self.provider} for User {self.user_id}>"


# Research and analysis models
class Analysis(db.Model):
    __tablename__ = 'analyses'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    analysis_type = Column(String(50), nullable=False)
    parameters = Column(JSON, nullable=True)
    results = Column(JSON, nullable=True)
    
    # Status tracking
    status = Column(String(20), default='pending')  # pending, processing, completed, failed
    error_message = Column(Text, nullable=True)
    
    # File references
    input_file = Column(String(255), nullable=True)
    output_file = Column(String(255), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.now)
    start_time = Column(DateTime, nullable=True)
    end_time = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    # Relationships
    user = relationship('User', back_populates='analyses')
    
    def __repr__(self):
        return f"<Analysis {self.name} ({self.status})>"
    
    @property
    def duration(self):
        """Return analysis duration in seconds or None if not completed."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None
    
    def to_dict(self):
        """Convert analysis to dictionary for API responses."""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'analysis_type': self.analysis_type,
            'status': self.status,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration': self.duration,
            'parameters': self.parameters,
            'results': self.results,
            'user_id': self.user_id
        }


class BatchJob(db.Model):
    __tablename__ = 'batch_jobs'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    
    # Job configuration
    input_file = Column(String(255), nullable=False)
    output_file = Column(String(255), nullable=True)
    parameters = Column(JSON, nullable=True)
    
    # Status tracking
    status = Column(String(20), default='pending')  # pending, processing, completed, failed
    error_message = Column(Text, nullable=True)
    total_records = Column(Integer, default=0)
    processed_records = Column(Integer, default=0)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.now)
    start_time = Column(DateTime, nullable=True)
    end_time = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    # Relationships
    user = relationship('User')
    
    def __repr__(self):
        return f"<BatchJob {self.name} ({self.status})>"
    
    @property
    def progress(self):
        """Return job progress as percentage."""
        if self.total_records > 0:
            return min(100, round((self.processed_records / self.total_records) * 100))
        return 0
    
    @property
    def duration(self):
        """Return job duration in seconds or None if not completed."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None


class ImageAnalysis(db.Model):
    __tablename__ = 'image_analyses'
    
    id = Column(Integer, primary_key=True)
    analysis_id = Column(Integer, ForeignKey('analyses.id'), nullable=False)
    
    # Image information
    image_path = Column(String(255), nullable=False)
    processed_image_path = Column(String(255), nullable=True)
    
    # Analysis options
    analyze_species = Column(Boolean, default=True)
    analyze_morphology = Column(Boolean, default=True)
    analyze_color = Column(Boolean, default=True)
    analyze_growth = Column(Boolean, default=True)
    
    # Results - Species identification
    primary_species = Column(String(100), nullable=True)
    primary_confidence = Column(Float, nullable=True)
    secondary_species = Column(String(100), nullable=True)
    secondary_confidence = Column(Float, nullable=True)
    
    # Results - Morphological measurements
    cap_diameter = Column(Float, nullable=True)
    stem_height = Column(Float, nullable=True)
    stem_width = Column(Float, nullable=True)
    cap_shape = Column(String(50), nullable=True)
    
    # Results - Growth stage
    growth_stage = Column(String(50), nullable=True)
    growth_progress = Column(Float, nullable=True)
    days_to_harvest = Column(Integer, nullable=True)
    
    # Additional data
    detailed_results = Column(JSON, nullable=True)
    notes = Column(Text, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    # Relationships
    analysis = relationship('Analysis')
    
    def __repr__(self):
        return f"<ImageAnalysis {self.id} for Analysis {self.analysis_id}>"


class LiteratureReference(db.Model):
    __tablename__ = 'literature_references'
    
    id = Column(Integer, primary_key=True)
    title = Column(String(255), nullable=False)
    authors = Column(String(255), nullable=True)
    journal = Column(String(100), nullable=True)
    year = Column(Integer, nullable=True)
    volume = Column(String(20), nullable=True)
    issue = Column(String(20), nullable=True)
    pages = Column(String(20), nullable=True)
    doi = Column(String(100), nullable=True, unique=True)
    abstract = Column(Text, nullable=True)
    url = Column(String(255), nullable=True)
    
    # Source of the reference
    source = Column(String(50), nullable=True)  # PubMed, Scopus, Science Direct, etc.
    external_id = Column(String(50), nullable=True)  # ID in the source database
    
    # Analysis related data
    compounds = Column(JSON, nullable=True)  # List of compounds mentioned
    species = Column(JSON, nullable=True)  # List of fungal species mentioned
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    # Relationships
    notes = relationship('LiteratureNote', back_populates='reference')
    
    def __repr__(self):
        return f"<LiteratureReference {self.title[:30]}... ({self.year})>"
    
    @property
    def citation(self):
        """Return formatted citation."""
        citation = f"{self.authors}. "
        citation += f"({self.year}). " if self.year else ""
        citation += f"{self.title}. "
        if self.journal:
            citation += f"{self.journal}"
            if self.volume:
                citation += f", {self.volume}"
                if self.issue:
                    citation += f"({self.issue})"
            if self.pages:
                citation += f", {self.pages}"
            citation += "."
        return citation


class LiteratureNote(db.Model):
    __tablename__ = 'literature_notes'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    reference_id = Column(Integer, ForeignKey('literature_references.id'), nullable=False)
    content = Column(Text, nullable=False)
    
    # Tags and categorization
    tags = Column(JSON, nullable=True)  # Array of tag strings
    importance = Column(Integer, nullable=True)  # User ranking 1-5
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    # Relationships
    user = relationship('User', back_populates='literature_notes')
    reference = relationship('LiteratureReference', back_populates='notes')
    
    def __repr__(self):
        return f"<LiteratureNote {self.id} by User {self.user_id}>"


class SavedSearch(db.Model):
    __tablename__ = 'saved_searches'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    name = Column(String(100), nullable=False)
    search_type = Column(String(50), nullable=False)  # literature, compounds, etc.
    query = Column(JSON, nullable=False)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.now)
    last_run_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    # Relationships
    user = relationship('User', back_populates='saved_searches')
    
    def __repr__(self):
        return f"<SavedSearch {self.name} by User {self.user_id}>"


class ResearchLog(db.Model):
    __tablename__ = 'research_logs'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(String, ForeignKey('users.id'), nullable=False)
    title = Column(String(100), nullable=False)
    content = Column(Text, nullable=False)
    category = Column(String(50), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    # Relationships
    user = relationship('User', back_populates='research_logs')
    
    def __repr__(self):
        return f"<ResearchLog {self.title[:30]}... by User {self.user_id}>"


class Version(db.Model):
    __tablename__ = 'versions'
    
    id = Column(Integer, primary_key=True)
    version = Column(String(20), nullable=False, unique=True)
    release_date = Column(DateTime, default=datetime.now)
    description = Column(Text, nullable=True)
    changelog = Column(JSON, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.now)
    
    def __repr__(self):
        return f"<Version {self.version}>"