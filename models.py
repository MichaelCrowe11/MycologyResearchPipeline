from datetime import datetime
from app import db
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, JSON, ForeignKey, Boolean
from sqlalchemy.orm import relationship


class Sample(db.Model):
    """Model representing a mycological sample."""
    __tablename__ = 'samples'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    species = Column(String(255), nullable=True)
    collection_date = Column(DateTime, default=datetime.utcnow)
    location = Column(String(255), nullable=True)
    sample_metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    compounds = relationship("Compound", back_populates="sample", cascade="all, delete-orphan")
    analyses = relationship("Analysis", back_populates="sample", cascade="all, delete-orphan")
    literature_references = relationship("LiteratureReference", back_populates="sample", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Sample {self.id}: {self.name}>"
        
    @property
    def metadata_dict(self):
        """Return sample_metadata as a Python dictionary for JSON serialization."""
        if self.sample_metadata is None:
            return {}
        # Convert SQLAlchemy JSON type to Python dict for serialization
        return self.sample_metadata


class Compound(db.Model):
    """Model representing a medicinal compound in a sample."""
    __tablename__ = 'compounds'
    
    id = Column(Integer, primary_key=True)
    sample_id = Column(Integer, ForeignKey('samples.id'), nullable=False)
    name = Column(String(255), nullable=False)
    formula = Column(String(255), nullable=True)
    molecular_weight = Column(Float, nullable=True)
    concentration = Column(Float, nullable=True)
    bioactivity_index = Column(Float, nullable=True)
    compound_metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    sample = relationship("Sample", back_populates="compounds")
    
    def __repr__(self):
        return f"<Compound {self.id}: {self.name}>"


class Analysis(db.Model):
    """Model representing an analysis performed on a sample."""
    __tablename__ = 'analyses'
    
    id = Column(Integer, primary_key=True)
    sample_id = Column(Integer, ForeignKey('samples.id'), nullable=False)
    analysis_type = Column(String(100), nullable=False)
    parameters = Column(JSON, nullable=True)
    results = Column(JSON, nullable=True)
    status = Column(String(50), default='pending')
    start_time = Column(DateTime, nullable=True)
    end_time = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    sample = relationship("Sample", back_populates="analyses")
    
    def __repr__(self):
        return f"<Analysis {self.id}: {self.analysis_type}>"


class BatchJob(db.Model):
    """Model representing a batch processing job."""
    __tablename__ = 'batch_jobs'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=True)
    description = Column(Text, nullable=True)
    input_file = Column(String(255), nullable=True)
    output_file = Column(String(255), nullable=True)
    parameters = Column(JSON, nullable=True)
    status = Column(String(50), default='pending')
    start_time = Column(DateTime, nullable=True)
    end_time = Column(DateTime, nullable=True)
    error_message = Column(Text, nullable=True)
    total_records = Column(Integer, default=0)
    processed_records = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<BatchJob {self.id}: {self.name}>"


class Version(db.Model):
    """Model for tracking pipeline versions."""
    __tablename__ = 'versions'
    
    id = Column(Integer, primary_key=True)
    version = Column(String(50), nullable=False, unique=True)
    description = Column(Text, nullable=True)
    is_current = Column(Boolean, default=False)
    release_date = Column(DateTime, default=datetime.utcnow)
    changelog = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<Version {self.version}>"


class ResearchLog(db.Model):
    """Model for research logs and notes."""
    __tablename__ = 'research_logs'
    
    id = Column(Integer, primary_key=True)
    title = Column(String(255), nullable=False)
    content = Column(Text, nullable=True)
    sample_id = Column(Integer, ForeignKey('samples.id'), nullable=True)
    analysis_id = Column(Integer, ForeignKey('analyses.id'), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<ResearchLog {self.id}: {self.title}>"


class LiteratureReference(db.Model):
    """Model representing a scientific literature reference."""
    __tablename__ = 'literature_references'
    
    id = Column(Integer, primary_key=True)
    sample_id = Column(Integer, ForeignKey('samples.id'), nullable=True)
    reference_id = Column(String(50), nullable=False)  # e.g., PubMed ID
    title = Column(String(500), nullable=False)
    authors = Column(Text, nullable=True)
    journal = Column(String(255), nullable=True)
    year = Column(Integer, nullable=True)
    url = Column(String(500), nullable=True)
    abstract = Column(Text, nullable=True)
    reference_type = Column(String(50), default='pubmed')  # pubmed, doi, etc.
    reference_metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    sample = relationship("Sample", back_populates="literature_references")
    
    def __repr__(self):
        return f"<LiteratureReference {self.id}: {self.title[:30]}...>"
