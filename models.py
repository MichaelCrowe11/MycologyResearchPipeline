from datetime import datetime
from app import db
from flask_login import UserMixin
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, JSON, ForeignKey, Boolean, Enum
from sqlalchemy.orm import relationship
import enum


class MembershipTier(enum.Enum):
    """Enumeration for membership tiers."""
    FREE = "free"
    BASIC = "basic"
    PRO = "pro"
    ENTERPRISE = "enterprise"


class SubscriptionStatus(enum.Enum):
    """Enumeration for subscription status."""
    ACTIVE = "active"
    CANCELLED = "cancelled"
    PAST_DUE = "past_due"
    INCOMPLETE = "incomplete"


class Sample(db.Model):
    """Model representing a mycological sample."""
    __tablename__ = 'samples'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    species = Column(String(255), nullable=True)
    collection_date = Column(DateTime, default=datetime.utcnow)
    location = Column(String(255), nullable=True)
    sample_metadata = Column(JSON, nullable=True)
    is_public = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="samples")
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


class User(UserMixin, db.Model):
    """Model representing a user of the platform."""
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    email = Column(String(255), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    first_name = Column(String(100), nullable=True)
    last_name = Column(String(100), nullable=True)
    organization = Column(String(255), nullable=True)
    role = Column(String(50), default='user')  # user, admin
    is_active = Column(Boolean, default=True)
    last_login = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    memberships = relationship("Membership", back_populates="user", cascade="all, delete-orphan")
    samples = relationship("Sample", back_populates="user")
    api_tokens = relationship("OAuthToken", back_populates="user", cascade="all, delete-orphan")
    ai_queries = relationship("AIAssistantQuery", back_populates="user", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<User {self.id}: {self.email}>"
    
    @property
    def full_name(self):
        """Return the user's full name."""
        if self.first_name and self.last_name:
            return f"{self.first_name} {self.last_name}"
        elif self.first_name:
            return self.first_name
        elif self.last_name:
            return self.last_name
        return "Unnamed User"
    
    @property
    def active_membership(self):
        """Return the user's active membership."""
        return Membership.query.filter_by(
            user_id=self.id, is_active=True
        ).first()


class Membership(db.Model):
    """Model representing a user's membership plan."""
    __tablename__ = 'memberships'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    plan = Column(String(50), nullable=False)  # free, basic, pro, enterprise
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=True)
    is_active = Column(Boolean, default=True)
    features_access = Column(JSON, nullable=True)  # JSON with feature access flags
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="memberships")
    
    def __repr__(self):
        return f"<Membership {self.id}: {self.plan} for User {self.user_id}>"
    
    @property
    def is_valid(self):
        """Check if the membership is valid (active and not expired)."""
        if not self.is_active:
            return False
        if self.end_date and self.end_date < datetime.utcnow():
            return False
        return True


class Subscription(db.Model):
    """Model representing a Stripe subscription."""
    __tablename__ = 'subscriptions'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    stripe_subscription_id = Column(String(255), unique=True, nullable=False)
    stripe_customer_id = Column(String(255), nullable=True)
    tier = Column(Enum(MembershipTier), nullable=False, default=MembershipTier.FREE)
    status = Column(Enum(SubscriptionStatus), nullable=False, default=SubscriptionStatus.ACTIVE)
    current_period_start = Column(DateTime, nullable=True)
    current_period_end = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", backref="subscriptions")
    
    def __repr__(self):
        return f"<Subscription {self.id}: {self.tier.value} for User {self.user_id}>"


class Payment(db.Model):
    """Model representing payment transactions."""
    __tablename__ = 'payments'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    subscription_id = Column(Integer, ForeignKey('subscriptions.id'), nullable=True)
    stripe_payment_intent_id = Column(String(255), unique=True, nullable=False)
    amount = Column(Float, nullable=False)
    currency = Column(String(3), default='usd')
    status = Column(String(50), nullable=False)  # succeeded, pending, failed
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User", backref="payments")
    subscription = relationship("Subscription", backref="payments")
    
    def __repr__(self):
        return f"<Payment {self.id}: ${self.amount} for User {self.user_id}>"
    
    @property
    def days_remaining(self):
        """Return the number of days remaining in the membership."""
        if not self.end_date:
            return None
        
        delta = self.end_date - datetime.utcnow()
        return max(0, delta.days)


class OAuthToken(db.Model):
    """Model representing an OAuth token for API access."""
    __tablename__ = 'oauth_tokens'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    token = Column(String(255), unique=True, nullable=False)
    name = Column(String(100), nullable=True)
    scopes = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True)
    revoked_at = Column(DateTime, nullable=True)
    is_active = Column(Boolean, default=True)
    
    # Relationships
    user = relationship("User", back_populates="api_tokens")
    
    def __repr__(self):
        return f"<OAuthToken {self.id}: for User {self.user_id}>"
    
    @property
    def is_valid(self):
        """Check if the token is valid (active and not expired)."""
        if not self.is_active or self.revoked_at is not None:
            return False
        
        if self.expires_at and self.expires_at < datetime.utcnow():
            return False
        
        return True


class AIAssistantQuery(db.Model):
    """Model representing a query to the AI assistant."""
    __tablename__ = 'ai_assistant_queries'
    
    id = Column(Integer, primary_key=True)
    query_type = Column(String(50), nullable=False)  # sample_analysis, hypothesis_generation, etc.
    input_data = Column(Text, nullable=True)  # JSON string of input
    result_data = Column(Text, nullable=True)  # JSON string of result
    sample_id = Column(Integer, ForeignKey('samples.id'), nullable=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships - using foreign_keys to avoid circular dependencies
    sample = relationship("Sample", foreign_keys=[sample_id])
    user = relationship("User", foreign_keys=[user_id])
    
    def __repr__(self):
        return f"<AIAssistantQuery {self.id}: {self.query_type}>"
