"""
Update database script for Mycology Research Pipeline
"""
from app import app, db
import models

def update_database():
    """Create or update database tables based on models."""
    print("Updating database tables...")
    with app.app_context():
        db.create_all()
    print("Database tables updated successfully!")

if __name__ == "__main__":
    update_database()