"""
Authentication Routes for the Mycology Research Pipeline.

This module handles user authentication, registration, and membership management.
"""

import os
from datetime import datetime, timedelta
from urllib.parse import urlparse, urljoin
from flask import (
    Blueprint, render_template, request, redirect, url_for, 
    flash, current_app, session, jsonify
)
from flask_login import (
    login_user, logout_user, login_required, current_user
)
from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy.exc import IntegrityError

from app import db
from models import User, Membership, OAuthToken

# Create the blueprint
auth_bp = Blueprint('auth', __name__)


def is_safe_url(target):
    """
    Check if the target URL is safe for redirecting.
    Only allows relative URLs or URLs from the same host.
    """
    if not target:
        return False
    
    # Parse the target URL
    parsed = urlparse(target)
    
    # Allow relative URLs (no netloc)
    if not parsed.netloc:
        return True
    
    # For absolute URLs, check if they're from the same host
    ref_url = urlparse(request.host_url)
    return parsed.netloc == ref_url.netloc


@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    """Handle user login."""
    if current_user.is_authenticated:
        return redirect(url_for('web.dashboard'))
    
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        remember = 'remember_me' in request.form
        
        # Validate input
        if not email or not password:
            flash('Please provide both email and password.', 'warning')
            return render_template('auth/login.html')
        
        # Find user by email
        user = User.query.filter_by(email=email).first()
        
        if user and check_password_hash(user.password_hash, password):
            # Log the user in
            login_user(user, remember=remember)
            
            # Record login time
            user.last_login = datetime.now()
            db.session.commit()
            
            # Redirect to the page they were trying to access
            next_page = request.args.get('next')
            if not next_page or next_page.startswith('/auth') or not is_safe_url(next_page):
                next_page = url_for('web.dashboard')
            
            flash(f'Welcome back, {user.first_name}!', 'success')
            return redirect(next_page)
        else:
            flash('Invalid email or password. Please try again.', 'danger')
    
    return render_template('auth/login.html')


@auth_bp.route('/logout')
@login_required
def logout():
    """Handle user logout."""
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('web.index'))


@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    """Handle user registration."""
    if current_user.is_authenticated:
        return redirect(url_for('web.dashboard'))
    
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')
        first_name = request.form.get('first_name', '').strip()
        last_name = request.form.get('last_name', '').strip()
        organization = request.form.get('organization', '').strip()
        
        # Basic validation
        if not email or not password or not confirm_password:
            flash('Please provide all required fields.', 'warning')
            return render_template('auth/register.html')
        
        if password != confirm_password:
            flash('Passwords do not match.', 'warning')
            return render_template('auth/register.html')
        
        # Check if user already exists
        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            flash('This email is already registered. Please use a different email or log in.', 'warning')
            return render_template('auth/register.html')
        
        try:
            # Create new user
            new_user = User(
                email=email,
                password_hash=generate_password_hash(password),
                first_name=first_name,
                last_name=last_name,
                organization=organization,
                role='user',
                is_active=True,
                created_at=datetime.now()
            )
            
            # Create free tier membership
            membership = Membership(
                user=new_user,
                plan='free',
                start_date=datetime.now(),
                end_date=datetime.now() + timedelta(days=365),  # 1 year free trial
                is_active=True,
                features_access={
                    'basic_search': True,
                    'literature_export': True,
                    'visualization': True,
                    'api_access': False,
                    'advanced_analytics': False,
                    'batch_processing': False
                }
            )
            
            db.session.add(new_user)
            db.session.add(membership)
            db.session.commit()
            
            # Log in the new user
            login_user(new_user)
            
            flash('Your account has been created successfully!', 'success')
            return redirect(url_for('web.dashboard'))
        
        except IntegrityError:
            db.session.rollback()
            flash('An error occurred during registration. Please try again.', 'danger')
        
        except Exception as e:
            db.session.rollback()
            current_app.logger.error(f"Registration error: {str(e)}")
            flash('An unexpected error occurred. Please try again later.', 'danger')
    
    return render_template('auth/register.html')


@auth_bp.route('/profile')
@login_required
def profile():
    """Show user profile."""
    return render_template('auth/profile.html')


@auth_bp.route('/profile/edit', methods=['GET', 'POST'])
@login_required
def edit_profile():
    """Edit user profile."""
    if request.method == 'POST':
        first_name = request.form.get('first_name', '').strip()
        last_name = request.form.get('last_name', '').strip()
        organization = request.form.get('organization', '').strip()
        
        try:
            current_user.first_name = first_name
            current_user.last_name = last_name
            current_user.organization = organization
            current_user.updated_at = datetime.now()
            
            db.session.commit()
            
            flash('Your profile has been updated successfully!', 'success')
            return redirect(url_for('auth.profile'))
        
        except Exception as e:
            db.session.rollback()
            current_app.logger.error(f"Profile update error: {str(e)}")
            flash('An error occurred while updating your profile.', 'danger')
    
    return render_template('auth/edit_profile.html')


@auth_bp.route('/change-password', methods=['GET', 'POST'])
@login_required
def change_password():
    """Change user password."""
    if request.method == 'POST':
        current_password = request.form.get('current_password', '')
        new_password = request.form.get('new_password', '')
        confirm_password = request.form.get('confirm_password', '')
        
        # Validate input
        if not current_password or not new_password or not confirm_password:
            flash('Please fill in all password fields.', 'warning')
            return render_template('auth/change_password.html')
        
        if new_password != confirm_password:
            flash('New passwords do not match.', 'warning')
            return render_template('auth/change_password.html')
        
        # Verify current password
        if not check_password_hash(current_user.password_hash, current_password):
            flash('Current password is incorrect.', 'danger')
            return render_template('auth/change_password.html')
        
        try:
            # Update password
            current_user.password_hash = generate_password_hash(new_password)
            current_user.updated_at = datetime.now()
            
            db.session.commit()
            
            flash('Your password has been changed successfully.', 'success')
            return redirect(url_for('auth.profile'))
        
        except Exception as e:
            db.session.rollback()
            current_app.logger.error(f"Password change error: {str(e)}")
            flash('An error occurred while changing your password.', 'danger')
    
    return render_template('auth/change_password.html')


@auth_bp.route('/membership')
@login_required
def membership():
    """Show user membership details."""
    # Get the user's active membership
    user_membership = Membership.query.filter_by(
        user_id=current_user.id, is_active=True
    ).first()
    
    return render_template('auth/membership.html', membership=user_membership)


@auth_bp.route('/upgrade', methods=['GET', 'POST'])
@login_required
def upgrade_membership():
    """Handle membership upgrades."""
    if request.method == 'POST':
        plan = request.form.get('plan', '')
        
        if not plan or plan not in ['free', 'basic', 'pro', 'enterprise']:
            flash('Invalid membership plan selected.', 'warning')
            return redirect(url_for('auth.membership'))
        
        # In a real implementation, this would connect to a payment gateway
        # For the MVP, we'll just update the membership
        
        try:
            # Get the user's current membership
            current_membership = Membership.query.filter_by(
                user_id=current_user.id, is_active=True
            ).first()
            
            if current_membership:
                # Deactivate the current membership
                current_membership.is_active = False
                current_membership.end_date = datetime.now()
            
            # Create features access based on the plan
            features = {
                'basic_search': True,
                'literature_export': True,
                'visualization': True,
                'api_access': False,
                'advanced_analytics': False,
                'batch_processing': False
            }
            
            if plan == 'basic':
                features['api_access'] = True
            elif plan == 'pro':
                features['api_access'] = True
                features['advanced_analytics'] = True
            elif plan == 'enterprise':
                features['api_access'] = True
                features['advanced_analytics'] = True
                features['batch_processing'] = True
            
            # Create new membership
            new_membership = Membership(
                user_id=current_user.id,
                plan=plan,
                start_date=datetime.now(),
                end_date=datetime.now() + timedelta(days=365),  # 1 year subscription
                is_active=True,
                features_access=features
            )
            
            db.session.add(new_membership)
            db.session.commit()
            
            flash(f'Your membership has been upgraded to {plan.capitalize()}!', 'success')
            return redirect(url_for('auth.membership'))
        
        except Exception as e:
            db.session.rollback()
            current_app.logger.error(f"Membership upgrade error: {str(e)}")
            flash('An error occurred while upgrading your membership.', 'danger')
    
    return render_template('auth/upgrade.html')


@auth_bp.route('/api-keys')
@login_required
def api_keys():
    """Manage API keys."""
    # Check if user has API access
    user_membership = Membership.query.filter_by(
        user_id=current_user.id, is_active=True
    ).first()
    
    has_api_access = False
    if user_membership and user_membership.features_access.get('api_access', False):
        has_api_access = True
    
    # Get user's API tokens
    api_tokens = OAuthToken.query.filter_by(user_id=current_user.id).all()
    
    return render_template(
        'auth/api_keys.html',
        has_api_access=has_api_access,
        api_tokens=api_tokens
    )


@auth_bp.route('/generate-api-key', methods=['POST'])
@login_required
def generate_api_key():
    """Generate a new API key."""
    # Check if user has API access
    user_membership = Membership.query.filter_by(
        user_id=current_user.id, is_active=True
    ).first()
    
    if not user_membership or not user_membership.features_access.get('api_access', False):
        flash('Your current membership does not include API access.', 'warning')
        return redirect(url_for('auth.api_keys'))
    
    # Generate token
    import secrets
    token = secrets.token_hex(16)
    token_name = request.form.get('token_name', 'API Key')
    
    try:
        # Save token
        new_token = OAuthToken(
            user_id=current_user.id,
            token=token,
            name=token_name,
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(days=90),  # 90-day token
            is_active=True
        )
        
        db.session.add(new_token)
        db.session.commit()
        
        flash('Your API key has been generated successfully!', 'success')
    except Exception as e:
        db.session.rollback()
        current_app.logger.error(f"API key generation error: {str(e)}")
        flash('An error occurred while generating your API key.', 'danger')
    
    return redirect(url_for('auth.api_keys'))


@auth_bp.route('/revoke-api-key/<int:token_id>', methods=['POST'])
@login_required
def revoke_api_key(token_id):
    """Revoke an API key."""
    token = OAuthToken.query.filter_by(id=token_id, user_id=current_user.id).first()
    
    if not token:
        flash('API key not found.', 'warning')
        return redirect(url_for('auth.api_keys'))
    
    try:
        token.is_active = False
        token.revoked_at = datetime.now()
        
        db.session.commit()
        
        flash('Your API key has been revoked.', 'success')
    except Exception as e:
        db.session.rollback()
        current_app.logger.error(f"API key revocation error: {str(e)}")
        flash('An error occurred while revoking your API key.', 'danger')
    
    return redirect(url_for('auth.api_keys'))