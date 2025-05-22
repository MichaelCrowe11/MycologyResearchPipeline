"""
Authentication routes for the Mycology Research Pipeline.
"""
import logging
import os
import uuid
from functools import wraps
from urllib.parse import urlencode

from flask import Blueprint, flash, g, redirect, render_template, request, session, url_for
from flask_dance.consumer import OAuth2ConsumerBlueprint, oauth_authorized, oauth_error
from flask_dance.consumer.storage import BaseStorage
from flask_login import LoginManager, current_user, login_user, logout_user
from oauthlib.oauth2.rfc6749.errors import InvalidGrantError
import jwt
from werkzeug.local import LocalProxy

from app import app, db
from models import Membership, User

# Configure logging
logger = logging.getLogger(__name__)

# Create Blueprint
auth_bp = Blueprint('auth', __name__)

# Initialize login manager
login_manager = LoginManager(app)
login_manager.login_view = 'replit_auth.login'


@login_manager.user_loader
def load_user(user_id):
    """Load user by ID."""
    return User.query.get(user_id)


class UserSessionStorage(BaseStorage):
    """Storage for user session data."""

    def get(self, blueprint):
        """Get token from storage."""
        try:
            token = db.session.query(OAuth).filter_by(
                user_id=current_user.get_id(),
                browser_session_key=g.browser_session_key,
                provider=blueprint.name,
            ).one().token
        except Exception:
            token = None
        return token

    def set(self, blueprint, token):
        """Set token in storage."""
        db.session.query(OAuth).filter_by(
            user_id=current_user.get_id(),
            browser_session_key=g.browser_session_key,
            provider=blueprint.name,
        ).delete()
        new_model = OAuth()
        new_model.user_id = current_user.get_id()
        new_model.browser_session_key = g.browser_session_key
        new_model.provider = blueprint.name
        new_model.token = token
        db.session.add(new_model)
        db.session.commit()

    def delete(self, blueprint):
        """Delete token from storage."""
        db.session.query(OAuth).filter_by(
            user_id=current_user.get_id(),
            browser_session_key=g.browser_session_key,
            provider=blueprint.name).delete()
        db.session.commit()


def make_replit_blueprint():
    """Create a Replit auth blueprint."""
    try:
        repl_id = os.environ['REPL_ID']
    except KeyError:
        raise SystemExit("the REPL_ID environment variable must be set")

    issuer_url = os.environ.get('ISSUER_URL', "https://replit.com/oidc")

    replit_bp = OAuth2ConsumerBlueprint(
        "replit_auth",
        __name__,
        client_id=repl_id,
        client_secret=None,
        base_url=issuer_url,
        authorization_url_params={
            "prompt": "login consent",
        },
        token_url=issuer_url + "/token",
        token_url_params={
            "auth": (),
            "include_client_id": True,
        },
        auto_refresh_url=issuer_url + "/token",
        auto_refresh_kwargs={
            "client_id": repl_id,
        },
        authorization_url=issuer_url + "/auth",
        use_pkce=True,
        code_challenge_method="S256",
        scope=["openid", "profile", "email", "offline_access"],
        storage=UserSessionStorage(),
    )

    @replit_bp.before_app_request
    def set_applocal_session():
        """Set app local session."""
        if '_browser_session_key' not in session:
            session['_browser_session_key'] = uuid.uuid4().hex
        session.modified = True
        g.browser_session_key = session['_browser_session_key']
        g.flask_dance_replit = replit_bp.session

    @replit_bp.route("/logout")
    def logout():
        """Handle logout."""
        del replit_bp.token
        logout_user()

        end_session_endpoint = issuer_url + "/session/end"
        encoded_params = urlencode({
            "client_id": repl_id,
            "post_logout_redirect_uri": request.url_root,
        })
        logout_url = f"{end_session_endpoint}?{encoded_params}"

        return redirect(logout_url)

    @replit_bp.route("/error")
    def error():
        """Handle auth error."""
        return render_template("auth/error.html"), 403

    return replit_bp


def save_user(user_claims):
    """Save or update user from JWT claims."""
    user = User()
    user.id = user_claims['sub']
    user.email = user_claims.get('email')
    user.first_name = user_claims.get('first_name')
    user.last_name = user_claims.get('last_name')
    user.profile_image_url = user_claims.get('profile_image_url')
    merged_user = db.session.merge(user)
    
    # Check if membership exists, if not create one
    if not Membership.query.filter_by(user_id=merged_user.id).first():
        membership = Membership(
            user_id=merged_user.id,
            tier='free'
        )
        db.session.add(membership)
    
    db.session.commit()
    return merged_user


@oauth_authorized.connect
def logged_in(blueprint, token):
    """Handle successful login."""
    user_claims = jwt.decode(token['id_token'],
                            options={"verify_signature": False})
    user = save_user(user_claims)
    login_user(user)
    blueprint.token = token
    next_url = session.pop("next_url", None)
    if next_url is not None:
        return redirect(next_url)


@oauth_error.connect
def handle_error(blueprint, error, error_description=None, error_uri=None):
    """Handle auth error."""
    return redirect(url_for('replit_auth.error'))


def require_login(f):
    """Decorator to require login."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated:
            session["next_url"] = get_next_navigation_url(request)
            return redirect(url_for('replit_auth.login'))

        # Token refresh would go here in a real app
        return f(*args, **kwargs)
    return decorated_function


def get_next_navigation_url(request):
    """Get the URL to redirect to after login."""
    is_navigation_url = request.headers.get(
        'Sec-Fetch-Mode') == 'navigate' and request.headers.get(
            'Sec-Fetch-Dest') == 'document'
    if is_navigation_url:
        return request.url
    return request.referrer or request.url


replit = LocalProxy(lambda: g.flask_dance_replit)


# Auth routes
@auth_bp.route('/profile')
@require_login
def profile():
    """User profile page."""
    return render_template('auth/profile.html')


@auth_bp.route('/upgrade_membership')
@require_login
def upgrade_membership():
    """Membership upgrade page."""
    return render_template('auth/upgrade_membership.html')


# Add OAuth model to database
class OAuth(db.Model):
    """OAuth model for user tokens."""
    __tablename__ = 'oauth'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String, db.ForeignKey(User.id))
    browser_session_key = db.Column(db.String, nullable=False)
    provider = db.Column(db.String(50), nullable=False)
    token = db.Column(db.JSON, nullable=False)
    
    user = db.relationship(User)
    
    __table_args__ = (
        db.UniqueConstraint(
            'user_id',
            'browser_session_key',
            'provider',
            name='uq_user_browser_session_key_provider',
        ),
    )