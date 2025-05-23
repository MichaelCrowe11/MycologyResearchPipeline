"""
Payment and Membership Routes for Mycology Research Pipeline.

This module handles Stripe payment integration, subscription management,
and membership tier functionality.
"""

import os
import stripe
from flask import Blueprint, render_template, request, redirect, url_for, flash, session, jsonify
from flask_login import login_required, current_user
from app import db
from models import User, Subscription, Payment
from datetime import datetime, timedelta

payment_bp = Blueprint('payment', __name__, url_prefix='/payment')

# Initialize Stripe
stripe.api_key = os.environ.get('STRIPE_SECRET_KEY')

# Get domain for Stripe redirect URLs
YOUR_DOMAIN = os.environ.get('REPLIT_DEV_DOMAIN') if os.environ.get('REPLIT_DEPLOYMENT') != 'true' else os.environ.get('REPLIT_DOMAINS').split(',')[0]

# Membership tiers and pricing
MEMBERSHIP_TIERS = {
    'basic': {
        'name': 'Basic Researcher',
        'price_id': 'price_basic_monthly',  # Replace with actual Stripe price ID
        'price': 29.99,
        'features': [
            'Access to basic analysis tools',
            'Up to 100 samples per month',
            'Community support',
            'Basic AI assistance (10 queries/month)'
        ]
    },
    'pro': {
        'name': 'Professional Researcher',
        'price_id': 'price_pro_monthly',  # Replace with actual Stripe price ID
        'price': 79.99,
        'features': [
            'All Basic features',
            'Advanced analysis tools',
            'Unlimited samples',
            'Priority support',
            'Full AI assistance (unlimited)',
            'Batch processing',
            'API access',
            'Literature search integration'
        ]
    },
    'enterprise': {
        'name': 'Enterprise Institution',
        'price_id': 'price_enterprise_monthly',  # Replace with actual Stripe price ID
        'price': 199.99,
        'features': [
            'All Professional features',
            'Multi-user accounts',
            'Custom integrations',
            'Dedicated support',
            'Advanced security features',
            'Custom reporting',
            'Data export capabilities'
        ]
    }
}

@payment_bp.route('/membership')
@login_required
def membership():
    """Display membership tiers and current subscription."""
    return render_template('payment/membership.html', 
                         membership_tiers=MEMBERSHIP_TIERS,
                         current_tier=getattr(current_user, 'membership_tier', 'free'))

@payment_bp.route('/create-checkout-session', methods=['POST'])
@login_required
def create_checkout_session():
    """Create a Stripe checkout session for subscription."""
    try:
        tier = request.form.get('tier')
        if tier not in MEMBERSHIP_TIERS:
            flash('Invalid membership tier selected.', 'error')
            return redirect(url_for('payment.membership'))
        
        # Create Stripe checkout session
        checkout_session = stripe.checkout.Session.create(
            customer_email=current_user.email,
            payment_method_types=['card'],
            line_items=[{
                'price': MEMBERSHIP_TIERS[tier]['price_id'],
                'quantity': 1,
            }],
            mode='subscription',
            success_url=f'https://{YOUR_DOMAIN}/payment/success?session_id={{CHECKOUT_SESSION_ID}}',
            cancel_url=f'https://{YOUR_DOMAIN}/payment/cancel',
            metadata={
                'user_id': current_user.id,
                'tier': tier
            }
        )
        
        return redirect(checkout_session.url, code=303)
        
    except Exception as e:
        flash(f'Error creating payment session: {str(e)}', 'error')
        return redirect(url_for('payment.membership'))

@payment_bp.route('/success')
@login_required
def payment_success():
    """Handle successful payment."""
    session_id = request.args.get('session_id')
    
    if session_id:
        try:
            # Retrieve the session from Stripe
            session = stripe.checkout.Session.retrieve(session_id)
            
            if session.payment_status == 'paid':
                # Update user's membership
                tier = session.metadata.get('tier', 'basic')
                current_user.membership_tier = tier
                current_user.subscription_active = True
                current_user.subscription_end_date = datetime.utcnow() + timedelta(days=30)
                
                # Create subscription record
                subscription = Subscription(
                    user_id=current_user.id,
                    stripe_subscription_id=session.subscription,
                    tier=tier,
                    status='active',
                    current_period_start=datetime.utcnow(),
                    current_period_end=datetime.utcnow() + timedelta(days=30)
                )
                
                db.session.add(subscription)
                db.session.commit()
                
                flash(f'Welcome to {MEMBERSHIP_TIERS[tier]["name"]}! Your subscription is now active.', 'success')
            else:
                flash('Payment was not completed. Please try again.', 'warning')
                
        except Exception as e:
            flash(f'Error processing payment: {str(e)}', 'error')
    
    return redirect(url_for('auth.membership'))

@payment_bp.route('/cancel')
@login_required
def payment_cancel():
    """Handle cancelled payment."""
    flash('Payment was cancelled. You can try again anytime.', 'info')
    return redirect(url_for('payment.membership'))

@payment_bp.route('/webhook', methods=['POST'])
def stripe_webhook():
    """Handle Stripe webhooks for subscription updates."""
    payload = request.get_data(as_text=True)
    sig_header = request.headers.get('Stripe-Signature')
    
    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, os.environ.get('STRIPE_WEBHOOK_SECRET')
        )
    except ValueError:
        return 'Invalid payload', 400
    except stripe.error.SignatureVerificationError:
        return 'Invalid signature', 400
    
    # Handle the event
    if event['type'] == 'customer.subscription.updated':
        subscription = event['data']['object']
        # Update subscription in database
        update_subscription_from_stripe(subscription)
    elif event['type'] == 'customer.subscription.deleted':
        subscription = event['data']['object']
        # Cancel subscription in database
        cancel_subscription_from_stripe(subscription)
    
    return 'Success', 200

@payment_bp.route('/portal')
@login_required
def customer_portal():
    """Redirect to Stripe customer portal for subscription management."""
    try:
        # Get user's Stripe customer ID
        if hasattr(current_user, 'stripe_customer_id') and current_user.stripe_customer_id:
            portal_session = stripe.billing_portal.Session.create(
                customer=current_user.stripe_customer_id,
                return_url=f'https://{YOUR_DOMAIN}/auth/membership'
            )
            return redirect(portal_session.url, code=303)
        else:
            flash('No active subscription found.', 'warning')
            return redirect(url_for('payment.membership'))
    except Exception as e:
        flash(f'Error accessing customer portal: {str(e)}', 'error')
        return redirect(url_for('auth.membership'))

def update_subscription_from_stripe(stripe_subscription):
    """Update local subscription from Stripe webhook data."""
    subscription = Subscription.query.filter_by(
        stripe_subscription_id=stripe_subscription['id']
    ).first()
    
    if subscription:
        subscription.status = stripe_subscription['status']
        subscription.current_period_start = datetime.fromtimestamp(
            stripe_subscription['current_period_start']
        )
        subscription.current_period_end = datetime.fromtimestamp(
            stripe_subscription['current_period_end']
        )
        db.session.commit()

def cancel_subscription_from_stripe(stripe_subscription):
    """Cancel local subscription from Stripe webhook data."""
    subscription = Subscription.query.filter_by(
        stripe_subscription_id=stripe_subscription['id']
    ).first()
    
    if subscription:
        subscription.status = 'cancelled'
        subscription.user.subscription_active = False
        db.session.commit()