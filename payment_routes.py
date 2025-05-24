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
        'price_id': os.environ.get('STRIPE_PRICE_BASIC', 'price_1OXyZ2ABC123DEF456GHI7J'),
        'price': 29.99,
        'features': [
            'Access to basic analysis tools',
            'Up to 100 samples per month',
            'Community support',
            'Basic AI assistance (10 queries/month)',
            'Standard computer vision analysis'
        ]
    },
    'pro': {
        'name': 'Professional Researcher',
        'price_id': os.environ.get('STRIPE_PRICE_PRO', 'price_1OXyZABC123DEF456GHI7JK'),
        'price': 79.99,
        'features': [
            'All Basic features',
            'Advanced analysis tools',
            'Unlimited samples',
            'Priority support',
            'Full AI assistance (unlimited)',
            'Batch processing',
            'API access',
            'Literature search integration',
            'Enhanced identification system'
        ]
    },
    'enterprise': {
        'name': 'Enterprise Institution',
        'price_id': os.environ.get('STRIPE_PRICE_ENTERPRISE', 'price_1OXyZMABC123DEF456GHI7J'),
        'price': 199.99,
        'features': [
            'All Professional features',
            'Multi-user accounts',
            'Custom integrations',
            'Dedicated support',
            'Advanced security features',
            'Custom reporting',
            'Data export capabilities',
            'White-label solutions'
        ]
    }
}

# Premium Services - Individual purchases
PREMIUM_SERVICES = {
    'dried_specimen_analysis': {
        'name': 'Professional Dried Specimen Analysis',
        'price_id': os.environ.get('STRIPE_PRICE_DRIED_ANALYSIS', 'price_dried_analysis_premium'),
        'price': 199.99,
        'description': 'Comprehensive analysis of dried mushroom specimens with expert-level identification, bioactivity assessment, and detailed scientific report',
        'features': [
            'Multi-database cross-validation (iNaturalist, GBIF, MycoBank)',
            'Specialized dried specimen morphology analysis',
            'Bioactivity prediction based on 30,000+ authentic records',
            'Scientific literature correlation',
            'Professional PDF report with citations',
            'Quality assessment and authenticity verification',
            '48-hour turnaround time',
            'Expert review included'
        ]
    },
    'bioactivity_report': {
        'name': 'Advanced Bioactivity Analysis',
        'price_id': os.environ.get('STRIPE_PRICE_BIOACTIVITY', 'price_bioactivity_analysis'),
        'price': 149.99,
        'description': 'Detailed bioactivity analysis using machine learning trained on authentic mycology datasets',
        'features': [
            'ML predictions based on 30,000 authentic bioactivity records',
            'Compound identification and classification',
            'Target pathway analysis',
            'Potential applications assessment',
            'Confidence scoring and validation',
            'Research recommendations',
            'Comparative analysis with similar species'
        ]
    },
    'batch_analysis_premium': {
        'name': 'Premium Batch Processing',
        'price_id': os.environ.get('STRIPE_PRICE_BATCH_PREMIUM', 'price_batch_premium'),
        'price': 299.99,
        'description': 'Professional batch processing service for large datasets with priority queue and enhanced accuracy',
        'features': [
            'Up to 10,000 samples processed',
            'Priority processing queue',
            'Enhanced accuracy algorithms',
            'Comprehensive results export',
            'Statistical analysis included',
            'Custom parameter optimization',
            'Dedicated support during processing'
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

@payment_bp.route('/premium-services')
@login_required
def premium_services():
    """Display premium services available for purchase."""
    return render_template('payment/premium_services.html', 
                         services=PREMIUM_SERVICES)

@payment_bp.route('/buy-service/<service_id>')
@login_required
def buy_premium_service(service_id):
    """Create checkout session for premium service."""
    if service_id not in PREMIUM_SERVICES:
        flash('Invalid service selected.', 'error')
        return redirect(url_for('payment.premium_services'))
    
    service = PREMIUM_SERVICES[service_id]
    
    try:
        checkout_session = stripe.checkout.Session.create(
            line_items=[{
                'price_data': {
                    'currency': 'usd',
                    'product_data': {
                        'name': service['name'],
                        'description': service['description'],
                    },
                    'unit_amount': int(service['price'] * 100),  # Convert to cents
                },
                'quantity': 1,
            }],
            mode='payment',
            success_url=f'https://{YOUR_DOMAIN}/payment/service-success?session_id={{CHECKOUT_SESSION_ID}}',
            cancel_url=f'https://{YOUR_DOMAIN}/payment/premium-services',
            metadata={
                'user_id': current_user.id,
                'service_id': service_id,
                'service_type': 'premium_analysis'
            }
        )
        
        return redirect(checkout_session.url, code=303)
        
    except Exception as e:
        flash(f'Error creating payment session: {str(e)}', 'error')
        return redirect(url_for('payment.premium_services'))

@payment_bp.route('/service-success')
@login_required
def service_payment_success():
    """Handle successful premium service payment."""
    session_id = request.args.get('session_id')
    
    if session_id:
        try:
            session = stripe.checkout.Session.retrieve(session_id)
            
            if session.payment_status == 'paid':
                service_id = session.metadata.get('service_id')
                service = PREMIUM_SERVICES.get(service_id)
                
                if service:
                    # Create payment record
                    payment = Payment(
                        user_id=current_user.id,
                        stripe_payment_intent_id=session.payment_intent,
                        amount=service['price'],
                        currency='usd',
                        status='completed',
                        service_type=service_id,
                        metadata={
                            'service_name': service['name'],
                            'session_id': session_id
                        }
                    )
                    
                    db.session.add(payment)
                    db.session.commit()
                    
                    # Store service access in session for processing
                    session['purchased_service'] = {
                        'service_id': service_id,
                        'payment_id': payment.id,
                        'expires_at': (datetime.utcnow() + timedelta(days=30)).isoformat()
                    }
                    
                    flash(f'Payment successful! You now have access to {service["name"]}. Upload your specimens to begin analysis.', 'success')
                    
                    # Redirect to appropriate analysis page
                    if service_id == 'dried_specimen_analysis':
                        return redirect(url_for('web.premium_dried_analysis'))
                    elif service_id == 'bioactivity_report':
                        return redirect(url_for('web.premium_bioactivity'))
                    else:
                        return redirect(url_for('web.dashboard'))
                else:
                    flash('Invalid service in payment metadata.', 'error')
            else:
                flash('Payment was not completed. Please try again.', 'warning')
                
        except Exception as e:
            flash(f'Error processing payment: {str(e)}', 'error')
    
    return redirect(url_for('payment.premium_services'))

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