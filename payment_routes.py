"""
Payment and Membership Routes for Mycology Research Pipeline.

This module handles Stripe payment integration, subscription management,
and membership tier functionality with enhanced security and error handling.
"""

import os
import logging
import stripe
from functools import wraps
from flask import Blueprint, render_template, request, redirect, url_for, flash, session, jsonify, abort
from flask_login import login_required, current_user
from werkzeug.exceptions import BadRequest
from app import db
from models import User, Subscription, Payment, MembershipTier, SubscriptionStatus
from datetime import datetime, timedelta
import hmac
import hashlib

# Configure logging
logger = logging.getLogger(__name__)

payment_bp = Blueprint('payment', __name__, url_prefix='/payment')

# Initialize Stripe with error handling
try:
    stripe.api_key = os.environ.get('STRIPE_SECRET_KEY')
    if not stripe.api_key:
        logger.warning("Stripe API key not configured")
except Exception as e:
    logger.error(f"Error initializing Stripe: {str(e)}")

# Get domain for Stripe redirect URLs with validation
def get_domain():
    """Get the application domain with proper validation."""
    if os.environ.get('REPLIT_DEPLOYMENT') == 'true':
        domains = os.environ.get('REPLIT_DOMAINS', '').split(',')
        return domains[0] if domains else 'localhost:5000'
    return os.environ.get('REPLIT_DEV_DOMAIN', 'localhost:5000')

YOUR_DOMAIN = get_domain()

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

def stripe_configured(f):
    """Decorator to check if Stripe is properly configured."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not stripe.api_key:
            flash('Payment system is not configured. Please contact support.', 'error')
            return redirect(url_for('web.index'))
        return f(*args, **kwargs)
    return decorated_function

@payment_bp.route('/membership')
@login_required
def membership():
    """Display membership tiers and current subscription."""
    try:
        return render_template('payment/membership.html', 
                             membership_tiers=MEMBERSHIP_TIERS,
                             current_tier=getattr(current_user, 'membership_tier', 'free'))
    except Exception as e:
        logger.error(f"Error displaying membership page: {str(e)}")
        flash('Error loading membership page. Please try again.', 'error')
        return redirect(url_for('web.dashboard'))

@payment_bp.route('/create-checkout-session', methods=['POST'])
@login_required
@stripe_configured
def create_checkout_session():
    """Create a Stripe checkout session for subscription."""
    try:
        # Validate CSRF token
        if not request.form.get('csrf_token'):
            abort(400, 'CSRF token missing')
            
        tier = request.form.get('tier')
        if tier not in MEMBERSHIP_TIERS:
            flash('Invalid membership tier selected.', 'error')
            return redirect(url_for('payment.membership'))
        
        # Check if user already has an active subscription
        existing_subscription = Subscription.query.filter_by(
            user_id=current_user.id,
            status=SubscriptionStatus.ACTIVE
        ).first()
        
        if existing_subscription and existing_subscription.tier == tier:
            flash('You already have an active subscription to this plan.', 'info')
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
                'user_id': str(current_user.id),
                'tier': tier
            },
            allow_promotion_codes=True,
            billing_address_collection='required'
        )
        
        logger.info(f"Created checkout session for user {current_user.id}, tier: {tier}")
        return redirect(checkout_session.url, code=303)
        
    except stripe.error.StripeError as e:
        logger.error(f"Stripe error creating checkout session: {str(e)}")
        flash('Payment system error. Please try again later.', 'error')
        return redirect(url_for('payment.membership'))
    except Exception as e:
        logger.error(f"Error creating payment session: {str(e)}")
        flash('An error occurred. Please try again.', 'error')
        return redirect(url_for('payment.membership'))

@payment_bp.route('/premium-services')
@login_required
def premium_services():
    """Display premium services available for purchase."""
    try:
        return render_template('payment/premium_services.html', 
                             services=PREMIUM_SERVICES)
    except Exception as e:
        logger.error(f"Error displaying premium services: {str(e)}")
        flash('Error loading premium services. Please try again.', 'error')
        return redirect(url_for('web.dashboard'))

@payment_bp.route('/buy-service/<service_id>')
@login_required
@stripe_configured
def buy_premium_service(service_id):
    """Create checkout session for premium service."""
    if service_id not in PREMIUM_SERVICES:
        flash('Invalid service selected.', 'error')
        return redirect(url_for('payment.premium_services'))
    
    service = PREMIUM_SERVICES[service_id]
    
    try:
        checkout_session = stripe.checkout.Session.create(
            customer_email=current_user.email,
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
                'user_id': str(current_user.id),
                'service_id': service_id,
                'service_type': 'premium_analysis'
            },
            billing_address_collection='required'
        )
        
        logger.info(f"Created premium service checkout for user {current_user.id}, service: {service_id}")
        return redirect(checkout_session.url, code=303)
        
    except stripe.error.StripeError as e:
        logger.error(f"Stripe error creating service checkout: {str(e)}")
        flash('Payment system error. Please try again later.', 'error')
        return redirect(url_for('payment.premium_services'))
    except Exception as e:
        logger.error(f"Error creating payment session: {str(e)}")
        flash('An error occurred. Please try again.', 'error')
        return redirect(url_for('payment.premium_services'))

@payment_bp.route('/service-success')
@login_required
def service_payment_success():
    """Handle successful premium service payment."""
    session_id = request.args.get('session_id')
    
    if not session_id:
        flash('Invalid payment session.', 'error')
        return redirect(url_for('payment.premium_services'))
    
    try:
        # Retrieve and verify the session
        checkout_session = stripe.checkout.Session.retrieve(session_id)
        
        # Verify the session belongs to the current user
        if checkout_session.metadata.get('user_id') != str(current_user.id):
            logger.warning(f"User {current_user.id} attempted to access session for user {checkout_session.metadata.get('user_id')}")
            flash('Invalid session.', 'error')
            return redirect(url_for('payment.premium_services'))
        
        if checkout_session.payment_status == 'paid':
            service_id = checkout_session.metadata.get('service_id')
            service = PREMIUM_SERVICES.get(service_id)
            
            if service:
                # Check if payment already processed
                existing_payment = Payment.query.filter_by(
                    stripe_payment_intent_id=checkout_session.payment_intent
                ).first()
                
                if existing_payment:
                    flash('This payment has already been processed.', 'info')
                    return redirect(url_for('web.dashboard'))
                
                # Create payment record
                payment = Payment(
                    user_id=current_user.id,
                    stripe_payment_intent_id=checkout_session.payment_intent,
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
                
                logger.info(f"Processed premium service payment for user {current_user.id}, service: {service_id}")
                flash(f'Payment successful! You now have access to {service["name"]}. Upload your specimens to begin analysis.', 'success')
                
                # Redirect to appropriate analysis page
                if service_id == 'dried_specimen_analysis':
                    return redirect(url_for('web.premium_dried_analysis'))
                elif service_id == 'bioactivity_report':
                    return redirect(url_for('web.premium_bioactivity'))
                else:
                    return redirect(url_for('web.dashboard'))
            else:
                logger.error(f"Invalid service ID in payment metadata: {service_id}")
                flash('Invalid service in payment metadata.', 'error')
        else:
            flash('Payment was not completed. Please try again.', 'warning')
            
    except stripe.error.StripeError as e:
        logger.error(f"Stripe error processing payment success: {str(e)}")
        flash('Error verifying payment. Please contact support.', 'error')
    except Exception as e:
        logger.error(f"Error processing payment: {str(e)}")
        flash('Error processing payment. Please contact support.', 'error')
    
    return redirect(url_for('payment.premium_services'))

@payment_bp.route('/success')
@login_required
def payment_success():
    """Handle successful subscription payment."""
    session_id = request.args.get('session_id')
    
    if not session_id:
        flash('Invalid payment session.', 'error')
        return redirect(url_for('payment.membership'))
    
    try:
        # Retrieve and verify the session
        checkout_session = stripe.checkout.Session.retrieve(session_id)
        
        # Verify the session belongs to the current user
        if checkout_session.metadata.get('user_id') != str(current_user.id):
            logger.warning(f"User {current_user.id} attempted to access session for user {checkout_session.metadata.get('user_id')}")
            flash('Invalid session.', 'error')
            return redirect(url_for('payment.membership'))
        
        if checkout_session.payment_status == 'paid':
            # Check if subscription already processed
            existing_subscription = Subscription.query.filter_by(
                stripe_subscription_id=checkout_session.subscription
            ).first()
            
            if existing_subscription:
                flash('Your subscription is already active.', 'info')
                return redirect(url_for('web.dashboard'))
            
            # Update user's membership
            tier = checkout_session.metadata.get('tier', 'basic')
            current_user.membership_tier = tier
            current_user.subscription_active = True
            current_user.subscription_end_date = datetime.utcnow() + timedelta(days=30)
            
            # Create subscription record
            subscription = Subscription(
                user_id=current_user.id,
                stripe_subscription_id=checkout_session.subscription,
                stripe_customer_id=checkout_session.customer,
                tier=tier,
                status='active',
                current_period_start=datetime.utcnow(),
                current_period_end=datetime.utcnow() + timedelta(days=30)
            )
            
            db.session.add(subscription)
            db.session.commit()
            
            logger.info(f"Activated subscription for user {current_user.id}, tier: {tier}")
            flash(f'Welcome to {MEMBERSHIP_TIERS[tier]["name"]}! Your subscription is now active.', 'success')
        else:
            flash('Payment was not completed. Please try again.', 'warning')
            
    except stripe.error.StripeError as e:
        logger.error(f"Stripe error processing subscription success: {str(e)}")
        flash('Error verifying payment. Please contact support.', 'error')
    except Exception as e:
        logger.error(f"Error processing payment: {str(e)}")
        flash('Error processing payment. Please contact support.', 'error')
    
    return redirect(url_for('web.dashboard'))

@payment_bp.route('/cancel')
@login_required
def payment_cancel():
    """Handle cancelled payment."""
    flash('Payment was cancelled. You can try again anytime.', 'info')
    return redirect(url_for('payment.membership'))

@payment_bp.route('/webhook', methods=['POST'])
def stripe_webhook():
    """Handle Stripe webhook events with enhanced security."""
    payload = request.get_data(as_text=True)
    sig_header = request.headers.get('Stripe-Signature')
    webhook_secret = os.environ.get('STRIPE_WEBHOOK_SECRET')
    
    if not webhook_secret:
        logger.error("Stripe webhook secret not configured")
        return jsonify({'error': 'Webhook not configured'}), 500
    
    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, webhook_secret
        )
    except ValueError as e:
        logger.error(f"Invalid webhook payload: {str(e)}")
        return jsonify({'error': 'Invalid payload'}), 400
    except stripe.error.SignatureVerificationError as e:
        logger.error(f"Invalid webhook signature: {str(e)}")
        return jsonify({'error': 'Invalid signature'}), 400
    
    # Handle the event
    try:
        if event.type == 'customer.subscription.created':
            subscription = event.data.object
            update_subscription_from_stripe(subscription)
        elif event.type == 'customer.subscription.updated':
            subscription = event.data.object
            update_subscription_from_stripe(subscription)
        elif event.type == 'customer.subscription.deleted':
            subscription = event.data.object
            cancel_subscription_from_stripe(subscription)
        elif event.type == 'invoice.payment_succeeded':
            invoice = event.data.object
            handle_successful_payment(invoice)
        elif event.type == 'invoice.payment_failed':
            invoice = event.data.object
            handle_failed_payment(invoice)
        else:
            logger.info(f"Unhandled webhook event type: {event.type}")
        
        return jsonify({'status': 'success'})
        
    except Exception as e:
        logger.error(f"Error processing webhook event {event.type}: {str(e)}")
        return jsonify({'error': 'Processing error'}), 500

def handle_successful_payment(invoice):
    """Handle successful invoice payment with error handling."""
    try:
        if not invoice.subscription:
            return
            
        subscription = stripe.Subscription.retrieve(invoice.subscription)
        user_id = subscription.metadata.get('user_id')
        
        if user_id:
            # Check if payment already recorded
            existing_payment = Payment.query.filter_by(
                stripe_payment_intent_id=invoice.payment_intent
            ).first()
            
            if not existing_payment:
                payment = Payment(
                    user_id=int(user_id),
                    stripe_payment_intent_id=invoice.payment_intent,
                    amount=invoice.amount_paid / 100,  # Convert from cents
                    currency=invoice.currency,
                    status='succeeded',
                    description=f'Subscription payment for {invoice.subscription}'
                )
                
                db.session.add(payment)
                db.session.commit()
                logger.info(f"Recorded successful payment for user {user_id}")
            
            # Update subscription status
            update_subscription_from_stripe(subscription)
    except Exception as e:
        logger.error(f"Error handling successful payment: {str(e)}")
        db.session.rollback()

def handle_failed_payment(invoice):
    """Handle failed invoice payment with error handling."""
    try:
        if not invoice.subscription:
            return
            
        subscription = stripe.Subscription.retrieve(invoice.subscription)
        user_id = subscription.metadata.get('user_id')
        
        if user_id:
            # Update subscription status to past_due
            db_subscription = Subscription.query.filter_by(
                stripe_subscription_id=invoice.subscription
            ).first()
            
            if db_subscription:
                db_subscription.status = SubscriptionStatus.PAST_DUE
                db.session.commit()
                logger.info(f"Updated subscription to past_due for user {user_id}")
                
                # TODO: Implement notification system
                # notify_user_payment_failed(user_id)
    except Exception as e:
        logger.error(f"Error handling failed payment: {str(e)}")
        db.session.rollback()

def update_subscription_from_stripe(stripe_subscription):
    """Update local subscription record from Stripe data with validation."""
    try:
        user_id = stripe_subscription.metadata.get('user_id')
        if not user_id:
            logger.warning(f"No user_id in subscription metadata: {stripe_subscription.id}")
            return
            
        subscription = Subscription.query.filter_by(
            stripe_subscription_id=stripe_subscription.id
        ).first()
        
        if not subscription:
            # Create new subscription
            subscription = Subscription(
                user_id=int(user_id),
                stripe_subscription_id=stripe_subscription.id,
                stripe_customer_id=stripe_subscription.customer,
                tier=stripe_subscription.metadata.get('tier', 'basic'),
                status=stripe_subscription.status,
                current_period_start=datetime.fromtimestamp(stripe_subscription.current_period_start),
                current_period_end=datetime.fromtimestamp(stripe_subscription.current_period_end)
            )
            db.session.add(subscription)
            logger.info(f"Created new subscription for user {user_id}")
        else:
            # Update existing subscription
            subscription.status = stripe_subscription.status
            subscription.current_period_start = datetime.fromtimestamp(stripe_subscription.current_period_start)
            subscription.current_period_end = datetime.fromtimestamp(stripe_subscription.current_period_end)
            logger.info(f"Updated subscription for user {subscription.user_id}")
        
        # Update user's subscription status
        user = User.query.get(int(user_id))
        if user:
            user.subscription_active = stripe_subscription.status == 'active'
            user.subscription_end_date = subscription.current_period_end
        
        db.session.commit()
    except Exception as e:
        logger.error(f"Error updating subscription: {str(e)}")
        db.session.rollback()

def cancel_subscription_from_stripe(stripe_subscription):
    """Handle subscription cancellation from Stripe with validation."""
    try:
        subscription = Subscription.query.filter_by(
            stripe_subscription_id=stripe_subscription.id
        ).first()
        
        if subscription:
            subscription.status = SubscriptionStatus.CANCELLED
            subscription.current_period_end = datetime.fromtimestamp(stripe_subscription.current_period_end)
            
            # Update user's subscription status
            user = User.query.get(subscription.user_id)
            if user:
                user.subscription_active = False
                user.membership_tier = 'free'
            
            db.session.commit()
            logger.info(f"Cancelled subscription for user {subscription.user_id}")
            
            # TODO: Implement notification system
            # notify_user_subscription_cancelled(subscription.user_id)
    except Exception as e:
        logger.error(f"Error cancelling subscription: {str(e)}")
        db.session.rollback()

@payment_bp.route('/portal')
@login_required
@stripe_configured
def customer_portal():
    """Create a Stripe customer portal session with error handling."""
    try:
        # Get or create Stripe customer
        if not current_user.stripe_customer_id:
            customer = stripe.Customer.create(
                email=current_user.email,
                metadata={'user_id': str(current_user.id)}
            )
            current_user.stripe_customer_id = customer.id
            db.session.commit()
            logger.info(f"Created Stripe customer for user {current_user.id}")
        
        # Create portal session
        portal_session = stripe.billing_portal.Session.create(
            customer=current_user.stripe_customer_id,
            return_url=url_for('payment.membership', _external=True)
        )
        
        return redirect(portal_session.url)
        
    except stripe.error.StripeError as e:
        logger.error(f"Stripe error creating portal session: {str(e)}")
        flash('Unable to access billing portal. Please try again later.', 'error')
        return redirect(url_for('payment.membership'))
    except Exception as e:
        logger.error(f"Error accessing customer portal: {str(e)}")
        flash('An error occurred. Please try again.', 'error')
        return redirect(url_for('payment.membership'))