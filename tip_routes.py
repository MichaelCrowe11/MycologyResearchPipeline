"""
Research Tip Routes for Mycology Research Pipeline.

This module provides routes for the contextual research tip generator.
"""

from flask import Blueprint, render_template, jsonify, request
from research_tips import tip_generator

tip_bp = Blueprint('tips', __name__, url_prefix='/tips')

@tip_bp.route('/contextual')
def get_contextual_tip():
    """Get a contextual research tip based on current activity."""
    context = request.args.get('context', 'general')
    user_level = request.args.get('level', 'beginner')
    
    if context == 'general':
        tip_data = tip_generator.get_random_tip()
    else:
        tip_data = tip_generator.get_contextual_tip(context, user_level)
    
    return jsonify(tip_data)

@tip_bp.route('/random')
def get_random_tip():
    """Get a random research tip."""
    tip_data = tip_generator.get_random_tip()
    return jsonify(tip_data)

@tip_bp.route('/widget')
def tip_widget():
    """Render the tip widget component."""
    context = request.args.get('context', 'general')
    user_level = request.args.get('level', 'beginner')
    
    if context == 'general':
        tip_data = tip_generator.get_random_tip()
    else:
        tip_data = tip_generator.get_contextual_tip(context, user_level)
    
    return render_template('components/tip_widget.html', tip=tip_data)