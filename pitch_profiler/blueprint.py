# app/modules/pitch_profiler/blueprint.py
"""
Pitch Profiler Module Blueprint
Flask blueprint for the pitch profiler module
"""

from flask import Blueprint, render_template, current_app
from flask_login import login_required

# Create blueprint
bp = Blueprint(
    'pitch_profiler',
    __name__,
    template_folder='templates',
    static_folder='static',
    url_prefix='/pitch_profiler'
)

@bp.route('/')
@login_required
def index():
    """Main pitch profiler page - redirects to Dash app"""
    return render_template('pitch_profiler/index.html', 
                         title='Pitch Profiler - Baseball Analytics')

@bp.route('/about')
@login_required 
def about():
    """About page for the pitch profiler module"""
    return render_template('pitch_profiler/about.html',
                         title='About Pitch Profiler')

# Module initialization function (called by module manager)
def init_module():
    """Initialize the pitch profiler module"""
    current_app.logger.info("Pitch Profiler module initialized")
    return True

# Module status function
def get_module_status():
    """Get module status information"""
    return {
        'name': 'pitch_profiler',
        'status': 'active',
        'features': [
            'CSV Upload',
            'Movement Analysis', 
            'Usage Patterns',
            'Location Heatmaps',
            'PDF Reports'
        ]
    }