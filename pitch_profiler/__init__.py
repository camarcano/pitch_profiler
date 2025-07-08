# app/modules/pitch_profiler/__init__.py
"""
Pitch Profiler Module for Datanalytics.pro
Advanced baseball pitching analysis and reporting tool
"""

__version__ = "1.0.0"
__author__ = "Datanalytics.pro"
__description__ = "Advanced baseball pitching analysis tool with movement profiles, usage patterns, and comprehensive reporting"

# Module metadata
MODULE_INFO = {
    "name": "pitch_profiler",
    "display_name": "Pitch Profiler", 
    "version": __version__,
    "description": __description__,
    "author": __author__,
    "features": [
        "CSV data upload",
        "Pitch movement analysis",
        "Usage pattern analysis",
        "Location heatmaps", 
        "Batted ball statistics",
        "PDF report generation",
        "Interactive visualizations"
    ],
    "data_sources": ["statcast", "csv_upload"],
    "output_formats": ["pdf", "png", "interactive"]
}

def get_module_info():
    """Return module information"""
    return MODULE_INFO