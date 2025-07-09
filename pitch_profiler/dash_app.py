# app/modules/pitch_profiler/dash_app.py
"""
Complete Enhanced Pitcher Profiler Dash Application
Advanced baseball pitching analysis tool with comprehensive features matching Streamlit version
"""

import dash
from dash import dcc, html, Input, Output, State, callback_context, dash_table
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import base64
import io
import tempfile
import os
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import json

def calculate_k_bb_percentages(df):
    """Calculate K% and BB% from the dataframe"""
    # Get final pitch of each at-bat
    at_bats_final_pitch = df.loc[df.dropna(subset=['events']).groupby(['game_date', 'at_bat_number'])['pitch_number'].idxmax()]

    # Define plate appearance events
    pa_events = ['single', 'double', 'triple', 'home_run', 'field_out', 'strikeout',
                 'strikeout_double_play', 'double_play', 'grounded_into_double_play',
                 'fielders_choice', 'fielders_choice_out', 'force_out', 'batter_interference',
                 'walk', 'intent_walk', 'hit_by_pitch', 'sac_fly', 'sac_fly_double_play']

    # Filter for plate appearances
    plate_appearances = at_bats_final_pitch[at_bats_final_pitch['events'].isin(pa_events)]
    total_pa = len(plate_appearances)
    strikeouts = len(plate_appearances[plate_appearances['events'].isin(['strikeout', 'strikeout_double_play'])])
    walks = len(plate_appearances[plate_appearances['events'].isin(['walk', 'intent_walk'])])
    k_pct = (strikeouts / total_pa * 100) if total_pa > 0 else 0
    bb_pct = (walks / total_pa * 100) if total_pa > 0 else 0
    return k_pct, bb_pct, total_pa, strikeouts, walks

def analyze_pitch_usage_by_count(df):
    """Analyze pitch usage patterns by count"""
    df['count'] = df['balls'].astype(str) + '-' + df['strikes'].astype(str)
    
    usage_by_count = df.groupby(['count', 'pitch_name_full']).size().unstack(fill_value=0)
    usage_by_count_pct = usage_by_count.div(usage_by_count.sum(axis=1), axis=0) * 100
    
    return usage_by_count_pct

def calculate_batting_stats(data):
    """Calculate batting statistics against the pitcher"""
    ab_events = ['single', 'double', 'triple', 'home_run', 'field_out', 'strikeout',
                 'strikeout_double_play', 'double_play', 'grounded_into_double_play',
                 'fielders_choice', 'fielders_choice_out', 'force_out', 'batter_interference']
    pa_events = ab_events + ['walk', 'intent_walk', 'hit_by_pitch', 'sac_fly', 'sac_fly_double_play']
    
    at_bats_count = data['events'].isin(ab_events).sum()
    plate_appearances_count = data['events'].isin(pa_events).sum()
    hits = data['events'].isin(['single', 'double', 'triple', 'home_run']).sum()
    walks = data['events'].isin(['walk', 'intent_walk']).sum()
    hbp = data['events'].isin(['hit_by_pitch']).sum()
    
    singles = data[data['events'] == 'single']['events'].count()
    doubles = data[data['events'] == 'double']['events'].count()
    triples = data[data['events'] == 'triple']['events'].count()
    home_runs = data[data['events'] == 'home_run']['events'].count()
    
    total_bases = singles + (2 * doubles) + (3 * triples) + (4 * home_runs)
    
    avg = hits / at_bats_count if at_bats_count > 0 else 0
    obp = (hits + walks + hbp) / plate_appearances_count if plate_appearances_count > 0 else 0
    slg = total_bases / at_bats_count if at_bats_count > 0 else 0
    ops = obp + slg
    
    return pd.Series({
        'AVG': avg, 'OBP': obp, 'SLG': slg, 'OPS': ops,
        'PA': plate_appearances_count, 'AB': at_bats_count, 'H': hits,
        '1B': singles, '2B': doubles, '3B': triples, 'HR': home_runs
    })

def calculate_detailed_stats(df):
    """Calculate comprehensive pitching statistics including batted ball and batting analysis"""
    stats = {}
    
    # Basic counts
    stats['total_pitches'] = len(df)
    stats['total_strikes'] = len(df[df['type'] == 'S'])
    stats['total_balls'] = len(df[df['type'] == 'B'])
    stats['total_hits'] = len(df[df['type'] == 'X'])
    
    # Strike rate
    stats['strike_rate'] = (stats['total_strikes'] / stats['total_pitches']) * 100
    
    # First strike percentage
    first_pitches = df[df['pitch_number'] == 1]
    first_strikes = first_pitches[first_pitches['type'] == 'S']
    stats['first_strike_pct'] = (len(first_strikes) / len(first_pitches)) * 100 if len(first_pitches) > 0 else 0
    
    # Swinging strike rate
    swinging_strikes = df[df['description'].isin(['swinging_strike', 'swinging_strike_blocked'])]
    swings = df[df['description'].isin(['swinging_strike', 'swinging_strike_blocked', 'foul', 'foul_tip', 'hit_into_play'])]
    stats['swinging_strike_rate'] = (len(swinging_strikes) / len(swings)) * 100 if len(swings) > 0 else 0
    
    # Zone analysis
    if 'zone' in df.columns:
        in_zone = df[df['zone'].isin([1, 2, 3, 4, 5, 6, 7, 8, 9])]
        stats['zone_rate'] = (len(in_zone) / len(df)) * 100
        
        # Chase rate (swings at pitches outside zone)
        out_zone = df[~df['zone'].isin([1, 2, 3, 4, 5, 6, 7, 8, 9])]
        out_zone_swings = out_zone[out_zone['description'].isin(['swinging_strike', 'swinging_strike_blocked', 'foul', 'foul_tip', 'hit_into_play'])]
        stats['chase_rate'] = (len(out_zone_swings) / len(out_zone)) * 100 if len(out_zone) > 0 else 0
    
    # Velocity stats
    stats['avg_velocity'] = df['release_speed'].mean()
    stats['max_velocity'] = df['release_speed'].max()
    stats['min_velocity'] = df['release_speed'].min()
    
    # Spin rate stats (if available)
    if 'release_spin_rate' in df.columns and not df['release_spin_rate'].isna().all():
        stats['avg_spin_rate'] = df['release_spin_rate'].mean()
        stats['max_spin_rate'] = df['release_spin_rate'].max()
        stats['min_spin_rate'] = df['release_spin_rate'].min()
    
    # BATTED BALL ANALYSIS
    batted_balls = df[df['description'] == 'hit_into_play'].copy()
    batted_ball_stats = {}
    
    if not batted_balls.empty and 'launch_speed' in batted_balls.columns:
        # Hard hit analysis
        hard_hit_threshold = 95
        batted_balls['is_hard_hit'] = batted_balls['launch_speed'] >= hard_hit_threshold
        
        total_hard_hit_pct = batted_balls['is_hard_hit'].mean() * 100
        hard_hit_by_pitch = batted_balls.groupby('pitch_name_full')['is_hard_hit'].mean() * 100
        
        # xwOBA analysis
        if 'estimated_woba_using_speedangle' in batted_balls.columns:
            xwoba_threshold = 0.350
            batted_balls['is_high_xwoba'] = batted_balls['estimated_woba_using_speedangle'] >= xwoba_threshold
            total_high_xwoba_pct = batted_balls['is_high_xwoba'].mean() * 100
            high_xwoba_by_pitch = batted_balls.groupby('pitch_name_full')['is_high_xwoba'].mean() * 100
        else:
            total_high_xwoba_pct = 0
            high_xwoba_by_pitch = pd.Series()
        
        batted_ball_stats = {
            'total_hard_hit_pct': total_hard_hit_pct,
            'hard_hit_by_pitch': hard_hit_by_pitch.to_dict(),
            'total_high_xwoba_pct': total_high_xwoba_pct,
            'high_xwoba_by_pitch': high_xwoba_by_pitch.to_dict() if not high_xwoba_by_pitch.empty else {},
            'hard_hit_threshold': hard_hit_threshold,
            'xwoba_threshold': 0.350,
            'total_batted_balls': len(batted_balls)
        }
    else:
        batted_ball_stats = {'no_data': True}
    
    # BATTING STATS AGAINST
    at_bats_final_pitch = df.loc[df.dropna(subset=['events']).groupby(['game_date', 'at_bat_number'])['pitch_number'].idxmax()]
    batting_stats = {}
    
    if not at_bats_final_pitch.empty:
        # Overall stats
        overall_stats = calculate_batting_stats(at_bats_final_pitch)
        
        # Stats by handedness
        if 'stand' in at_bats_final_pitch.columns:
            stats_by_handedness = at_bats_final_pitch.groupby('stand').apply(calculate_batting_stats)
        else:
            stats_by_handedness = pd.DataFrame()
        
        # Stats by pitch type (for final pitch of at-bat)
        stats_by_pitch = at_bats_final_pitch.groupby('pitch_name_full').apply(calculate_batting_stats)
        
        batting_stats = {
            'overall_stats': overall_stats.to_dict(),
            'stats_by_handedness': stats_by_handedness.to_dict() if not stats_by_handedness.empty else {},
            'stats_by_pitch': stats_by_pitch.to_dict()
        }
    else:
        batting_stats = {'no_data': True}
    
    stats['batted_ball_stats'] = batted_ball_stats
    stats['batting_stats'] = batting_stats
    
    return stats

def create_movement_plot(df, pitcher_name, IVB_COL, HB_COL):
    """Create the horizontal vs vertical break plot using Plotly"""
    fig = go.Figure()

    # Add concentric circles for break magnitude
    theta = np.linspace(0, 2*np.pi, 100)
    for radius in range(4, 21, 4):
        x_circle = radius * np.cos(theta)
        y_circle = radius * np.sin(theta)
        fig.add_trace(go.Scatter(
            x=x_circle, y=y_circle,
            mode='lines',
            line=dict(color='grey', dash='dash', width=1),
            showlegend=False,
            opacity=0.6
        ))

    # Color mapping for pitch types
    pitch_types = df['pitch_name_full'].unique()
    colors = px.colors.qualitative.Set1[:len(pitch_types)]
    color_map = dict(zip(pitch_types, colors))

    # Main scatter plot by pitch type
    for pitch_type in pitch_types:
        pitch_data = df[df['pitch_name_full'] == pitch_type]
        fig.add_trace(go.Scatter(
            x=pitch_data[HB_COL],
            y=pitch_data[IVB_COL],
            mode='markers',
            marker=dict(
                size=8,
                color=color_map[pitch_type],
                opacity=0.6
            ),
            name=pitch_type,
            text=[f"{pitch_type}<br>Velo: {v:.1f} mph" for v in pitch_data['release_speed']],
            hovertemplate='<b>%{text}</b><br>HB: %{x:.1f}"<br>IVB: %{y:.1f}"<extra></extra>'
        ))

    # Calculate centroids and add annotations
    pitch_summary_for_plot = df.groupby('pitch_name_full').agg(
        avg_ivb=(IVB_COL, 'mean'),
        avg_hb=(HB_COL, 'mean'),
        avg_velocity=('release_speed', 'mean'),
        avg_spin_rate=('release_spin_rate', 'mean'),
        avg_extension=('release_extension', 'mean')
    )

    for pitch, data in pitch_summary_for_plot.iterrows():
        fig.add_annotation(
            x=data['avg_hb'],
            y=data['avg_ivb'],
            text=f"<b>{pitch}</b><br>Velo: {data['avg_velocity']:.1f} mph<br>Spin: {data['avg_spin_rate']:.0f} rpm<br>Break: {data['avg_hb']:.1f}\" / {data['avg_ivb']:.1f}\"<br>Ext: {data['avg_extension']:.1f}\"",
            showarrow=True,
            arrowhead=2,
            arrowcolor="black",
            arrowwidth=1,
            arrowsize=1,
            ax=20,
            ay=-30,
            bgcolor="white",
            bordercolor="black",
            borderwidth=1,
            font=dict(size=10)
        )

    # Set equal aspect ratio and center the origin
    max_hb = df[HB_COL].abs().max()
    max_ivb = df[IVB_COL].abs().max()
    max_limit = max(max_hb, max_ivb) + 5

    fig.update_layout(
        title=f"{pitcher_name} - Pitch Movement Profile (2025)",
        xaxis_title="Horizontal Break (HB) in Inches",
        yaxis_title="Induced Vertical Break (IVB) in Inches",
        xaxis=dict(range=[-max_limit, max_limit], zeroline=True, zerolinewidth=2, zerolinecolor='black'),
        yaxis=dict(range=[-max_limit, max_limit], zeroline=True, zerolinewidth=2, zerolinecolor='black'),
        width=800,
        height=800,
        showlegend=True,
        legend=dict(x=1.02, y=1),
        plot_bgcolor='white'
    )

    # Add grid
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

    return fig

def create_frequency_plot(df, pitcher_name, pitch_summary, total_csw_pct):
    """Create the pitch frequency and averages plot using Plotly"""
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=pitch_summary['pitch_name_full'],
        y=pitch_summary['count'],
        text=[f"CSW: {row['csw_pct']:.1f}%<br>Velo: {row['avg_velocity']:.1f} mph<br>Spin: {row['avg_spin_rate']:.0f} rpm<br>IVB: {row['avg_ivb']:.1f}\" | HB: {row['avg_hb']:.1f}\"" 
              for _, row in pitch_summary.iterrows()],
        textposition='middle',
        textfont=dict(color='white', size=10),
        marker_color=px.colors.sequential.Viridis[:len(pitch_summary)],
        hovertemplate='<b>%{x}</b><br>Count: %{y}<br>%{text}<extra></extra>'
    ))

    fig.update_layout(
        title=f'{pitcher_name} - Pitch Usage & Profile (2025) | Total CSW: {total_csw_pct:.1f}%',
        xaxis_title='Pitch Type',
        yaxis_title='Frequency (Total Thrown)',
        xaxis_tickangle=-45,
        width=900,
        height=600,
        showlegend=False
    )

    return fig

def create_usage_by_count_plot(usage_by_count_pct):
    """Create pitch usage by count heatmap"""
    fig = go.Figure(data=go.Heatmap(
        z=usage_by_count_pct.values,
        x=usage_by_count_pct.columns,
        y=usage_by_count_pct.index,
        colorscale='Viridis',
        text=usage_by_count_pct.round(1).values,
        texttemplate="%{text}%",
        textfont={"size": 10},
        colorbar=dict(title="Usage %")
    ))

    fig.update_layout(
        title='Pitch Usage by Count (%)',
        xaxis_title='Pitch Type',
        yaxis_title='Count',
        width=800,
        height=500
    )

    return fig

def create_enhanced_heatmap_plot(df, pitcher_name):
    """Create enhanced pitch location heatmap with better gradient"""
    fig = go.Figure()

    # Create more granular 2D histogram with better color scale
    fig.add_trace(go.Histogram2d(
        x=df['plate_x'],
        y=df['plate_z'],
        colorscale=[[0, 'blue'], [0.3, 'lightblue'], [0.6, 'yellow'], [0.8, 'orange'], [1, 'red']],
        showscale=True,
        colorbar=dict(title="Pitch Density", titleside="right"),
        nbinsx=40,  # More granular
        nbinsy=40,  # More granular
        hovertemplate='<b>Location Density</b><br>X: %{x:.1f}"<br>Z: %{y:.1f}"<br>Count: %{z}<extra></extra>'
    ))

    # Add strike zone
    strike_zone_width_inches = 17.0
    strike_zone_top_feet = 3.2
    strike_zone_bottom_feet = 1.6
    strike_zone_x = [-strike_zone_width_inches / 2, strike_zone_width_inches / 2]
    strike_zone_z = [strike_zone_bottom_feet * 12, strike_zone_top_feet * 12]
    
    fig.add_shape(
        type="rect",
        x0=strike_zone_x[0], y0=strike_zone_z[0],
        x1=strike_zone_x[1], y1=strike_zone_z[1],
        line=dict(color="black", width=3),
        fillcolor="rgba(0,0,0,0)"
    )

    # Add plate
    plate_width = 17.0
    fig.add_shape(
        type="rect",
        x0=-plate_width/2, y0=0,
        x1=plate_width/2, y1=2,
        line=dict(color="black", width=2),
        fillcolor="rgba(139,69,19,0.3)"  # Brown plate
    )

    fig.update_layout(
        title=f"{pitcher_name} - Enhanced Pitch Location Heatmap (Catcher's View)",
        xaxis_title='Horizontal Location (Inches from Center)',
        yaxis_title='Vertical Location (Inches from Ground)',
        xaxis=dict(range=[-30, 30]),
        yaxis=dict(range=[-2, 60]),
        width=700,
        height=700,
        plot_bgcolor='lightgray'
    )

    return fig

def process_data(df):
    """Process the uploaded data and return all analysis results"""
    # Extract pitcher name
    if 'player_name' in df.columns:
        pitcher_name = df['player_name'].iloc[0]
    else:
        pitcher_name = "Unknown Pitcher"

    # Define column names
    IVB_COL = 'pfx_z'
    HB_COL = 'pfx_x'

    # Data transformations
    for col in [HB_COL, 'plate_x']:
        if col in df.columns:
            df[col] = df[col] * -1

    # Convert measurements from feet to inches
    columns_to_convert = [IVB_COL, HB_COL, 'plate_x', 'plate_z', 'release_extension']
    for col in columns_to_convert:
        if col in df.columns:
            df[col] = df[col] * 12

    # Pitch types mapping
    PITCH_TYPES_MAP = {
        "Four-Seam Fastball": "FF", "Sinker": "SI", "Cutter": "FC", "Changeup": "CH",
        "Split-Finger": "FS", "Forkball": "FO", "Screwball": "SC", "Curveball": "CU",
        "Knuckle Curve": "KC", "Slow Curve": "CS", "Slider": "SL", "Sweeper": "ST",
        "Slurve": "SV", "Knuckleball": "KN", "Eephus": "EP", "Fastball": "FA",
        "Intentional Ball": "IN", "Pitchout": "PO",
    }
    df['pitch_name_full'] = df['pitch_name'].map(PITCH_TYPES_MAP).fillna(df['pitch_name'])

    # Calculate CSW%
    csw_events = ['called_strike', 'swinging_strike', 'foul_tip', 'swinging_strike_blocked']
    df['is_csw'] = df['description'].isin(csw_events)
    total_csw_pct = df['is_csw'].mean() * 100

    # Calculate K% and BB%
    k_pct, bb_pct, total_pa, strikeouts, walks = calculate_k_bb_percentages(df)

    # Pitch summary
    csw_by_pitch = df.groupby('pitch_name_full')['is_csw'].mean().reset_index(name='csw_pct')
    csw_by_pitch['csw_pct'] *= 100
    pitch_summary = df.groupby('pitch_name_full').agg(
        count=('pitch_name', 'size'),
        avg_velocity=('release_speed', 'mean'),
        avg_spin_rate=('release_spin_rate', 'mean'),
        avg_ivb=(IVB_COL, 'mean'),
        avg_hb=(HB_COL, 'mean'),
        avg_extension=('release_extension', 'mean')
    ).merge(csw_by_pitch, on='pitch_name_full').sort_values('count', ascending=False)

    # Additional analysis
    usage_by_count = analyze_pitch_usage_by_count(df)
    detailed_stats = calculate_detailed_stats(df)

    return df, pitcher_name, IVB_COL, HB_COL, {
        'pitch_summary': pitch_summary.to_dict('records'),
        'total_csw_pct': total_csw_pct,
        'k_pct': k_pct,
        'bb_pct': bb_pct,
        'total_pa': total_pa,
        'strikeouts': strikeouts,
        'walks': walks,
        'usage_by_count': usage_by_count.to_dict(),
        'detailed_stats': detailed_stats
    }

def parse_uploaded_file(contents, filename):
    """Parse uploaded CSV file"""
    try:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        
        if 'csv' in filename:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            return df, None
        else:
            return None, "Please upload a CSV file"
    except Exception as e:
        return None, f"Error processing file: {str(e)}"

def generate_comprehensive_pdf_report(df, pitcher_name, analysis_data):
    """Generate comprehensive PDF report matching Streamlit version"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
    
    # Get styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=TA_CENTER,
        textColor=colors.darkblue
    )
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        spaceAfter=12,
        textColor=colors.darkblue
    )
    subheading_style = ParagraphStyle(
        'CustomSubHeading',
        parent=styles['Heading3'],
        fontSize=14,
        spaceAfter=10,
        textColor=colors.black
    )
    
    # Build story
    story = []
    
    # Title Page
    story.append(Paragraph(pitcher_name, title_style))
    story.append(Paragraph("Pitching Performance Analysis Report", heading_style))
    story.append(Paragraph(f"Generated on: {datetime.now().strftime('%B %d, %Y')}", styles['Normal']))
    story.append(Spacer(1, 0.5*inch))
    
    # Executive Summary
    story.append(Paragraph("Executive Summary", heading_style))
    
    total_pitches = len(df)
    unique_pitch_types = df['pitch_name_full'].nunique()
    pitch_summary = analysis_data['pitch_summary']
    detailed_stats = analysis_data['detailed_stats']
    
    summary_text = f"""
    This report analyzes {total_pitches:,} pitches thrown by {pitcher_name}, featuring {unique_pitch_types} different pitch types. 
    The analysis includes pitch movement profiles, usage patterns, location tendencies, and performance metrics against opposing batters.
    
    Key Highlights:
    • Total Called Strike + Whiff Rate (CSW%): {analysis_data['total_csw_pct']:.1f}%
    • Strikeout Rate (K%): {analysis_data['k_pct']:.1f}%
    • Walk Rate (BB%): {analysis_data['bb_pct']:.1f}%
    • Most used pitch: {pitch_summary[0]['pitch_name_full']} ({pitch_summary[0]['count']} pitches)
    • Average velocity across all pitches: {df['release_speed'].mean():.1f} mph
    """
    
    # Add batted ball stats if available
    if 'no_data' not in detailed_stats['batted_ball_stats']:
        bb_stats = detailed_stats['batted_ball_stats']
        summary_text += f"""
    • Hard hit rate allowed: {bb_stats['total_hard_hit_pct']:.1f}%
    • Total batted balls: {bb_stats['total_batted_balls']}
        """
        if bb_stats['total_high_xwoba_pct'] > 0:
            summary_text += f"    • High xwOBA rate allowed: {bb_stats['total_high_xwoba_pct']:.1f}%"
    
    # Add batting stats if available
    if 'no_data' not in detailed_stats['batting_stats']:
        batting_stats = detailed_stats['batting_stats']['overall_stats']
        summary_text += f"""
    • Opponent batting average: {batting_stats['AVG']:.3f}
    • Opponent OPS: {batting_stats['OPS']:.3f}
        """
    
    story.append(Paragraph(summary_text, styles['Normal']))
    story.append(Spacer(1, 12))
    
    # Key Metrics Table
    summary_data = [
        ['Metric', 'Value'],
        ['Total Pitches', f"{total_pitches:,}"],
        ['CSW%', f"{analysis_data['total_csw_pct']:.1f}%"],
        ['K%', f"{analysis_data['k_pct']:.1f}%"],
        ['BB%', f"{analysis_data['bb_pct']:.1f}%"],
        ['Average Velocity', f"{df['release_speed'].mean():.1f} mph"],
        ['Pitch Types', str(unique_pitch_types)],
        ['Strike Rate', f"{detailed_stats['strike_rate']:.1f}%"],
        ['First Strike %', f"{detailed_stats['first_strike_pct']:.1f}%"]
    ]
    
    summary_table = Table(summary_data)
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(summary_table)
    story.append(PageBreak())
    
    # Pitch Breakdown
    story.append(Paragraph("Pitch Type Breakdown", heading_style))
    
    pitch_data = [['Pitch Type', 'Count', 'Usage %', 'Avg Velocity', 'CSW%', 'Avg Spin']]
    
    for pitch_info in pitch_summary:
        usage_pct = (pitch_info['count'] / total_pitches) * 100
        pitch_data.append([
            pitch_info['pitch_name_full'],
            str(pitch_info['count']),
            f"{usage_pct:.1f}%",
            f"{pitch_info['avg_velocity']:.1f} mph",
            f"{pitch_info['csw_pct']:.1f}%",
            f"{pitch_info['avg_spin_rate']:.0f} rpm"
        ])
    
    pitch_table = Table(pitch_data)
    pitch_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(pitch_table)
    story.append(PageBreak())
    
    # Batted Ball Analysis
    if 'no_data' not in detailed_stats['batted_ball_stats']:
        story.append(Paragraph("Batted Ball Analysis", heading_style))
        bb_stats = detailed_stats['batted_ball_stats']
        
        bb_text = f"""
        Analysis of {bb_stats['total_batted_balls']} batted balls:
        
        • Hard Hit Rate (≥95 mph): {bb_stats['total_hard_hit_pct']:.1f}%
        • High xwOBA Rate (≥0.350): {bb_stats['total_high_xwoba_pct']:.1f}%
        
        Hard Hit Rate by Pitch Type:
        """
        
        story.append(Paragraph(bb_text, styles['Normal']))
        
        # Hard hit by pitch table
        if bb_stats['hard_hit_by_pitch']:
            hh_data = [['Pitch Type', 'Hard Hit %']]
            for pitch, rate in bb_stats['hard_hit_by_pitch'].items():
                hh_data.append([pitch, f"{rate:.1f}%"])
            
            hh_table = Table(hh_data)
            hh_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(hh_table)
        
        story.append(PageBreak())
    
    # Opponent Batting Performance
    if 'no_data' not in detailed_stats['batting_stats']:
        story.append(Paragraph("Opponent Batting Performance", heading_style))
        batting_stats = detailed_stats['batting_stats']
        
        # Overall stats
        story.append(Paragraph("Overall Statistics", subheading_style))
        overall = batting_stats['overall_stats']
        
        overall_data = [
            ['Statistic', 'Value'],
            ['Plate Appearances', str(int(overall['PA']))],
            ['At Bats', str(int(overall['AB']))],
            ['Hits', str(int(overall['H']))],
            ['Batting Average', f"{overall['AVG']:.3f}"],
            ['On-Base Percentage', f"{overall['OBP']:.3f}"],
            ['Slugging Percentage', f"{overall['SLG']:.3f}"],
            ['OPS', f"{overall['OPS']:.3f}"]
        ]
        
        overall_table = Table(overall_data)
        overall_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(overall_table)
        story.append(Spacer(1, 12))
        
        # By handedness if available
        if batting_stats['stats_by_handedness']:
            story.append(Paragraph("Performance vs Batter Handedness", subheading_style))
            
            hand_data = [['Handedness', 'PA', 'AVG', 'OBP', 'SLG', 'OPS']]
            for hand, stats in batting_stats['stats_by_handedness'].items():
                hand_data.append([
                    'Left' if hand == 'L' else 'Right',
                    str(int(stats['PA'])),
                    f"{stats['AVG']:.3f}",
                    f"{stats['OBP']:.3f}",
                    f"{stats['SLG']:.3f}",
                    f"{stats['OPS']:.3f}"
                ])
            
            hand_table = Table(hand_data)
            hand_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(hand_table)
    
    # Footer
    story.append(Spacer(1, 20))
    story.append(Paragraph(f"Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Paragraph("Generated by Datanalytics.pro Pitcher Profiler", styles['Normal']))
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer

def init_dash_app(server, route_prefix):
    """Initialize the complete enhanced Dash app"""
    
    app = dash.Dash(
        __name__,
        server=server,
        url_base_pathname=route_prefix,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        suppress_callback_exceptions=True
    )

    # App layout
    app.layout = dbc.Container([
        dcc.Store(id='processed-data-store'),
        dcc.Store(id='pitcher-name-store'),
        dcc.Download(id="download-pdf"),
        
        # Header with navigation
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.A("← Back to Dashboard", href="/dashboard/", className="btn btn-outline-secondary btn-sm me-2"),
                    html.A("← Pitcher Profiler Home", href="/dashboard/pitch_profiler/", className="btn btn-outline-primary btn-sm"),
                ], className="mb-3"),
                html.H1("Baseball Pitcher Profiler", className="text-center mb-4 text-primary"),
                html.P("Upload your Statcast CSV data to generate comprehensive pitching analysis", 
                       className="text-center text-muted mb-4")
            ])
        ]),
        
        # Upload section
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("📁 Upload Data", className="card-title"),
                        dcc.Upload(
                            id='upload-data',
                            children=html.Div([
                                'Drag and Drop or ',
                                html.A('Select CSV File')
                            ]),
                            style={
                                'width': '100%',
                                'height': '60px',
                                'lineHeight': '60px',
                                'borderWidth': '1px',
                                'borderStyle': 'dashed',
                                'borderRadius': '5px',
                                'textAlign': 'center',
                                'margin': '10px'
                            },
                            multiple=False
                        ),
                        html.Div(id='upload-status', className="mt-2")
                    ])
                ])
            ], width=12)
        ], className="mb-4"),
        
        # Main content (hidden until data is uploaded)
        html.Div(id='main-content', style={'display': 'none'}, children=[
            # Metrics row
            html.Div(id='metrics-row'),
            
            html.Hr(),
            
            # Tabs for different analyses
            dbc.Tabs([
                dbc.Tab(label="🎯 Movement Profile", tab_id="movement"),
                dbc.Tab(label="📈 Usage & Performance", tab_id="usage"),
                dbc.Tab(label="🗺️ Location Heatmap", tab_id="location"),
                dbc.Tab(label="📋 Detailed Stats", tab_id="stats")
            ], id="analysis-tabs", active_tab="movement"),
            
            html.Div(id="tab-content", className="mt-4"),
            
            html.Hr(),
            
            # PDF Generation section
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("📄 Complete Report"),
                            html.P("Generate a comprehensive PDF report with all analysis, visualizations, batted ball stats, and opponent batting performance."),
                            dbc.Button("🎯 Generate Complete PDF Report", id="generate-pdf-btn", color="primary", className="me-2"),
                            html.Div(id="pdf-status")
                        ])
                    ])
                ])
            ], className="mb-4")
        ])
    ], fluid=True)

    # Callbacks
    @app.callback(
        [Output('upload-status', 'children'),
         Output('processed-data-store', 'data'),
         Output('pitcher-name-store', 'data'),
         Output('main-content', 'style')],
        [Input('upload-data', 'contents')],
        [State('upload-data', 'filename')]
    )
    def process_uploaded_file(contents, filename):
        if contents is None:
            return "", {}, "", {'display': 'none'}
        
        df, error = parse_uploaded_file(contents, filename)
        
        if error:
            return dbc.Alert(f"Error: {error}", color="danger"), {}, "", {'display': 'none'}
        
        try:
            df_processed, pitcher_name, IVB_COL, HB_COL, analysis_data = process_data(df)
            
            # Store processed data - ensure all data is JSON serializable
            stored_data = {
                'df': df_processed.to_json(date_format='iso', orient='split'),
                'IVB_COL': IVB_COL,
                'HB_COL': HB_COL,
                'analysis': analysis_data
            }
            
            success_message = dbc.Alert(
                f"✅ File uploaded successfully! {len(df_processed)} rows processed for {pitcher_name}",
                color="success"
            )
            
            return success_message, stored_data, pitcher_name, {'display': 'block'}
            
        except Exception as e:
            error_message = dbc.Alert(f"Error processing data: {str(e)}", color="danger")
            return error_message, {}, "", {'display': 'none'}

    @app.callback(
        Output('metrics-row', 'children'),
        [Input('processed-data-store', 'data'),
         Input('pitcher-name-store', 'data')]
    )
    def update_metrics(stored_data, pitcher_name):
        if not stored_data:
            return ""
        
        df = pd.read_json(io.StringIO(stored_data['df']), orient='split')
        analysis = stored_data['analysis']
        
        # Additional metrics
        detailed_stats = analysis['detailed_stats']
        
        return html.Div([
            dbc.Row([
                dbc.Col([
                    html.H2(f"📊 Analysis for: {pitcher_name}", className="text-primary mb-3")
                ], width=12)
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(str(len(df)), className="text-primary mb-0"),
                            html.P("Total Pitches", className="text-muted mb-0")
                        ])
                    ])
                ], width=2),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(f"{analysis['total_csw_pct']:.1f}%", className="text-primary mb-0"),
                            html.P("CSW%", className="text-muted mb-0")
                        ])
                    ])
                ], width=2),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(f"{analysis['k_pct']:.1f}%", className="text-primary mb-0"),
                            html.P("K%", className="text-muted mb-0")
                        ])
                    ])
                ], width=2),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(f"{analysis['bb_pct']:.1f}%", className="text-primary mb-0"),
                            html.P("BB%", className="text-muted mb-0")
                        ])
                    ])
                ], width=2),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(f"{df['release_speed'].mean():.1f} mph", className="text-primary mb-0"),
                            html.P("Avg Velocity", className="text-muted mb-0")
                        ])
                    ])
                ], width=2),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(str(df['pitch_name_full'].nunique()), className="text-primary mb-0"),
                            html.P("Pitch Types", className="text-muted mb-0")
                        ])
                    ])
                ], width=2)
            ], className="mb-3"),
            
            # Additional detailed metrics
            dbc.Card([
                dbc.CardBody([
                    html.H5("📈 Detailed Statistics", className="mb-3"),
                    dbc.Row([
                        dbc.Col([
                            html.P(f"• Total Plate Appearances: {analysis['total_pa']}", className="mb-1"),
                            html.P(f"• Total Strikeouts: {analysis['strikeouts']}", className="mb-1"),
                            html.P(f"• Total Walks: {analysis['walks']}", className="mb-1"),
                        ], width=4),
                        dbc.Col([
                            html.P(f"• Strike Rate: {detailed_stats['strike_rate']:.1f}%", className="mb-1"),
                            html.P(f"• First Strike %: {detailed_stats['first_strike_pct']:.1f}%", className="mb-1"),
                            html.P(f"• Swinging Strike Rate: {detailed_stats['swinging_strike_rate']:.1f}%", className="mb-1"),
                        ], width=4),
                        dbc.Col([
                            html.P(f"• Zone Rate: {detailed_stats.get('zone_rate', 'N/A'):.1f}%" if isinstance(detailed_stats.get('zone_rate'), (int, float)) else "• Zone Rate: N/A", className="mb-1"),
                            html.P(f"• Max Velocity: {detailed_stats['max_velocity']:.1f} mph", className="mb-1"),
                            html.P(f"• Min Velocity: {detailed_stats['min_velocity']:.1f} mph", className="mb-1"),
                        ], width=4)
                    ])
                ])
            ], className="mb-3")
        ])

    @app.callback(
        Output('tab-content', 'children'),
        [Input('analysis-tabs', 'active_tab'),
         Input('processed-data-store', 'data'),
         Input('pitcher-name-store', 'data')]
    )
    def update_tab_content(active_tab, stored_data, pitcher_name):
        if not stored_data:
            return ""
        
        df = pd.read_json(io.StringIO(stored_data['df']), orient='split')
        IVB_COL = stored_data['IVB_COL']
        HB_COL = stored_data['HB_COL']
        analysis = stored_data['analysis']
        
        if active_tab == "movement":
            fig = create_movement_plot(df, pitcher_name, IVB_COL, HB_COL)
            return dcc.Graph(figure=fig, style={'height': '800px'})
            
        elif active_tab == "usage":
            # Convert pitch_summary back to DataFrame for plotting
            pitch_summary_df = pd.DataFrame(analysis['pitch_summary'])
            fig1 = create_frequency_plot(df, pitcher_name, pitch_summary_df, analysis['total_csw_pct'])
            
            # Create usage by count plot
            usage_by_count_df = pd.DataFrame(analysis['usage_by_count'])
            fig2 = create_usage_by_count_plot(usage_by_count_df)
            
            # Round values for display
            display_summary = pitch_summary_df.round(1)
            
            return html.Div([
                dcc.Graph(figure=fig1, style={'height': '600px'}),
                html.Hr(),
                dcc.Graph(figure=fig2, style={'height': '500px'}),
                html.Hr(),
                html.H4("Pitch Summary Table", className="mt-4"),
                dash_table.DataTable(
                    data=display_summary.to_dict('records'),
                    columns=[{"name": i.replace('_', ' ').title(), "id": i} for i in display_summary.columns],
                    style_cell={'textAlign': 'center'},
                    style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
                    style_data_conditional=[
                        {
                            'if': {'row_index': 0},
                            'backgroundColor': 'rgb(248, 248, 255)',
                        }
                    ]
                )
            ])
            
        elif active_tab == "location":
            fig = create_enhanced_heatmap_plot(df, pitcher_name)
            return html.Div([
                dcc.Graph(figure=fig, style={'height': '700px'}),
                html.Div([
                    html.H5("Location Analysis", className="mt-4"),
                    html.P("• Red areas indicate higher pitch density"),
                    html.P("• Blue areas indicate lower pitch density"),
                    html.P("• Black rectangle shows the strike zone"),
                    html.P("• Brown rectangle shows home plate")
                ], className="mt-3")
            ])
            
        elif active_tab == "stats":
            detailed_stats = analysis['detailed_stats']
            
            components = []
            
            # Basic Statistics Section
            components.extend([
                html.H4("Basic Statistics", className="mb-3"),
                dash_table.DataTable(
                    data=[
                        {"Metric": "Total Pitches", "Value": f"{detailed_stats['total_pitches']:,}"},
                        {"Metric": "Total Strikes", "Value": f"{detailed_stats['total_strikes']:,}"},
                        {"Metric": "Total Balls", "Value": f"{detailed_stats['total_balls']:,}"},
                        {"Metric": "Strike Rate", "Value": f"{detailed_stats['strike_rate']:.1f}%"},
                        {"Metric": "First Strike %", "Value": f"{detailed_stats['first_strike_pct']:.1f}%"},
                        {"Metric": "Swinging Strike Rate", "Value": f"{detailed_stats['swinging_strike_rate']:.1f}%"}
                    ],
                    columns=[{"name": "Metric", "id": "Metric"}, {"name": "Value", "id": "Value"}],
                    style_cell={'textAlign': 'left'},
                    style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'}
                ),
                html.Hr()
            ])
            
            # Batted Ball Analysis Section
            if 'no_data' not in detailed_stats['batted_ball_stats']:
                bb_stats = detailed_stats['batted_ball_stats']
                components.extend([
                    html.H4("🏏 Batted Ball Analysis", className="mb-3"),
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.H3(f"{bb_stats['total_hard_hit_pct']:.1f}%", className="text-danger mb-0"),
                                    html.P("Hard Hit Rate", className="text-muted mb-0"),
                                    html.Small(f"(≥{bb_stats['hard_hit_threshold']} mph)")
                                ])
                            ])
                        ], width=4),
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.H3(f"{bb_stats['total_high_xwoba_pct']:.1f}%", className="text-warning mb-0"),
                                    html.P("High xwOBA Rate", className="text-muted mb-0"),
                                    html.Small(f"(≥{bb_stats['xwoba_threshold']:.3f})")
                                ])
                            ])
                        ], width=4),
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.H3(f"{bb_stats['total_batted_balls']}", className="text-info mb-0"),
                                    html.P("Total Batted Balls", className="text-muted mb-0")
                                ])
                            ])
                        ], width=4)
                    ], className="mb-3"),
                    
                    html.H5("Hard Hit % by Pitch Type"),
                    dash_table.DataTable(
                        data=[{"Pitch Type": pitch, "Hard Hit %": f"{rate:.1f}%"} 
                              for pitch, rate in bb_stats['hard_hit_by_pitch'].items()],
                        columns=[{"name": "Pitch Type", "id": "Pitch Type"}, {"name": "Hard Hit %", "id": "Hard Hit %"}],
                        style_cell={'textAlign': 'center'},
                        style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'}
                    ),
                    html.Hr()
                ])
            
            # Opponent Batting Performance Section
            if 'no_data' not in detailed_stats['batting_stats']:
                batting_stats = detailed_stats['batting_stats']
                components.extend([
                    html.H4("🏏 Opponent Batting Performance", className="mb-3"),
                    
                    html.H5("Overall Statistics"),
                    dash_table.DataTable(
                        data=[
                            {"Statistic": "Plate Appearances", "Value": str(int(batting_stats['overall_stats']['PA']))},
                            {"Statistic": "At Bats", "Value": str(int(batting_stats['overall_stats']['AB']))},
                            {"Statistic": "Hits", "Value": str(int(batting_stats['overall_stats']['H']))},
                            {"Statistic": "Batting Average", "Value": f"{batting_stats['overall_stats']['AVG']:.3f}"},
                            {"Statistic": "On-Base Percentage", "Value": f"{batting_stats['overall_stats']['OBP']:.3f}"},
                            {"Statistic": "Slugging Percentage", "Value": f"{batting_stats['overall_stats']['SLG']:.3f}"},
                            {"Statistic": "OPS", "Value": f"{batting_stats['overall_stats']['OPS']:.3f}"}
                        ],
                        columns=[{"name": "Statistic", "id": "Statistic"}, {"name": "Value", "id": "Value"}],
                        style_cell={'textAlign': 'left'},
                        style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'}
                    ),
                    
                    html.Hr(),
                    
                    # By handedness if available
                    html.H5("Performance vs Batter Handedness") if batting_stats['stats_by_handedness'] else html.Div(),
                    dash_table.DataTable(
                        data=[
                            {
                                "Handedness": "Left" if hand == "L" else "Right",
                                "PA": str(int(stats['PA'])),
                                "AVG": f"{stats['AVG']:.3f}",
                                "OBP": f"{stats['OBP']:.3f}", 
                                "SLG": f"{stats['SLG']:.3f}",
                                "OPS": f"{stats['OPS']:.3f}"
                            }
                            for hand, stats in batting_stats['stats_by_handedness'].items()
                        ],
                        columns=[
                            {"name": "Handedness", "id": "Handedness"},
                            {"name": "PA", "id": "PA"},
                            {"name": "AVG", "id": "AVG"},
                            {"name": "OBP", "id": "OBP"},
                            {"name": "SLG", "id": "SLG"},
                            {"name": "OPS", "id": "OPS"}
                        ],
                        style_cell={'textAlign': 'center'},
                        style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'}
                    ) if batting_stats['stats_by_handedness'] else html.Div()
                ])
            
            return html.Div(components)
        
        return ""

    @app.callback(
        [Output('pdf-status', 'children'),
         Output('download-pdf', 'data')],
        [Input('generate-pdf-btn', 'n_clicks')],
        [State('processed-data-store', 'data'),
         State('pitcher-name-store', 'data')]
    )
    def generate_pdf_report_callback(n_clicks, stored_data, pitcher_name):
        if not n_clicks or not stored_data:
            return "", None
        
        try:
            df = pd.read_json(io.StringIO(stored_data['df']), orient='split')
            analysis = stored_data['analysis']
            
            # Generate comprehensive PDF
            pdf_buffer = generate_comprehensive_pdf_report(df, pitcher_name, analysis)
            
            # Create download data
            download_data = dict(
                content=base64.b64encode(pdf_buffer.getvalue()).decode(),
                filename=f"comprehensive_pitch_report_{pitcher_name.replace(' ', '_').replace(',', '')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                type="application/pdf",
                base64=True
            )
            
            success_msg = dbc.Alert("✅ Comprehensive PDF report generated successfully! Download should start automatically.", color="success")
            
            return success_msg, download_data
            
        except Exception as e:
            error_msg = dbc.Alert(f"❌ Error generating comprehensive PDF report: {str(e)}", color="danger")
            return error_msg, None

    return app