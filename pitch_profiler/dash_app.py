# app/modules/pitch_profiler/dash_app.py
"""
Complete Enhanced Pitcher Profiler Dash Application
Exact match to Streamlit functionality with comprehensive features
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
    """Calculate K% and BB% from the dataframe - exact match to Streamlit"""
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

def calculate_batting_stats(data):
    """Calculate batting statistics against the pitcher - exact match to Streamlit"""
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
    return pd.Series({'AVG': avg, 'OBP': obp, 'SLG': slg, 'OPS': ops,
                      'PA': plate_appearances_count, 'AB': at_bats_count, 'H': hits})

def create_movement_plot(df, pitcher_name, IVB_COL, HB_COL):
    """Create the horizontal vs vertical break plot using Plotly (exact match to Streamlit)"""
    fig = go.Figure()

    # Add concentric circles for break magnitude (matching Streamlit exactly)
    theta = np.linspace(0, 2*np.pi, 100)
    for radius in range(4, 21, 4):
        x_circle = radius * np.cos(theta)
        y_circle = radius * np.sin(theta)
        fig.add_trace(go.Scatter(
            x=x_circle, y=y_circle,
            mode='lines',
            line=dict(color='grey', dash='dash', width=1),
            showlegend=False,
            opacity=0.6,
            hoverinfo='skip'
        ))

    # Color mapping for pitch types (using viridis like Streamlit)
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
                size=10,
                color=color_map[pitch_type],
                opacity=0.6
            ),
            name=pitch_type,
            text=[f"Velo: {v:.1f} mph" for v in pitch_data['release_speed']],
            hovertemplate=f'<b>{pitch_type}</b><br>%{{text}}<br>HB: %{{x:.1f}}"<br>IVB: %{{y:.1f}}"<extra></extra>'
        ))

    # Calculate centroids and add annotations (matching Streamlit exactly)
    pitch_summary_for_plot = df.groupby('pitch_name_full').agg(
        avg_ivb=(IVB_COL, 'mean'),
        avg_hb=(HB_COL, 'mean'),
        avg_velocity=('release_speed', 'mean'),
        avg_spin_rate=('release_spin_rate', 'mean'),
        avg_extension=('release_extension', 'mean')
    )

    for pitch, data in pitch_summary_for_plot.iterrows():
        annotation_text = (
            f"<b>{pitch}</b><br>"
            f"Velo: {data['avg_velocity']:.1f} mph<br>"
            f"Spin: {data['avg_spin_rate']:.0f} rpm<br>"
            f"Break (HB/IVB): {data['avg_hb']:.1f}\" / {data['avg_ivb']:.1f}\"<br>"
            f"Extension: {data['avg_extension']:.1f}\""
        )
        fig.add_annotation(
            x=data['avg_hb'],
            y=data['avg_ivb'],
            text=annotation_text,
            showarrow=True,
            arrowhead=2,
            arrowcolor="black",
            arrowwidth=1,
            arrowsize=1,
            ax=10,
            ay=-15,
            bgcolor="white",
            bordercolor="black",
            borderwidth=1,
            font=dict(size=9)
        )

    # Set 1:1 scale and center the origin (matching Streamlit exactly)
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

    # Add grid (matching Streamlit)
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

    return fig

def create_frequency_plot(df, pitcher_name, pitch_summary, total_csw_pct):
    """Create the pitch frequency and averages plot using Plotly (exact match to Streamlit)"""
    fig = go.Figure()

    # Create bar chart
    fig.add_trace(go.Bar(
        x=pitch_summary['pitch_name_full'],
        y=pitch_summary['count'],
        marker_color=px.colors.sequential.Plasma[:len(pitch_summary)],
        hovertemplate='<b>%{x}</b><br>Count: %{y}<extra></extra>'
    ))

    # Add annotations for each bar (matching Streamlit exactly)
    for i, row in pitch_summary.iterrows():
        annotation_text = (
            f"CSW: {row['csw_pct']:.1f}%<br>"
            f"Velo: {row['avg_velocity']:.1f} mph<br>"
            f"Spin: {row['avg_spin_rate']:.0f} rpm<br>"
            f"IVB: {row['avg_ivb']:.1f}\" | HB: {row['avg_hb']:.1f}\""
        )
        # Add annotation at the center of each bar
        fig.add_annotation(
            x=row['pitch_name_full'],
            y=row['count'] / 2,  # Middle of the bar
            text=annotation_text,
            showarrow=False,
            font=dict(color='white', size=10, family="Arial"),
            align="center"
        )

    fig.update_layout(
        title=f'{pitcher_name} - Pitch Usage & Profile (2025) | Total CSW: {total_csw_pct:.1f}%',
        xaxis_title='Pitch Type',
        yaxis_title='Frequency (Total Thrown)',
        xaxis_tickangle=-45,
        width=900,
        height=600,
        showlegend=False,
        font=dict(size=12)
    )

    return fig

def create_heatmap_plot(df, pitcher_name):
    """Create the pitch location heatmap (exact match to Streamlit)"""
    fig = go.Figure()

    # Create 2D histogram for heatmap (matching Streamlit kde density)
    fig.add_trace(go.Histogram2d(
        x=df['plate_x'],
        y=df['plate_z'],
        colorscale='viridis_r',  # Using valid Plotly colorscale
        showscale=True,
        colorbar=dict(title="Pitch Density"),
        nbinsx=50,  # Higher resolution like kde
        nbinsy=50,
        hovertemplate='<b>Location</b><br>X: %{x:.1f}"<br>Z: %{y:.1f}"<br>Count: %{z}<extra></extra>'
    ))

    # Add strike zone (matching Streamlit exactly)
    strike_zone_width_inches = 17.0
    strike_zone_top_feet = 3.2
    strike_zone_bottom_feet = 1.6
    strike_zone_x = [-strike_zone_width_inches / 2, strike_zone_width_inches / 2]
    strike_zone_z = [strike_zone_bottom_feet * 12, strike_zone_top_feet * 12]
    
    # Strike zone outline
    fig.add_shape(
        type="rect",
        x0=strike_zone_x[0], y0=strike_zone_z[0],
        x1=strike_zone_x[1], y1=strike_zone_z[1],
        line=dict(color="red", width=2, dash="dash"),
        fillcolor="rgba(0,0,0,0)"
    )

    fig.update_layout(
        title=f"{pitcher_name} - Pitch Heatmap (Catcher's View)",
        xaxis_title='Horizontal Location (Inches from Center)',
        yaxis_title='Vertical Location (Inches from Ground)',
        xaxis=dict(range=[-24, 24]),
        yaxis=dict(range=[0, 54]),
        width=600,
        height=600
    )

    # Set equal aspect ratio
    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
    )

    return fig

def process_data(df):
    """Process the uploaded data and return all analysis results - exact match to Streamlit"""
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
    
    # Batted ball analysis
    batted_balls = df[df['description'] == 'hit_into_play'].copy()
    batted_ball_stats = {}
    
    if not batted_balls.empty:
        hard_hit_threshold = 95
        batted_balls['is_hard_hit'] = batted_balls['launch_speed'] >= hard_hit_threshold
        total_hard_hit_pct = batted_balls['is_hard_hit'].mean() * 100
        hard_hit_by_pitch = batted_balls.groupby('pitch_name_full')['is_hard_hit'].mean() * 100
        
        xwoba_threshold = 0.350
        batted_balls['is_high_xwoba'] = batted_balls['estimated_woba_using_speedangle'] >= xwoba_threshold
        total_high_xwoba_pct = batted_balls['is_high_xwoba'].mean() * 100
        high_xwoba_by_pitch = batted_balls.groupby('pitch_name_full')['is_high_xwoba'].mean() * 100
        
        batted_ball_stats = {
            'total_hard_hit_pct': total_hard_hit_pct,
            'hard_hit_by_pitch': hard_hit_by_pitch,
            'total_high_xwoba_pct': total_high_xwoba_pct,
            'high_xwoba_by_pitch': high_xwoba_by_pitch,
            'hard_hit_threshold': hard_hit_threshold,
            'xwoba_threshold': xwoba_threshold
        }
    else:
        batted_ball_stats = {'no_data': True}
    
    # Batting stats calculation - define function inside to match Streamlit exactly
    def calculate_batting_stats_inner(data):
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
        return pd.Series({'AVG': avg, 'OBP': obp, 'SLG': slg, 'OPS': ops,
                          'PA': plate_appearances_count, 'AB': at_bats_count, 'H': hits})
    
    at_bats_final_pitch = df.loc[df.dropna(subset=['events']).groupby(['game_date', 'at_bat_number'])['pitch_number'].idxmax()]
    batting_stats = {}
    
    if not at_bats_final_pitch.empty:
        overall_stats = calculate_batting_stats_inner(at_bats_final_pitch)
        stats_by_pitch = at_bats_final_pitch.groupby('pitch_name_full').apply(calculate_batting_stats_inner)
        stats_by_handedness = at_bats_final_pitch.groupby('stand').apply(calculate_batting_stats_inner)
        
        batting_stats = {
            'overall_stats': overall_stats,
            'stats_by_pitch': stats_by_pitch,
            'stats_by_handedness': stats_by_handedness
        }
    else:
        batting_stats = {'no_data': True}
    
    # Compile all data
    pdf_data = {
        'pitch_summary': pitch_summary,
        'total_csw_pct': total_csw_pct,
        'k_pct': k_pct,
        'bb_pct': bb_pct,
        'total_pa': total_pa,
        'strikeouts': strikeouts,
        'walks': walks,
        'batted_ball_stats': batted_ball_stats,
        'batting_stats': batting_stats
    }
    
    return df, pitcher_name, IVB_COL, HB_COL, pdf_data

def create_pdf_report(df, pitcher_name, pdf_data):
    """Create the PDF report (exact match to Streamlit)"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
    
    story = []
    
    # Define styles (matching Streamlit)
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
    
    # Title Page (exact match to Streamlit)
    story.append(Paragraph(pitcher_name, title_style))
    story.append(Paragraph("Pitching Performance Analysis Report", heading_style))
    story.append(Paragraph(f"Generated on: {datetime.now().strftime('%B %d, %Y')}", styles['Normal']))
    story.append(Spacer(1, 0.5*inch))
    
    # Executive Summary (exact match to Streamlit)
    story.append(Paragraph("Executive Summary", heading_style))
    
    total_pitches = len(df)
    unique_pitch_types = df['pitch_name_full'].nunique()
    
    summary_text = f"""
    This report analyzes {total_pitches} pitches thrown by {pitcher_name}, featuring {unique_pitch_types} different pitch types. 
    The analysis includes pitch movement profiles, usage patterns, location tendencies, and performance metrics against opposing batters.
    
    Key Highlights:
    • Total Called Strike + Whiff Rate (CSW%): {pdf_data['total_csw_pct']:.1f}%
    • Strikeout Rate (K%): {pdf_data['k_pct']:.1f}%
    • Walk Rate (BB%): {pdf_data['bb_pct']:.1f}%
    • Most used pitch: {pdf_data['pitch_summary'].iloc[0]['pitch_name_full']} ({pdf_data['pitch_summary'].iloc[0]['count']} pitches)
    • Average velocity across all pitches: {df['release_speed'].mean():.1f} mph
    """
    
    if 'no_data' not in pdf_data['batted_ball_stats']:
        summary_text += f"""
    • Hard hit rate allowed: {pdf_data['batted_ball_stats']['total_hard_hit_pct']:.1f}%
    • High xwOBA rate allowed: {pdf_data['batted_ball_stats']['total_high_xwoba_pct']:.1f}%
        """
    
    if 'no_data' not in pdf_data['batting_stats']:
        summary_text += f"""
    • Opponent batting average: {pdf_data['batting_stats']['overall_stats']['AVG']:.3f}
    • Opponent OPS: {pdf_data['batting_stats']['overall_stats']['OPS']:.3f}
        """
    
    story.append(Paragraph(summary_text, styles['Normal']))
    story.append(PageBreak())
    
    # Pitch breakdown section (adding missing sections from Streamlit)
    story.append(Paragraph("Pitch Type Breakdown", heading_style))
    
    pitch_data = [['Pitch Type', 'Count', 'Usage %', 'Avg Velocity', 'CSW%', 'Avg Spin']]
    
    for _, pitch_info in pdf_data['pitch_summary'].iterrows():
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
    
    # Batted ball analysis (exact match to Streamlit)
    if 'no_data' not in pdf_data['batted_ball_stats']:
        story.append(Paragraph("Batted Ball Analysis", heading_style))
        bb_stats = pdf_data['batted_ball_stats']
        
        bb_text = f"""
        Analysis of batted balls in play:
        
        • Hard Hit Rate (≥95 mph): {bb_stats['total_hard_hit_pct']:.1f}%
        • High xwOBA Rate (≥0.350): {bb_stats['total_high_xwoba_pct']:.1f}%
        
        Hard Hit Rate by Pitch Type:
        """
        
        story.append(Paragraph(bb_text, styles['Normal']))
        
        # Hard hit by pitch table
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
    
    # Opponent batting performance (exact match to Streamlit)
    if 'no_data' not in pdf_data['batting_stats']:
        story.append(Paragraph("Opponent Batting Performance", heading_style))
        batting_stats = pdf_data['batting_stats']
        
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
        
        # By handedness
        if not batting_stats['stats_by_handedness'].empty:
            story.append(Paragraph("Performance vs Batter Handedness", subheading_style))
            
            hand_data = [['Handedness', 'PA', 'AVG', 'OBP', 'SLG', 'OPS']]
            for hand, stats in batting_stats['stats_by_handedness'].iterrows():
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
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

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

def init_dash_app(server, route_prefix):
    """Initialize the complete Dash app with exact Streamlit functionality"""
    
    app = dash.Dash(
        __name__,
        server=server,
        url_base_pathname=route_prefix,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        suppress_callback_exceptions=True
    )

    # App layout (matching Streamlit structure)
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
                html.H1("⚾ Baseball Pitching Analysis Tool", className="text-center mb-4 text-primary"),
                html.P("Upload your Statcast CSV data to generate comprehensive pitching analysis including:", 
                       className="text-center text-muted"),
                html.Div([
                    html.P("• Pitch movement profiles", className="mb-1"),
                    html.P("• Usage patterns and performance metrics", className="mb-1"),
                    html.P("• Location heatmaps", className="mb-1"),
                    html.P("• Opponent batting statistics", className="mb-1"),
                    html.P("• Complete PDF report", className="mb-1"),
                ], className="text-center text-muted mb-4")
            ])
        ]),
        
        # Upload section (matching Streamlit sidebar style)
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("📁 Upload Data", className="card-title"),
                        dcc.Upload(
                            id='upload-data',
                            children=html.Div([
                                'Drag and Drop or ',
                                html.A('Select CSV File'),
                                html.Br(),
                                html.Small("Upload your Statcast CSV data file", className="text-muted")
                            ]),
                            style={
                                'width': '100%',
                                'height': '80px',
                                'lineHeight': '80px',
                                'borderWidth': '1px',
                                'borderStyle': 'dashed',
                                'borderRadius': '5px',
                                'textAlign': 'center',
                                'margin': '10px',
                                'backgroundColor': '#f8f9fa'
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
            # Metrics row (matching Streamlit exactly)
            html.Div(id='metrics-row'),
            
            html.Hr(),
            
            # Tabs for different analyses (matching Streamlit tabs exactly)
            dbc.Tabs([
                dbc.Tab(label="🎯 Movement Profile", tab_id="movement"),
                dbc.Tab(label="📈 Usage & Performance", tab_id="usage"),
                dbc.Tab(label="🗺️ Location Heatmap", tab_id="location"),
                dbc.Tab(label="📋 Detailed Stats", tab_id="stats")
            ], id="analysis-tabs", active_tab="movement"),
            
            html.Div(id="tab-content", className="mt-4"),
            
            html.Hr(),
            
            # PDF Generation section (matching Streamlit exactly)
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("📄 Complete Report"),
                            html.P("Generate a comprehensive PDF report with all analysis and visualizations."),
                            dbc.Button("🎯 Generate PDF Report", id="generate-pdf-btn", color="primary", className="me-2"),
                            html.Div(id="pdf-status")
                        ])
                    ])
                ])
            ], className="mb-4")
        ])
    ], fluid=True)

    # Callbacks (matching Streamlit functionality exactly)
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
            return dbc.Alert(f"❌ Error: {error}", color="danger"), {}, "", {'display': 'none'}
        
        try:
            df_processed, pitcher_name, IVB_COL, HB_COL, pdf_data = process_data(df)
            
            # Properly serialize the batted ball stats
            batted_ball_serialized = {}
            if 'no_data' not in pdf_data['batted_ball_stats']:
                bb_stats = pdf_data['batted_ball_stats']
                batted_ball_serialized = {
                    'total_hard_hit_pct': bb_stats['total_hard_hit_pct'],
                    'hard_hit_by_pitch': bb_stats['hard_hit_by_pitch'].to_dict() if hasattr(bb_stats['hard_hit_by_pitch'], 'to_dict') else dict(bb_stats['hard_hit_by_pitch']),
                    'total_high_xwoba_pct': bb_stats['total_high_xwoba_pct'],
                    'high_xwoba_by_pitch': bb_stats['high_xwoba_by_pitch'].to_dict() if hasattr(bb_stats['high_xwoba_by_pitch'], 'to_dict') else dict(bb_stats['high_xwoba_by_pitch']),
                    'hard_hit_threshold': bb_stats['hard_hit_threshold'],
                    'xwoba_threshold': bb_stats['xwoba_threshold']
                }
            else:
                batted_ball_serialized = pdf_data['batted_ball_stats']
            
            # Properly serialize the batting stats
            batting_stats_serialized = {}
            if 'no_data' not in pdf_data['batting_stats']:
                bs = pdf_data['batting_stats']
                batting_stats_serialized = {
                    'overall_stats': bs['overall_stats'].to_dict() if hasattr(bs['overall_stats'], 'to_dict') else dict(bs['overall_stats']),
                    'stats_by_pitch': bs['stats_by_pitch'].to_dict() if hasattr(bs['stats_by_pitch'], 'to_dict') else dict(bs['stats_by_pitch']),
                    'stats_by_handedness': bs['stats_by_handedness'].to_dict() if hasattr(bs['stats_by_handedness'], 'to_dict') else dict(bs['stats_by_handedness'])
                }
            else:
                batting_stats_serialized = pdf_data['batting_stats']
            
            # Store processed data
            stored_data = {
                'df': df_processed.to_json(date_format='iso', orient='split'),
                'IVB_COL': IVB_COL,
                'HB_COL': HB_COL,
                'pdf_data': {
                    'pitch_summary': pdf_data['pitch_summary'].to_dict('records'),
                    'total_csw_pct': pdf_data['total_csw_pct'],
                    'k_pct': pdf_data['k_pct'],
                    'bb_pct': pdf_data['bb_pct'],
                    'total_pa': pdf_data['total_pa'],
                    'strikeouts': pdf_data['strikeouts'],
                    'walks': pdf_data['walks'],
                    'batted_ball_stats': batted_ball_serialized,
                    'batting_stats': batting_stats_serialized
                }
            }
            
            success_message = dbc.Alert(
                f"✅ File uploaded successfully! {len(df_processed)} rows loaded.",
                color="success"
            )
            
            return success_message, stored_data, pitcher_name, {'display': 'block'}
            
        except Exception as e:
            # Print the full error for debugging
            import traceback
            print(f"Full error in process_uploaded_file: {traceback.format_exc()}")
            error_message = dbc.Alert(f"❌ Error processing file: {str(e)}", color="danger")
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
        pdf_data = stored_data['pdf_data']
        
        return html.Div([
            # Pitcher name and title (matching Streamlit exactly)
            dbc.Row([
                dbc.Col([
                    html.H2(f"📊 Analysis for: {pitcher_name}", className="text-primary mb-3")
                ], width=12)
            ]),
            
            # Key metrics in columns (matching Streamlit exactly)
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
                            html.H4(f"{pdf_data['total_csw_pct']:.1f}%", className="text-primary mb-0"),
                            html.P("CSW%", className="text-muted mb-0")
                        ])
                    ])
                ], width=2),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(f"{pdf_data['k_pct']:.1f}%", className="text-primary mb-0"),
                            html.P("K%", className="text-muted mb-0")
                        ])
                    ])
                ], width=2),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(f"{pdf_data['bb_pct']:.1f}%", className="text-primary mb-0"),
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
            
            # Additional detailed metrics (matching Streamlit exactly)
            dbc.Card([
                dbc.CardBody([
                    html.H5("📈 Detailed Statistics:", className="mb-3"),
                    html.P(f"• Total Plate Appearances: {pdf_data['total_pa']}", className="mb-1"),
                    html.P(f"• Total Strikeouts: {pdf_data['strikeouts']}", className="mb-1"),
                    html.P(f"• Total Walks: {pdf_data['walks']}", className="mb-1"),
                    html.P(f"• Unique Pitch Types: {df['pitch_name_full'].nunique()}", className="mb-1"),
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
        pdf_data = stored_data['pdf_data']
        
        if active_tab == "movement":
            # Movement Profile tab (exact match to Streamlit)
            fig = create_movement_plot(df, pitcher_name, IVB_COL, HB_COL)
            return html.Div([
                html.H3("Pitch Movement Profile"),
                dcc.Graph(figure=fig, style={'height': '800px'}),
                dbc.Button("📥 Download Movement Plot", color="info", className="mt-3"),
            ])
            
        elif active_tab == "usage":
            # Usage & Performance tab (exact match to Streamlit)
            pitch_summary_df = pd.DataFrame(pdf_data['pitch_summary'])
            fig1 = create_frequency_plot(df, pitcher_name, pitch_summary_df, pdf_data['total_csw_pct'])
            
            return html.Div([
                html.H3("Pitch Usage & Performance"),
                dcc.Graph(figure=fig1, style={'height': '600px'}),
                dbc.Button("📥 Download Usage Plot", color="info", className="mt-3 me-2"),
                
                html.Hr(),
                html.H4("Pitch Summary Table", className="mt-4"),
                dash_table.DataTable(
                    data=pitch_summary_df.round(1).to_dict('records'),
                    columns=[{"name": i.replace('_', ' ').title(), "id": i} for i in pitch_summary_df.columns],
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
            # Location Heatmap tab (exact match to Streamlit)
            fig = create_heatmap_plot(df, pitcher_name)
            return html.Div([
                html.H3("Pitch Location Heatmap"),
                dcc.Graph(figure=fig, style={'height': '700px'}),
                dbc.Button("📥 Download Heatmap", color="info", className="mt-3"),
            ])
            
        elif active_tab == "stats":
            # Detailed Stats tab (exact match to Streamlit)
            components = []
            
            # Batted ball stats section (exact match to Streamlit)
            if 'no_data' not in pdf_data['batted_ball_stats']:
                bb_stats = pdf_data['batted_ball_stats']
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
                        ], width=6),
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.H3(f"{bb_stats['total_high_xwoba_pct']:.1f}%", className="text-warning mb-0"),
                                    html.P("High xwOBA Rate", className="text-muted mb-0"),
                                    html.Small(f"(≥{bb_stats['xwoba_threshold']:.3f})")
                                ])
                            ])
                        ], width=6)
                    ], className="mb-3"),
                    
                    html.H5("**Hard Hit % by Pitch Type:**"),
                    dash_table.DataTable(
                        data=[{"Pitch Type": pitch, "Hard Hit %": f"{rate:.1f}%"} 
                              for pitch, rate in bb_stats['hard_hit_by_pitch'].items()],
                        columns=[{"name": "Pitch Type", "id": "Pitch Type"}, {"name": "Hard Hit %", "id": "Hard Hit %"}],
                        style_cell={'textAlign': 'center'},
                        style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'}
                    ),
                    html.Hr()
                ])
            
            # Opponent batting performance section (exact match to Streamlit)
            if 'no_data' not in pdf_data['batting_stats']:
                batting_stats = pdf_data['batting_stats']
                
                # Safety check for overall_stats
                if 'overall_stats' in batting_stats and batting_stats['overall_stats']:
                    overall_stats = batting_stats['overall_stats']
                    components.extend([
                        html.H4("🏏 Opponent Batting Performance", className="mb-3"),
                        
                        html.H5("**Overall Stats:**"),
                        dash_table.DataTable(
                            data=[
                                {"Statistic": "Plate Appearances", "Value": str(int(overall_stats.get('PA', 0)))},
                                {"Statistic": "At Bats", "Value": str(int(overall_stats.get('AB', 0)))},
                                {"Statistic": "Hits", "Value": str(int(overall_stats.get('H', 0)))},
                                {"Statistic": "Batting Average", "Value": f"{overall_stats.get('AVG', 0):.3f}"},
                                {"Statistic": "On-Base Percentage", "Value": f"{overall_stats.get('OBP', 0):.3f}"},
                                {"Statistic": "Slugging Percentage", "Value": f"{overall_stats.get('SLG', 0):.3f}"},
                                {"Statistic": "OPS", "Value": f"{overall_stats.get('OPS', 0):.3f}"}
                            ],
                            columns=[{"name": "Statistic", "id": "Statistic"}, {"name": "Value", "id": "Value"}],
                            style_cell={'textAlign': 'left'},
                            style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'}
                        ),
                        
                        html.Hr(),
                    ])
                    
                    # By handedness if available (exact match to Streamlit)
                    if 'stats_by_handedness' in batting_stats and batting_stats['stats_by_handedness']:
                        handedness_data = []
                        for hand, stats in batting_stats['stats_by_handedness'].items():
                            if isinstance(stats, dict) and 'PA' in stats:
                                handedness_data.append({
                                    "Handedness": "Left" if hand == "L" else "Right",
                                    "PA": str(int(stats.get('PA', 0))),
                                    "AVG": f"{stats.get('AVG', 0):.3f}",
                                    "OBP": f"{stats.get('OBP', 0):.3f}", 
                                    "SLG": f"{stats.get('SLG', 0):.3f}",
                                    "OPS": f"{stats.get('OPS', 0):.3f}"
                                })
                        
                        if handedness_data:
                            components.extend([
                                html.H5("**By Batter Handedness:**"),
                                dash_table.DataTable(
                                    data=handedness_data,
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
                                )
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
            
            # Reconstruct pdf_data from stored data
            pdf_data = {
                'pitch_summary': pd.DataFrame(stored_data['pdf_data']['pitch_summary']),
                'total_csw_pct': stored_data['pdf_data']['total_csw_pct'],
                'k_pct': stored_data['pdf_data']['k_pct'],
                'bb_pct': stored_data['pdf_data']['bb_pct'],
                'total_pa': stored_data['pdf_data']['total_pa'],
                'strikeouts': stored_data['pdf_data']['strikeouts'],
                'walks': stored_data['pdf_data']['walks'],
                'batted_ball_stats': stored_data['pdf_data']['batted_ball_stats'],
                'batting_stats': stored_data['pdf_data']['batting_stats']
            }
            
            # Convert batting stats back to pandas Series/DataFrame if needed
            if 'no_data' not in pdf_data['batting_stats']:
                bs = pdf_data['batting_stats']
                
                # Safely convert overall_stats
                if 'overall_stats' in bs and bs['overall_stats']:
                    pdf_data['batting_stats']['overall_stats'] = pd.Series(bs['overall_stats'])
                else:
                    # Create empty series with default values
                    pdf_data['batting_stats']['overall_stats'] = pd.Series({
                        'PA': 0, 'AB': 0, 'H': 0, 'AVG': 0.0, 'OBP': 0.0, 'SLG': 0.0, 'OPS': 0.0
                    })
                
                # Handle stats_by_handedness
                if 'stats_by_handedness' in bs and bs['stats_by_handedness']:
                    try:
                        pdf_data['batting_stats']['stats_by_handedness'] = pd.DataFrame(bs['stats_by_handedness']).T
                    except:
                        pdf_data['batting_stats']['stats_by_handedness'] = pd.DataFrame()
                else:
                    pdf_data['batting_stats']['stats_by_handedness'] = pd.DataFrame()
                    
                # Handle stats_by_pitch  
                if 'stats_by_pitch' in bs and bs['stats_by_pitch']:
                    try:
                        pdf_data['batting_stats']['stats_by_pitch'] = pd.DataFrame(bs['stats_by_pitch']).T
                    except:
                        pdf_data['batting_stats']['stats_by_pitch'] = pd.DataFrame()
                else:
                    pdf_data['batting_stats']['stats_by_pitch'] = pd.DataFrame()
            
            # Convert batted ball stats back to pandas Series if needed
            if 'no_data' not in pdf_data['batted_ball_stats']:
                bb_stats = pdf_data['batted_ball_stats']
                if 'hard_hit_by_pitch' in bb_stats:
                    pdf_data['batted_ball_stats']['hard_hit_by_pitch'] = pd.Series(bb_stats['hard_hit_by_pitch'])
                if 'high_xwoba_by_pitch' in bb_stats:
                    pdf_data['batted_ball_stats']['high_xwoba_by_pitch'] = pd.Series(bb_stats['high_xwoba_by_pitch'])
            
            # Generate PDF (exact match to Streamlit)
            pdf_content = create_pdf_report(df, pitcher_name, pdf_data)
            
            # Create download data
            download_data = dict(
                content=base64.b64encode(pdf_content).decode(),
                filename=f"{pitcher_name.replace(' ', '_')}_Pitching_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                type="application/pdf",
                base64=True
            )
            
            success_msg = dbc.Alert("✅ PDF report generated successfully!", color="success")
            
            return success_msg, download_data
            
        except Exception as e:
            error_msg = dbc.Alert(f"❌ Error generating PDF: {str(e)}", color="danger")
            return error_msg, None

    return app