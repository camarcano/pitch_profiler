import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.patches import Circle
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from datetime import datetime
import os
import streamlit as st
import tempfile
import io
from PIL import Image as PILImage

# Streamlit page configuration

st.set_page_config(
page_title=“Baseball Pitching Analysis Tool”,
page_icon=“⚾”,
layout=“wide”,
initial_sidebar_state=“expanded”
)

# Custom CSS for better styling

st.markdown(”””

<style>
    .main-header {
        font-size: 3rem;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2e75b6;
        margin: 1rem 0;
    }
    .metric-box {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #2e75b6;
    }
</style>

“””, unsafe_allow_html=True)

def calculate_k_bb_percentages(df):
“”“Calculate K% and BB% from the dataframe”””
# Get final pitch of each at-bat
at_bats_final_pitch = df.loc[df.dropna(subset=[‘events’]).groupby([‘game_date’, ‘at_bat_number’])[‘pitch_number’].idxmax()]

```
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
```

def create_movement_plot(df, pitcher_name, IVB_COL, HB_COL):
“”“Create the horizontal vs vertical break plot”””
fig, ax = plt.subplots(figsize=(14, 14))

```
# Add concentric circles for break magnitude
for radius in range(4, 21, 4):
    circle = Circle((0, 0), radius, color='grey', linestyle='--', fill=False, alpha=0.6)
    ax.add_patch(circle)

# Main scatter plot
sns.scatterplot(data=df, x=HB_COL, y=IVB_COL, hue='pitch_name_full', alpha=0.6, s=120, ax=ax, palette='viridis')

# Calculate centroids and add annotations
pitch_summary_for_plot = df.groupby('pitch_name_full').agg(
    avg_ivb=(IVB_COL, 'mean'),
    avg_hb=(HB_COL, 'mean'),
    avg_velocity=('release_speed', 'mean'),
    avg_spin_rate=('release_spin_rate', 'mean'),
    avg_extension=('release_extension', 'mean')
)

for pitch, data in pitch_summary_for_plot.iterrows():
    annotation_text = (
        f"{pitch}\n"
        f"Velo: {data['avg_velocity']:.1f} mph\n"
        f"Spin: {data['avg_spin_rate']:.0f} rpm\n"
        f"Break (HB/IVB): {data['avg_hb']:.1f}\" / {data['avg_ivb']:.1f}\"\n"
        f"Extension: {data['avg_extension']:.1f}\""
    )
    ax.annotate(annotation_text, (data['avg_hb'], data['avg_ivb']),
                textcoords="offset points", xytext=(10, -15), ha='left',
                fontsize=9, color='black',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1, alpha=0.7))

# Set 1:1 scale and center the origin
ax.set_aspect('equal', adjustable='box')
plt.grid(True, linestyle='--', alpha=0.6)
plt.axhline(0, color='black', linestyle='-', lw=1.5)
plt.axvline(0, color='black', linestyle='-', lw=1.5)

# Dynamically set plot limits to be centered around the origin
max_hb = df[HB_COL].abs().max()
max_ivb = df[IVB_COL].abs().max()
max_limit = max(max_hb, max_ivb) + 5
ax.set_xlim(-max_limit, max_limit)
ax.set_ylim(-max_limit, max_limit)

ax.set_title(f"{pitcher_name} - Pitch Movement Profile (2025)", fontsize=18)
ax.set_xlabel("Horizontal Break (HB) in Inches", fontsize=14)
ax.set_ylabel("Induced Vertical Break (IVB) in Inches", fontsize=14)
ax.legend(title='Pitch Type', bbox_to_anchor=(1.02, 1), loc='upper left')

plt.tight_layout()
return fig
```

def create_frequency_plot(df, pitcher_name, pitch_summary, total_csw_pct):
“”“Create the pitch frequency and averages plot”””
plt.figure(figsize=(16, 9))
ax = sns.barplot(x=‘pitch_name_full’, y=‘count’, data=pitch_summary, palette=‘magma’, hue=‘pitch_name_full’, legend=False)

```
plt.title(f'{pitcher_name} - Pitch Usage & Profile (2025) | Total CSW: {total_csw_pct:.1f}%', fontsize=18)
plt.xlabel('Pitch Type', fontsize=14)
plt.ylabel('Frequency (Total Thrown)', fontsize=14)
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(fontsize=12)

# Add CSW% to bar annotations
for i, p in enumerate(ax.patches):
    data = pitch_summary.iloc[i]
    annotation = (f"CSW: {data['csw_pct']:.1f}%\n"
                  f"Velo: {data['avg_velocity']:.1f} mph\n"
                  f"Spin: {data['avg_spin_rate']:.0f} rpm\n"
                  f"IVB: {data['avg_ivb']:.1f}\" | HB: {data['avg_hb']:.1f}\"")
    ax.annotate(annotation, (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, -45), textcoords='offset points',
                color='white', fontsize=10, weight='bold')
plt.tight_layout()
return plt.gcf()
```

def create_heatmap_plot(df, pitcher_name):
“”“Create the pitch location heatmap”””
plt.figure(figsize=(8, 8))
sns.kdeplot(data=df, x=‘plate_x’, y=‘plate_z’, fill=True, cmap=‘mako_r’, levels=50)

```
strike_zone_width_inches = 17.0
strike_zone_top_feet = 3.2
strike_zone_bottom_feet = 1.6
strike_zone_x = [-strike_zone_width_inches / 2, strike_zone_width_inches / 2]
strike_zone_z = [strike_zone_bottom_feet * 12, strike_zone_top_feet * 12]

plt.plot([strike_zone_x[0], strike_zone_x[1], strike_zone_x[1], strike_zone_x[0], strike_zone_x[0]],
         [strike_zone_z[0], strike_zone_z[0], strike_zone_z[1], strike_zone_z[1], strike_zone_z[0]],
         color='red', linestyle='--', linewidth=2, label='Strike Zone')
plt.title(f"{pitcher_name} - Pitch Heatmap (Catcher's View)", fontsize=16)
plt.xlabel('Horizontal Location (Inches from Center)', fontsize=12)
plt.ylabel('Vertical Location (Inches from Ground)', fontsize=12)
plt.gca().set_aspect('equal', adjustable='box')
plt.xlim(-24, 24)
plt.ylim(0, 54)
plt.legend()
return plt.gcf()
```

def create_pdf_report(df, pitcher_name, pdf_data):
“”“Create the PDF report”””
# Create temporary files for plots
with tempfile.TemporaryDirectory() as temp_dir:
# Generate plots and save to temp directory
IVB_COL = ‘pfx_z’
HB_COL = ‘pfx_x’

```
    # 1. Movement plot
    movement_fig = create_movement_plot(df, pitcher_name, IVB_COL, HB_COL)
    movement_path = os.path.join(temp_dir, 'movement.png')
    movement_fig.savefig(movement_path, dpi=300, bbox_inches='tight')
    plt.close(movement_fig)
    
    # 2. Frequency plot
    frequency_fig = create_frequency_plot(df, pitcher_name, pdf_data['pitch_summary'], pdf_data['total_csw_pct'])
    frequency_path = os.path.join(temp_dir, 'frequency.png')
    frequency_fig.savefig(frequency_path, dpi=300, bbox_inches='tight')
    plt.close(frequency_fig)
    
    # 3. Heatmap plot
    heatmap_fig = create_heatmap_plot(df, pitcher_name)
    heatmap_path = os.path.join(temp_dir, 'heatmap.png')
    heatmap_fig.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.close(heatmap_fig)
    
    # Create temporary file for PDF
    temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
    
    doc = SimpleDocTemplate(temp_pdf.name, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
    
    story = []
    
    # Define styles
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
    
    # Title Page
    story.append(Paragraph(pitcher_name, title_style))
    story.append(Paragraph("Pitching Performance Analysis Report", heading_style))
    story.append(Paragraph(f"Generated on: {datetime.now().strftime('%B %d, %Y')}", styles['Normal']))
    story.append(Spacer(1, 0.5*inch))
    
    # Executive Summary
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
    
    # Pitch Movement Profile
    story.append(Paragraph("Pitch Movement Profile", heading_style))
    story.append(Paragraph("Horizontal vs. Vertical Break Analysis", subheading_style))
    
    if os.path.exists(movement_path):
        img = Image(movement_path, width=7*inch, height=7*inch)
        story.append(img)
    
    story.append(Spacer(1, 0.2*inch))
    movement_text = """
    The pitch movement profile shows the relationship between horizontal break (HB) and induced vertical break (IVB) for all pitch types. 
    Concentric circles indicate break magnitude, helping visualize the overall movement profile. Each pitch type is annotated with 
    key metrics including velocity, spin rate, and break measurements.
    """
    story.append(Paragraph(movement_text, styles['Normal']))
    story.append(PageBreak())
    
    # Pitch Usage and Frequency
    story.append(Paragraph("Pitch Usage and Performance", heading_style))
    story.append(Paragraph("Frequency and Key Metrics by Pitch Type", subheading_style))
    
    if os.path.exists(frequency_path):
        img = Image(frequency_path, width=8*inch, height=4.5*inch)
        story.append(img)
    
    story.append(Spacer(1, 0.3*inch))
    
    # Pitch Summary Table
    story.append(Paragraph("Detailed Pitch Metrics", subheading_style))
    
    # Create table data
    table_data = [['Pitch Type', 'Count', 'CSW%', 'Avg Velo', 'Avg Spin', 'Avg IVB', 'Avg HB']]
    for _, row in pdf_data['pitch_summary'].iterrows():
        table_data.append([
            row['pitch_name_full'],
            str(int(row['count'])),
            f"{row['csw_pct']:.1f}%",
            f"{row['avg_velocity']:.1f}",
            f"{row['avg_spin_rate']:.0f}",
            f"{row['avg_ivb']:.1f}\"",
            f"{row['avg_hb']:.1f}\""
        ])
    
    table = Table(table_data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(table)
    story.append(PageBreak())
    
    # Pitch Location
    story.append(Paragraph("Pitch Location Analysis", heading_style))
    story.append(Paragraph("Heat Map - Catcher's View", subheading_style))
    
    if os.path.exists(heatmap_path):
        img = Image(heatmap_path, width=6*inch, height=6*inch)
        story.append(img)
    
    story.append(Spacer(1, 0.2*inch))
    location_text = """
    The pitch location heatmap shows the density of pitches across the strike zone from the catcher's perspective. 
    The red dashed line indicates the official strike zone boundaries. Higher density areas (darker colors) indicate 
    preferred pitch locations.
    """
    story.append(Paragraph(location_text, styles['Normal']))
    story.append(PageBreak())
    
    # Batted Ball Analysis
    if 'no_data' not in pdf_data['batted_ball_stats']:
        story.append(Paragraph("Batted Ball Analysis", heading_style))
        
        bb_stats = pdf_data['batted_ball_stats']
        
        # Hard Hit Analysis Table
        story.append(Paragraph(f"Hard Hit Analysis (Exit Velocity ≥ {bb_stats['hard_hit_threshold']} mph)", subheading_style))
        
        story.append(Paragraph(f"Overall Hard Hit Rate: {bb_stats['total_hard_hit_pct']:.1f}%", styles['Normal']))
        story.append(Spacer(1, 0.1*inch))
        
        # Hard hit by pitch table
        hh_table_data = [['Pitch Type', 'Hard Hit %']]
        for pitch, pct in bb_stats['hard_hit_by_pitch'].items():
            hh_table_data.append([pitch, f"{pct:.1f}%"])
        
        hh_table = Table(hh_table_data)
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
        story.append(Spacer(1, 0.3*inch))
        
        # xwOBA Analysis
        story.append(Paragraph(f"Expected Weighted On-Base Average (xwOBA ≥ {bb_stats['xwoba_threshold']})", subheading_style))
        story.append(Paragraph(f"Overall High xwOBA Rate: {bb_stats['total_high_xwoba_pct']:.1f}%", styles['Normal']))
        story.append(Spacer(1, 0.1*inch))
        
        # xwOBA by pitch table
        xw_table_data = [['Pitch Type', f'xwOBA ≥ {bb_stats["xwoba_threshold"]} %']]
        for pitch, pct in bb_stats['high_xwoba_by_pitch'].items():
            xw_table_data.append([pitch, f"{pct:.1f}%"])
        
        xw_table = Table(xw_table_data)
        xw_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(xw_table)
        story.append(PageBreak())
    
    # Opponent Batting Statistics
    if 'no_data' not in pdf_data['batting_stats']:
        story.append(Paragraph("Opponent Batting Performance", heading_style))
        
        bs = pdf_data['batting_stats']
        
        # Overall Stats
        story.append(Paragraph("Overall Performance Against All Batters", subheading_style))
        overall_table_data = [['Metric', 'Value']]
        for stat, value in bs['overall_stats'].items():
            if stat in ['AVG', 'OBP', 'SLG', 'OPS']:
                overall_table_data.append([stat, f"{value:.3f}"])
            else:
                overall_table_data.append([stat, str(int(value))])
        
        overall_table = Table(overall_table_data)
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
        story.append(Spacer(1, 0.3*inch))
        
        # Stats by Handedness
        story.append(Paragraph("Performance vs. Batter Handedness", subheading_style))
        hand_table_data = [['Handedness', 'AVG', 'OBP', 'SLG', 'OPS', 'PA']]
        for hand, stats in bs['stats_by_handedness'].iterrows():
            hand_table_data.append([
                'Left' if hand == 'L' else 'Right',
                f"{stats['AVG']:.3f}",
                f"{stats['OBP']:.3f}",
                f"{stats['SLG']:.3f}",
                f"{stats['OPS']:.3f}",
                str(int(stats['PA']))
            ])
        
        hand_table = Table(hand_table_data)
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
        story.append(PageBreak())
        
        # Stats by Final Pitch Type
        story.append(Paragraph("Performance by Final Pitch Type of At-Bat", subheading_style))
        pitch_stats_table_data = [['Pitch Type', 'AVG', 'OBP', 'SLG', 'OPS', 'AB']]
        for pitch, stats in bs['stats_by_pitch'].iterrows():
            pitch_stats_table_data.append([
                pitch,
                f"{stats['AVG']:.3f}",
                f"{stats['OBP']:.3f}",
                f"{stats['SLG']:.3f}",
                f"{stats['OPS']:.3f}",
                str(int(stats['AB']))
            ])
        
        pitch_stats_table = Table(pitch_stats_table_data)
        pitch_stats_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(pitch_stats_table)
    
    # Build PDF
    doc.build(story)
    
    # Read the PDF content
    with open(temp_pdf.name, 'rb') as f:
        pdf_content = f.read()
    
    # Clean up temp file
    os.unlink(temp_pdf.name)
    
    return pdf_content
```

def process_data(df):
“”“Process the uploaded data and return all analysis results”””
# Extract pitcher name
if ‘player_name’ in df.columns:
pitcher_name = df[‘player_name’].iloc[0]
else:
pitcher_name = “Unknown Pitcher”

```
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

# Batting stats calculation
def calculate_batting_stats(data):
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
    overall_stats = calculate_batting_stats(at_bats_final_pitch)
    stats_by_pitch = at_bats_final_pitch.groupby('pitch_name_full').apply(calculate_batting_stats, include_groups=False)
    stats_by_handedness = at_bats_final_pitch.groupby('stand').apply(calculate_batting_stats, include_groups=False)
    
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
```

def save_plot_to_bytes(fig):
“”“Save a matplotlib figure to bytes for download”””
buf = io.BytesIO()
fig.savefig(buf, format=‘png’, dpi=300, bbox_inches=‘tight’)
buf.seek(0)
return buf.getvalue()

# Main Streamlit App

def main():
st.markdown(’<h1 class="main-header">⚾ Baseball Pitching Analysis Tool</h1>’, unsafe_allow_html=True)

```
st.markdown("""
Upload your Statcast CSV data to generate comprehensive pitching analysis including:
- Pitch movement profiles
- Usage patterns and performance metrics  
- Location heatmaps
- Opponent batting statistics
- Complete PDF report
""")

# Sidebar for file upload
st.sidebar.markdown("## 📁 Upload Data")
uploaded_file = st.sidebar.file_uploader(
    "Choose a CSV file",
    type="csv",
    help="Upload your Statcast CSV data file"
)

if uploaded_file is not None:
    try:
        # Load the data
        df = pd.read_csv(uploaded_file)
        st.sidebar.success(f"✅ File uploaded successfully! {len(df)} rows loaded.")
        
        # Process the data
        with st.spinner("🔄 Processing data and generating analysis..."):
            df_processed, pitcher_name, IVB_COL, HB_COL, pdf_data = process_data(df)
        
        # Display pitcher information
        st.markdown(f'<h2 class="sub-header">📊 Analysis for: {pitcher_name}</h2>', unsafe_allow_html=True)
        
        # Key metrics in columns
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Pitches", len(df_processed))
        with col2:
            st.metric("CSW%", f"{pdf_data['total_csw_pct']:.1f}%")
        with col3:
            st.metric("K%", f"{pdf_data['k_pct']:.1f}%")
        with col4:
            st.metric("BB%", f"{pdf_data['bb_pct']:.1f}%")
        with col5:
            st.metric("Avg Velocity", f"{df_processed['release_speed'].mean():.1f} mph")
        
        # Additional metrics
        st.markdown(f"""
        <div class="metric-box">
            <strong>Detailed Statistics:</strong><br>
            • Total Plate Appearances: {pdf_data['total_pa']}<br>
            • Total Strikeouts: {pdf_data['strikeouts']}<br>
            • Total Walks: {pdf_data['walks']}<br>
            • Unique Pitch Types: {df_processed['pitch_name_full'].nunique()}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Tabs for different analyses
        tab1, tab2, tab3, tab4 = st.tabs(["🎯 Movement Profile", "📈 Usage & Performance", "🗺️ Location Heatmap", "📋 Detailed Stats"])
        
        with tab1:
            st.markdown("### Pitch Movement Profile")
            with st.spinner("Generating movement plot..."):
                movement_fig = create_movement_plot(df_processed, pitcher_name, IVB_COL, HB_COL)
                st.pyplot(movement_fig)
            
            # Download button for movement plot
            movement_bytes = save_plot_to_bytes(movement_fig)
            st.download_button(
                label="📥 Download Movement Plot",
                data=movement_bytes,
                file_name=f"{pitcher_name.replace(' ', '_')}_movement_plot.png",
                mime="image/png"
            )
        
        with tab2:
            st.markdown("### Pitch Usage & Performance")
            with st.spinner("Generating usage plot..."):
                frequency_fig = create_frequency_plot(df_processed, pitcher_name, pdf_data['pitch_summary'], pdf_data['total_csw_pct'])
                st.pyplot(frequency_fig)
            
            # Download button for frequency plot
            frequency_bytes = save_plot_to_bytes(frequency_fig)
            st.download_button(
                label="📥 Download Usage Plot",
                data=frequency_bytes,
                file_name=f"{pitcher_name.replace(' ', '_')}_usage_plot.png",
                mime="image/png"
            )
            
            # Display pitch summary table
            st.markdown("#### Pitch Summary Table")
            st.dataframe(pdf_data['pitch_summary'], use_container_width=True)
        
        with tab3:
            st.markdown("### Pitch Location Heatmap")
            with st.spinner("Generating heatmap..."):
                heatmap_fig = create_heatmap_plot(df_processed, pitcher_name)
                st.pyplot(heatmap_fig)
            
            # Download button for heatmap
            heatmap_bytes = save_plot_to_bytes(heatmap_fig)
            st.download_button(
                label="📥 Download Heatmap",
                data=heatmap_bytes,
                file_name=f"{pitcher_name.replace(' ', '_')}_heatmap.png",
                mime="image/png"
            )
        
        with tab4:
            st.markdown("### Detailed Statistics")
            
            # Batted ball stats
            if 'no_data' not in pdf_data['batted_ball_stats']:
                st.markdown("#### Batted Ball Analysis")
                bb_stats = pdf_data['batted_ball_stats']
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Hard Hit Rate", f"{bb_stats['total_hard_hit_pct']:.1f}%")
                with col2:
                    st.metric("High xwOBA Rate", f"{bb_stats['total_high_xwoba_pct']:.1f}%")
                
                # Hard hit by pitch type
                st.markdown("**Hard Hit % by Pitch Type:**")
                hh_df = pd.DataFrame(bb_stats['hard_hit_by_pitch']).reset_index()
                hh_df.columns = ['Pitch Type', 'Hard Hit %']
                hh_df['Hard Hit %'] = hh_df['Hard Hit %'].round(1)
                st.dataframe(hh_df, use_container_width=True)
            
            # Batting stats against
            if 'no_data' not in pdf_data['batting_stats']:
                st.markdown("#### Opponent Batting Performance")
                bs = pdf_data['batting_stats']
                
                # Overall stats
                st.markdown("**Overall Stats:**")
                overall_df = pd.DataFrame(bs['overall_stats']).transpose()
                st.dataframe(overall_df, use_container_width=True)
                
                # By handedness
                st.markdown("**By Batter Handedness:**")
                hand_df = bs['stats_by_handedness']
                st.dataframe(hand_df, use_container_width=True)
        
        # PDF Report Generation
        st.markdown("---")
        st.markdown("### 📄 Complete Report")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("Generate a comprehensive PDF report with all analysis and visualizations.")
        
        with col2:
            if st.button("🎯 Generate PDF Report", type="primary"):
                with st.spinner("🔄 Generating PDF report..."):
                    try:
                        pdf_content = create_pdf_report(df_processed, pitcher_name, pdf_data)
                        st.download_button(
                            label="📥 Download Complete PDF Report",
                            data=pdf_content,
                            file_name=f"{pitcher_name.replace(' ', '_')}_Pitching_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                            mime="application/pdf"
                        )
                        st.success("✅ PDF report generated successfully!")
                    except Exception as e:
                        st.error(f"❌ Error generating PDF: {str(e)}")
        
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        st.info("Please make sure your CSV file contains the required Statcast columns.")

else:
    st.info("👆 Please upload a CSV file to begin analysis")
    
    # Sample data information
    st.markdown("### 📝 Expected Data Format")
    st.markdown("""
    Your CSV file should contain Statcast data with columns including:
    - `player_name`: Pitcher's name
    - `pitch_name`: Pitch type abbreviation (e.g., 'FF', 'SL', 'CH')
    - `release_speed`: Pitch velocity
    - `pfx_x`, `pfx_z`: Pitch movement data
    - `plate_x`, `plate_z`: Pitch location data
    - `description`: Pitch outcome description
    - `events`: At-bat outcome events
    - And other standard Statcast columns...
    """)
```

if **name** == “**main**”:
main()

if __name__ == "__main__":
    main()
