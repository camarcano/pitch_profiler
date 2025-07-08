# app/modules/pitch_profiler/dash_app.py
"""
Pitch Profiler Dash Application
Advanced baseball pitching analysis tool converted from Streamlit to Dash
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

def create_heatmap_plot(df, pitcher_name):
    """Create the pitch location heatmap using Plotly"""
    fig = go.Figure()

    # Create 2D histogram for heatmap
    fig.add_trace(go.Histogram2d(
        x=df['plate_x'],
        y=df['plate_z'],
        colorscale='Blues',
        showscale=True,
        colorbar=dict(title="Pitch Density"),
        nbinsx=30,
        nbinsy=30
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
        line=dict(color="red", width=3, dash="dash"),
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

    return df, pitcher_name, IVB_COL, HB_COL, {
        'pitch_summary': pitch_summary,
        'total_csw_pct': total_csw_pct,
        'k_pct': k_pct,
        'bb_pct': bb_pct,
        'total_pa': total_pa,
        'strikeouts': strikeouts,
        'walks': walks
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

def init_dash_app(server, route_prefix):
    """Initialize the Dash app"""
    
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
        
        # Header
        dbc.Row([
            dbc.Col([
                html.H1("⚾ Baseball Pitch Profiler", className="text-center mb-4 text-primary"),
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
                            html.H4("📄 Generate Report"),
                            html.P("Create a comprehensive PDF report with all analysis and visualizations."),
                            dbc.Button("Generate PDF Report", id="generate-pdf-btn", color="primary", className="me-2"),
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
            
            # Store processed data
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
        
        return dbc.Row([
            dbc.Col([
                html.H2(f"📊 Analysis for: {pitcher_name}", className="text-primary mb-3")
            ], width=12)
        ] + [
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(str(value), className="text-primary mb-0"),
                        html.P(label, className="text-muted mb-0")
                    ])
                ])
            ], width=2) for label, value in [
                ("Total Pitches", len(df)),
                ("CSW%", f"{analysis['total_csw_pct']:.1f}%"),
                ("K%", f"{analysis['k_pct']:.1f}%"),
                ("BB%", f"{analysis['bb_pct']:.1f}%"),
                ("Avg Velocity", f"{df['release_speed'].mean():.1f} mph"),
                ("Pitch Types", df['pitch_name_full'].nunique())
            ]
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
            fig = create_frequency_plot(df, pitcher_name, analysis['pitch_summary'], analysis['total_csw_pct'])
            
            # Convert pitch summary to display table
            display_summary = analysis['pitch_summary'].round(1)
            
            return html.Div([
                dcc.Graph(figure=fig, style={'height': '600px'}),
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
            fig = create_heatmap_plot(df, pitcher_name)
            return dcc.Graph(figure=fig, style={'height': '600px'})
            
        elif active_tab == "stats":
            return html.Div([
                html.H4("Detailed Statistics"),
                html.P("Additional statistical analysis will be displayed here."),
                dbc.Alert("More detailed statistics coming soon!", color="info")
            ])
        
        return ""

    @app.callback(
        Output('pdf-status', 'children'),
        [Input('generate-pdf-btn', 'n_clicks')],
        [State('processed-data-store', 'data'),
         State('pitcher-name-store', 'data')]
    )
    def generate_pdf_report(n_clicks, stored_data, pitcher_name):
        if not n_clicks or not stored_data:
            return ""
        
        return dbc.Alert("PDF generation is being implemented. Check back soon!", color="info")

    return app