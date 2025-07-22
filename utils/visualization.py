import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import cv2
import io
import base64
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import plotly.io as pio
from PIL import Image, ImageDraw, ImageFont

def create_result_card(result: Dict[str, Any]) -> str:
    """Create HTML result card for detection results"""
    confidence = result.get('confidence', 0.0)
    is_fake = result.get('is_fake', False)
    is_duplicate = result.get('is_duplicate', False)
    explanation = result.get('explanation', 'No explanation available')
    
    # Determine card style
    if is_fake or is_duplicate:
        card_class = "alert-danger"
        icon = "⚠️"
        status = "SUSPICIOUS" if is_fake else "DUPLICATE"
    else:
        card_class = "alert-success"
        icon = "✅"
        status = "AUTHENTIC"
    
    html_card = f"""
    <div class="alert {card_class}" style="border-radius: 10px; padding: 20px; margin: 10px 0;">
        <h3>{icon} {status}</h3>
        <p><strong>Confidence:</strong> {confidence:.1%}</p>
        <p><strong>Analysis:</strong> {explanation}</p>
        <div class="progress" style="height: 10px; margin-top: 10px;">
            <div class="progress-bar" role="progressbar" 
                 style="width: {confidence*100}%; background-color: {'#dc3545' if confidence > 0.5 else '#28a745'};"></div>
        </div>
    </div>
    """
    
    return html_card

def create_confidence_chart(confidence_data: Dict[str, float]) -> go.Figure:
    """Create interactive confidence visualization chart"""
    
    # Extract confidence scores
    models = list(confidence_data.keys())
    scores = list(confidence_data.values())
    
    # Create color mapping
    colors = ['red' if score > 0.5 else 'green' for score in scores]
    
    fig = go.Figure(data=[
        go.Bar(
            x=models,
            y=scores,
            marker_color=colors,
            text=[f'{score:.1%}' for score in scores],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title='Model Confidence Scores',
        xaxis_title='Detection Models',
        yaxis_title='Confidence Score',
        yaxis=dict(range=[0, 1]),
        template='plotly_white'
    )
    
    # Add threshold line
    fig.add_hline(y=0.5, line_dash="dash", line_color="orange", 
                  annotation_text="Decision Threshold")
    
    return fig

def create_frame_analysis_chart(frame_results: List[Dict[str, Any]]) -> go.Figure:
    """Create timeline chart showing frame-by-frame analysis"""
    
    frame_indices = [r['frame_idx'] for r in frame_results]
    confidences = [r.get('ensemble_confidence', r.get('confidence', 0)) for r in frame_results]
    
    fig = go.Figure()
    
    # Add confidence timeline
    fig.add_trace(go.Scatter(
        x=frame_indices,
        y=confidences,
        mode='lines+markers',
        name='Deepfake Confidence',
        line=dict(color='red', width=2),
        marker=dict(size=6)
    ))
    
    # Add threshold line
    fig.add_hline(y=0.5, line_dash="dash", line_color="orange", 
                  annotation_text="Decision Threshold")
    
    fig.update_layout(
        title='Frame-by-Frame Deepfake Detection Analysis',
        xaxis_title='Frame Index',
        yaxis_title='Deepfake Confidence',
        yaxis=dict(range=[0, 1]),
        template='plotly_white',
        hovermode='x unified'
    )
    
    return fig

def create_model_comparison_chart(xception_scores: List[float], 
                                 meso_scores: List[float],
                                 ensemble_scores: List[float]) -> go.Figure:
    """Create comparison chart for different model outputs"""
    
    frames = list(range(len(xception_scores)))
    
    fig = go.Figure()
    
    # Add traces for each model
    fig.add_trace(go.Scatter(
        x=frames, y=xception_scores,
        mode='lines', name='XceptionNet',
        line=dict(color='blue', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=frames, y=meso_scores,
        mode='lines', name='MesoNet',
        line=dict(color='green', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=frames, y=ensemble_scores,
        mode='lines', name='Ensemble',
        line=dict(color='red', width=3)
    ))
    
    fig.add_hline(y=0.5, line_dash="dash", line_color="orange", 
                  annotation_text="Decision Threshold")
    
    fig.update_layout(
        title='Model Comparison - Deepfake Detection Confidence',
        xaxis_title='Frame Index',
        yaxis_title='Confidence Score',
        yaxis=dict(range=[0, 1]),
        template='plotly_white'
    )
    
    return fig

def create_confidence_distribution(confidence_scores: List[float]) -> go.Figure:
    """Create histogram showing distribution of confidence scores"""
    
    fig = go.Figure(data=[
        go.Histogram(
            x=confidence_scores,
            nbinsx=20,
            marker_color='skyblue',
            marker_line_color='navy',
            marker_line_width=1
        )
    ])
    
    fig.add_vline(x=0.5, line_dash="dash", line_color="red", 
                  annotation_text="Decision Threshold")
    
    fig.update_layout(
        title='Distribution of Confidence Scores',
        xaxis_title='Confidence Score',
        yaxis_title='Number of Frames',
        template='plotly_white'
    )
    
    return fig

def create_detection_summary_chart(summary_stats: Dict[str, Any]) -> go.Figure:
    """Create summary visualization for detection results"""
    
    # Create pie chart for fake vs real frames
    fake_frames = summary_stats.get('fake_frames', 0)
    total_frames = summary_stats.get('total_frames', 1)
    real_frames = total_frames - fake_frames
    
    fig = go.Figure(data=[go.Pie(
        labels=['Authentic Frames', 'Suspicious Frames'],
        values=[real_frames, fake_frames],
        colors=['green', 'red'],
        textinfo='label+percent',
        textfont_size=12
    )])
    
    fig.update_layout(
        title='Frame Classification Summary',
        template='plotly_white'
    )
    
    return fig

def create_attention_heatmap(image: np.ndarray, attention_map: np.ndarray) -> np.ndarray:
    """Create heatmap overlay on image using OpenCV"""
    heatmap = cv2.applyColorMap(np.uint8(255 * attention_map), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)
    return overlay


# Function to create progress indicator

def create_progress_indicator(current: int, total: int) -> str:
    """Creates a simple progress indicator"""
    progress = (current / total) * 100
    return f"Progress: {progress:.2f}%"


# Function to export figures

def export_figure(fig: go.Figure, filename: str) -> None:
    """Export a Plotly figure to a file"""
    pio.write_image(fig, filename)
    print(f"Figure exported to {filename}")

def create_technical_metrics_table(metrics: Dict[str, Any]) -> str:
    """Create HTML table for technical metrics"""
    
    html_table = """
    <table class="table table-striped" style="margin-top: 20px;">
        <thead>
            <tr>
                <th>Metric</th>
                <th>Value</th>
                <th>Description</th>
            </tr>
        </thead>
        <tbody>
    """
    
    for key, value in metrics.items():
        description = get_metric_description(key)
        html_table += f"""
            <tr>
                <td><strong>{key.replace('_', ' ').title()}</strong></td>
                <td>{format_metric_value(value)}</td>
                <td>{description}</td>
            </tr>
        """
    
    html_table += """
        </tbody>
    </table>
    """
    
    return html_table

def get_metric_description(metric_key: str) -> str:
    """Get description for metric keys"""
    descriptions = {
        'processing_time': 'Time taken to analyze the media file',
        'model_accuracy': 'Expected accuracy of the detection model',
        'total_frames': 'Total number of frames in the video',
        'processed_frames': 'Number of frames that were analyzed',
        'faces_detected': 'Total number of faces found in the video',
        'fake_frames': 'Number of frames classified as deepfake',
        'confidence': 'Overall confidence in the detection result',
        'threshold_used': 'Decision threshold used for classification'
    }
    
    return descriptions.get(metric_key, 'Technical metric from analysis')

def format_metric_value(value: Any) -> str:
    """Format metric values for display"""
    if isinstance(value, float):
        if 0 <= value <= 1:
            return f"{value:.1%}"  # Format as percentage
        else:
            return f"{value:.3f}"  # Format as decimal
    elif isinstance(value, int):
        return f"{value:,}"  # Format with commas
    else:
        return str(value)
