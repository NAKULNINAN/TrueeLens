import streamlit as st
import cv2
import numpy as np
import hashlib
import imagehash
from PIL import Image
import io
import tempfile
import os
import logging
import moviepy.editor as mp
from datetime import datetime
import json
from settings import Config
from config_manager import config_manager
import base64
import torch

# Configure logging
Config.configure_logging()

# Import custom modules
from detection.ai_image_detector import AIImageDetector
from detection.ai_video_detector import AIVideoDetector
from detection.face_extractor import FaceExtractor
from models.xception_net import load_xception_model
from models.meso_net import load_meso_model
from detection.deepfake_detector import DeepfakeDetector
from detection.duplicate_detector import DuplicateDetector
from utils.media_processor import MediaProcessor
from utils.visualization import create_result_card, create_confidence_chart
from reports.analysis_summary import AnalysisSummaryGenerator
from reports.advanced_pdf_generator import AdvancedPDFGenerator
from reports.technical_exporter import TechnicalDetailsExporter
from reports.advanced_batch_processor import AdvancedBatchProcessor

# Configure Streamlit page
st.set_page_config(
    page_title="TrueLens",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    .result-card {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid;
    }
    .real-result {
        background-color: #d4edda;
        border-left-color: #28a745;
    }
    .fake-result {
        background-color: #f8d7da;
        border-left-color: #dc3545;
    }
    .warning-result {
        background-color: #fff3cd;
        border-left-color: #ffc107;
    }
    .stProgress .st-bo {
        background-color: #667eea;
    }
</style>
""", unsafe_allow_html=True)

# Log system status at startup
Config.log_system_status()

class MediaForensicsApp:
    def __init__(self):
        self.ai_detector = AIImageDetector()
        self.ai_video_detector = AIVideoDetector()
        self.deepfake_detector = DeepfakeDetector()
        self.duplicate_detector = DuplicateDetector()
        self.media_processor = MediaProcessor()
        self.face_extractor = FaceExtractor()
        # Load XceptionNet for deepfake detection
        self.xception_model = load_xception_model()
        # Load MesoNet for deepfake detection
        self.meso_model = load_meso_model()
        
        # Initialize comprehensive report generators
        self.summary_generator = AnalysisSummaryGenerator()
        self.pdf_generator = AdvancedPDFGenerator()
        self.tech_exporter = TechnicalDetailsExporter()
        self.batch_processor = AdvancedBatchProcessor()
        
    def run(self):
        # Header
        st.markdown("""
        <div class="main-header">
            <h1>🔍 TrueLens</h1>
            <p>Advanced AI-powered detection for synthetic media, deepfakes, and duplicates</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Sidebar configuration
        with st.sidebar:
            st.header("⚙️ Configuration")
            
            # Load user preferences
            user_prefs = config_manager.load_preferences()
            
            # Detection mode selection
            detection_mode = st.selectbox(
                "🎯 Select Detection Mode:",
                [
                    "Detect AI-Generated Image",
                    "Detect AI-Generated Video",
                    "Detect Deepfake Video", 
                    "Detect Duplicate Image/Video"
                ],
                index=[
                    "Detect AI-Generated Image",
                    "Detect AI-Generated Video",
                    "Detect Deepfake Video", 
                    "Detect Duplicate Image/Video"
                ].index(user_prefs.get('detection_mode', 'Detect AI-Generated Image'))
            )
            
            # Advanced options
            st.subheader("🔧 Advanced Options")
            confidence_threshold = st.slider(
                "Confidence Threshold", 
                min_value=0.1, 
                max_value=1.0, 
                value=config_manager.get_preference('confidence_threshold', Config.CONFIDENCE_THRESHOLD), 
                step=0.05
            )
            
            enable_visualization = st.checkbox(
                "Enable Visual Analysis", 
                value=config_manager.get_preference('enable_visualization', Config.ENABLE_VISUALIZATION)
            )
            save_results = st.checkbox(
                "Save Results", 
                value=config_manager.get_preference('save_results', Config.SAVE_RESULTS)
            )
            
            # Set fixed default models
            config_manager.set_model_preference('deepfake_detection', 'xception')
            config_manager.set_model_preference('ai_image_detection', 'efficientnet')
            
            # Save preferences when they change
            current_prefs = {
                'detection_mode': detection_mode,
                'confidence_threshold': confidence_threshold,
                'enable_visualization': enable_visualization,
                'save_results': save_results
            }
            
            # Update preferences if they changed
            if current_prefs != {k: user_prefs.get(k) for k in current_prefs.keys()}:
                user_prefs.update(current_prefs)
                config_manager.save_preferences(user_prefs)
            
            # Configuration status
            with st.expander("📊 Configuration Status"):
                status = config_manager.get_config_status()
                st.json(status)
            
        # Main content area
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.header("📁 Upload Media")
            
            # File uploader based on detection mode
            if "Image" in detection_mode:
                uploaded_file = st.file_uploader(
                    "Choose an image file",
                    type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
                    help="Upload JPG, PNG, or other image formats"
                )
            else:
                uploaded_file = st.file_uploader(
                    "Choose a video file",
                    type=['mp4', 'avi', 'mov', 'mkv', 'wmv'],
                    help="Upload MP4, AVI, MOV, or other video formats"
                )
            
        
        with col2:
            st.header("👁️ Preview")
            
            if uploaded_file is not None:
                if "image" in uploaded_file.type:
                    try:
                        # Properly handle Streamlit uploaded file
                        image_bytes = uploaded_file.read()
                        uploaded_file.seek(0)  # Reset file pointer for later use
                        image = Image.open(io.BytesIO(image_bytes))
                        st.image(image, caption="Uploaded Image", use_container_width=True)
                    except Exception as e:
                        st.error(f"Error loading image preview: {str(e)}")
                        st.info("The image file may be corrupted or in an unsupported format.")
                else:
                    st.video(uploaded_file)
                    st.info("Video preview shown above. Analysis will process key frames.")
        
        # Analysis section
        if uploaded_file is not None:
            st.header("🔍 Analysis Results")
            
            with st.spinner("🧠 Analyzing media... This may take a moment."):
                results = self.process_media(
                    uploaded_file, 
                    detection_mode, 
                    confidence_threshold,
                    enable_visualization
                )
            
            # Display results
            self.display_results(results, enable_visualization)
            
            # Comprehensive Report Generation Options
            if save_results:
                st.subheader("📄 Report Generation")
                
                report_col1, report_col2, report_col3 = st.columns(3)
                
                with report_col1:
                    # JSON Report
                    if st.button("📊 Generate JSON Report"):
                        with st.spinner("Generating comprehensive JSON report..."):
                            detailed_report = self.summary_generator.generate_detailed_report(results, results['file_info'])
                            json_report = json.dumps(detailed_report, indent=2, default=str)
                        
                        st.download_button(
                            label="📊 Download JSON Report",
                            data=json_report,
                            file_name=f"comprehensive_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
                
                with report_col2:
                    # PDF Report
                    if st.button("📋 Generate PDF Report"):
                        with st.spinner("Generating professional PDF report..."):
                            try:
                                pdf_path = f"temp_pdf_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                                self.pdf_generator.generate_comprehensive_pdf_report(
                                    results, results['file_info'], pdf_path
                                )
                                
                                # Read the PDF file
                                with open(pdf_path, "rb") as pdf_file:
                                    pdf_data = pdf_file.read()
                                
                                st.download_button(
                                    label="📋 Download PDF Report",
                                    data=pdf_data,
                                    file_name=f"forensics_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                    mime="application/pdf"
                                )
                                
                                # Clean up temp file
                                if os.path.exists(pdf_path):
                                    os.remove(pdf_path)
                                    
                            except Exception as e:
                                st.error(f"Error generating PDF: {str(e)}")
                
                with report_col3:
                    # Technical Export
                    export_format = st.selectbox(
                        "Technical Export Format:",
                        ['json', 'csv', 'xml', 'excel', 'txt']
                    )
                    
                    if st.button("⚙️ Export Technical Details"):
                        with st.spinner(f"Exporting technical details as {export_format.upper()}..."):
                            try:
                                temp_path = f"temp_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{export_format}"
                                self.tech_exporter.export_technical_details(
                                    results, temp_path, export_format
                                )
                                
                                # Read the exported file
                                with open(temp_path, "rb") as export_file:
                                    export_data = export_file.read()
                                
                                st.download_button(
                                    label=f"⚙️ Download {export_format.upper()} Export",
                                    data=export_data,
                                    file_name=f"technical_details_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{export_format}",
                                    mime=f"application/{export_format}"
                                )
                                
                                # Clean up temp file
                                if os.path.exists(temp_path):
                                    os.remove(temp_path)
                                    
                            except Exception as e:
                                st.error(f"Error exporting technical details: {str(e)}")
                
                # Executive Summary Display
                st.subheader("📈 Executive Summary")
                exec_summary = self.summary_generator.generate_executive_summary(results)
                
                with st.expander("View Executive Summary", expanded=True):
                    st.json(exec_summary)
    
    def process_media(self, uploaded_file, detection_mode, threshold, enable_viz):
        """Process uploaded media based on selected detection mode"""
        logging.info(f"Processing media: {uploaded_file.name} with mode: {detection_mode}")
        
        results = {
            "mode": detection_mode,
            "timestamp": datetime.now().isoformat(),
            "file_info": {
                "name": uploaded_file.name,
                "size": uploaded_file.size,
                "type": uploaded_file.type
            }
        }
        
        try:
            # Reset file pointer to beginning before processing
            uploaded_file.seek(0)
            
            if detection_mode == "Detect AI-Generated Image":
                detection_result = self.ai_detector.detect(uploaded_file, threshold, enable_viz)
                if detection_result is None:
                    raise ValueError("AI detector returned None result")
                # Check if detector returned an error
                if 'error' in detection_result:
                    raise ValueError(f"AI detection failed: {detection_result['error']}")
                results.update(detection_result)
                
            elif detection_mode == "Detect AI-Generated Video":
                detection_result = self.ai_video_detector.detect(uploaded_file, threshold, enable_viz)
                if detection_result is None:
                    raise ValueError("AI video detector returned None result")
                # Check if detector returned an error
                if 'error' in detection_result:
                    raise ValueError(f"AI video detection failed: {detection_result['error']}")
                results.update(detection_result)
                
            elif detection_mode == "Detect Deepfake Video":
                detection_result = self.deepfake_detector.detect(uploaded_file, threshold, enable_viz)
                if detection_result is None:
                    raise ValueError("Deepfake detector returned None result")
                # Check if detector returned an error
                if 'error' in detection_result:
                    raise ValueError(f"Deepfake detection failed: {detection_result['error']}")
                results.update(detection_result)
                
            elif detection_mode == "Detect Duplicate Image/Video":
                detection_result = self.duplicate_detector.detect(uploaded_file, threshold, enable_viz)
                if detection_result is None:
                    raise ValueError("Duplicate detector returned None result")
                # Check if detector returned an error
                if 'error' in detection_result:
                    raise ValueError(f"Duplicate detection failed: {detection_result['error']}")
                results.update(detection_result)
                
        except Exception as e:
            error_msg = str(e) if str(e) else f"Unknown error of type {type(e).__name__}"
            logging.error(f"Detection failed for {uploaded_file.name}: {error_msg}", exc_info=True)
            results["error"] = error_msg
            results["status"] = "error"
            
        return results
    
    def display_results(self, results, enable_visualization):
        """Display analysis results with visual components"""
        if "error" in results:
            logging.error(f"Analysis error: {results['error']}")
        
        if "error" in results:
            st.error(f"❌ Analysis failed: {results['error']}")
            return
            
        # Main result card
        if results.get("is_fake", False):
            card_class = "fake-result"
            icon = "❌"
            status = "SYNTHETIC/FAKE DETECTED"
        elif results.get("is_duplicate", False):
            card_class = "warning-result"
            icon = "⚠️"
            status = "DUPLICATE DETECTED"
        else:
            card_class = "real-result"
            icon = "✅"
            status = "AUTHENTIC/ORIGINAL"
            
        st.markdown(f"""
        <div class="result-card {card_class}">
            <h2>{icon} {status}</h2>
            <h3>Confidence: {results.get('confidence', 0):.1%}</h3>
            <p><strong>Analysis:</strong> {results.get('explanation', 'No explanation available')}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Detailed metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Confidence Score",
                f"{results.get('confidence', 0):.1%}",
                delta=f"{results.get('confidence', 0) - 0.5:.1%}"
            )
        
        with col2:
            st.metric(
                "Processing Time",
                f"{results.get('processing_time', 0):.2f}s"
            )
            
        with col3:
            st.metric(
                "Model Accuracy",
                f"{results.get('model_accuracy', 0.85):.1%}"
            )
        
        # Visualizations
        if enable_visualization and results.get('visualizations'):
            st.subheader("📊 Visual Analysis")
            
            if 'heatmap' in results['visualizations']:
                st.image(
                    results['visualizations']['heatmap'],
                    caption="Attention Heatmap - Red areas indicate suspicious regions"
                )
            
            if 'confidence_chart' in results['visualizations']:
                st.plotly_chart(
                    results['visualizations']['confidence_chart'],
                    use_container_width=True
                )
        
        # Technical details
        with st.expander("🔬 Technical Details"):
            st.json(results.get('technical_details', {}))

    def generate_report(self, results, filename):
        """Generate downloadable JSON report"""
        logging.info(f"Generating report for {filename}")
        logging.info("Error handling is active.")
        report = {
            "media_forensics_report": {
                "version": "1.0",
                "generated_at": datetime.now().isoformat(),
                "file_analyzed": filename,
                "results": results
            }
        }
        return json.dumps(report, indent=2)

# Initialize and run the app
if __name__ == "__main__":
    app = MediaForensicsApp()
    app.run()
