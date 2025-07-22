from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
from reportlab.platypus.flowables import HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.pdfgen import canvas
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from datetime import datetime
import os
from typing import Dict, Any, List, Optional
import logging
import numpy as np
import pandas as pd

class AdvancedPDFGenerator:
    """Advanced PDF report generator with charts and professional formatting"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Setup custom paragraph styles"""
        self.custom_styles = {
            'CustomTitle': ParagraphStyle(
                'CustomTitle',
                parent=self.styles['Heading1'],
                fontSize=24,
                spaceAfter=30,
                alignment=TA_CENTER,
                textColor=colors.HexColor('#2E5266')
            ),
            'CustomHeading': ParagraphStyle(
                'CustomHeading',
                parent=self.styles['Heading2'],
                fontSize=16,
                spaceBefore=20,
                spaceAfter=12,
                textColor=colors.HexColor('#2E5266')
            ),
            'CustomSubheading': ParagraphStyle(
                'CustomSubheading',
                parent=self.styles['Heading3'],
                fontSize=12,
                spaceBefore=12,
                spaceAfter=6,
                textColor=colors.HexColor('#4A7C8C')
            ),
            'CustomBody': ParagraphStyle(
                'CustomBody',
                parent=self.styles['Normal'],
                fontSize=10,
                spaceAfter=6,
                alignment=TA_JUSTIFY
            ),
            'ExecutiveSummary': ParagraphStyle(
                'ExecutiveSummary',
                parent=self.styles['Normal'],
                fontSize=11,
                spaceAfter=8,
                leftIndent=20,
                rightIndent=20,
                alignment=TA_JUSTIFY,
                backColor=colors.HexColor('#F0F8FF')
            )
        }
    
    def generate_comprehensive_pdf_report(self, analysis_data: Dict[str, Any], 
                                        file_info: Dict[str, Any],
                                        output_path: str) -> str:
        """Generate a comprehensive PDF report with visualizations"""
        try:
            doc = SimpleDocTemplate(output_path, pagesize=A4, 
                                  rightMargin=72, leftMargin=72,
                                  topMargin=72, bottomMargin=18)
            
            story = []
            
            # Title page
            story.extend(self._create_title_page(file_info))
            story.append(PageBreak())
            
            # Executive summary
            story.extend(self._create_executive_summary(analysis_data))
            story.append(PageBreak())
            
            # Technical analysis section
            story.extend(self._create_technical_analysis_section(analysis_data))
            
            # Visualizations section
            story.extend(self._create_visualizations_section(analysis_data))
            
            # Detailed results section
            story.extend(self._create_detailed_results_section(analysis_data))
            story.append(PageBreak())
            
            # Recommendations section
            story.extend(self._create_recommendations_section(analysis_data))
            
            # Appendix
            story.extend(self._create_appendix_section(analysis_data, file_info))
            
            doc.build(story)
            
            self.logger.info(f"PDF report generated successfully: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error generating PDF report: {str(e)}")
            raise
    
    def _create_title_page(self, file_info: Dict[str, Any]) -> List:
        """Create title page"""
        story = []
        
        # Title
        story.append(Paragraph("Media Forensics Analysis Report", self.custom_styles['CustomTitle']))
        story.append(Spacer(1, 0.5*inch))
        
        # File information table
        file_data = [
            ['File Name:', file_info.get('name', 'Unknown')],
            ['File Size:', f"{file_info.get('size', 0):,} bytes"],
            ['File Type:', file_info.get('type', 'Unknown')],
            ['Analysis Date:', datetime.now().strftime('%B %d, %Y')],
            ['Analysis Time:', datetime.now().strftime('%I:%M %p')],
        ]
        
        file_table = Table(file_data, colWidths=[2*inch, 3*inch])
        file_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#E6F3FF')),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        
        story.append(file_table)
        story.append(Spacer(1, 1*inch))
        
        # Disclaimer
        disclaimer = """
        <b>CONFIDENTIAL ANALYSIS REPORT</b><br/><br/>
        This report contains the results of automated media forensics analysis. 
        The findings presented here are based on computational analysis and should 
        be interpreted by qualified professionals. This report is intended for 
        authorized personnel only.
        """
        story.append(Paragraph(disclaimer, self.custom_styles['CustomBody']))
        
        return story
    
    def _create_executive_summary(self, analysis_data: Dict[str, Any]) -> List:
        """Create executive summary section"""
        story = []
        
        story.append(Paragraph("Executive Summary", self.custom_styles['CustomTitle']))
        story.append(Spacer(1, 0.2*inch))
        
        # Overall verdict
        verdict = self._determine_verdict(analysis_data)
        confidence = analysis_data.get('confidence', 0)
        
        verdict_color = colors.red if 'DETECTED' in verdict.upper() else colors.green
        
        summary_text = f"""
        <b>Analysis Verdict:</b> <font color="{verdict_color.hexval()}">{verdict}</font><br/>
        <b>Confidence Level:</b> {confidence:.1%}<br/>
        <b>Risk Assessment:</b> {self._assess_risk_level(analysis_data)}<br/>
        <b>Processing Time:</b> {analysis_data.get('processing_time', 0):.2f} seconds<br/>
        """
        
        story.append(Paragraph(summary_text, self.custom_styles['ExecutiveSummary']))
        story.append(Spacer(1, 0.3*inch))
        
        # Key findings
        story.append(Paragraph("Key Findings", self.custom_styles['CustomHeading']))
        findings = self._extract_key_findings(analysis_data)
        
        for i, finding in enumerate(findings[:5], 1):
            story.append(Paragraph(f"{i}. {finding}", self.custom_styles['CustomBody']))
        
        story.append(Spacer(1, 0.2*inch))
        
        # Recommendations summary
        story.append(Paragraph("Immediate Recommendations", self.custom_styles['CustomHeading']))
        recommendations = self._generate_recommendations(analysis_data)
        
        for rec in recommendations[:3]:
            story.append(Paragraph(f"• {rec}", self.custom_styles['CustomBody']))
        
        return story
    
    def _create_technical_analysis_section(self, analysis_data: Dict[str, Any]) -> List:
        """Create technical analysis section"""
        story = []
        
        story.append(Paragraph("Technical Analysis", self.custom_styles['CustomTitle']))
        story.append(Spacer(1, 0.2*inch))
        
        # Detection metrics
        story.append(Paragraph("Detection Metrics", self.custom_styles['CustomHeading']))
        
        metrics_data = [
            ['Metric', 'Value', 'Interpretation'],
            ['Confidence Score', f"{analysis_data.get('confidence', 0):.3f}", 
             'High' if analysis_data.get('confidence', 0) > 0.8 else 'Medium' if analysis_data.get('confidence', 0) > 0.5 else 'Low'],
            ['Model Accuracy', f"{analysis_data.get('model_accuracy', 0.85):.3f}", 
             'Reliable' if analysis_data.get('model_accuracy', 0.85) > 0.8 else 'Moderate'],
            ['Processing Quality', 'High', 'Optimal analysis conditions'],
        ]
        
        # Add model-specific metrics if available
        if 'model_predictions' in analysis_data:
            for model, pred in analysis_data['model_predictions'].items():
                if isinstance(pred, dict):
                    metrics_data.append([
                        f'{model} Score', 
                        f"{pred.get('confidence', 0):.3f}",
                        pred.get('prediction', 'Unknown')
                    ])
        
        metrics_table = Table(metrics_data, colWidths=[2*inch, 1.5*inch, 2*inch])
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2E5266')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(metrics_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Technical indicators
        if analysis_data.get('technical_details'):
            story.append(Paragraph("Technical Indicators", self.custom_styles['CustomHeading']))
            
            tech_details = analysis_data['technical_details']
            for key, value in tech_details.items():
                if isinstance(value, (int, float, str)) and len(str(value)) < 100:
                    story.append(Paragraph(f"<b>{key.replace('_', ' ').title()}:</b> {value}", 
                                         self.custom_styles['CustomBody']))
        
        return story
    
    def _create_visualizations_section(self, analysis_data: Dict[str, Any]) -> List:
        """Create visualizations section with charts"""
        story = []
        
        story.append(Paragraph("Visual Analysis", self.custom_styles['CustomTitle']))
        story.append(Spacer(1, 0.2*inch))
        
        # Confidence visualization
        if 'confidence' in analysis_data:
            story.extend(self._create_confidence_chart(analysis_data['confidence']))
        
        # Model comparison chart
        if 'model_predictions' in analysis_data:
            story.extend(self._create_model_comparison_chart(analysis_data['model_predictions']))
        
        # Processing time chart
        if 'processing_time' in analysis_data:
            story.extend(self._create_processing_time_chart(analysis_data['processing_time']))
        
        return story
    
    def _create_confidence_chart(self, confidence: float) -> List:
        """Create confidence level visualization"""
        story = []
        
        story.append(Paragraph("Confidence Level Analysis", self.custom_styles['CustomHeading']))
        
        # Create gauge-like chart using matplotlib
        fig, ax = plt.subplots(figsize=(8, 4))
        
        # Create gauge chart
        categories = ['Low\n(0-50%)', 'Medium\n(50-80%)', 'High\n(80-100%)']
        values = [min(confidence * 100, 50), 
                 max(0, min((confidence * 100) - 50, 30)),
                 max(0, (confidence * 100) - 80)]
        colors_list = ['#ff9999', '#ffcc99', '#99ff99']
        
        bars = ax.bar(categories, [50, 30, 20], color=['#ffeeee', '#fff5ee', '#eeffee'], alpha=0.3)
        filled_bars = ax.bar(categories, values, color=colors_list, alpha=0.8)
        
        # Add confidence indicator
        ax.axhline(y=confidence * 100, color='red', linestyle='--', linewidth=2, 
                  label=f'Current Confidence: {confidence:.1%}')
        
        ax.set_ylabel('Confidence Level (%)')
        ax.set_title('Detection Confidence Analysis')
        ax.legend()
        ax.set_ylim(0, 100)
        
        # Save to bytes
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        plt.close()
        
        # Add to story
        story.append(Image(img_buffer, width=6*inch, height=3*inch))
        story.append(Spacer(1, 0.2*inch))
        
        return story
    
    def _create_model_comparison_chart(self, model_predictions: Dict[str, Any]) -> List:
        """Create model comparison chart"""
        story = []
        
        story.append(Paragraph("Model Performance Comparison", self.custom_styles['CustomHeading']))
        
        # Extract model scores
        models = []
        scores = []
        
        for model, pred in model_predictions.items():
            if isinstance(pred, dict) and 'confidence' in pred:
                models.append(model.replace('_', ' ').title())
                scores.append(pred['confidence'])
        
        if models and scores:
            fig, ax = plt.subplots(figsize=(8, 4))
            
            bars = ax.bar(models, scores, color=['#4472C4', '#E70000', '#FFC000', '#70AD47'])
            
            # Add value labels on bars
            for bar, score in zip(bars, scores):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{score:.3f}', ha='center', va='bottom')
            
            ax.set_ylabel('Confidence Score')
            ax.set_title('Model Prediction Comparison')
            ax.set_ylim(0, 1)
            plt.xticks(rotation=45, ha='right')
            
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
            img_buffer.seek(0)
            plt.close()
            
            story.append(Image(img_buffer, width=6*inch, height=3*inch))
            story.append(Spacer(1, 0.2*inch))
        
        return story
    
    def _create_processing_time_chart(self, processing_time: float) -> List:
        """Create processing time visualization"""
        story = []
        
        story.append(Paragraph("Processing Performance", self.custom_styles['CustomHeading']))
        
        # Benchmarks for different processing categories
        categories = ['Fast\n(<2s)', 'Normal\n(2-10s)', 'Slow\n(>10s)']
        benchmarks = [2, 8, 10]  # Upper limits for each category
        current_category = 0 if processing_time < 2 else 1 if processing_time < 10 else 2
        
        fig, ax = plt.subplots(figsize=(8, 3))
        
        bars = ax.bar(categories, benchmarks, color=['#90EE90', '#FFD700', '#FFB6C1'], alpha=0.5)
        
        # Highlight current performance
        bars[current_category].set_color(['#00FF00', '#FFA500', '#FF0000'][current_category])
        bars[current_category].set_alpha(0.8)
        
        ax.axhline(y=processing_time, color='red', linestyle='--', linewidth=2,
                  label=f'Current: {processing_time:.2f}s')
        
        ax.set_ylabel('Processing Time (seconds)')
        ax.set_title('Processing Time Analysis')
        ax.legend()
        
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        plt.close()
        
        story.append(Image(img_buffer, width=6*inch, height=2.5*inch))
        story.append(Spacer(1, 0.2*inch))
        
        return story
    
    def _create_detailed_results_section(self, analysis_data: Dict[str, Any]) -> List:
        """Create detailed results section"""
        story = []
        
        story.append(Paragraph("Detailed Analysis Results", self.custom_styles['CustomTitle']))
        story.append(Spacer(1, 0.2*inch))
        
        # Detection results
        story.append(Paragraph("Detection Results", self.custom_styles['CustomHeading']))
        
        detection_data = [
            ['Parameter', 'Result', 'Details']
        ]
        
        # Add detection results
        if analysis_data.get('is_fake') is not None:
            detection_data.append([
                'Manipulation Detected',
                'Yes' if analysis_data['is_fake'] else 'No',
                f"Confidence: {analysis_data.get('confidence', 0):.1%}"
            ])
        
        if analysis_data.get('is_duplicate') is not None:
            detection_data.append([
                'Duplicate Content',
                'Yes' if analysis_data['is_duplicate'] else 'No',
                f"Matches: {analysis_data.get('duplicate_count', 0)}"
            ])
        
        if analysis_data.get('deepfake_detected') is not None:
            detection_data.append([
                'Deepfake Indicators',
                'Yes' if analysis_data['deepfake_detected'] else 'No',
                'Facial analysis performed'
            ])
        
        # Quality metrics
        detection_data.append([
            'Processing Quality',
            'High',
            f"Time: {analysis_data.get('processing_time', 0):.2f}s"
        ])
        
        detection_table = Table(detection_data, colWidths=[2*inch, 1.5*inch, 2.5*inch])
        detection_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2E5266')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(detection_table)
        story.append(Spacer(1, 0.3*inch))
        
        return story
    
    def _create_recommendations_section(self, analysis_data: Dict[str, Any]) -> List:
        """Create recommendations section"""
        story = []
        
        story.append(Paragraph("Recommendations & Next Steps", self.custom_styles['CustomTitle']))
        story.append(Spacer(1, 0.2*inch))
        
        recommendations = self._generate_recommendations(analysis_data)
        
        for i, rec in enumerate(recommendations, 1):
            story.append(Paragraph(f"{i}. {rec}", self.custom_styles['CustomBody']))
            story.append(Spacer(1, 0.1*inch))
        
        # Additional considerations
        story.append(Spacer(1, 0.2*inch))
        story.append(Paragraph("Additional Considerations", self.custom_styles['CustomHeading']))
        
        considerations = [
            "Results should be interpreted in context of use case and requirements",
            "Consider manual review for high-stakes decisions",
            "Archive analysis results for audit and compliance purposes",
            "Monitor for updates to detection algorithms and models"
        ]
        
        for consideration in considerations:
            story.append(Paragraph(f"• {consideration}", self.custom_styles['CustomBody']))
        
        return story
    
    def _create_appendix_section(self, analysis_data: Dict[str, Any], file_info: Dict[str, Any]) -> List:
        """Create appendix section"""
        story = []
        
        story.append(Paragraph("Appendix", self.custom_styles['CustomTitle']))
        story.append(Spacer(1, 0.2*inch))
        
        # Technical specifications
        story.append(Paragraph("Technical Specifications", self.custom_styles['CustomHeading']))
        
        tech_specs = [
            ['Analysis Framework', 'Media Forensics Suite v1.0'],
            ['Detection Models', 'EfficientNet, XceptionNet, MesoNet'],
            ['Processing Environment', f"Python {analysis_data.get('python_version', '3.x')}"],
            ['Report Generated', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
        ]
        
        tech_table = Table(tech_specs, colWidths=[2*inch, 4*inch])
        tech_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#E6F3FF')),
        ]))
        
        story.append(tech_table)
        story.append(Spacer(1, 0.2*inch))
        
        # Glossary
        story.append(Paragraph("Glossary", self.custom_styles['CustomHeading']))
        
        glossary_terms = [
            ("Confidence Score", "Numerical measure (0-1) indicating the certainty of detection"),
            ("Deepfake", "AI-generated synthetic media that replaces a person's likeness"),
            ("Manipulation", "Any alteration to the original media content"),
            ("Model Accuracy", "Performance metric of the detection algorithm")
        ]
        
        for term, definition in glossary_terms:
            story.append(Paragraph(f"<b>{term}:</b> {definition}", self.custom_styles['CustomBody']))
        
        return story
    
    def generate_batch_pdf_report(self, batch_results: Dict[str, Any], output_path: str) -> str:
        """Generate PDF report for batch processing results"""
        try:
            doc = SimpleDocTemplate(output_path, pagesize=A4,
                                  rightMargin=72, leftMargin=72,
                                  topMargin=72, bottomMargin=18)
            
            story = []
            
            # Title
            story.append(Paragraph("Batch Processing Report", self.custom_styles['CustomTitle']))
            story.append(Spacer(1, 0.3*inch))
            
            # Batch summary
            job_info = batch_results.get('job_info', {})
            story.append(Paragraph(f"Job ID: {job_info.get('job_id', 'Unknown')}", self.custom_styles['CustomBody']))
            story.append(Paragraph(f"Processing Time: {job_info.get('duration', 0):.2f} seconds", self.custom_styles['CustomBody']))
            story.append(Paragraph(f"Files Processed: {job_info.get('total_files', 0)}", self.custom_styles['CustomBody']))
            story.append(Spacer(1, 0.2*inch))
            
            # Statistics section
            stats = batch_results.get('statistics', {})
            story.append(Paragraph("Processing Statistics", self.custom_styles['CustomHeading']))
            
            stats_data = [
                ['Metric', 'Count', 'Percentage'],
                ['Total Processed', str(stats.get('total_processed', 0)), '100%'],
                ['Successful', str(stats.get('successful', 0)), 
                 f"{stats.get('successful', 0) / max(stats.get('total_processed', 1), 1) * 100:.1f}%"],
                ['Failed', str(stats.get('failed', 0)), 
                 f"{stats.get('failed', 0) / max(stats.get('total_processed', 1), 1) * 100:.1f}%"],
                ['Fake Detected', str(stats.get('fake_detected', 0)), 
                 f"{stats.get('fake_detected', 0) / max(stats.get('successful', 1), 1) * 100:.1f}%"],
                ['Duplicates Found', str(stats.get('duplicate_detected', 0)), 
                 f"{stats.get('duplicate_detected', 0) / max(stats.get('successful', 1), 1) * 100:.1f}%"],
            ]
            
            stats_table = Table(stats_data, colWidths=[2.5*inch, 1*inch, 1.5*inch])
            stats_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2E5266')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ]))
            
            story.append(stats_table)
            story.append(PageBreak())
            
            # Individual results summary (first 20 files)
            story.append(Paragraph("Individual Results Summary", self.custom_styles['CustomHeading']))
            
            detailed_results = batch_results.get('detailed_results', [])[:20]  # Limit to first 20
            
            if detailed_results:
                results_data = [['File Name', 'Status', 'Fake', 'Duplicate', 'Confidence']]
                
                for result in detailed_results:
                    file_name = result.get('file_info', {}).get('name', 'Unknown')[:20]  # Truncate long names
                    status = result.get('status', 'Unknown')
                    is_fake = 'Yes' if result.get('is_fake', False) else 'No'
                    is_duplicate = 'Yes' if result.get('is_duplicate', False) else 'No'
                    confidence = f"{result.get('confidence', 0):.2f}"
                    
                    results_data.append([file_name, status, is_fake, is_duplicate, confidence])
                
                results_table = Table(results_data, colWidths=[2*inch, 1*inch, 0.8*inch, 0.8*inch, 1*inch])
                results_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2E5266')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 8),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ]))
                
                story.append(results_table)
            
            doc.build(story)
            
            self.logger.info(f"Batch PDF report generated successfully: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error generating batch PDF report: {str(e)}")
            raise
    
    # Helper methods
    def _determine_verdict(self, results: Dict[str, Any]) -> str:
        """Determine overall verdict from analysis results"""
        if results.get("is_fake", False) or results.get("is_synthetic", False):
            return "SYNTHETIC/MANIPULATED CONTENT DETECTED"
        elif results.get("is_duplicate", False):
            return "DUPLICATE CONTENT DETECTED"
        elif results.get("suspicious_indicators", 0) > 0:
            return "SUSPICIOUS CONTENT - REQUIRES REVIEW"
        else:
            return "AUTHENTIC CONTENT - NO MANIPULATION DETECTED"
    
    def _assess_risk_level(self, results: Dict[str, Any]) -> str:
        """Assess risk level based on analysis results"""
        confidence = results.get('confidence', 0)
        
        if results.get("is_fake", False) and confidence > 0.8:
            return "HIGH"
        elif results.get("is_fake", False) and confidence > 0.6:
            return "MEDIUM"
        elif results.get("suspicious_indicators", 0) > 0:
            return "LOW"
        else:
            return "MINIMAL"
    
    def _extract_key_findings(self, results: Dict[str, Any]) -> List[str]:
        """Extract key findings from analysis results"""
        findings = []
        
        if results.get("is_fake", False):
            findings.append(f"AI-generated content detected with {results.get('confidence', 0):.1%} confidence")
        
        if results.get("deepfake_detected", False):
            findings.append("Deepfake indicators found in facial analysis")
            
        if results.get("is_duplicate", False):
            findings.append(f"Content matches {results.get('duplicate_count', 0)} existing files")
        
        if results.get("compression_artifacts", 0) > 0.7:
            findings.append("High compression artifacts detected")
            
        if results.get("metadata_inconsistencies"):
            findings.append("Metadata inconsistencies found")
        
        return findings if findings else ["No significant anomalies detected"]
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on analysis"""
        recommendations = []
        
        risk_level = self._assess_risk_level(results)
        
        if risk_level == "HIGH":
            recommendations.extend([
                "Content should be flagged as potentially manipulated",
                "Additional verification through alternative detection methods recommended",
                "Consider forensic analysis of source metadata and compression artifacts"
            ])
        elif risk_level == "MEDIUM":
            recommendations.extend([
                "Content requires manual review by trained analyst",
                "Cross-reference with original source if available",
                "Apply additional detection algorithms for confirmation"
            ])
        else:
            recommendations.extend([
                "Content appears authentic based on current analysis",
                "Continue standard processing workflow",
                "Archive analysis results for audit trail"
            ])
            
        return recommendations
