import json
from fpdf import FPDF
import os
from datetime import datetime
import logging
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class ComprehensiveReportGenerator:
    def __init__(self):
        logging.basicConfig(level=logging.DEBUG)
        self.logger = logging.getLogger(__name__)

    def generate_json_report(self, results: dict, file_name: str) -> str:
        """Generate a complete report in JSON format"""
        self.logger.debug("Generating comprehensive JSON report...")
        report = {
            "report": {
                "generated_at": datetime.now().isoformat(),
                "file_analyzed": file_name,
                "summary": self._generate_analysis_summary(results),
                "technical_details": results,
            }
        }
        json_report = json.dumps(report, indent=2)
        return json_report

    def save_json_report(self, report: str, directory: str) -> str:
        """Save JSON report to the specified directory"""
        if not os.path.exists(directory):
            os.makedirs(directory)
        file_path = os.path.join(directory, f"complete_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(file_path, 'w') as json_file:
            json_file.write(report)
        self.logger.debug(f"JSON report saved to {file_path}")
        return file_path

    def generate_pdf_report(self, results: dict, file_name: str) -> str:
        """Generate a complete report in PDF format"""
        self.logger.debug("Generating comprehensive PDF report...")
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Comprehensive Media Forensics Report", ln=True, align='C')
        pdf.cell(200, 10, txt=f"Generated: {datetime.now().isoformat()}", ln=True, align='C')
        pdf.ln(10)
        pdf.multi_cell(0, 10, txt=f"File Analyzed: {file_name}")
        pdf.ln(5)
        
        summary = self._generate_analysis_summary(results)
        pdf.multi_cell(0, 10, txt=f"Summary: {summary}")
        pdf.ln(5)

        # Result details
        for key, value in results.items():
            pdf.cell(0, 10, txt=f"{key}: {value}", ln=True)

        # Save to a file
        directory = os.path.join(os.getcwd(), 'comprehensive_reports')
        if not os.path.exists(directory):
            os.makedirs(directory)
        file_path = os.path.join(directory, f"complete_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
        pdf.output(file_path)
        self.logger.debug(f"PDF report saved to {file_path}")
        return file_path

    def _generate_analysis_summary(self, results: dict) -> str:
        """Generate a summary of the analysis results"""
        # Placeholder for more complex summary logic
        summary_lines = [f"{key}: {value}" for key, value in results.items() if key != 'technical_details']
        return " | ".join(summary_lines)

