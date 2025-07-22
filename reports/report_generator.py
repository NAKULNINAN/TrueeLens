import json
from fpdf import FPDF
import os
from datetime import datetime
import logging

class ReportGenerator:
    def __init__(self):
        logging.basicConfig(level=logging.DEBUG)
        self.logger = logging.getLogger(__name__)

    def generate_json_report(self, results: dict, file_name: str) -> str:
        """Generate report in JSON format"""
        self.logger.debug("Generating JSON report...")
        report = {
            "report": {
                "generated_at": datetime.now().isoformat(),
                "file_analyzed": file_name,
                "results": results
            }
        }
        json_report = json.dumps(report, indent=2)
        return json_report

    def save_json_report(self, report: str, directory: str) -> str:
        """Save JSON report to the specified directory"""
        if not os.path.exists(directory):
            os.makedirs(directory)
        file_path = os.path.join(directory, f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(file_path, 'w') as json_file:
            json_file.write(report)
        self.logger.debug(f"JSON report saved to {file_path}")
        return file_path

    def generate_pdf_report(self, results: dict, file_name: str) -> str:
        """Generate report in PDF format"""
        self.logger.debug("Generating PDF report...")
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Media Forensics Report", ln=True, align='C')
        pdf.cell(200, 10, txt=f"Generated: {datetime.now().isoformat()}", ln=True, align='C')
        pdf.ln(10)
        pdf.multi_cell(0, 10, txt=f"File Analyzed: {file_name}")
        pdf.ln(5)
        
        # Result details
        for key, value in results.items():
            pdf.cell(0, 10, txt=f"{key}: {value}", ln=True)

        # Save to a file
        directory = os.path.join(os.getcwd(), 'reports')
        if not os.path.exists(directory):
            os.makedirs(directory)
        file_path = os.path.join(directory, f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
        pdf.output(file_path)
        self.logger.debug(f"PDF report saved to {file_path}")
        return file_path

