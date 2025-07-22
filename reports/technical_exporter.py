import json
import csv
import xml.etree.ElementTree as ET
import pandas as pd
import os
from datetime import datetime
import logging
from typing import Dict, Any, List, Optional
import numpy as np

class TechnicalDetailsExporter:
    """Export technical details in various structured formats"""
    
    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.supported_formats = ['json', 'csv', 'xml', 'excel', 'txt']
    
    def export_technical_details(self, technical_data: Dict[str, Any], 
                               output_path: str, format_type: str = 'json') -> str:
        """Export technical details in specified format"""
        if format_type not in self.supported_formats:
            raise ValueError(f"Unsupported format: {format_type}. Supported: {self.supported_formats}")
        
        try:
            if format_type == 'json':
                return self._export_json(technical_data, output_path)
            elif format_type == 'csv':
                return self._export_csv(technical_data, output_path)
            elif format_type == 'xml':
                return self._export_xml(technical_data, output_path)
            elif format_type == 'excel':
                return self._export_excel(technical_data, output_path)
            elif format_type == 'txt':
                return self._export_txt(technical_data, output_path)
                
        except Exception as e:
            self.logger.error(f"Error exporting technical details: {str(e)}")
            raise
    
    def _export_json(self, data: Dict[str, Any], output_path: str) -> str:
        """Export to JSON format with proper formatting"""
        structured_data = {
            "export_info": {
                "timestamp": datetime.now().isoformat(),
                "format": "json",
                "version": "1.0"
            },
            "technical_analysis": self._structure_technical_data(data),
            "metadata": data.get("metadata", {}),
            "model_outputs": data.get("model_predictions", {}),
            "quality_metrics": data.get("quality_metrics", {}),
            "processing_details": data.get("processing_details", {})
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(structured_data, f, indent=2, default=self._json_serializer)
        
        self.logger.info(f"Technical details exported to JSON: {output_path}")
        return output_path
    
    def _export_csv(self, data: Dict[str, Any], output_path: str) -> str:
        """Export to CSV format with flattened structure"""
        flattened_data = self._flatten_dict(data)
        
        # Create DataFrame
        df = pd.DataFrame([flattened_data])
        df.to_csv(output_path, index=False)
        
        self.logger.info(f"Technical details exported to CSV: {output_path}")
        return output_path
    
    def _export_xml(self, data: Dict[str, Any], output_path: str) -> str:
        """Export to XML format"""
        root = ET.Element("TechnicalAnalysis")
        
        # Add export info
        export_info = ET.SubElement(root, "ExportInfo")
        ET.SubElement(export_info, "Timestamp").text = datetime.now().isoformat()
        ET.SubElement(export_info, "Format").text = "xml"
        ET.SubElement(export_info, "Version").text = "1.0"
        
        # Add technical data
        self._dict_to_xml(data, root, "Data")
        
        # Write to file
        tree = ET.ElementTree(root)
        tree.write(output_path, encoding='utf-8', xml_declaration=True)
        
        self.logger.info(f"Technical details exported to XML: {output_path}")
        return output_path
    
    def _export_excel(self, data: Dict[str, Any], output_path: str) -> str:
        """Export to Excel with multiple sheets"""
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Summary sheet
            summary_data = self._create_summary_sheet(data)
            summary_df = pd.DataFrame([summary_data])
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Model predictions sheet
            if "model_predictions" in data:
                model_df = pd.json_normalize(data["model_predictions"])
                model_df.to_excel(writer, sheet_name='Model_Predictions', index=False)
            
            # Quality metrics sheet
            if "quality_metrics" in data:
                quality_df = pd.DataFrame([data["quality_metrics"]])
                quality_df.to_excel(writer, sheet_name='Quality_Metrics', index=False)
            
            # Raw data sheet
            flattened_data = self._flatten_dict(data)
            raw_df = pd.DataFrame([flattened_data])
            raw_df.to_excel(writer, sheet_name='Raw_Data', index=False)
        
        self.logger.info(f"Technical details exported to Excel: {output_path}")
        return output_path
    
    def _export_txt(self, data: Dict[str, Any], output_path: str) -> str:
        """Export to structured text format"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("MEDIA FORENSICS TECHNICAL ANALYSIS REPORT\n")
            f.write("="*80 + "\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n\n")
            
            # Write structured data
            self._write_dict_to_text(data, f, level=0)
        
        self.logger.info(f"Technical details exported to TXT: {output_path}")
        return output_path
    
    def _structure_technical_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Structure technical data for better organization"""
        structured = {
            "analysis_results": {},
            "confidence_scores": {},
            "technical_metrics": {},
            "detection_details": {}
        }
        
        # Organize data by category
        for key, value in data.items():
            if "confidence" in key.lower():
                structured["confidence_scores"][key] = value
            elif key in ["processing_time", "model_accuracy", "quality_score"]:
                structured["technical_metrics"][key] = value
            elif key in ["is_fake", "is_duplicate", "deepfake_detected"]:
                structured["analysis_results"][key] = value
            else:
                structured["detection_details"][key] = value
        
        return structured
    
    def _flatten_dict(self, data: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
        """Flatten nested dictionary for CSV export"""
        items = []
        for k, v in data.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                for i, item in enumerate(v):
                    if isinstance(item, dict):
                        items.extend(self._flatten_dict(item, f"{new_key}_{i}", sep=sep).items())
                    else:
                        items.append((f"{new_key}_{i}", item))
            else:
                items.append((new_key, v))
        return dict(items)
    
    def _dict_to_xml(self, data: Dict[str, Any], parent: ET.Element, root_name: str = "item"):
        """Convert dictionary to XML elements"""
        for key, value in data.items():
            # Sanitize key for XML
            clean_key = str(key).replace(" ", "_").replace("-", "_")
            
            if isinstance(value, dict):
                child = ET.SubElement(parent, clean_key)
                self._dict_to_xml(value, child)
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    child = ET.SubElement(parent, f"{clean_key}_item_{i}")
                    if isinstance(item, dict):
                        self._dict_to_xml(item, child)
                    else:
                        child.text = str(item)
            else:
                child = ET.SubElement(parent, clean_key)
                child.text = str(value)
    
    def _create_summary_sheet(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary data for Excel export"""
        summary = {
            "Export_Timestamp": datetime.now().isoformat(),
            "Total_Fields": len(self._flatten_dict(data)),
            "Has_Confidence_Score": "confidence" in data,
            "Has_Model_Predictions": "model_predictions" in data,
            "Has_Quality_Metrics": "quality_metrics" in data,
            "Processing_Status": "completed" if not data.get("error") else "error"
        }
        
        # Add key metrics if available
        if "confidence" in data:
            summary["Primary_Confidence"] = data["confidence"]
        if "processing_time" in data:
            summary["Processing_Time_Seconds"] = data["processing_time"]
        if "is_fake" in data:
            summary["Manipulation_Detected"] = data["is_fake"]
        
        return summary
    
    def _write_dict_to_text(self, data: Dict[str, Any], file_handle, level: int = 0):
        """Write dictionary to text file with proper indentation"""
        indent = "  " * level
        
        for key, value in data.items():
            if isinstance(value, dict):
                file_handle.write(f"{indent}{key}:\n")
                self._write_dict_to_text(value, file_handle, level + 1)
            elif isinstance(value, list):
                file_handle.write(f"{indent}{key}:\n")
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        file_handle.write(f"{indent}  [{i}]:\n")
                        self._write_dict_to_text(item, file_handle, level + 2)
                    else:
                        file_handle.write(f"{indent}  [{i}]: {item}\n")
            else:
                file_handle.write(f"{indent}{key}: {value}\n")
        
        if level == 0:
            file_handle.write("\n")
    
    def _json_serializer(self, obj):
        """JSON serializer for numpy and other types"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, datetime):
            return obj.isoformat()
        return str(obj)
    
    def export_batch_results(self, batch_results: List[Dict[str, Any]], 
                           output_dir: str, format_type: str = 'json') -> List[str]:
        """Export batch processing results"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        exported_files = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for i, result in enumerate(batch_results):
            filename = f"batch_result_{i+1}_{timestamp}.{format_type}"
            output_path = os.path.join(output_dir, filename)
            
            try:
                exported_path = self.export_technical_details(result, output_path, format_type)
                exported_files.append(exported_path)
            except Exception as e:
                self.logger.error(f"Error exporting batch result {i+1}: {str(e)}")
        
        # Create batch summary
        summary_data = {
            "batch_info": {
                "total_files": len(batch_results),
                "exported_files": len(exported_files),
                "export_timestamp": datetime.now().isoformat(),
                "format": format_type
            },
            "results_summary": self._create_batch_summary(batch_results)
        }
        
        summary_path = os.path.join(output_dir, f"batch_summary_{timestamp}.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, default=self._json_serializer)
        
        exported_files.append(summary_path)
        self.logger.info(f"Batch export completed. {len(exported_files)} files exported to {output_dir}")
        
        return exported_files
    
    def _create_batch_summary(self, batch_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create summary statistics for batch results"""
        summary = {
            "total_processed": len(batch_results),
            "successful_analyses": 0,
            "failed_analyses": 0,
            "fake_detections": 0,
            "duplicate_detections": 0,
            "average_confidence": 0.0,
            "processing_times": []
        }
        
        confidence_scores = []
        
        for result in batch_results:
            if result.get("error"):
                summary["failed_analyses"] += 1
            else:
                summary["successful_analyses"] += 1
                
            if result.get("is_fake"):
                summary["fake_detections"] += 1
                
            if result.get("is_duplicate"):
                summary["duplicate_detections"] += 1
                
            if "confidence" in result:
                confidence_scores.append(result["confidence"])
                
            if "processing_time" in result:
                summary["processing_times"].append(result["processing_time"])
        
        if confidence_scores:
            summary["average_confidence"] = float(np.mean(confidence_scores))
            summary["confidence_std"] = float(np.std(confidence_scores))
            summary["min_confidence"] = float(np.min(confidence_scores))
            summary["max_confidence"] = float(np.max(confidence_scores))
        
        if summary["processing_times"]:
            summary["average_processing_time"] = float(np.mean(summary["processing_times"]))
            summary["total_processing_time"] = float(np.sum(summary["processing_times"]))
        
        return summary
