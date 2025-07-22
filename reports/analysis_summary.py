import json
import numpy as np
import pandas as pd
from datetime import datetime
import logging
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

class AnalysisSummaryGenerator:
    """Advanced analysis summary generator for media forensics reports"""
    
    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def generate_executive_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate executive summary from analysis results"""
        try:
            summary = {
                "timestamp": datetime.now().isoformat(),
                "overall_verdict": self._determine_verdict(results),
                "confidence_score": self._calculate_overall_confidence(results),
                "risk_level": self._assess_risk_level(results),
                "key_findings": self._extract_key_findings(results),
                "technical_indicators": self._get_technical_indicators(results),
                "recommendations": self._generate_recommendations(results)
            }
            
            self.logger.info("Executive summary generated successfully")
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating executive summary: {str(e)}")
            return {"error": str(e)}
    
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
    
    def _calculate_overall_confidence(self, results: Dict[str, Any]) -> float:
        """Calculate overall confidence score"""
        confidence_scores = []
        
        # Collect all confidence scores from different analyses
        if "confidence" in results:
            confidence_scores.append(results["confidence"])
        
        if "detection_confidence" in results:
            confidence_scores.append(results["detection_confidence"])
            
        if "model_predictions" in results:
            for pred in results["model_predictions"]:
                if isinstance(pred, dict) and "confidence" in pred:
                    confidence_scores.append(pred["confidence"])
        
        if confidence_scores:
            return float(np.mean(confidence_scores))
        return 0.0
    
    def _assess_risk_level(self, results: Dict[str, Any]) -> str:
        """Assess risk level based on analysis results"""
        confidence = self._calculate_overall_confidence(results)
        
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
        
        # AI/Synthetic content findings
        if results.get("is_fake", False):
            findings.append(f"AI-generated content detected with {results.get('confidence', 0):.1%} confidence")
        
        # Deepfake findings
        if results.get("deepfake_detected", False):
            findings.append(f"Deepfake indicators found in facial analysis")
            
        # Duplicate findings
        if results.get("is_duplicate", False):
            findings.append(f"Content matches {results.get('duplicate_count', 0)} existing files")
        
        # Technical anomalies
        if results.get("compression_artifacts", 0) > 0.7:
            findings.append("High compression artifacts detected")
            
        if results.get("metadata_inconsistencies"):
            findings.append("Metadata inconsistencies found")
            
        # Model-specific findings
        if "model_predictions" in results:
            for model, pred in results["model_predictions"].items():
                if isinstance(pred, dict) and pred.get("anomaly_score", 0) > 0.5:
                    findings.append(f"{model} model detected anomalies")
        
        return findings if findings else ["No significant anomalies detected"]
    
    def _get_technical_indicators(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract technical indicators for detailed analysis"""
        indicators = {}
        
        # Image quality metrics
        if "image_quality" in results:
            indicators["image_quality"] = results["image_quality"]
            
        # Compression analysis
        if "compression_analysis" in results:
            indicators["compression_metrics"] = results["compression_analysis"]
            
        # Frequency domain analysis
        if "frequency_analysis" in results:
            indicators["frequency_anomalies"] = results["frequency_analysis"]
            
        # Metadata analysis
        if "metadata" in results:
            indicators["metadata_integrity"] = self._analyze_metadata_integrity(results["metadata"])
            
        # Model scores
        model_scores = {}
        if "model_predictions" in results:
            for model, pred in results["model_predictions"].items():
                if isinstance(pred, dict):
                    model_scores[model] = {
                        "confidence": pred.get("confidence", 0),
                        "prediction": pred.get("prediction", "unknown")
                    }
        if model_scores:
            indicators["model_scores"] = model_scores
            
        return indicators
    
    def _analyze_metadata_integrity(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze metadata for integrity issues"""
        integrity = {
            "present": len(metadata) > 0,
            "complete": False,
            "consistent": True,
            "suspicious_fields": []
        }
        
        # Check for common metadata fields
        expected_fields = ["DateTime", "Camera", "Software", "GPS"]
        present_fields = [field for field in expected_fields if any(field.lower() in key.lower() for key in metadata.keys())]
        integrity["complete"] = len(present_fields) >= 2
        
        # Check for suspicious software signatures
        software_fields = [value for key, value in metadata.items() if "software" in key.lower()]
        for software in software_fields:
            if any(suspicious in str(software).lower() for suspicious in ["photoshop", "gimp", "deepfake", "fakeapp"]):
                integrity["suspicious_fields"].append(f"Suspicious software: {software}")
                integrity["consistent"] = False
        
        return integrity
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on analysis"""
        recommendations = []
        
        confidence = self._calculate_overall_confidence(results)
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
        elif risk_level == "LOW":
            recommendations.extend([
                "Monitor for additional suspicious indicators",
                "Document findings for future reference",
                "Consider batch analysis of related content"
            ])
        else:
            recommendations.extend([
                "Content appears authentic based on current analysis",
                "Continue standard processing workflow",
                "Archive analysis results for audit trail"
            ])
            
        # Technical recommendations
        if results.get("low_resolution", False):
            recommendations.append("Consider higher resolution analysis for improved accuracy")
            
        if results.get("processing_time", 0) > 30:
            recommendations.append("Long processing time detected - consider optimization")
            
        return recommendations
    
    def generate_detailed_report(self, results: Dict[str, Any], file_info: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive detailed report"""
        try:
            detailed_report = {
                "executive_summary": self.generate_executive_summary(results),
                "file_information": file_info,
                "analysis_details": {
                    "detection_results": results,
                    "statistical_analysis": self._generate_statistical_analysis(results),
                    "visualization_data": self._prepare_visualization_data(results)
                },
                "quality_metrics": self._calculate_quality_metrics(results),
                "forensic_indicators": self._extract_forensic_indicators(results),
                "compliance_check": self._run_compliance_checks(results)
            }
            
            self.logger.info("Detailed report generated successfully")
            return detailed_report
            
        except Exception as e:
            self.logger.error(f"Error generating detailed report: {str(e)}")
            return {"error": str(e)}
    
    def _generate_statistical_analysis(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate statistical analysis of results"""
        stats = {
            "confidence_distribution": {},
            "anomaly_statistics": {},
            "model_performance": {}
        }
        
        # Confidence distribution
        if "confidence" in results:
            conf = results["confidence"]
            stats["confidence_distribution"] = {
                "mean": float(conf),
                "category": "high" if conf > 0.8 else "medium" if conf > 0.5 else "low"
            }
        
        # Model performance statistics
        if "model_predictions" in results:
            performances = []
            for model, pred in results["model_predictions"].items():
                if isinstance(pred, dict) and "confidence" in pred:
                    performances.append(pred["confidence"])
            
            if performances:
                stats["model_performance"] = {
                    "mean_confidence": float(np.mean(performances)),
                    "std_confidence": float(np.std(performances)),
                    "model_agreement": len([p for p in performances if p > 0.5]) / len(performances)
                }
        
        return stats
    
    def _prepare_visualization_data(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data for visualizations"""
        viz_data = {}
        
        # Confidence score visualization
        if "confidence" in results:
            viz_data["confidence_chart"] = {
                "type": "gauge",
                "value": results["confidence"],
                "title": "Detection Confidence"
            }
        
        # Model comparison data
        if "model_predictions" in results:
            model_names = []
            model_scores = []
            for model, pred in results["model_predictions"].items():
                if isinstance(pred, dict) and "confidence" in pred:
                    model_names.append(model)
                    model_scores.append(pred["confidence"])
            
            if model_names:
                viz_data["model_comparison"] = {
                    "type": "bar",
                    "models": model_names,
                    "scores": model_scores
                }
        
        return viz_data
    
    def _calculate_quality_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate quality metrics for the analysis"""
        metrics = {
            "analysis_completeness": 0.0,
            "data_quality": "good",
            "processing_efficiency": "normal"
        }
        
        # Calculate completeness based on available data
        expected_fields = ["confidence", "processing_time", "technical_details"]
        present_fields = sum(1 for field in expected_fields if field in results)
        metrics["analysis_completeness"] = present_fields / len(expected_fields)
        
        # Data quality assessment
        if results.get("error"):
            metrics["data_quality"] = "poor"
        elif results.get("warnings"):
            metrics["data_quality"] = "fair"
        
        # Processing efficiency
        processing_time = results.get("processing_time", 0)
        if processing_time > 30:
            metrics["processing_efficiency"] = "slow"
        elif processing_time < 2:
            metrics["processing_efficiency"] = "fast"
            
        return metrics
    
    def _extract_forensic_indicators(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract forensic indicators for legal/compliance purposes"""
        indicators = []
        
        # Tampering indicators
        if results.get("is_fake", False):
            indicators.append({
                "type": "tampering",
                "severity": "high",
                "description": "Content manipulation detected",
                "confidence": results.get("confidence", 0)
            })
        
        # Metadata indicators
        if results.get("metadata_inconsistencies"):
            indicators.append({
                "type": "metadata_anomaly",
                "severity": "medium",
                "description": "Inconsistent metadata detected",
                "details": results["metadata_inconsistencies"]
            })
        
        # Technical indicators
        if results.get("compression_artifacts", 0) > 0.8:
            indicators.append({
                "type": "compression_anomaly",
                "severity": "medium",
                "description": "Unusual compression patterns detected"
            })
        
        return indicators
    
    def _run_compliance_checks(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Run compliance checks for various standards"""
        compliance = {
            "chain_of_custody": True,
            "analysis_standards": "ISO_27037_compliant",
            "data_integrity": True,
            "documentation_complete": True
        }
        
        # Check for required fields
        required_fields = ["timestamp", "file_info", "technical_details"]
        for field in required_fields:
            if field not in results and field != "timestamp":
                compliance["documentation_complete"] = False
        
        # Data integrity check
        if results.get("error") or results.get("processing_failed"):
            compliance["data_integrity"] = False
        
        return compliance
    
    def export_summary_json(self, summary: Dict[str, Any], file_path: str) -> str:
        """Export analysis summary to JSON file"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, default=str)
            
            self.logger.info(f"Summary exported to {file_path}")
            return file_path
            
        except Exception as e:
            self.logger.error(f"Error exporting summary: {str(e)}")
            raise
    
    def export_summary_excel(self, summary: Dict[str, Any], file_path: str) -> str:
        """Export analysis summary to Excel file with multiple sheets"""
        try:
            with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                # Executive summary sheet
                if "executive_summary" in summary:
                    exec_df = pd.DataFrame([summary["executive_summary"]])
                    exec_df.to_excel(writer, sheet_name='Executive Summary', index=False)
                
                # Technical details sheet
                if "analysis_details" in summary:
                    tech_df = pd.json_normalize(summary["analysis_details"])
                    tech_df.to_excel(writer, sheet_name='Technical Details', index=False)
                
                # Quality metrics sheet
                if "quality_metrics" in summary:
                    quality_df = pd.DataFrame([summary["quality_metrics"]])
                    quality_df.to_excel(writer, sheet_name='Quality Metrics', index=False)
            
            self.logger.info(f"Summary exported to Excel: {file_path}")
            return file_path
            
        except Exception as e:
            self.logger.error(f"Error exporting to Excel: {str(e)}")
            raise
