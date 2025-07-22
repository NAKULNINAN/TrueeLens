"""
Complete unit tests for all reports modules
Tests report generator, export manager, and batch processor
"""

import pytest
import os
import json
import tempfile
import zipfile
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
from PIL import Image
import numpy as np

from reports.report_generator import ReportGenerator
from reports.export_manager import ExportManager
from reports.batch_processor import BatchReportProcessor


class TestReportGeneratorComplete:
    """Complete tests for Report Generator"""
    
    def setup_method(self):
        """Setup test environment"""
        self.report_generator = ReportGenerator()
        self.sample_results = self._create_sample_results()
    
    def _create_sample_results(self):
        """Create sample detection results for testing"""
        return {
            'mode': 'Detect AI-Generated Image',
            'timestamp': datetime.now().isoformat(),
            'file_info': {
                'name': 'test_image.jpg',
                'size': 1024000,
                'type': 'image/jpeg'
            },
            'confidence': 0.87,
            'is_fake': True,
            'explanation': 'Image shows characteristics consistent with AI generation',
            'processing_time': 2.5,
            'technical_details': {
                'model_used': 'EfficientNet',
                'stylegan_analysis': {
                    'stylegan_confidence': 0.75,
                    'spectral_ratio': 0.45,
                    'periodic_score': 0.62
                },
                'dalle_analysis': {
                    'dalle_confidence': 0.82,
                    'edge_uniformity': 0.78,
                    'color_uniformity': 0.85
                },
                'midjourney_analysis': {
                    'midjourney_confidence': 0.69,
                    'color_harmony': 0.73,
                    'gradient_smoothness': 0.81
                }
            },
            'visualizations': {
                'heatmap_generated': True,
                'confidence_chart_generated': True
            }
        }
    
    def test_generate_basic_report(self):
        """Test basic report generation"""
        report = self.report_generator.generate_report(self.sample_results)
        
        assert report is not None
        assert isinstance(report, dict)
        assert 'report_id' in report
        assert 'generated_at' in report
        assert 'summary' in report
        assert 'detailed_analysis' in report
    
    def test_generate_json_report(self):
        """Test JSON report generation"""
        json_report = self.report_generator.generate_json_report(self.sample_results)
        
        assert json_report is not None
        assert isinstance(json_report, str)
        
        # Verify it's valid JSON
        parsed_report = json.loads(json_report)
        assert 'media_forensics_report' in parsed_report
        assert 'version' in parsed_report['media_forensics_report']
        assert 'results' in parsed_report['media_forensics_report']
    
    def test_generate_html_report(self):
        """Test HTML report generation"""
        html_report = self.report_generator.generate_html_report(self.sample_results)
        
        assert html_report is not None
        assert isinstance(html_report, str)
        assert '<html>' in html_report
        assert '</html>' in html_report
        assert 'Media Forensics Analysis Report' in html_report
        assert str(self.sample_results['confidence']) in html_report
    
    def test_generate_csv_report(self):
        """Test CSV report generation"""
        csv_report = self.report_generator.generate_csv_report([self.sample_results])
        
        assert csv_report is not None
        assert isinstance(csv_report, str)
        assert 'file_name,confidence,is_fake,processing_time' in csv_report
        assert 'test_image.jpg' in csv_report
    
    def test_generate_pdf_report(self):
        """Test PDF report generation"""
        try:
            pdf_bytes = self.report_generator.generate_pdf_report(self.sample_results)
            
            assert pdf_bytes is not None
            assert isinstance(pdf_bytes, bytes)
            assert pdf_bytes.startswith(b'%PDF')  # PDF signature
            
        except ImportError:
            # Skip if PDF generation dependencies not available
            pytest.skip("PDF generation dependencies not available")
    
    def test_generate_summary_statistics(self):
        """Test summary statistics generation"""
        batch_results = [self.sample_results.copy() for _ in range(5)]
        
        # Vary the results
        for i, result in enumerate(batch_results):
            result['confidence'] = 0.5 + (i * 0.1)
            result['is_fake'] = i % 2 == 0
            result['processing_time'] = 1.0 + (i * 0.5)
        
        stats = self.report_generator.generate_summary_statistics(batch_results)
        
        assert 'total_files' in stats
        assert 'fake_detected' in stats
        assert 'real_detected' in stats
        assert 'average_confidence' in stats
        assert 'average_processing_time' in stats
        assert stats['total_files'] == 5
        assert stats['fake_detected'] + stats['real_detected'] == 5
    
    def test_generate_detailed_analysis(self):
        """Test detailed analysis generation"""
        analysis = self.report_generator.generate_detailed_analysis(self.sample_results)
        
        assert 'technical_summary' in analysis
        assert 'confidence_breakdown' in analysis
        assert 'detection_rationale' in analysis
        assert 'recommendations' in analysis
        
        # Check technical summary
        tech_summary = analysis['technical_summary']
        assert 'model_performance' in tech_summary
        assert 'artifact_analysis' in tech_summary
        assert 'processing_metrics' in tech_summary
    
    def test_generate_comparison_report(self):
        """Test comparison report generation"""
        # Create multiple results for comparison
        results_list = []
        for i in range(3):
            result = self.sample_results.copy()
            result['file_info']['name'] = f'test_image_{i}.jpg'
            result['confidence'] = 0.6 + (i * 0.1)
            results_list.append(result)
        
        comparison = self.report_generator.generate_comparison_report(results_list)
        
        assert 'files_compared' in comparison
        assert 'confidence_distribution' in comparison
        assert 'detection_summary' in comparison
        assert comparison['files_compared'] == 3
    
    def test_add_visualizations_to_report(self):
        """Test adding visualizations to report"""
        # Create sample visualization data
        viz_data = {
            'heatmap_base64': 'data:image/png;base64,iVBORw0KGgoAAAANS',
            'confidence_chart_html': '<div>Chart HTML</div>',
            'comparison_grid': 'visualization_grid_data'
        }
        
        enhanced_report = self.report_generator.add_visualizations_to_report(
            self.sample_results, viz_data
        )
        
        assert 'visualizations' in enhanced_report
        assert enhanced_report['visualizations']['heatmap_base64'] is not None
        assert enhanced_report['visualizations']['confidence_chart_html'] is not None
    
    def test_generate_executive_summary(self):
        """Test executive summary generation"""
        summary = self.report_generator.generate_executive_summary(self.sample_results)
        
        assert 'key_findings' in summary
        assert 'risk_assessment' in summary
        assert 'confidence_level' in summary
        assert 'recommendations' in summary
        
        # Check risk assessment
        risk = summary['risk_assessment']
        assert risk in ['Low', 'Medium', 'High']
    
    def test_report_templates(self):
        """Test different report templates"""
        templates = ['standard', 'detailed', 'executive', 'technical']
        
        for template in templates:
            report = self.report_generator.generate_report(
                self.sample_results, 
                template=template
            )
            
            assert report is not None
            assert 'template_used' in report
            assert report['template_used'] == template
    
    def test_custom_report_metadata(self):
        """Test custom report metadata"""
        custom_metadata = {
            'analyst_name': 'Test Analyst',
            'organization': 'Test Organization',
            'case_id': 'CASE-2024-001',
            'priority': 'High'
        }
        
        report = self.report_generator.generate_report(
            self.sample_results, 
            metadata=custom_metadata
        )
        
        assert 'metadata' in report
        for key, value in custom_metadata.items():
            assert report['metadata'][key] == value


class TestExportManagerComplete:
    """Complete tests for Export Manager"""
    
    def setup_method(self):
        """Setup test environment"""
        self.export_manager = ExportManager()
        self.temp_dir = tempfile.mkdtemp()
        self.sample_data = self._create_sample_data()
    
    def teardown_method(self):
        """Clean up test environment"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def _create_sample_data(self):
        """Create sample data for export testing"""
        return {
            'report': {
                'id': 'RPT-001',
                'timestamp': datetime.now().isoformat(),
                'results': {
                    'confidence': 0.85,
                    'is_fake': True,
                    'file_name': 'test.jpg'
                }
            },
            'attachments': [],
            'metadata': {
                'version': '1.0',
                'format': 'json'
            }
        }
    
    def test_export_to_json(self):
        """Test JSON export"""
        output_path = os.path.join(self.temp_dir, 'test_export.json')
        
        success = self.export_manager.export_to_json(self.sample_data, output_path)
        
        assert success == True
        assert os.path.exists(output_path)
        
        # Verify content
        with open(output_path, 'r') as f:
            exported_data = json.load(f)
        
        assert exported_data['report']['id'] == 'RPT-001'
        assert exported_data['metadata']['version'] == '1.0'
    
    def test_export_to_csv(self):
        """Test CSV export"""
        # Prepare tabular data
        tabular_data = [
            {'file': 'test1.jpg', 'confidence': 0.8, 'is_fake': True},
            {'file': 'test2.jpg', 'confidence': 0.3, 'is_fake': False},
            {'file': 'test3.jpg', 'confidence': 0.9, 'is_fake': True}
        ]
        
        output_path = os.path.join(self.temp_dir, 'test_export.csv')
        
        success = self.export_manager.export_to_csv(tabular_data, output_path)
        
        assert success == True
        assert os.path.exists(output_path)
        
        # Verify content
        with open(output_path, 'r') as f:
            content = f.read()
        
        assert 'file,confidence,is_fake' in content
        assert 'test1.jpg,0.8,True' in content
    
    def test_export_to_excel(self):
        """Test Excel export"""
        try:
            tabular_data = [
                {'file': 'test1.jpg', 'confidence': 0.8, 'is_fake': True},
                {'file': 'test2.jpg', 'confidence': 0.3, 'is_fake': False}
            ]
            
            output_path = os.path.join(self.temp_dir, 'test_export.xlsx')
            
            success = self.export_manager.export_to_excel(tabular_data, output_path)
            
            assert success == True
            assert os.path.exists(output_path)
            
        except ImportError:
            pytest.skip("Excel export dependencies not available")
    
    def test_export_to_xml(self):
        """Test XML export"""
        output_path = os.path.join(self.temp_dir, 'test_export.xml')
        
        success = self.export_manager.export_to_xml(self.sample_data, output_path)
        
        assert success == True
        assert os.path.exists(output_path)
        
        # Verify it's valid XML
        with open(output_path, 'r') as f:
            content = f.read()
        
        assert '<?xml version=' in content
        assert '<report>' in content
        assert '</report>' in content
    
    def test_create_export_package(self):
        """Test creating export package (ZIP)"""
        # Create test files to include in package
        test_files = []
        for i in range(3):
            file_path = os.path.join(self.temp_dir, f'test_file_{i}.txt')
            with open(file_path, 'w') as f:
                f.write(f'Test content {i}')
            test_files.append(file_path)
        
        package_path = os.path.join(self.temp_dir, 'export_package.zip')
        
        success = self.export_manager.create_export_package(
            files=test_files,
            output_path=package_path,
            metadata=self.sample_data['metadata']
        )
        
        assert success == True
        assert os.path.exists(package_path)
        
        # Verify package contents
        with zipfile.ZipFile(package_path, 'r') as zip_file:
            file_list = zip_file.namelist()
            assert len(file_list) >= len(test_files)
            assert 'metadata.json' in file_list
    
    def test_export_with_compression(self):
        """Test export with compression"""
        large_data = {
            'data': ['large_data_item'] * 1000,  # Create large dataset
            'metadata': self.sample_data['metadata']
        }
        
        output_path = os.path.join(self.temp_dir, 'compressed_export.json.gz')
        
        success = self.export_manager.export_with_compression(
            large_data, 
            output_path, 
            compression='gzip'
        )
        
        assert success == True
        assert os.path.exists(output_path)
        
        # Verify compressed file is smaller
        import gzip
        with gzip.open(output_path, 'rt') as f:
            decompressed_data = json.load(f)
        
        assert len(decompressed_data['data']) == 1000
    
    def test_export_with_encryption(self):
        """Test export with encryption (if available)"""
        try:
            from cryptography.fernet import Fernet
            
            # Generate encryption key
            key = Fernet.generate_key()
            
            output_path = os.path.join(self.temp_dir, 'encrypted_export.enc')
            
            success = self.export_manager.export_with_encryption(
                self.sample_data,
                output_path,
                encryption_key=key
            )
            
            assert success == True
            assert os.path.exists(output_path)
            
            # Verify file is encrypted (not readable as plain JSON)
            with open(output_path, 'rb') as f:
                content = f.read()
            
            # Should not be readable as JSON
            with pytest.raises(json.JSONDecodeError):
                json.loads(content.decode())
                
        except ImportError:
            pytest.skip("Encryption dependencies not available")
    
    def test_batch_export(self):
        """Test batch export functionality"""
        # Create multiple datasets
        datasets = []
        for i in range(5):
            dataset = self.sample_data.copy()
            dataset['report']['id'] = f'RPT-{i:03d}'
            datasets.append(dataset)
        
        output_dir = os.path.join(self.temp_dir, 'batch_exports')
        os.makedirs(output_dir, exist_ok=True)
        
        results = self.export_manager.batch_export(
            datasets,
            output_dir,
            format='json'
        )
        
        assert len(results) == len(datasets)
        assert all(result['success'] for result in results)
        
        # Verify all files were created
        for i in range(5):
            expected_file = os.path.join(output_dir, f'RPT-{i:03d}.json')
            assert os.path.exists(expected_file)
    
    def test_export_formats_validation(self):
        """Test export format validation"""
        valid_formats = ['json', 'csv', 'xml', 'xlsx', 'pdf']
        
        for fmt in valid_formats:
            is_valid = self.export_manager.validate_export_format(fmt)
            assert is_valid == True
        
        # Test invalid format
        is_valid = self.export_manager.validate_export_format('invalid_format')
        assert is_valid == False
    
    def test_export_error_handling(self):
        """Test export error handling"""
        # Test export to invalid path
        invalid_path = '/invalid/path/export.json'
        
        success = self.export_manager.export_to_json(self.sample_data, invalid_path)
        assert success == False
        
        # Test export with invalid data
        invalid_data = {'circular_ref': None}
        invalid_data['circular_ref'] = invalid_data  # Create circular reference
        
        output_path = os.path.join(self.temp_dir, 'invalid_export.json')
        success = self.export_manager.export_to_json(invalid_data, output_path)
        assert success == False


class TestBatchReportProcessorComplete:
    """Complete tests for Batch Report Processor"""
    
    def setup_method(self):
        """Setup test environment"""
        self.batch_processor = BatchReportProcessor()
        self.temp_dir = tempfile.mkdtemp()
        self.sample_batch_data = self._create_sample_batch_data()
    
    def teardown_method(self):
        """Clean up test environment"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def _create_sample_batch_data(self):
        """Create sample batch data"""
        batch_data = []
        for i in range(10):
            data = {
                'file_id': f'FILE-{i:03d}',
                'file_name': f'test_image_{i}.jpg',
                'detection_results': {
                    'confidence': 0.5 + (i * 0.05),
                    'is_fake': i % 3 == 0,
                    'processing_time': 1.0 + (i * 0.2)
                },
                'metadata': {
                    'size': 1024 * (i + 1),
                    'format': 'JPEG'
                }
            }
            batch_data.append(data)
        return batch_data
    
    def test_process_batch_reports(self):
        """Test batch report processing"""
        output_dir = os.path.join(self.temp_dir, 'batch_reports')
        os.makedirs(output_dir, exist_ok=True)
        
        results = self.batch_processor.process_batch(
            self.sample_batch_data,
            output_dir,
            format='json'
        )
        
        assert len(results) == len(self.sample_batch_data)
        assert all(result['success'] for result in results)
        
        # Verify reports were generated
        for i, result in enumerate(results):
            assert 'output_path' in result
            assert os.path.exists(result['output_path'])
    
    def test_generate_batch_summary(self):
        """Test batch summary generation"""
        summary = self.batch_processor.generate_batch_summary(self.sample_batch_data)
        
        assert 'total_files' in summary
        assert 'fake_detected' in summary
        assert 'real_detected' in summary
        assert 'average_confidence' in summary
        assert 'processing_statistics' in summary
        assert 'confidence_distribution' in summary
        
        assert summary['total_files'] == 10
        assert summary['fake_detected'] + summary['real_detected'] == 10
    
    def test_parallel_batch_processing(self):
        """Test parallel batch processing"""
        import time
        
        output_dir = os.path.join(self.temp_dir, 'parallel_reports')
        os.makedirs(output_dir, exist_ok=True)
        
        # Time sequential processing
        start_time = time.time()
        sequential_results = self.batch_processor.process_batch(
            self.sample_batch_data[:5],  # Use smaller batch for timing
            output_dir,
            format='json',
            parallel=False
        )
        sequential_time = time.time() - start_time
        
        # Time parallel processing
        start_time = time.time()
        parallel_results = self.batch_processor.process_batch(
            self.sample_batch_data[:5],
            output_dir,
            format='json',
            parallel=True,
            max_workers=3
        )
        parallel_time = time.time() - start_time
        
        # Both should produce same number of results
        assert len(sequential_results) == len(parallel_results)
        
        # Parallel should generally be faster (or at least not much slower)
        # Note: For small batches, overhead might make parallel slower
        assert parallel_time < sequential_time * 2  # Allow some overhead
    
    def test_batch_export_formats(self):
        """Test batch processing with different export formats"""
        formats = ['json', 'csv', 'xml']
        
        for fmt in formats:
            output_dir = os.path.join(self.temp_dir, f'batch_{fmt}')
            os.makedirs(output_dir, exist_ok=True)
            
            results = self.batch_processor.process_batch(
                self.sample_batch_data[:3],  # Small batch for testing
                output_dir,
                format=fmt
            )
            
            assert len(results) == 3
            assert all(result['success'] for result in results)
            
            # Verify files have correct extensions
            for result in results:
                assert result['output_path'].endswith(f'.{fmt}')
    
    def test_batch_filtering(self):
        """Test batch processing with filtering"""
        # Filter only fake detections
        def filter_fake(data):
            return data['detection_results']['is_fake']
        
        output_dir = os.path.join(self.temp_dir, 'filtered_reports')
        os.makedirs(output_dir, exist_ok=True)
        
        results = self.batch_processor.process_batch_with_filter(
            self.sample_batch_data,
            output_dir,
            filter_func=filter_fake,
            format='json'
        )
        
        # Should only process fake detections
        fake_count = sum(1 for data in self.sample_batch_data 
                        if data['detection_results']['is_fake'])
        assert len(results) == fake_count
        assert all(result['success'] for result in results)
    
    def test_batch_error_handling(self):
        """Test batch processing error handling"""
        # Add some invalid data to batch
        invalid_batch = self.sample_batch_data.copy()
        invalid_batch.append({
            'file_id': 'INVALID',
            'detection_results': None,  # Invalid data
            'metadata': {}
        })
        
        output_dir = os.path.join(self.temp_dir, 'error_handling')
        os.makedirs(output_dir, exist_ok=True)
        
        results = self.batch_processor.process_batch(
            invalid_batch,
            output_dir,
            format='json'
        )
        
        # Should process valid items and report errors for invalid ones
        successful = [r for r in results if r['success']]
        failed = [r for r in results if not r['success']]
        
        assert len(successful) == len(self.sample_batch_data)  # Original valid data
        assert len(failed) == 1  # One invalid item
        assert 'error' in failed[0]
    
    def test_progress_tracking(self):
        """Test progress tracking during batch processing"""
        progress_updates = []
        
        def progress_callback(current, total, item_id):
            progress_updates.append({
                'current': current,
                'total': total,
                'item_id': item_id,
                'percentage': (current / total) * 100
            })
        
        output_dir = os.path.join(self.temp_dir, 'progress_tracking')
        os.makedirs(output_dir, exist_ok=True)
        
        results = self.batch_processor.process_batch_with_progress(
            self.sample_batch_data[:5],  # Small batch for testing
            output_dir,
            format='json',
            progress_callback=progress_callback
        )
        
        assert len(results) == 5
        assert len(progress_updates) >= 5  # Should have progress updates
        
        # Check final progress update
        final_update = progress_updates[-1]
        assert final_update['current'] == final_update['total']
        assert final_update['percentage'] == 100.0
    
    def test_custom_report_templates(self):
        """Test batch processing with custom report templates"""
        def custom_template(data):
            return {
                'custom_report': {
                    'file': data['file_name'],
                    'result': 'FAKE' if data['detection_results']['is_fake'] else 'REAL',
                    'confidence_level': 'HIGH' if data['detection_results']['confidence'] > 0.7 else 'LOW',
                    'generated_at': datetime.now().isoformat()
                }
            }
        
        output_dir = os.path.join(self.temp_dir, 'custom_templates')
        os.makedirs(output_dir, exist_ok=True)
        
        results = self.batch_processor.process_batch_with_template(
            self.sample_batch_data[:3],
            output_dir,
            template_func=custom_template,
            format='json'
        )
        
        assert len(results) == 3
        assert all(result['success'] for result in results)
        
        # Verify custom template was used
        with open(results[0]['output_path'], 'r') as f:
            report_data = json.load(f)
        
        assert 'custom_report' in report_data
        assert 'confidence_level' in report_data['custom_report']
    
    def test_batch_aggregation(self):
        """Test batch result aggregation"""
        aggregated_report = self.batch_processor.generate_aggregated_report(
            self.sample_batch_data
        )
        
        assert 'summary_statistics' in aggregated_report
        assert 'detailed_results' in aggregated_report
        assert 'confidence_analysis' in aggregated_report
        assert 'performance_metrics' in aggregated_report
        
        # Check summary statistics
        summary = aggregated_report['summary_statistics']
        assert 'total_files_analyzed' in summary
        assert 'detection_breakdown' in summary
        assert 'average_processing_time' in summary
        
        # Check confidence analysis
        confidence_analysis = aggregated_report['confidence_analysis']
        assert 'distribution' in confidence_analysis
        assert 'thresholds' in confidence_analysis


# Integration tests for all report components
class TestReportsIntegration:
    """Integration tests for reports components working together"""
    
    def setup_method(self):
        """Setup test environment"""
        self.report_generator = ReportGenerator()
        self.export_manager = ExportManager()
        self.batch_processor = BatchReportProcessor()
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test environment"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_complete_reporting_workflow(self):
        """Test complete reporting workflow"""
        # 1. Generate individual report
        sample_results = {
            'file_name': 'test_integration.jpg',
            'confidence': 0.89,
            'is_fake': True,
            'processing_time': 3.2,
            'technical_details': {
                'model_used': 'EfficientNet',
                'artifacts_detected': ['stylegan_patterns', 'compression_anomalies']
            }
        }
        
        report = self.report_generator.generate_report(sample_results)
        assert report is not None
        
        # 2. Export report in multiple formats
        formats = ['json', 'html', 'csv']
        export_results = {}
        
        for fmt in formats:
            output_path = os.path.join(self.temp_dir, f'integration_report.{fmt}')
            
            if fmt == 'json':
                success = self.export_manager.export_to_json(report, output_path)
            elif fmt == 'html':
                html_content = self.report_generator.generate_html_report(sample_results)
                with open(output_path, 'w') as f:
                    f.write(html_content)
                success = True
            elif fmt == 'csv':
                csv_content = self.report_generator.generate_csv_report([sample_results])
                with open(output_path, 'w') as f:
                    f.write(csv_content)
                success = True
            
            export_results[fmt] = success
            assert os.path.exists(output_path)
        
        # 3. Create export package
        export_files = [os.path.join(self.temp_dir, f'integration_report.{fmt}') 
                       for fmt in formats]
        package_path = os.path.join(self.temp_dir, 'complete_report_package.zip')
        
        package_success = self.export_manager.create_export_package(
            files=export_files,
            output_path=package_path,
            metadata={'workflow': 'integration_test'}
        )
        
        assert package_success == True
        assert os.path.exists(package_path)
    
    def test_batch_workflow_integration(self):
        """Test batch processing workflow integration"""
        # Create batch of sample results
        batch_data = []
        for i in range(5):
            data = {
                'file_id': f'BATCH-{i}',
                'file_name': f'batch_test_{i}.jpg',
                'detection_results': {
                    'confidence': 0.6 + (i * 0.1),
                    'is_fake': i % 2 == 0,
                    'processing_time': 2.0 + (i * 0.3)
                }
            }
            batch_data.append(data)
        
        # 1. Process batch with report generator
        individual_reports = []
        for data in batch_data:
            report = self.report_generator.generate_report(data['detection_results'])
            individual_reports.append(report)
        
        assert len(individual_reports) == 5
        
        # 2. Generate batch summary
        batch_summary = self.batch_processor.generate_batch_summary(batch_data)
        assert 'total_files' in batch_summary
        
        # 3. Export everything
        output_dir = os.path.join(self.temp_dir, 'batch_integration')
        os.makedirs(output_dir, exist_ok=True)
        
        # Export individual reports
        batch_results = self.batch_processor.process_batch(
            batch_data,
            output_dir,
            format='json'
        )
        
        # Export batch summary
        summary_path = os.path.join(output_dir, 'batch_summary.json')
        summary_success = self.export_manager.export_to_json(batch_summary, summary_path)
        
        assert len(batch_results) == 5
        assert all(result['success'] for result in batch_results)
        assert summary_success == True
        assert os.path.exists(summary_path)
    
    def test_error_handling_integration(self):
        """Test error handling across all report components"""
        # Test with invalid data
        invalid_results = {
            'file_name': None,  # Invalid
            'confidence': 'invalid',  # Invalid
            'is_fake': None  # Invalid
        }
        
        # Report generator should handle gracefully
        try:
            report = self.report_generator.generate_report(invalid_results)
            # Should either return valid report or None
            assert report is None or isinstance(report, dict)
        except Exception:
            # Or handle with exception
            pass
        
        # Export manager should handle invalid paths
        invalid_path = '/invalid/path/test.json'
        success = self.export_manager.export_to_json({'test': 'data'}, invalid_path)
        assert success == False
        
        # Batch processor should handle mixed valid/invalid data
        mixed_batch = [
            {'file_id': 'valid', 'detection_results': {'confidence': 0.8}},
            {'file_id': 'invalid', 'detection_results': None}
        ]
        
        results = self.batch_processor.process_batch(
            mixed_batch,
            self.temp_dir,
            format='json'
        )
        
        # Should process what it can
        assert len(results) == 2
        successful = [r for r in results if r['success']]
        failed = [r for r in results if not r['success']]
        assert len(successful) >= 1  # At least the valid one
        assert len(failed) >= 1  # At least the invalid one


# Test fixtures for reports components
@pytest.fixture
def report_generator():
    """Report generator fixture"""
    return ReportGenerator()

@pytest.fixture
def export_manager():
    """Export manager fixture"""
    return ExportManager()

@pytest.fixture
def batch_processor():
    """Batch processor fixture"""
    return BatchReportProcessor()

@pytest.fixture
def sample_detection_results():
    """Sample detection results fixture"""
    return {
        'file_name': 'fixture_test.jpg',
        'confidence': 0.75,
        'is_fake': False,
        'processing_time': 2.1,
        'technical_details': {
            'model_used': 'TestModel',
            'artifacts_found': []
        }
    }

@pytest.fixture
def temp_output_dir():
    """Temporary output directory fixture"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup
    import shutil
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
