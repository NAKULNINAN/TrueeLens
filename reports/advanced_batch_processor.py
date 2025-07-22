import os
import asyncio
import concurrent.futures
from pathlib import Path
from typing import List, Dict, Any, Callable, Optional
import logging
from datetime import datetime
import time
import json
import pandas as pd
from dataclasses import dataclass
import threading
from queue import Queue

# Import your existing detection modules
from detection.ai_image_detector import AIImageDetector
from detection.deepfake_detector import DeepfakeDetector
from detection.duplicate_detector import DuplicateDetector
from utils.media_processor import MediaProcessor

@dataclass
class BatchJob:
    """Represents a batch processing job"""
    job_id: str
    files: List[str]
    detection_mode: str
    threshold: float
    output_dir: str
    status: str = "pending"
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    results: List[Dict[str, Any]] = None
    error: Optional[str] = None

class AdvancedBatchProcessor:
    """Advanced batch processor for media forensics analysis"""
    
    def __init__(self, max_workers: int = 4):
        self.logger = logging.getLogger(__name__)
        self.max_workers = max_workers
        self.jobs = {}
        self.job_queue = Queue()
        
        # Initialize detectors
        self.ai_detector = AIImageDetector()
        self.deepfake_detector = DeepfakeDetector()
        self.duplicate_detector = DuplicateDetector()
        self.media_processor = MediaProcessor()
        
        # Start worker threads
        self.workers = []
        self.shutdown_flag = threading.Event()
        self._start_workers()
    
    def _start_workers(self):
        """Start worker threads for processing jobs"""
        for i in range(self.max_workers):
            worker = threading.Thread(target=self._worker_loop, name=f"BatchWorker-{i}")
            worker.daemon = True
            worker.start()
            self.workers.append(worker)
    
    def _worker_loop(self):
        """Worker thread main loop"""
        while not self.shutdown_flag.is_set():
            try:
                job = self.job_queue.get(timeout=1)
                if job is None:
                    break
                self._process_job(job)
                self.job_queue.task_done()
            except:
                continue
    
    def submit_batch_job(self, files: List[str], detection_mode: str, 
                        threshold: float = 0.5, output_dir: str = None) -> str:
        """Submit a batch processing job"""
        job_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.jobs)}"
        
        if output_dir is None:
            output_dir = os.path.join(os.getcwd(), "batch_reports", job_id)
        
        os.makedirs(output_dir, exist_ok=True)
        
        job = BatchJob(
            job_id=job_id,
            files=files,
            detection_mode=detection_mode,
            threshold=threshold,
            output_dir=output_dir
        )
        
        self.jobs[job_id] = job
        self.job_queue.put(job)
        
        self.logger.info(f"Submitted batch job {job_id} with {len(files)} files")
        return job_id
    
    def _process_job(self, job: BatchJob):
        """Process a single batch job"""
        try:
            job.status = "running"
            job.start_time = datetime.now()
            job.results = []
            
            self.logger.info(f"Starting batch job {job.job_id}")
            
            for i, file_path in enumerate(job.files):
                try:
                    self.logger.info(f"Processing file {i+1}/{len(job.files)}: {file_path}")
                    
                    # Process single file
                    result = self._process_single_file(file_path, job.detection_mode, job.threshold)
                    result["batch_info"] = {
                        "job_id": job.job_id,
                        "file_index": i + 1,
                        "total_files": len(job.files),
                        "file_path": file_path
                    }
                    job.results.append(result)
                    
                    # Save individual result
                    self._save_individual_result(result, job.output_dir, i + 1)
                    
                except Exception as e:
                    error_result = {
                        "error": str(e),
                        "file_path": file_path,
                        "status": "failed",
                        "timestamp": datetime.now().isoformat(),
                        "batch_info": {
                            "job_id": job.job_id,
                            "file_index": i + 1,
                            "total_files": len(job.files),
                            "file_path": file_path
                        }
                    }
                    job.results.append(error_result)
                    self.logger.error(f"Error processing {file_path}: {str(e)}")
            
            job.status = "completed"
            job.end_time = datetime.now()
            
            # Generate batch summary report
            self._generate_batch_summary(job)
            
            self.logger.info(f"Completed batch job {job.job_id}")
            
        except Exception as e:
            job.status = "failed"
            job.error = str(e)
            job.end_time = datetime.now()
            self.logger.error(f"Batch job {job.job_id} failed: {str(e)}")
    
    def _process_single_file(self, file_path: str, detection_mode: str, threshold: float) -> Dict[str, Any]:
        """Process a single file based on detection mode"""
        start_time = time.time()
        
        # Create file info
        file_info = {
            "name": os.path.basename(file_path),
            "path": file_path,
            "size": os.path.getsize(file_path),
            "modified": datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()
        }
        
        result = {
            "timestamp": datetime.now().isoformat(),
            "file_info": file_info,
            "detection_mode": detection_mode,
            "threshold": threshold
        }
        
        try:
            # Load file data
            with open(file_path, 'rb') as f:
                file_data = f.read()
            
            # Create a mock uploaded file object
            class MockFile:
                def __init__(self, data, name):
                    self.data = data
                    self.name = name
                    self.size = len(data)
                    self.type = self._get_mime_type(name)
                
                def read(self):
                    return self.data
                
                def _get_mime_type(self, filename):
                    ext = os.path.splitext(filename)[1].lower()
                    if ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                        return f"image/{ext[1:]}"
                    elif ext in ['.mp4', '.avi', '.mov']:
                        return f"video/{ext[1:]}"
                    return "application/octet-stream"
            
            mock_file = MockFile(file_data, os.path.basename(file_path))
            
            # Process based on detection mode
            if detection_mode == "Detect AI-Generated Image":
                detection_result = self.ai_detector.detect(mock_file, threshold, enable_visualization=False)
                result.update(detection_result)
                
            elif detection_mode == "Detect Deepfake Video":
                detection_result = self.deepfake_detector.detect(mock_file, threshold, enable_visualization=False)
                result.update(detection_result)
                
            elif detection_mode == "Detect Duplicate Image/Video":
                detection_result = self.duplicate_detector.detect(mock_file, threshold, enable_visualization=False)
                result.update(detection_result)
            
            result["processing_time"] = time.time() - start_time
            result["status"] = "completed"
            
        except Exception as e:
            result["error"] = str(e)
            result["status"] = "failed"
            result["processing_time"] = time.time() - start_time
        
        return result
    
    def _save_individual_result(self, result: Dict[str, Any], output_dir: str, file_index: int):
        """Save individual file result"""
        filename = f"result_{file_index:04d}.json"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, default=str)
    
    def _generate_batch_summary(self, job: BatchJob):
        """Generate comprehensive batch summary"""
        summary = {
            "job_info": {
                "job_id": job.job_id,
                "start_time": job.start_time.isoformat(),
                "end_time": job.end_time.isoformat(),
                "duration": (job.end_time - job.start_time).total_seconds(),
                "detection_mode": job.detection_mode,
                "threshold": job.threshold,
                "total_files": len(job.files),
                "status": job.status
            },
            "statistics": self._calculate_batch_statistics(job.results),
            "detailed_results": job.results
        }
        
        # Save summary
        summary_path = os.path.join(job.output_dir, "batch_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Generate Excel report
        self._generate_batch_excel_report(summary, job.output_dir)
        
        self.logger.info(f"Generated batch summary for job {job.job_id}")
    
    def _calculate_batch_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate comprehensive batch statistics"""
        stats = {
            "total_processed": len(results),
            "successful": 0,
            "failed": 0,
            "fake_detected": 0,
            "duplicate_detected": 0,
            "authentic": 0,
            "confidence_scores": [],
            "processing_times": [],
            "error_types": {}
        }
        
        for result in results:
            if result.get("error"):
                stats["failed"] += 1
                error_type = type(Exception(result["error"])).__name__
                stats["error_types"][error_type] = stats["error_types"].get(error_type, 0) + 1
            else:
                stats["successful"] += 1
                
                if result.get("is_fake", False):
                    stats["fake_detected"] += 1
                elif result.get("is_duplicate", False):
                    stats["duplicate_detected"] += 1
                else:
                    stats["authentic"] += 1
                
                if "confidence" in result:
                    stats["confidence_scores"].append(result["confidence"])
                
                if "processing_time" in result:
                    stats["processing_times"].append(result["processing_time"])
        
        # Calculate aggregated metrics
        if stats["confidence_scores"]:
            import numpy as np
            scores = np.array(stats["confidence_scores"])
            stats["confidence_stats"] = {
                "mean": float(np.mean(scores)),
                "median": float(np.median(scores)),
                "std": float(np.std(scores)),
                "min": float(np.min(scores)),
                "max": float(np.max(scores))
            }
        
        if stats["processing_times"]:
            times = np.array(stats["processing_times"])
            stats["timing_stats"] = {
                "total_time": float(np.sum(times)),
                "average_time": float(np.mean(times)),
                "median_time": float(np.median(times)),
                "fastest": float(np.min(times)),
                "slowest": float(np.max(times))
            }
        
        return stats
    
    def _generate_batch_excel_report(self, summary: Dict[str, Any], output_dir: str):
        """Generate Excel report for batch results"""
        excel_path = os.path.join(output_dir, "batch_report.xlsx")
        
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Job info sheet
            job_df = pd.DataFrame([summary["job_info"]])
            job_df.to_excel(writer, sheet_name='Job_Info', index=False)
            
            # Statistics sheet
            stats_df = pd.json_normalize(summary["statistics"])
            stats_df.to_excel(writer, sheet_name='Statistics', index=False)
            
            # Results summary sheet
            results_summary = []
            for result in summary["detailed_results"]:
                summary_row = {
                    "file_name": result.get("file_info", {}).get("name", "unknown"),
                    "status": result.get("status", "unknown"),
                    "is_fake": result.get("is_fake", False),
                    "is_duplicate": result.get("is_duplicate", False),
                    "confidence": result.get("confidence", 0),
                    "processing_time": result.get("processing_time", 0),
                    "error": result.get("error", "")
                }
                results_summary.append(summary_row)
            
            results_df = pd.DataFrame(results_summary)
            results_df.to_excel(writer, sheet_name='Results_Summary', index=False)
        
        self.logger.info(f"Generated Excel report: {excel_path}")
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get status of a batch job"""
        if job_id not in self.jobs:
            return {"error": "Job not found"}
        
        job = self.jobs[job_id]
        status = {
            "job_id": job.job_id,
            "status": job.status,
            "total_files": len(job.files),
            "detection_mode": job.detection_mode
        }
        
        if job.start_time:
            status["start_time"] = job.start_time.isoformat()
        
        if job.end_time:
            status["end_time"] = job.end_time.isoformat()
            status["duration"] = (job.end_time - job.start_time).total_seconds()
        
        if job.results:
            status["completed_files"] = len([r for r in job.results if not r.get("error")])
            status["failed_files"] = len([r for r in job.results if r.get("error")])
        
        if job.error:
            status["error"] = job.error
        
        return status
    
    def list_jobs(self) -> List[Dict[str, Any]]:
        """List all batch jobs"""
        return [self.get_job_status(job_id) for job_id in self.jobs.keys()]
    
    def get_job_results(self, job_id: str) -> Dict[str, Any]:
        """Get complete results for a job"""
        if job_id not in self.jobs:
            return {"error": "Job not found"}
        
        job = self.jobs[job_id]
        if job.status != "completed":
            return {"error": "Job not completed yet"}
        
        return {
            "job_info": {
                "job_id": job.job_id,
                "status": job.status,
                "start_time": job.start_time.isoformat(),
                "end_time": job.end_time.isoformat(),
                "duration": (job.end_time - job.start_time).total_seconds()
            },
            "results": job.results,
            "output_dir": job.output_dir
        }
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a pending job"""
        if job_id not in self.jobs:
            return False
        
        job = self.jobs[job_id]
        if job.status == "pending":
            job.status = "cancelled"
            return True
        
        return False
    
    def shutdown(self):
        """Shutdown the batch processor"""
        self.shutdown_flag.set()
        
        # Add None to queue to wake up workers
        for _ in range(self.max_workers):
            self.job_queue.put(None)
        
        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=5)
        
        self.logger.info("Batch processor shutdown complete")
