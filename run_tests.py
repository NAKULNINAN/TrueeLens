#!/usr/bin/env python3
"""
Comprehensive Test Runner for Media Forensics App
=================================================

This script runs all test suites including unit tests, integration tests,
and performance benchmarks. It also generates test reports and manages
test data.

Usage:
    python run_tests.py [options]
    
Options:
    --unit-only          Run only unit tests
    --integration-only   Run only integration tests  
    --performance-only   Run only performance tests
    --quick             Skip slow tests
    --with-coverage     Generate coverage reports
    --generate-samples  Generate sample test data first
    --clean             Clean previous test results
    --verbose           Enable verbose output
    --html-report       Generate HTML test report
"""

import argparse
import os
import sys
import subprocess
import shutil
from pathlib import Path
import json
import time
from datetime import datetime


class TestRunner:
    """Comprehensive test runner for Media Forensics app"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.tests_dir = self.project_root / "tests"
        self.results_dir = self.project_root / "test_results"
        
    def setup_environment(self):
        """Setup test environment"""
        print("Setting up test environment...")
        
        # Create results directory
        self.results_dir.mkdir(exist_ok=True)
        
        # Create test data directories
        test_data_dir = self.tests_dir / "test_data"
        test_data_dir.mkdir(exist_ok=True)
        (test_data_dir / "images").mkdir(exist_ok=True)
        (test_data_dir / "videos").mkdir(exist_ok=True)
        (test_data_dir / "benchmarks").mkdir(exist_ok=True)
        
    def clean_previous_results(self):
        """Clean previous test results"""
        print("Cleaning previous test results...")
        
        patterns_to_clean = [
            "test_results/**/*",
            "tests/**/__pycache__",
            "tests/**/*.pyc",
            ".coverage",
            "htmlcov/**/*",
            "tests/benchmark_results/**/*"
        ]
        
        for pattern in patterns_to_clean:
            for path in self.project_root.glob(pattern):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    shutil.rmtree(path, ignore_errors=True)
    
    def generate_sample_data(self):
        """Generate sample test data"""
        print("Generating sample test data...")
        
        try:
            subprocess.run([
                sys.executable, 
                str(self.tests_dir / "sample_data_generator.py")
            ], check=True, cwd=self.project_root)
            print("✅ Sample data generated successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to generate sample data: {e}")
            return False
        
        return True
    
    def run_unit_tests(self, verbose=False, with_coverage=False):
        """Run unit tests"""
        print("🧪 Running unit tests...")
        
        cmd = [sys.executable, "-m", "pytest"]
        cmd.extend(["-m", "unit"])
        
        if verbose:
            cmd.append("-v")
        
        if with_coverage:
            cmd.extend([
                "--cov=detection", 
                "--cov=utils", 
                "--cov=database",
                "--cov=models",
                "--cov=reports",
                "--cov-report=html:test_results/coverage_html",
                "--cov-report=json:test_results/coverage.json"
            ])
        
        # Add test results
        cmd.extend([
            "--junit-xml=test_results/unit_tests.xml",
            "--html=test_results/unit_tests.html",
            "--self-contained-html"
        ])
        
        try:
            result = subprocess.run(cmd, cwd=self.project_root)
            success = result.returncode == 0
            print(f"✅ Unit tests {'passed' if success else 'failed'}")
            return success
        except subprocess.CalledProcessError as e:
            print(f"❌ Unit tests failed: {e}")
            return False
    
    def run_integration_tests(self, verbose=False):
        """Run integration tests"""
        print("🔗 Running integration tests...")
        
        cmd = [sys.executable, "-m", "pytest"]
        cmd.extend(["-m", "integration"])
        
        if verbose:
            cmd.append("-v")
        
        cmd.extend([
            "--junit-xml=test_results/integration_tests.xml",
            "--html=test_results/integration_tests.html",
            "--self-contained-html"
        ])
        
        try:
            result = subprocess.run(cmd, cwd=self.project_root)
            success = result.returncode == 0
            print(f"✅ Integration tests {'passed' if success else 'failed'}")
            return success
        except subprocess.CalledProcessError as e:
            print(f"❌ Integration tests failed: {e}")
            return False
    
    def run_performance_tests(self, verbose=False, quick=False):
        """Run performance benchmarks"""
        print("⚡ Running performance benchmarks...")
        
        cmd = [sys.executable, "-m", "pytest"]
        cmd.extend(["-m", "performance"])
        
        if quick:
            cmd.extend(["-m", "not slow"])
        
        if verbose:
            cmd.append("-v")
        
        cmd.extend([
            "--junit-xml=test_results/performance_tests.xml",
            "--html=test_results/performance_tests.html", 
            "--self-contained-html"
        ])
        
        try:
            result = subprocess.run(cmd, cwd=self.project_root)
            success = result.returncode == 0
            print(f"✅ Performance tests {'passed' if success else 'failed'}")
            return success
        except subprocess.CalledProcessError as e:
            print(f"❌ Performance tests failed: {e}")
            return False
    
    def run_all_tests(self, verbose=False, with_coverage=False, quick=False):
        """Run all test suites"""
        print("🚀 Running complete test suite...")
        
        cmd = [sys.executable, "-m", "pytest"]
        cmd.append("tests/")
        
        if quick:
            cmd.extend(["-m", "not slow"])
        
        if verbose:
            cmd.append("-v")
        
        if with_coverage:
            cmd.extend([
                "--cov=detection",
                "--cov=utils", 
                "--cov=database",
                "--cov=models",
                "--cov=reports",
                "--cov-report=html:test_results/coverage_html",
                "--cov-report=json:test_results/coverage.json",
                "--cov-report=term"
            ])
        
        cmd.extend([
            "--junit-xml=test_results/all_tests.xml",
            "--html=test_results/all_tests.html",
            "--self-contained-html"
        ])
        
        try:
            result = subprocess.run(cmd, cwd=self.project_root)
            success = result.returncode == 0
            print(f"✅ All tests {'passed' if success else 'failed'}")
            return success
        except subprocess.CalledProcessError as e:
            print(f"❌ Test suite failed: {e}")
            return False
    
    def generate_summary_report(self, results: dict):
        """Generate summary test report"""
        print("📊 Generating summary report...")
        
        summary = {
            "timestamp": datetime.now().isoformat(),
            "results": results,
            "environment": {
                "python_version": sys.version,
                "platform": sys.platform,
                "working_directory": str(self.project_root)
            }
        }
        
        # Save JSON report
        with open(self.results_dir / "test_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        # Generate HTML summary
        html_content = self._generate_html_summary(summary)
        with open(self.results_dir / "test_summary.html", "w") as f:
            f.write(html_content)
        
        print(f"📄 Summary report saved to {self.results_dir}")
    
    def _generate_html_summary(self, summary: dict) -> str:
        """Generate HTML summary report"""
        results = summary["results"]
        timestamp = summary["timestamp"]
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Media Forensics Test Summary</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background: #333; color: white; padding: 20px; border-radius: 5px; }}
        .result {{ margin: 10px 0; padding: 15px; border-radius: 5px; }}
        .passed {{ background: #d4edda; border-left: 5px solid #28a745; }}
        .failed {{ background: #f8d7da; border-left: 5px solid #dc3545; }}
        .info {{ background: #d1ecf1; border-left: 5px solid #17a2b8; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔍 Media Forensics Test Suite Report</h1>
        <p>Generated on: {timestamp}</p>
    </div>
    
    <h2>Test Results Overview</h2>
    <table>
        <tr><th>Test Suite</th><th>Status</th><th>Details</th></tr>
"""
        
        for test_name, status in results.items():
            status_class = "passed" if status else "failed"
            status_text = "✅ PASSED" if status else "❌ FAILED"
            html += f"<tr><td>{test_name.replace('_', ' ').title()}</td><td class='{status_class}'>{status_text}</td><td>See detailed reports</td></tr>"
        
        html += """
    </table>
    
    <h2>Available Reports</h2>
    <ul>
        <li><a href="unit_tests.html">Unit Tests Report</a></li>
        <li><a href="integration_tests.html">Integration Tests Report</a></li>
        <li><a href="performance_tests.html">Performance Tests Report</a></li>
        <li><a href="coverage_html/index.html">Code Coverage Report</a></li>
    </ul>
</body>
</html>
"""
        return html


def main():
    """Main test runner function"""
    parser = argparse.ArgumentParser(
        description="Comprehensive test runner for Media Forensics app"
    )
    
    # Test selection options
    parser.add_argument("--unit-only", action="store_true", help="Run only unit tests")
    parser.add_argument("--integration-only", action="store_true", help="Run only integration tests")
    parser.add_argument("--performance-only", action="store_true", help="Run only performance tests")
    
    # Test configuration options  
    parser.add_argument("--quick", action="store_true", help="Skip slow tests")
    parser.add_argument("--with-coverage", action="store_true", help="Generate coverage reports")
    parser.add_argument("--generate-samples", action="store_true", help="Generate sample test data first")
    parser.add_argument("--clean", action="store_true", help="Clean previous test results")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument("--html-report", action="store_true", help="Generate HTML test report")
    
    args = parser.parse_args()
    
    # Initialize test runner
    runner = TestRunner()
    
    print("🔍 Media Forensics Test Suite")
    print("=" * 50)
    
    # Setup environment
    runner.setup_environment()
    
    # Clean previous results if requested
    if args.clean:
        runner.clean_previous_results()
    
    # Generate sample data if requested
    if args.generate_samples:
        if not runner.generate_sample_data():
            sys.exit(1)
    
    # Track test results
    results = {}
    overall_success = True
    
    try:
        # Run specific test suites based on arguments
        if args.unit_only:
            results["unit_tests"] = runner.run_unit_tests(args.verbose, args.with_coverage)
            
        elif args.integration_only:
            results["integration_tests"] = runner.run_integration_tests(args.verbose)
            
        elif args.performance_only:
            results["performance_tests"] = runner.run_performance_tests(args.verbose, args.quick)
            
        else:
            # Run all tests
            results["unit_tests"] = runner.run_unit_tests(args.verbose, args.with_coverage)
            results["integration_tests"] = runner.run_integration_tests(args.verbose)
            results["performance_tests"] = runner.run_performance_tests(args.verbose, args.quick)
        
        # Check overall success
        overall_success = all(results.values())
        
        # Generate summary report
        if args.html_report or not any([args.unit_only, args.integration_only, args.performance_only]):
            runner.generate_summary_report(results)
        
        # Print final summary
        print("\n" + "=" * 50)
        print("📋 TEST SUITE SUMMARY")
        print("=" * 50)
        
        for test_name, success in results.items():
            status = "✅ PASSED" if success else "❌ FAILED"
            print(f"{test_name.replace('_', ' ').title()}: {status}")
        
        print(f"\nOverall Status: {'✅ ALL TESTS PASSED' if overall_success else '❌ SOME TESTS FAILED'}")
        print(f"Test results saved to: {runner.results_dir}")
        
        if args.with_coverage:
            print("Coverage report: test_results/coverage_html/index.html")
        
    except KeyboardInterrupt:
        print("\n❌ Test execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test execution failed: {e}")
        sys.exit(1)
    
    # Exit with appropriate code
    sys.exit(0 if overall_success else 1)


if __name__ == "__main__":
    main()
