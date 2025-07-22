#!/usr/bin/env python3
"""
Configuration Management Test Script
Tests all aspects of the configuration management system
"""

import os
import sys
import json
from pathlib import Path

# Add current directory to path to import local modules
sys.path.insert(0, os.path.dirname(__file__))

from settings import Config
from config_manager import config_manager

def test_environment_variables():
    """Test environment variable loading and defaults"""
    print("=== Testing Environment Variables ===")
    
    # Test detection thresholds
    print(f"Confidence Threshold: {Config.CONFIDENCE_THRESHOLD}")
    print(f"AI Image Threshold: {Config.AI_IMAGE_THRESHOLD}")
    print(f"Deepfake Threshold: {Config.DEEPFAKE_THRESHOLD}")
    print(f"Duplicate Threshold: {Config.DUPLICATE_THRESHOLD}")
    
    # Test UI preferences
    print(f"Enable Visualization: {Config.ENABLE_VISUALIZATION}")
    print(f"Save Results: {Config.SAVE_RESULTS}")
    print(f"Enable Webcam: {Config.ENABLE_WEBCAM}")
    print(f"UI Theme: {Config.UI_THEME}")
    print(f"Show Technical Details: {Config.SHOW_TECHNICAL_DETAILS}")
    
    # Test advanced features
    print(f"GPU Acceleration: {Config.ENABLE_GPU_ACCELERATION}")
    print(f"Max Concurrent Processes: {Config.MAX_CONCURRENT_PROCESSES}")
    print(f"Cache Predictions: {Config.CACHE_MODEL_PREDICTIONS}")
    print(f"Auto Backup: {Config.AUTO_BACKUP_PREFERENCES}")
    print(f"Debug Logging: {Config.ENABLE_DEBUG_LOGGING}")
    print(f"Development Mode: {Config.DEVELOPMENT_MODE}")
    print(f"Profiling Enabled: {Config.ENABLE_PROFILING}")

def test_model_paths():
    """Test model path configuration"""
    print("\n=== Testing Model Paths ===")
    
    print(f"Model Base Directory: {Config.MODEL_BASE_DIR}")
    print(f"Xception Model: {Config.XCEPTION_MODEL_PATH}")
    print(f"Meso Model: {Config.MESO_MODEL_PATH}")
    print(f"EfficientNet Model: {Config.EFFICIENTNET_MODEL_PATH}")
    print(f"MTCNN Model: {Config.MTCNN_MODEL_PATH}")
    
    # Test model validation
    model_status = Config.validate_model_paths()
    print(f"Model Files Status: {model_status}")
    
    # Test model configuration
    model_config = Config.get_model_config()
    for name, config in model_config.items():
        print(f"{name.capitalize()} Config: {config}")

def test_file_storage():
    """Test file storage configuration"""
    print("\n=== Testing File Storage Configuration ===")
    
    print(f"Preferences Storage: {Config.PREFERENCES_STORAGE_PATH}")
    print(f"Reports Directory: {Config.REPORTS_OUTPUT_DIR}")
    print(f"Temp Upload Directory: {Config.TEMP_UPLOAD_DIR}")
    print(f"Database Path: {Config.DATABASE_PATH}")
    print(f"Log File Path: {Config.LOG_FILE_PATH}")

def test_processing_limits():
    """Test processing configuration limits"""
    print("\n=== Testing Processing Configuration ===")
    
    print(f"Max File Size (MB): {Config.MAX_FILE_SIZE_MB}")
    print(f"Max Video Duration (s): {Config.MAX_VIDEO_DURATION_SECONDS}")
    print(f"Batch Size: {Config.BATCH_SIZE}")
    print(f"Image Resize Dimensions: {Config.IMAGE_RESIZE_DIMENSIONS}")
    print(f"Video Frame Sample Rate: {Config.VIDEO_FRAME_SAMPLE_RATE}")
    
    # Test file extension validation
    print(f"Allowed Image Extensions: {Config.ALLOWED_IMAGE_EXTENSIONS}")
    print(f"Allowed Video Extensions: {Config.ALLOWED_VIDEO_EXTENSIONS}")
    
    # Test validation methods
    test_files = [
        ("test.jpg", "image"),
        ("test.mp4", "video"),
        ("test.txt", "image"),
        ("video.avi", "video")
    ]
    
    for filename, file_type in test_files:
        is_allowed = Config.is_allowed_file_extension(filename, file_type)
        print(f"File '{filename}' ({file_type}): {'Allowed' if is_allowed else 'Not Allowed'}")

def test_user_preferences():
    """Test user preferences management"""
    print("\n=== Testing User Preferences Management ===")
    
    # Load default preferences
    default_prefs = Config.get_default_preferences()
    print(f"Default Preferences: {json.dumps(default_prefs, indent=2)}")
    
    # Test loading preferences
    current_prefs = config_manager.load_preferences()
    print(f"Current Preferences Loaded: {len(current_prefs)} items")
    
    # Test setting a preference
    test_key = "test_setting"
    test_value = "test_value"
    success = config_manager.set_preference(test_key, test_value)
    print(f"Set preference '{test_key}': {'Success' if success else 'Failed'}")
    
    # Test getting the preference
    retrieved_value = config_manager.get_preference(test_key)
    print(f"Retrieved preference '{test_key}': {retrieved_value}")
    
    # Test nested preference
    nested_key = "nested.test.value"
    nested_value = 42
    success = config_manager.set_preference(nested_key, nested_value)
    print(f"Set nested preference '{nested_key}': {'Success' if success else 'Failed'}")
    
    retrieved_nested = config_manager.get_preference(nested_key)
    print(f"Retrieved nested preference: {retrieved_nested}")

def test_threshold_management():
    """Test threshold configuration management"""
    print("\n=== Testing Threshold Management ===")
    
    # Get current thresholds
    current_thresholds = config_manager.get_all_thresholds()
    print(f"Current Thresholds: {current_thresholds}")
    
    # Test updating thresholds
    test_threshold_updates = [
        ("general", 0.75),
        ("ai_image", 0.85),
        ("deepfake", 0.9),
        ("duplicate", 0.95)
    ]
    
    for threshold_type, value in test_threshold_updates:
        success = config_manager.update_threshold(threshold_type, value)
        print(f"Update {threshold_type} threshold to {value}: {'Success' if success else 'Failed'}")
    
    # Get updated thresholds
    updated_thresholds = config_manager.get_all_thresholds()
    print(f"Updated Thresholds: {updated_thresholds}")

def test_model_preferences():
    """Test model preference management"""
    print("\n=== Testing Model Preference Management ===")
    
    # Test getting model preferences
    deepfake_model = config_manager.get_model_preference('deepfake_detection')
    ai_image_model = config_manager.get_model_preference('ai_image_detection')
    
    print(f"Current Deepfake Model: {deepfake_model}")
    print(f"Current AI Image Model: {ai_image_model}")
    
    # Test setting model preferences
    success1 = config_manager.set_model_preference('deepfake_detection', 'meso')
    success2 = config_manager.set_model_preference('ai_image_detection', 'efficientnet')
    
    print(f"Set deepfake model to 'meso': {'Success' if success1 else 'Failed'}")
    print(f"Set AI image model to 'efficientnet': {'Success' if success2 else 'Failed'}")

def test_file_validation():
    """Test file upload validation"""
    print("\n=== Testing File Validation ===")
    
    # Test file validation scenarios
    test_scenarios = [
        ("valid_image.jpg", 5 * 1024 * 1024, "image"),  # 5MB image
        ("large_video.mp4", 150 * 1024 * 1024, "video"),  # 150MB video (over limit)
        ("invalid.txt", 1024, "image"),  # Invalid extension
        ("warning_size.png", 85 * 1024 * 1024, "image"),  # Large but within limits
    ]
    
    for filename, file_size, file_type in test_scenarios:
        result = config_manager.validate_file_upload(filename, file_size, file_type)
        print(f"File: {filename} ({file_size / (1024*1024):.1f}MB, {file_type})")
        print(f"  Valid: {result['valid']}")
        if result['errors']:
            print(f"  Errors: {result['errors']}")
        if result['warnings']:
            print(f"  Warnings: {result['warnings']}")

def test_advanced_settings():
    """Test advanced settings management"""
    print("\n=== Testing Advanced Settings ===")
    
    # Get advanced settings
    advanced_settings = config_manager.get_advanced_settings()
    print(f"Advanced Settings: {json.dumps(advanced_settings, indent=2)}")
    
    # Get processing limits
    processing_limits = config_manager.get_processing_limits()
    print(f"Processing Limits: {json.dumps(processing_limits, indent=2)}")
    
    # Test updating advanced settings
    test_updates = [
        ("ui_theme", "dark"),
        ("gpu_acceleration", False),
        ("debug_mode", True)
    ]
    
    for setting_name, value in test_updates:
        success = config_manager.update_advanced_setting(setting_name, value)
        print(f"Update {setting_name} to {value}: {'Success' if success else 'Failed'}")

def test_configuration_export_import():
    """Test configuration export and import"""
    print("\n=== Testing Configuration Export/Import ===")
    
    # Export configuration
    config_export = config_manager.export_config(include_system=True)
    print(f"Exported config keys: {list(config_export.keys())}")
    
    # Create config report
    config_report = config_manager.create_config_report()
    print(f"Config report sections: {list(config_report.keys())}")
    
    # Test backup functionality
    backup_success = config_manager.backup_preferences_if_enabled()
    print(f"Preferences backup: {'Success' if backup_success else 'Skipped/Failed'}")

def test_api_configuration():
    """Test API configuration settings"""
    print("\n=== Testing API Configuration ===")
    
    # Get API configuration
    api_config = Config.get_api_config()
    print("API Configuration:")
    for key, value in api_config.items():
        display_value = "***SET***" if value else "Not Set"
        print(f"  {key}: {display_value}")

def test_config_status():
    """Test configuration status and health check"""
    print("\n=== Testing Configuration Status ===")
    
    # Get configuration status
    status = config_manager.get_config_status()
    print("Configuration Status:")
    for key, value in status.items():
        print(f"  {key}: {value}")

def test_directory_creation():
    """Test directory creation functionality"""
    print("\n=== Testing Directory Creation ===")
    
    # Test directory creation
    Config.create_required_directories()
    
    required_dirs = [
        Config.MODEL_BASE_DIR,
        Config.REPORTS_OUTPUT_DIR,
        Config.TEMP_UPLOAD_DIR,
    ]
    
    for directory in required_dirs:
        exists = os.path.exists(directory)
        print(f"Directory '{directory}': {'Exists' if exists else 'Missing'}")

def main():
    """Run all configuration tests"""
    print("Media Forensics App - Configuration Management Test")
    print("=" * 60)
    
    try:
        test_environment_variables()
        test_model_paths()
        test_file_storage()
        test_processing_limits()
        test_user_preferences()
        test_threshold_management()
        test_model_preferences()
        test_file_validation()
        test_advanced_settings()
        test_configuration_export_import()
        test_api_configuration()
        test_config_status()
        test_directory_creation()
        
        print("\n" + "=" * 60)
        print("Configuration Management Test Complete!")
        
        # Print final configuration summary
        print("\n=== Final Configuration Summary ===")
        Config.print_config_summary()
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
