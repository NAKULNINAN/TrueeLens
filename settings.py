import os
import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(dotenv_path=Path('.env'), override=False)


class Config:
    """Configuration management for Media Forensics Application"""
    
    # ===== ENVIRONMENT VARIABLES =====
    
    # Detection thresholds
    CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', '0.5'))
    AI_IMAGE_THRESHOLD = float(os.getenv('AI_IMAGE_THRESHOLD', '0.6'))
    DEEPFAKE_THRESHOLD = float(os.getenv('DEEPFAKE_THRESHOLD', '0.7'))
    DUPLICATE_THRESHOLD = float(os.getenv('DUPLICATE_THRESHOLD', '0.8'))
    
    # UI preferences
    ENABLE_VISUALIZATION = os.getenv('ENABLE_VISUALIZATION', 'True').lower() == 'true'
    SAVE_RESULTS = os.getenv('SAVE_RESULTS', 'False').lower() == 'true'
    ENABLE_WEBCAM = os.getenv('ENABLE_WEBCAM', 'True').lower() == 'true'
    UI_THEME = os.getenv('UI_THEME', 'light')
    SHOW_TECHNICAL_DETAILS = os.getenv('SHOW_TECHNICAL_DETAILS', 'True').lower() == 'true'
    
# ===== DEBUGGING UTILITIES =====
    @staticmethod
    def log_system_status():
        try:
            import psutil
            memory = psutil.virtual_memory()
            logging.info(f"Memory: {memory.percent}% used")
            cpu = psutil.cpu_percent(interval=1)
            logging.info(f"CPU: {cpu}% used")
        except ImportError:
            logging.warning("psutil not available for detailed system status.")

    # ===== MODEL PATHS =====
    
    # Base model directory
    MODEL_BASE_DIR = os.getenv('MODEL_BASE_DIR', 'models')
    
    # Deepfake detection models
    XCEPTION_MODEL_PATH = os.getenv('XCEPTION_MODEL_PATH', 
                                  os.path.join(MODEL_BASE_DIR, 'xception_net.pth'))
    MESO_MODEL_PATH = os.getenv('MESO_MODEL_PATH', 
                              os.path.join(MODEL_BASE_DIR, 'meso_net.pth'))
    
    # AI image detection models
    EFFICIENTNET_MODEL_PATH = os.getenv('EFFICIENTNET_MODEL_PATH', 
                                       os.path.join(MODEL_BASE_DIR, 'efficientnet_b0.pth'))
    
    # Face detection models
    MTCNN_MODEL_PATH = os.getenv('MTCNN_MODEL_PATH', 
                               os.path.join(MODEL_BASE_DIR, 'mtcnn'))
    
# ===== LOGGING CONFIGURATION =====
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    @staticmethod
    def configure_logging():
        logging.basicConfig(level=logging.DEBUG if Config.DEBUG_MODE else logging.INFO,
                            format=Config.LOG_FORMAT,
                            handlers=[logging.FileHandler(Config.LOG_FILE_PATH),
                                      logging.StreamHandler()])

    # ===== FILE STORAGE PATHS =====
    
    # User preferences and data
    PREFERENCES_STORAGE_PATH = os.getenv('PREFERENCES_STORAGE_PATH', 'user_preferences.json')
    REPORTS_OUTPUT_DIR = os.getenv('REPORTS_OUTPUT_DIR', 'reports')
    TEMP_UPLOAD_DIR = os.getenv('TEMP_UPLOAD_DIR', 'temp_uploads')
    
    # Database configuration
    DATABASE_PATH = os.getenv('DATABASE_PATH', 'forensics_database.db')
    
    # ===== PROCESSING CONFIGURATION =====
    
    # Performance settings
    MAX_FILE_SIZE_MB = int(os.getenv('MAX_FILE_SIZE_MB', '100'))
    MAX_VIDEO_DURATION_SECONDS = int(os.getenv('MAX_VIDEO_DURATION_SECONDS', '300'))
    BATCH_SIZE = int(os.getenv('BATCH_SIZE', '32'))
    
    # Image processing
    IMAGE_RESIZE_DIMENSIONS = tuple(map(int, os.getenv('IMAGE_RESIZE_DIMENSIONS', '224,224').split(',')))
    VIDEO_FRAME_SAMPLE_RATE = int(os.getenv('VIDEO_FRAME_SAMPLE_RATE', '30'))
    
    # ===== SECURITY CONFIGURATION =====
    
    # File type restrictions
    ALLOWED_IMAGE_EXTENSIONS = os.getenv('ALLOWED_IMAGE_EXTENSIONS', 
                                       'jpg,jpeg,png,bmp,tiff,webp').split(',')
    ALLOWED_VIDEO_EXTENSIONS = os.getenv('ALLOWED_VIDEO_EXTENSIONS', 
                                       'mp4,avi,mov,mkv,wmv,flv').split(',')
    
    # Logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE_PATH = os.getenv('LOG_FILE_PATH', 'forensics_app.log')
    ENABLE_DEBUG_LOGGING = os.getenv('ENABLE_DEBUG_LOGGING', 'False').lower() == 'true'
    
    # ===== ADVANCED FEATURES =====
    
    # Performance and GPU settings
    ENABLE_GPU_ACCELERATION = os.getenv('ENABLE_GPU_ACCELERATION', 'True').lower() == 'true'
    MAX_CONCURRENT_PROCESSES = int(os.getenv('MAX_CONCURRENT_PROCESSES', '4'))
    CACHE_MODEL_PREDICTIONS = os.getenv('CACHE_MODEL_PREDICTIONS', 'True').lower() == 'true'
    AUTO_BACKUP_PREFERENCES = os.getenv('AUTO_BACKUP_PREFERENCES', 'True').lower() == 'true'
    
    # ===== API CONFIGURATION =====
    
    # External API keys (optional)
    GOOGLE_VISION_API_KEY = os.getenv('GOOGLE_VISION_API_KEY')
    AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
    AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
    
    # ===== DEVELOPMENT SETTINGS =====
    
    DEBUG_MODE = os.getenv('DEBUG_MODE', 'False').lower() == 'true'
    DEVELOPMENT_MODE = os.getenv('DEVELOPMENT_MODE', 'False').lower() == 'true'
    ENABLE_PROFILING = os.getenv('ENABLE_PROFILING', 'False').lower() == 'true'
    
    # ===== USER PREFERENCES MANAGEMENT =====
    
    @classmethod
    def load_user_preferences(cls) -> Dict[str, Any]:
        """Load user preferences from storage file"""
        try:
            if os.path.exists(cls.PREFERENCES_STORAGE_PATH):
                with open(cls.PREFERENCES_STORAGE_PATH, 'r') as f:
                    return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading user preferences: {e}")
        
        # Return default preferences if file doesn't exist or is corrupted
        return cls.get_default_preferences()
    
    @classmethod
    def save_user_preferences(cls, preferences: Dict[str, Any]) -> None:
        """Save user preferences to storage file"""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(cls.PREFERENCES_STORAGE_PATH), exist_ok=True)
            
            with open(cls.PREFERENCES_STORAGE_PATH, 'w') as f:
                json.dump(preferences, f, indent=2)
        except Exception as e:
            print(f"Error saving user preferences: {e}")
    
    @classmethod
    def get_default_preferences(cls) -> Dict[str, Any]:
        """Get default user preferences"""
        return {
            "confidence_threshold": cls.CONFIDENCE_THRESHOLD,
            "enable_visualization": cls.ENABLE_VISUALIZATION,
            "save_results": cls.SAVE_RESULTS,
            "detection_mode": "Detect AI-Generated Image",
            "ui_theme": "light",
            "show_technical_details": True,
            "auto_save_reports": False,
            "preferred_models": {
                "deepfake_detection": "xception",
                "ai_image_detection": "efficientnet"
            }
        }
    
    # ===== UTILITY METHODS =====
    
    @classmethod
    def create_required_directories(cls) -> None:
        """Create required directories if they don't exist"""
        directories = [
            cls.MODEL_BASE_DIR,
            cls.REPORTS_OUTPUT_DIR,
            cls.TEMP_UPLOAD_DIR,
            os.path.dirname(cls.PREFERENCES_STORAGE_PATH),
            os.path.dirname(cls.LOG_FILE_PATH)
        ]
        
        for directory in directories:
            if directory:  # Only create if directory path is not empty
                os.makedirs(directory, exist_ok=True)
    
    @classmethod
    def validate_model_paths(cls) -> Dict[str, bool]:
        """Validate that required model files exist"""
        model_paths = {
            "xception": cls.XCEPTION_MODEL_PATH,
            "meso": cls.MESO_MODEL_PATH,
            "efficientnet": cls.EFFICIENTNET_MODEL_PATH,
            "mtcnn": cls.MTCNN_MODEL_PATH
        }
        
        return {name: os.path.exists(path) for name, path in model_paths.items()}
    
    @classmethod
    def get_model_config(cls) -> Dict[str, Any]:
        """Get model configuration dictionary"""
        return {
            "xception": {
                "path": cls.XCEPTION_MODEL_PATH,
                "threshold": cls.DEEPFAKE_THRESHOLD,
                "input_size": (224, 224),
                "preprocessing": "normalize"
            },
            "meso": {
                "path": cls.MESO_MODEL_PATH,
                "threshold": cls.DEEPFAKE_THRESHOLD,
                "input_size": (256, 256),
                "preprocessing": "normalize"
            },
            "efficientnet": {
                "path": cls.EFFICIENTNET_MODEL_PATH,
                "threshold": cls.AI_IMAGE_THRESHOLD,
                "input_size": cls.IMAGE_RESIZE_DIMENSIONS,
                "preprocessing": "imagenet"
            }
        }
    
    @classmethod
    def get_advanced_config(cls) -> Dict[str, Any]:
        """Get advanced configuration settings"""
        return {
            "gpu_acceleration": cls.ENABLE_GPU_ACCELERATION,
            "max_concurrent_processes": cls.MAX_CONCURRENT_PROCESSES,
            "cache_predictions": cls.CACHE_MODEL_PREDICTIONS,
            "auto_backup": cls.AUTO_BACKUP_PREFERENCES,
            "debug_logging": cls.ENABLE_DEBUG_LOGGING,
            "development_mode": cls.DEVELOPMENT_MODE,
            "profiling_enabled": cls.ENABLE_PROFILING
        }
    
    @classmethod
    def get_api_config(cls) -> Dict[str, Optional[str]]:
        """Get API configuration settings"""
        return {
            "google_vision_api": cls.GOOGLE_VISION_API_KEY,
            "aws_access_key": cls.AWS_ACCESS_KEY_ID,
            "aws_secret_key": cls.AWS_SECRET_ACCESS_KEY
        }
    
    @classmethod
    def backup_user_preferences(cls) -> bool:
        """Backup user preferences to a timestamped file"""
        if not cls.AUTO_BACKUP_PREFERENCES:
            return False
            
        try:
            from datetime import datetime
            current_prefs = cls.load_user_preferences()
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_path = f"{cls.PREFERENCES_STORAGE_PATH}.backup_{timestamp}"
            
            with open(backup_path, 'w') as f:
                json.dump(current_prefs, f, indent=2)
            return True
        except Exception as e:
            print(f"Error backing up preferences: {e}")
            return False
    
    @classmethod
    def update_default_preferences(cls, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update default preferences with new values"""
        defaults = cls.get_default_preferences()
        defaults.update(updates)
        return defaults
    
    @classmethod
    def validate_file_size(cls, file_size_bytes: int) -> bool:
        """Validate if file size is within allowed limits"""
        max_size_bytes = cls.MAX_FILE_SIZE_MB * 1024 * 1024
        return file_size_bytes <= max_size_bytes
    
    @classmethod
    def validate_video_duration(cls, duration_seconds: int) -> bool:
        """Validate if video duration is within allowed limits"""
        return duration_seconds <= cls.MAX_VIDEO_DURATION_SECONDS
    
    @classmethod
    def is_allowed_file_extension(cls, filename: str, file_type: str = 'image') -> bool:
        """Check if file extension is allowed"""
        extension = filename.lower().split('.')[-1] if '.' in filename else ''
        
        if file_type.lower() == 'image':
            return extension in cls.ALLOWED_IMAGE_EXTENSIONS
        elif file_type.lower() == 'video':
            return extension in cls.ALLOWED_VIDEO_EXTENSIONS
        return False
    
    @classmethod
    def print_config_summary(cls) -> None:
        """Print configuration summary for debugging"""
        print("=== Media Forensics App Configuration ===\n")
        
        print("Detection Thresholds:")
        print(f"  - General Confidence: {cls.CONFIDENCE_THRESHOLD}")
        print(f"  - AI Image Detection: {cls.AI_IMAGE_THRESHOLD}")
        print(f"  - Deepfake Detection: {cls.DEEPFAKE_THRESHOLD}")
        print(f"  - Duplicate Detection: {cls.DUPLICATE_THRESHOLD}\n")
        
        print("Model Paths:")
        for name, path in cls.get_model_config().items():
            exists = "✓" if os.path.exists(path["path"]) else "✗"
            print(f"  - {name.capitalize()}: {path['path']} {exists}")
        
        print(f"\nStorage Paths:")
        print(f"  - Preferences: {cls.PREFERENCES_STORAGE_PATH}")
        print(f"  - Reports: {cls.REPORTS_OUTPUT_DIR}")
        print(f"  - Temp Uploads: {cls.TEMP_UPLOAD_DIR}")
        print(f"  - Database: {cls.DATABASE_PATH}")
        print(f"  - Logs: {cls.LOG_FILE_PATH}")
        
        print(f"\nAdvanced Features:")
        advanced = cls.get_advanced_config()
        for key, value in advanced.items():
            print(f"  - {key.replace('_', ' ').title()}: {value}")
        
        print(f"\nFile Limits:")
        print(f"  - Max File Size: {cls.MAX_FILE_SIZE_MB}MB")
        print(f"  - Max Video Duration: {cls.MAX_VIDEO_DURATION_SECONDS}s")
        print(f"  - Batch Size: {cls.BATCH_SIZE}")
        print(f"  - Image Resize: {cls.IMAGE_RESIZE_DIMENSIONS}")


# Initialize configuration on import
Config.create_required_directories()
