import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from settings import Config


class ConfigManager:
    """Dynamic configuration manager for user preferences and runtime settings"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._preferences_cache: Optional[Dict[str, Any]] = None
        self._last_loaded: Optional[datetime] = None
        
        # Set up logging
        self._setup_logging()
        
        # Load initial preferences
        self.load_preferences()
    
    def _setup_logging(self) -> None:
        """Setup logging configuration"""
        logging.basicConfig(
            level=getattr(logging, Config.LOG_LEVEL),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(Config.LOG_FILE_PATH),
                logging.StreamHandler()
            ]
        )
    
    def load_preferences(self, force_reload: bool = False) -> Dict[str, Any]:
        """
        Load user preferences from storage
        
        Args:
            force_reload: Force reload from disk even if cached
            
        Returns:
            Dictionary of user preferences
        """
        # Check if we need to reload
        if not force_reload and self._preferences_cache is not None:
            # Check if file has been modified since last load
            if os.path.exists(Config.PREFERENCES_STORAGE_PATH):
                file_modified = datetime.fromtimestamp(
                    os.path.getmtime(Config.PREFERENCES_STORAGE_PATH)
                )
                if self._last_loaded and file_modified <= self._last_loaded:
                    return self._preferences_cache
        
        # Load preferences from config
        self._preferences_cache = Config.load_user_preferences()
        self._last_loaded = datetime.now()
        
        self.logger.info("User preferences loaded successfully")
        return self._preferences_cache
    
    def save_preferences(self, preferences: Dict[str, Any]) -> bool:
        """
        Save user preferences to storage
        
        Args:
            preferences: Dictionary of preferences to save
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            # Validate preferences before saving
            if self._validate_preferences(preferences):
                Config.save_user_preferences(preferences)
                self._preferences_cache = preferences.copy()
                self._last_loaded = datetime.now()
                
                self.logger.info("User preferences saved successfully")
                return True
            else:
                self.logger.error("Invalid preferences data - not saved")
                return False
                
        except Exception as e:
            self.logger.error(f"Error saving preferences: {e}")
            return False
    
    def get_preference(self, key: str, default: Any = None) -> Any:
        """
        Get a specific preference value
        
        Args:
            key: Preference key (supports dot notation for nested keys)
            default: Default value if key not found
            
        Returns:
            Preference value or default
        """
        preferences = self.load_preferences()
        
        # Handle nested keys with dot notation
        if '.' in key:
            keys = key.split('.')
            value = preferences
            try:
                for k in keys:
                    value = value[k]
                return value
            except (KeyError, TypeError):
                return default
        
        return preferences.get(key, default)
    
    def set_preference(self, key: str, value: Any) -> bool:
        """
        Set a specific preference value
        
        Args:
            key: Preference key (supports dot notation for nested keys)
            value: Value to set
            
        Returns:
            True if set successfully, False otherwise
        """
        preferences = self.load_preferences()
        
        # Handle nested keys with dot notation
        if '.' in key:
            keys = key.split('.')
            target = preferences
            
            # Navigate to the parent of the final key
            for k in keys[:-1]:
                if k not in target:
                    target[k] = {}
                target = target[k]
            
            # Set the final key
            target[keys[-1]] = value
        else:
            preferences[key] = value
        
        return self.save_preferences(preferences)
    
    def reset_preferences(self) -> bool:
        """
        Reset preferences to defaults
        
        Returns:
            True if reset successfully, False otherwise
        """
        default_preferences = Config.get_default_preferences()
        return self.save_preferences(default_preferences)
    
    def get_model_preference(self, detection_type: str) -> str:
        """
        Get preferred model for a detection type
        
        Args:
            detection_type: Type of detection (deepfake_detection, ai_image_detection)
            
        Returns:
            Preferred model name
        """
        return self.get_preference(f'preferred_models.{detection_type}', 'default')
    
    def set_model_preference(self, detection_type: str, model_name: str) -> bool:
        """
        Set preferred model for a detection type
        
        Args:
            detection_type: Type of detection
            model_name: Name of the preferred model
            
        Returns:
            True if set successfully, False otherwise
        """
        return self.set_preference(f'preferred_models.{detection_type}', model_name)
    
    def get_detection_thresholds(self) -> Dict[str, float]:
        """
        Get current detection thresholds
        
        Returns:
            Dictionary of detection thresholds
        """
        return {
            'general': self.get_preference('confidence_threshold', Config.CONFIDENCE_THRESHOLD),
            'ai_image': Config.AI_IMAGE_THRESHOLD,
            'deepfake': Config.DEEPFAKE_THRESHOLD,
            'duplicate': Config.DUPLICATE_THRESHOLD
        }
    
    def update_threshold(self, threshold_type: str, value: float) -> bool:
        """
        Update a specific threshold value
        
        Args:
            threshold_type: Type of threshold (general, ai_image, deepfake, duplicate)
            value: New threshold value (0.0 to 1.0)
            
        Returns:
            True if updated successfully, False otherwise
        """
        if not (0.0 <= value <= 1.0):
            self.logger.error(f"Invalid threshold value: {value}. Must be between 0.0 and 1.0")
            return False
        
        # Store all threshold updates in user preferences for persistence
        threshold_key = f"thresholds.{threshold_type}"
        return self.set_preference(threshold_key, value)
    
    def get_all_thresholds(self) -> Dict[str, float]:
        """
        Get all current detection thresholds including user overrides
        
        Returns:
            Dictionary of all threshold values
        """
        return {
            'general': self.get_preference('thresholds.general', Config.CONFIDENCE_THRESHOLD),
            'ai_image': self.get_preference('thresholds.ai_image', Config.AI_IMAGE_THRESHOLD),
            'deepfake': self.get_preference('thresholds.deepfake', Config.DEEPFAKE_THRESHOLD),
            'duplicate': self.get_preference('thresholds.duplicate', Config.DUPLICATE_THRESHOLD)
        }
    
    def validate_file_upload(self, filename: str, file_size: int, file_type: str = 'image') -> Dict[str, Any]:
        """
        Validate uploaded file against configuration constraints
        
        Args:
            filename: Name of the uploaded file
            file_size: Size of file in bytes
            file_type: Type of file ('image' or 'video')
            
        Returns:
            Validation result dictionary
        """
        result = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Check file extension
        if not Config.is_allowed_file_extension(filename, file_type):
            result['valid'] = False
            allowed_exts = Config.ALLOWED_IMAGE_EXTENSIONS if file_type == 'image' else Config.ALLOWED_VIDEO_EXTENSIONS
            result['errors'].append(f"File extension not allowed. Allowed: {', '.join(allowed_exts)}")
        
        # Check file size
        if not Config.validate_file_size(file_size):
            result['valid'] = False
            result['errors'].append(f"File size ({file_size / (1024*1024):.1f}MB) exceeds maximum allowed ({Config.MAX_FILE_SIZE_MB}MB)")
        
        # Add warning if file is large but within limits
        file_size_mb = file_size / (1024 * 1024)
        if file_size_mb > Config.MAX_FILE_SIZE_MB * 0.8:
            result['warnings'].append(f"Large file detected ({file_size_mb:.1f}MB). Processing may take longer.")
        
        return result
    
    def export_config(self, include_system: bool = False) -> Dict[str, Any]:
        """
        Export current configuration
        
        Args:
            include_system: Whether to include system/environment configuration
            
        Returns:
            Dictionary of current configuration
        """
        config_data = {
            'user_preferences': self.load_preferences(),
            'export_timestamp': datetime.now().isoformat(),
            'app_version': '1.0'
        }
        
        if include_system:
            config_data['system_config'] = {
                'model_paths': Config.get_model_config(),
                'detection_thresholds': self.get_detection_thresholds(),
                'file_limits': {
                    'max_file_size_mb': Config.MAX_FILE_SIZE_MB,
                    'max_video_duration': Config.MAX_VIDEO_DURATION_SECONDS
                },
                'allowed_extensions': {
                    'images': Config.ALLOWED_IMAGE_EXTENSIONS,
                    'videos': Config.ALLOWED_VIDEO_EXTENSIONS
                }
            }
        
        return config_data
    
    def import_config(self, config_data: Dict[str, Any]) -> bool:
        """
        Import configuration from exported data
        
        Args:
            config_data: Configuration data to import
            
        Returns:
            True if imported successfully, False otherwise
        """
        try:
            if 'user_preferences' in config_data:
                return self.save_preferences(config_data['user_preferences'])
            else:
                self.logger.error("No user preferences found in import data")
                return False
                
        except Exception as e:
            self.logger.error(f"Error importing configuration: {e}")
            return False
    
    def _validate_preferences(self, preferences: Dict[str, Any]) -> bool:
        """
        Validate preferences data structure
        
        Args:
            preferences: Preferences to validate
            
        Returns:
            True if valid, False otherwise
        """
        required_keys = ['confidence_threshold', 'enable_visualization', 'save_results']
        
        # Check required keys exist
        for key in required_keys:
            if key not in preferences:
                self.logger.error(f"Missing required preference key: {key}")
                return False
        
        # Validate confidence threshold range
        threshold = preferences.get('confidence_threshold')
        if not isinstance(threshold, (int, float)) or not (0.0 <= threshold <= 1.0):
            self.logger.error(f"Invalid confidence threshold: {threshold}")
            return False
        
        return True
    
    def backup_preferences_if_enabled(self) -> bool:
        """
        Backup preferences if auto-backup is enabled
        
        Returns:
            True if backup was created or not needed, False if backup failed
        """
        return Config.backup_user_preferences()
    
    def get_advanced_settings(self) -> Dict[str, Any]:
        """
        Get advanced application settings
        
        Returns:
            Dictionary of advanced settings
        """
        return {
            'gpu_acceleration': Config.ENABLE_GPU_ACCELERATION,
            'max_concurrent_processes': Config.MAX_CONCURRENT_PROCESSES,
            'cache_predictions': Config.CACHE_MODEL_PREDICTIONS,
            'debug_logging': Config.ENABLE_DEBUG_LOGGING,
            'development_mode': Config.DEVELOPMENT_MODE,
            'profiling_enabled': Config.ENABLE_PROFILING,
            'ui_theme': Config.UI_THEME,
            'show_technical_details': Config.SHOW_TECHNICAL_DETAILS
        }
    
    def update_advanced_setting(self, setting_name: str, value: Any) -> bool:
        """
        Update an advanced setting in user preferences
        
        Args:
            setting_name: Name of the advanced setting
            value: New value for the setting
            
        Returns:
            True if updated successfully, False otherwise
        """
        advanced_key = f"advanced_settings.{setting_name}"
        return self.set_preference(advanced_key, value)
    
    def get_processing_limits(self) -> Dict[str, Any]:
        """
        Get current processing limits and constraints
        
        Returns:
            Dictionary of processing limits
        """
        return {
            'max_file_size_mb': Config.MAX_FILE_SIZE_MB,
            'max_video_duration_seconds': Config.MAX_VIDEO_DURATION_SECONDS,
            'batch_size': Config.BATCH_SIZE,
            'image_resize_dimensions': Config.IMAGE_RESIZE_DIMENSIONS,
            'video_frame_sample_rate': Config.VIDEO_FRAME_SAMPLE_RATE,
            'allowed_image_extensions': Config.ALLOWED_IMAGE_EXTENSIONS,
            'allowed_video_extensions': Config.ALLOWED_VIDEO_EXTENSIONS
        }
    
    def create_config_report(self) -> Dict[str, Any]:
        """
        Create a comprehensive configuration report
        
        Returns:
            Complete configuration report
        """
        return {
            'timestamp': datetime.now().isoformat(),
            'basic_config': {
                'thresholds': self.get_all_thresholds(),
                'model_preferences': {
                    'deepfake': self.get_model_preference('deepfake_detection'),
                    'ai_image': self.get_model_preference('ai_image_detection')
                }
            },
            'advanced_settings': self.get_advanced_settings(),
            'processing_limits': self.get_processing_limits(),
            'system_status': self.get_config_status(),
            'user_preferences': self.load_preferences(),
            'environment_variables': {
                'model_base_dir': Config.MODEL_BASE_DIR,
                'log_level': Config.LOG_LEVEL,
                'debug_mode': Config.DEBUG_MODE
            }
        }
    
    def get_config_status(self) -> Dict[str, Any]:
        """
        Get current configuration status and health check
        
        Returns:
            Dictionary with configuration status information
        """
        model_status = Config.validate_model_paths()
        preferences = self.load_preferences()
        
        return {
            'preferences_loaded': self._preferences_cache is not None,
            'last_loaded': self._last_loaded.isoformat() if self._last_loaded else None,
            'preferences_file_exists': os.path.exists(Config.PREFERENCES_STORAGE_PATH),
            'model_files_status': model_status,
            'models_available': sum(model_status.values()),
            'total_models': len(model_status),
            'current_thresholds': self.get_all_thresholds(),
            'directories_created': all([
                os.path.exists(Config.MODEL_BASE_DIR),
                os.path.exists(Config.REPORTS_OUTPUT_DIR),
                os.path.exists(Config.TEMP_UPLOAD_DIR)
            ]),
            'env_file_loaded': os.path.exists('.env'),
            'advanced_features': Config.get_advanced_config()
        }


# Global configuration manager instance
config_manager = ConfigManager()
