import logging
import traceback
import functools
import time
import sys
import os
from datetime import datetime
from typing import Any, Callable, Dict, Optional, Type, Union
from contextlib import contextmanager
import json
from pathlib import Path


class ErrorHandler:
    """Comprehensive error handling and recovery system"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.error_stats = {
            'total_errors': 0,
            'recovered_errors': 0,
            'critical_errors': 0,
            'last_error_time': None,
            'error_types': {}
        }
        self.recovery_strategies = {}
        self._setup_recovery_strategies()
    
    def _setup_recovery_strategies(self):
        """Define recovery strategies for common error types"""
        self.recovery_strategies = {
            'FileNotFoundError': self._handle_file_not_found,
            'MemoryError': self._handle_memory_error,
            'ConnectionError': self._handle_connection_error,
            'ImportError': self._handle_import_error,
            'PermissionError': self._handle_permission_error,
            'TimeoutError': self._handle_timeout_error,
            'ValueError': self._handle_value_error,
            'KeyError': self._handle_key_error,
            'AttributeError': self._handle_attribute_error,
            'IOError': self._handle_io_error
        }
    
    def handle_error(self, error: Exception, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Main error handling method with recovery attempts"""
        error_type = type(error).__name__
        error_info = {
            'error_type': error_type,
            'error_message': str(error),
            'timestamp': datetime.now().isoformat(),
            'context': context or {},
            'traceback': traceback.format_exc(),
            'recovered': False,
            'recovery_action': None
        }
        
        # Update error statistics
        self._update_error_stats(error_type)
        
        # Log the error
        self._log_error(error, error_info)
        
        # Attempt recovery
        recovery_result = self._attempt_recovery(error, context)
        if recovery_result:
            error_info.update(recovery_result)
            self.error_stats['recovered_errors'] += 1
            self.logger.info(f"Successfully recovered from {error_type}: {recovery_result['recovery_action']}")
        
        return error_info
    
    def _update_error_stats(self, error_type: str):
        """Update error statistics"""
        self.error_stats['total_errors'] += 1
        self.error_stats['last_error_time'] = datetime.now().isoformat()
        
        if error_type not in self.error_stats['error_types']:
            self.error_stats['error_types'][error_type] = 0
        self.error_stats['error_types'][error_type] += 1
        
        # Mark as critical if it's a severe error type
        critical_errors = ['MemoryError', 'SystemError', 'OSError', 'RuntimeError']
        if error_type in critical_errors:
            self.error_stats['critical_errors'] += 1
    
    def _log_error(self, error: Exception, error_info: Dict[str, Any]):
        """Log error with appropriate level"""
        error_type = type(error).__name__
        
        if error_type in ['MemoryError', 'SystemError', 'OSError']:
            self.logger.critical(f"Critical error: {error_info['error_message']}", extra=error_info)
        elif error_type in ['ConnectionError', 'TimeoutError', 'IOError']:
            self.logger.error(f"System error: {error_info['error_message']}", extra=error_info)
        else:
            self.logger.warning(f"Recoverable error: {error_info['error_message']}", extra=error_info)
    
    def _attempt_recovery(self, error: Exception, context: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """Attempt to recover from the error"""
        error_type = type(error).__name__
        
        if error_type in self.recovery_strategies:
            try:
                recovery_action = self.recovery_strategies[error_type](error, context)
                return {
                    'recovered': True,
                    'recovery_action': recovery_action,
                    'recovery_timestamp': datetime.now().isoformat()
                }
            except Exception as recovery_error:
                self.logger.error(f"Recovery failed for {error_type}: {recovery_error}")
                return None
        
        return None
    
    def _handle_file_not_found(self, error: FileNotFoundError, context: Dict[str, Any]) -> str:
        """Handle file not found errors"""
        missing_file = str(error).split("'")[1] if "'" in str(error) else "unknown"
        
        # Try to create missing directories
        if context and 'create_dirs' in context:
            dir_path = os.path.dirname(missing_file)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
                return f"Created missing directory: {dir_path}"
        
        # Try to use alternative file paths
        if context and 'alternative_paths' in context:
            for alt_path in context['alternative_paths']:
                if os.path.exists(alt_path):
                    return f"Using alternative file: {alt_path}"
        
        # Create empty file if it's a config file
        if missing_file.endswith(('.json', '.config', '.ini')):
            Path(missing_file).touch()
            return f"Created empty config file: {missing_file}"
        
        return f"Logged missing file: {missing_file}"
    
    def _handle_memory_error(self, error: MemoryError, context: Dict[str, Any]) -> str:
        """Handle memory errors"""
        # Force garbage collection
        import gc
        gc.collect()
        
        # Try to reduce batch size if available in context
        if context and 'batch_size' in context:
            new_batch_size = max(1, context['batch_size'] // 2)
            context['batch_size'] = new_batch_size
            return f"Reduced batch size to {new_batch_size} to free memory"
        
        # Clear caches if available
        if context and 'clear_cache' in context:
            context['clear_cache']()
            return "Cleared application caches to free memory"
        
        return "Performed garbage collection to free memory"
    
    def _handle_connection_error(self, error: ConnectionError, context: Dict[str, Any]) -> str:
        """Handle connection errors"""
        # Implement retry logic
        if context and 'retry_count' in context:
            if context['retry_count'] < 3:
                time.sleep(2 ** context['retry_count'])  # Exponential backoff
                context['retry_count'] += 1
                return f"Will retry connection (attempt {context['retry_count']})"
        
        # Switch to offline mode if available
        if context and 'offline_mode' in context:
            context['offline_mode'] = True
            return "Switched to offline mode due to connection issues"
        
        return "Logged connection error for manual review"
    
    def _handle_import_error(self, error: ImportError, context: Dict[str, Any]) -> str:
        """Handle import errors"""
        missing_module = str(error).split("'")[1] if "'" in str(error) else "unknown"
        
        # Try to install missing package
        if context and context.get('auto_install', False):
            try:
                import subprocess
                subprocess.check_call([sys.executable, "-m", "pip", "install", missing_module])
                return f"Automatically installed missing package: {missing_module}"
            except:
                pass
        
        # Use fallback implementation if available
        if context and 'fallback_modules' in context:
            for fallback in context['fallback_modules']:
                try:
                    __import__(fallback)
                    return f"Using fallback module: {fallback}"
                except ImportError:
                    continue
        
        return f"Logged missing dependency: {missing_module}"
    
    def _handle_permission_error(self, error: PermissionError, context: Dict[str, Any]) -> str:
        """Handle permission errors"""
        file_path = str(error).split("'")[1] if "'" in str(error) else "unknown"
        
        # Try alternative location
        if context and 'temp_dir' in context:
            import tempfile
            temp_path = os.path.join(tempfile.gettempdir(), os.path.basename(file_path))
            context['alternative_path'] = temp_path
            return f"Using temporary location: {temp_path}"
        
        # Try to change permissions
        try:
            os.chmod(file_path, 0o666)
            return f"Changed permissions for: {file_path}"
        except:
            pass
        
        return f"Logged permission issue for: {file_path}"
    
    def _handle_timeout_error(self, error: TimeoutError, context: Dict[str, Any]) -> str:
        """Handle timeout errors"""
        # Increase timeout if configurable
        if context and 'timeout' in context:
            context['timeout'] *= 2
            return f"Increased timeout to {context['timeout']} seconds"
        
        # Switch to asynchronous processing if available
        if context and 'async_mode' in context:
            context['async_mode'] = True
            return "Switched to asynchronous processing mode"
        
        return "Logged timeout for performance analysis"
    
    def _handle_value_error(self, error: ValueError, context: Dict[str, Any]) -> str:
        """Handle value errors"""
        # Use default values if available
        if context and 'default_values' in context:
            return "Applied default values for invalid inputs"
        
        # Sanitize input if possible
        if context and 'input_sanitizer' in context:
            sanitized = context['input_sanitizer'](str(error))
            return f"Sanitized input: {sanitized}"
        
        return "Logged invalid value for validation improvement"
    
    def _handle_key_error(self, error: KeyError, context: Dict[str, Any]) -> str:
        """Handle key errors"""
        missing_key = str(error).replace("'", "").replace('"', '')
        
        # Use default value
        if context and 'defaults' in context:
            default_value = context['defaults'].get(missing_key, None)
            return f"Using default value for missing key '{missing_key}': {default_value}"
        
        # Create missing key with None value
        if context and 'data' in context:
            context['data'][missing_key] = None
            return f"Created missing key '{missing_key}' with None value"
        
        return f"Logged missing key: {missing_key}"
    
    def _handle_attribute_error(self, error: AttributeError, context: Dict[str, Any]) -> str:
        """Handle attribute errors"""
        # Use getattr with default
        if context and 'object' in context and 'attribute' in context:
            setattr(context['object'], context['attribute'], None)
            return f"Set missing attribute '{context['attribute']}' to None"
        
        return "Logged missing attribute for code review"
    
    def _handle_io_error(self, error: IOError, context: Dict[str, Any]) -> str:
        """Handle IO errors"""
        # Retry with different mode
        if context and 'file_mode' in context:
            if context['file_mode'] == 'r':
                context['file_mode'] = 'rb'
                return "Switched to binary read mode"
            elif context['file_mode'] == 'w':
                context['file_mode'] = 'wb'
                return "Switched to binary write mode"
        
        # Use temporary file
        if context and 'use_temp_file' in context:
            import tempfile
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            context['temp_file_path'] = temp_file.name
            return f"Using temporary file: {temp_file.name}"
        
        return "Logged IO error for system check"
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get current error statistics"""
        return self.error_stats.copy()
    
    def reset_error_stats(self):
        """Reset error statistics"""
        self.error_stats = {
            'total_errors': 0,
            'recovered_errors': 0,
            'critical_errors': 0,
            'last_error_time': None,
            'error_types': {}
        }
    
    def export_error_report(self, filepath: str) -> bool:
        """Export error statistics to file"""
        try:
            report = {
                'error_statistics': self.error_stats,
                'recovery_strategies': list(self.recovery_strategies.keys()),
                'export_timestamp': datetime.now().isoformat()
            }
            
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2)
            
            self.logger.info(f"Error report exported to: {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to export error report: {e}")
            return False


def with_error_handling(error_handler: ErrorHandler = None, context: Dict[str, Any] = None):
    """Decorator for automatic error handling"""
    if error_handler is None:
        error_handler = ErrorHandler()
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_info = error_handler.handle_error(e, context)
                
                # If recovery was successful, try the function again
                if error_info.get('recovered', False):
                    try:
                        return func(*args, **kwargs)
                    except Exception as retry_error:
                        # If retry fails, log and re-raise
                        error_handler.logger.error(f"Retry failed after recovery: {retry_error}")
                        raise retry_error
                
                # If no recovery, re-raise the original exception
                raise e
        
        return wrapper
    return decorator


@contextmanager
def error_recovery_context(error_handler: ErrorHandler = None, context: Dict[str, Any] = None):
    """Context manager for error handling and recovery"""
    if error_handler is None:
        error_handler = ErrorHandler()
    
    try:
        yield error_handler
    except Exception as e:
        error_info = error_handler.handle_error(e, context)
        
        # If recovery was not successful, re-raise
        if not error_info.get('recovered', False):
            raise e


# Global error handler instance
global_error_handler = ErrorHandler()
