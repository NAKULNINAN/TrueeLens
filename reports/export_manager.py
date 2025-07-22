import os
import logging
from typing import Dict, Any

class ExportManager:
    def __init__(self):
        logging.basicConfig(level=logging.DEBUG)
        self.logger = logging.getLogger(__name__)

    def export_to_file(self, data: Dict[str, Any], directory: str, file_name: str) -> str:
        """Export data to a specified file within a directory"""
        self.logger.debug(f"Exporting data to file: {file_name} in directory: {directory}")
        if not os.path.exists(directory):
            self.logger.debug(f"Directory does not exist, creating: {directory}")
            os.makedirs(directory)
        file_path = os.path.join(directory, file_name)
        with open(file_path, 'w') as file:
            file.write(str(data))
        self.logger.debug(f"Data exported to {file_path}")
        return file_path

