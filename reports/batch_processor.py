import logging
from typing import List, Dict, Any

class BatchProcessor:
    def __init__(self):
        logging.basicConfig(level=logging.DEBUG)
        self.logger = logging.getLogger(__name__)

    def process_multiple_files(self, files: List[str], processor_function) -> List[Dict[str, Any]]:
        """Process multiple files and collect results using the provided processor_function"""
        results = []
        for file_path in files:
            try:
                self.logger.debug(f"Processing file: {file_path}")
                result = processor_function(file_path)
                results.append(result)
                self.logger.debug(f"Processed file: {file_path}, result: {result}")
            except Exception as e:
                self.logger.error(f"Error processing file {file_path}: {str(e)}")
        return results

    def export_results(self, results: List[Dict[str, Any]], export_function) -> None:
        """Use the export function to handle results storage or further processing"""
        for result in results:
            try:
                self.logger.debug(f"Exporting result: {result}")
                export_function(result)
            except Exception as e:
                self.logger.error(f"Error exporting result: {str(e)}")

