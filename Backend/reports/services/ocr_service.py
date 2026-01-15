import logging
import os
import sys
from django.conf import settings
from ..models import ExtractedReportData

# Add ML/src to path to import the engine
ml_src_path = os.path.join(settings.BASE_DIR, 'ML', 'src')
if ml_src_path not in sys.path:
    sys.path.append(ml_src_path)

try:
    from ocr_engine.ocr_main import extract_text_with_layout, parse_lab_report
except ImportError:
    # Fallback to avoid breaking the whole app if ML folder is missing
    extract_text_with_layout = None
    parse_lab_report = None

logger = logging.getLogger('security')

class OCRService:
    @staticmethod
    def process_report(report_instance):
        """
        Main entry point to process a report.
        """
        try:
            file_path = report_instance.file.path
            
            if not extract_text_with_layout or not parse_lab_report:
                raise ImportError("OCR Engine functions could not be imported. Check ML folder structure.")

            # 1. Call Custom OCR Engine
            raw_data = OCRService._call_custom_engine(file_path)
            
            # 2. Parse/Standardize Output
            # In your engine, parse_lab_report is already doing the structuring.
            # We'll store the structured data.
            final_data = OCRService._parse_output(raw_data)
            
            # 3. Save Data
            ExtractedReportData.objects.create(
                report=report_instance,
                raw_ocr_data={"text_lines": raw_data['lines']}, # We store lines in raw
                final_data=final_data
            )
            
            # 4. Update Report Status
            report_instance.status = 'PROCESSED'
            report_instance.save()
            return True

        except Exception as e:
            logger.error(f"OCR Processing failed for report {report_instance.id}: {str(e)}")
            report_instance.status = 'FAILED'
            report_instance.save()
            return False

    @staticmethod
    def _call_custom_engine(file_path):
        """
        Executes the OCR engine logic from ML/src/ocr_engine/ocr_main.py
        """
        # 1. Extract lines
        lines = extract_text_with_layout(file_path)
        
        # 2. Parse lines into structured JSON
        structured_data = parse_lab_report(lines)
        
        return {
            "lines": lines,
            "structured": structured_data
        }

    @staticmethod
    def _parse_output(raw_data):
        """
        Returns the structured part of the OCR output.
        """
        return raw_data.get('structured', {})

