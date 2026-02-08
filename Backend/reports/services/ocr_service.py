import logging
import requests
from django.conf import settings
from ..models import ExtractedReportData

logger = logging.getLogger('security')

class OCRService:
    @staticmethod
    def process_report(report_instance):
        """
        Main entry point to process a report by calling the external FastAPI OCR service.
        """
        try:
            file_path = report_instance.file.path
            
            # 1. Call External FastAPI OCR Engine
            with open(file_path, 'rb') as f:
                files = {'file': (report_instance.file.name, f, 'application/pdf')}
                response = requests.post(settings.OCR_API_URL, files=files)
            
            response.raise_for_status()
            data = response.json()

            if data.get('status') != 'Success':
                raise Exception(data.get('message', 'OCR Extraction failed at microservice.'))

            # 2. Extract structured data
            extracted_data = data.get('extracted_data', {})
            
            # 3. Save Data
            ExtractedReportData.objects.update_or_create(
                report=report_instance,
                defaults={
                    "raw_ocr_data": data,
                    "final_data": extracted_data,
                    "is_corrected": False
                }
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

