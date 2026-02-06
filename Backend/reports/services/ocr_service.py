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
            print(f"DEBUG: Processing report {report_instance.id}, file path: {file_path}")
            
            # 1. Call External FastAPI OCR Engine
            import os
            with open(file_path, 'rb') as f:
                filename = os.path.basename(report_instance.file.name)
                files = {'file': (filename, f, 'application/pdf')}
                print(f"DEBUG: Calling OCR service at {settings.OCR_API_URL} with filename: {filename}")
                response = requests.post(settings.OCR_API_URL, files=files)
            
            print(f"DEBUG: OCR service response status: {response.status_code}")
            response.raise_for_status()
            data = response.json()
            print(f"DEBUG: OCR service response data keys: {list(data.keys())}")

            if data.get('status') != 'Success':
                print(f"DEBUG: OCR service returned non-success status: {data.get('status')}, message: {data.get('message')}")
                raise Exception(data.get('message', 'OCR Extraction failed at microservice.'))

            # 2. Extract structured data
            extracted_data = data.get('extracted_data', {})
            print(f"DEBUG: Extracted data tests count: {len(extracted_data.get('tests', []))}")
            
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
            print(f"DEBUG: Report {report_instance.id} successfully processed")
            return True

        except Exception as e:
            print(f"DEBUG: OCR Processing error: {str(e)}")
            logger.error(f"OCR Processing failed for report {report_instance.id}: {str(e)}")
            report_instance.status = 'FAILED'
            report_instance.save()
            return False

