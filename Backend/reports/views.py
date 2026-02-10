from rest_framework import viewsets, status, permissions
import logging
import requests
from django.conf import settings

# OCR service URL (adjust if needed)
OCR_API_URL = getattr(settings, 'OCR_API_URL', 'http://localhost:8001/extract_from_pdf')
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser, JSONParser
from drf_spectacular.utils import extend_schema

from .models import Report, ExtractedReportData
from .serializers import ReportSerializer, ExtractedDataSerializer
from .services.ocr_service import OCRService
from .services.analysis_service import AnalysisService

class ReportViewSet(viewsets.ModelViewSet):
    serializer_class = ReportSerializer
    permission_classes = [permissions.IsAuthenticated]
    parser_classes = [MultiPartParser, FormParser, JSONParser]
    queryset = Report.objects.none()

    def get_queryset(self):
        return Report.objects.filter(user=self.request.user).order_by('-uploaded_at')

    def perform_create(self, serializer):
        serializer.save(user=self.request.user)

    @action(detail=True, methods=['post'], url_path='process')
    def process(self, request, pk=None):
        """
        Manually trigger OCR processing for a report.
        """
        report = self.get_object()
        
        # Reset status for re-processing if needed
        report.status = 'PENDING'
        report.save()

        success = OCRService.process_report(report)
        
        if success:
            # Refresh report from DB to get extracted_data relation
            report.refresh_from_db()
            return Response(ReportSerializer(report).data)
        
        return Response({
            "error": "OCR Processing failed",
            "status": report.status
        }, status=status.HTTP_400_BAD_REQUEST)

    @action(detail=True, methods=['post'], url_path='ocr')
    def ocr_process(self, request, pk=None):
        """Send the report PDF to external OCR service and store extracted data."""
        report = self.get_object()
        try:
            import os
            with open(report.file.path, 'rb') as f:
                filename = os.path.basename(report.file.name)
                files = {'file': (filename, f, 'application/pdf')}
                print(f"DEBUG: ocr_process calling {OCR_API_URL} with filename: {filename}")
                response = requests.post(OCR_API_URL, files=files)
            print(f"DEBUG: ocr_process response status: {response.status_code}")
            response.raise_for_status()
            data = response.json()
            # Expecting 'extracted_data' key per ML service
            extracted = data.get('extracted_data', {})
            # Save raw OCR response
            ExtractedReportData.objects.create(
                report=report,
                raw_ocr_data=data,
                final_data=extracted
            )
            report.status = 'PROCESSED'
            report.save()
            return Response(ReportSerializer(report).data)
        except Exception as e:
            logger = logging.getLogger('security')
            logger.error(f"External OCR call failed for report {report.id}: {str(e)}")
            report.status = 'FAILED'
            report.save()
            return Response({"error": "External OCR processing failed", "detail": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


    @extend_schema(request=ExtractedDataSerializer)
    @action(detail=True, methods=['put'], url_path='correct')
    def correct_data(self, request, pk=None):
        """
        Endpoint for users to correct the OCR data.
        Body: { "final_data": { ... } }
        """
        report = self.get_object()
        
        if not hasattr(report, 'extracted_data'):
            return Response({"error": "No extracted data found for this report."}, status=status.HTTP_404_NOT_FOUND)
        
        extracted_data_obj = report.extracted_data
        
        # Serialize with partial update to allow updating only final_data
        serializer = ExtractedDataSerializer(extracted_data_obj, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save(is_corrected=True)
            
            # --- CALL AI ANALYSIS SERVICE (Step 4) ---
            success, result = AnalysisService.run_analysis(
                report_instance=report,
                validated_data=serializer.validated_data.get('final_data', {}),
                user_id=request.user.id
            )
            
            if not success:
               return Response({"error": "AI Analysis failed", "detail": result}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            
            return Response(ReportSerializer(report).data)
        
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    @action(detail=True, methods=['get'])
    def result(self, request, pk=None):
        """
        Get the AI-generated result for a report.
        """
        report = self.get_object()
        if not hasattr(report, 'result'):
            return Response({"error": "Analysis result not ready yet."}, status=status.HTTP_404_NOT_FOUND)
        
        from .serializers import ReportResultSerializer
        return Response(ReportResultSerializer(report.result).data)
