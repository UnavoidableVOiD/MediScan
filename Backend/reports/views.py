from rest_framework import viewsets, status, permissions
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from drf_spectacular.utils import extend_schema

from .models import Report, ExtractedReportData
from .serializers import ReportSerializer, ExtractedDataSerializer
from .services.ocr_service import OCRService

class ReportViewSet(viewsets.ModelViewSet):
    serializer_class = ReportSerializer
    permission_classes = [permissions.IsAuthenticated]
    parser_classes = [MultiPartParser, FormParser]

    def get_queryset(self):
        return Report.objects.filter(user=self.request.user).order_by('-uploaded_at')

    def perform_create(self, serializer):
        report = serializer.save(user=self.request.user)
        
        # Trigger OCR Processing (Synchronous for now, ideally async)
        success = OCRService.process_report(report)
        if not success:
            # We don't fail the request, just the processing status
            # Status is already updated inside process_report
            pass

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
        
        extracted_data = report.extracted_data
        
        # Serialize with partial update to allow updating only final_data
        serializer = ExtractedDataSerializer(extracted_data, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save(is_corrected=True)
            return Response(serializer.data)
        
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    @extend_schema(request=ReportSerializer)
    @action(detail=True, methods=['post'], url_path='upload', parser_classes=[MultiPartParser])
    def upload_file(self, request, pk=None):
        # Redundant? Standard 'create' handles upload. 
        # But if specifically requested as separate action:
        pass
