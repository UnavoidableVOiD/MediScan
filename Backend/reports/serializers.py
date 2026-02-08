from rest_framework import serializers
from .models import Report, ExtractedReportData, ReportResult

class ReportResultSerializer(serializers.ModelSerializer):
    class Meta:
        model = ReportResult
        fields = ['id', 'summary', 'doctor_summary', 'key_findings', 'conditions', 'risk_level', 'confidence_score', 'created_at']

class ExtractedDataSerializer(serializers.ModelSerializer):
    class Meta:
        model = ExtractedReportData
        fields = ['id', 'raw_ocr_data', 'final_data', 'is_corrected', 'created_at', 'updated_at']
        read_only_fields = ['id', 'raw_ocr_data', 'created_at', 'updated_at']

class ReportSerializer(serializers.ModelSerializer):
    extracted_data = ExtractedDataSerializer(read_only=True)
    result = ReportResultSerializer(read_only=True)
    doctor_comment = serializers.StringRelatedField(read_only=True)
    
    class Meta:
        model = Report
        fields = ['id', 'file', 'uploaded_at', 'status', 'extracted_data', 'result', 'doctor_comment']
        read_only_fields = ['id', 'uploaded_at', 'status', 'extracted_data', 'result', 'doctor_comment']


    def validate_file(self, value):
        valid_extensions = ['.pdf', '.jpg', '.jpeg', '.png']
        valid = False
        for ext in valid_extensions:
            if value.name.lower().endswith(ext):
                valid = True
                break
        if not valid:
            raise serializers.ValidationError("Unsupported file type. Allowed: PDF, JPG, PNG.")
        return value
