from rest_framework import serializers
from drf_spectacular.utils import extend_schema_field
from .models import Report, ExtractedReportData, ReportResult

class ReportResultSerializer(serializers.ModelSerializer):
    class Meta:
        model = ReportResult
        fields = ['id', 'summary', 'doctor_summary', 'key_findings', 'conditions', 'risk_level', 'confidence_score', 'suggested_specialization', 'created_at']

    def to_representation(self, instance):
        ret = super().to_representation(instance)
        request = self.context.get('request')
        if request and request.user.is_authenticated:
            if request.user.role == 'DOCTOR':
                # Doctors only see doctor_summary, mapped to 'summary' for frontend simplicity or kept as is?
                # The user said "doctor must be served appropriate doctor summary". 
                # Let's keep doctor_summary but also make 'summary' return doctor_summary if requested.
                ret['summary'] = ret.get('doctor_summary')
                # Optional: Remove doctor_summary if we want to be clean
            elif request.user.role == 'PATIENT':
                # Patients should NOT see doctor_summary
                ret.pop('doctor_summary', None)
        return ret

class ExtractedDataSerializer(serializers.ModelSerializer):
    class Meta:
        model = ExtractedReportData
        fields = ['id', 'raw_ocr_data', 'final_data', 'is_corrected', 'created_at', 'updated_at']
        read_only_fields = ['id', 'raw_ocr_data', 'created_at', 'updated_at']

class ReportSerializer(serializers.ModelSerializer):
    extracted_data = ExtractedDataSerializer(read_only=True)
    result = ReportResultSerializer(read_only=True)
    
    # We use a MethodField to handle the OneToOne relation cleanly
    doctor_comment = serializers.SerializerMethodField()
    
    class Meta:
        model = Report
        fields = ['id', 'file', 'uploaded_at', 'status', 'extracted_data', 'result', 'doctor_comment']
        read_only_fields = ['id', 'uploaded_at', 'status', 'extracted_data', 'result', 'doctor_comment']

    @extend_schema_field(serializers.DictField())
    def get_doctor_comment(self, obj):
        from doctor.models import DoctorComment
        from doctor.serializers import DoctorCommentSerializer
        try:
            comment = obj.doctor_comment
            return DoctorCommentSerializer(comment).data
        except DoctorComment.DoesNotExist:
            return None


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
