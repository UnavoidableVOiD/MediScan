from rest_framework import serializers
from django.contrib.auth import get_user_model
from .models import DoctorPatientLink, DoctorComment, DoctorLicense, SupportingDocument, DoctorAvailability, Appointment
from reports.serializers import ReportSerializer
from drf_spectacular.utils import extend_schema_field

User = get_user_model()

class PatientUserSerializer(serializers.ModelSerializer):
    condition = serializers.SerializerMethodField()
    status = serializers.SerializerMethodField()
    last_visit = serializers.SerializerMethodField()
    notes = serializers.SerializerMethodField()

    class Meta:
        model = User
        fields = ['id', 'email', 'first_name', 'last_name', 'phone_number', 'condition', 'status', 'last_visit', 'notes']

    @extend_schema_field(serializers.CharField())
    def get_condition(self, obj):
        # Get latest report's risk level/finding
        from reports.models import ReportResult
        latest_result = ReportResult.objects.filter(report__user=obj).order_by('-report__uploaded_at').first()
        return latest_result.risk_level if latest_result else "Unknown"

    @extend_schema_field(serializers.CharField())
    def get_status(self, obj):
        # Get status from the link
        doctor = self.context['request'].user
        link = DoctorPatientLink.objects.filter(doctor=doctor, patient=obj).first()
        return link.status if link else "N/A"

    @extend_schema_field(serializers.DateTimeField())
    def get_last_visit(self, obj):
        link = DoctorPatientLink.objects.filter(patient=obj).first()
        return link.linked_at if link else None

    @extend_schema_field(serializers.CharField())
    def get_notes(self, obj):
        doctor = self.context['request'].user
        link = DoctorPatientLink.objects.filter(doctor=doctor, patient=obj).first()
        return link.notes if link else ""


class DoctorDashboardSerializer(serializers.Serializer):
    total_patients = serializers.IntegerField()
    ongoing_patients = serializers.IntegerField()
    completed_patients = serializers.IntegerField()
    new_patients_7_days = serializers.IntegerField()


class DoctorUserSerializer(serializers.ModelSerializer):
    
    class Meta:
        model = User
        fields = ['id', 'email', 'first_name', 'last_name', 'specialization', 'consultation_fee']

class DoctorPatientLinkSerializer(serializers.ModelSerializer):
    patient = PatientUserSerializer(read_only=True)
    doctor = DoctorUserSerializer(read_only=True)
    doctor_id = serializers.PrimaryKeyRelatedField(
        queryset=User.objects.filter(role='DOCTOR'), 
        source='doctor', 
        write_only=True
    )

    class Meta:
        model = DoctorPatientLink
        fields = ['id', 'patient', 'doctor', 'doctor_id', 'status', 'notes', 'linked_at']
        read_only_fields = ['id', 'patient', 'doctor', 'linked_at']

    def validate_patient(self, value):
        if value.role != 'PATIENT':
            raise serializers.ValidationError("Only patients can be linked to doctors.")
        return value

class DoctorCommentSerializer(serializers.ModelSerializer):
    class Meta:
        model = DoctorComment
        fields = ['id', 'report', 'doctor', 'comment', 'created_at', 'updated_at']
        read_only_fields = ['id', 'doctor', 'created_at', 'updated_at']

    def validate(self, data):
        # Ensure the doctor is linked to the patient who owns the report
        doctor = self.context['request'].user
        report = data.get('report')
        if not DoctorPatientLink.objects.filter(doctor=doctor, patient=report.user).exists():
            raise serializers.ValidationError("You can only comment on reports of your linked patients.")
        return data

class SupportingDocumentSerializer(serializers.ModelSerializer):
    class Meta:
        model = SupportingDocument
        fields = ['id', 'file', 'uploaded_at']
        read_only_fields = ['id', 'uploaded_at']


class DoctorLicenseSerializer(serializers.ModelSerializer):
    supporting_documents = SupportingDocumentSerializer(many=True, read_only=True)
    supporting_documents_upload = serializers.ListField(
        child=serializers.FileField(),
        write_only=True,
        required=False,
        help_text="Upload multiple verification documents."
    )

    class Meta:
        model = DoctorLicense
        fields = [
            'id', 'license_number', 'license_file', 'supporting_documents', 
            'supporting_documents_upload',
            'status', 'rejection_reason', 'submitted_at', 'updated_at'
        ]
        read_only_fields = ['id', 'status', 'rejection_reason', 'submitted_at', 'updated_at']

    def validate(self, data):
        user = self.context['request'].user
        if user.role != 'DOCTOR':
            raise serializers.ValidationError("Only doctors can submit license information.")
        return data

    def create(self, validated_data):
        supporting_docs = validated_data.pop('supporting_documents_upload', [])
        license_obj = DoctorLicense.objects.create(**validated_data)
        self._save_supporting_docs(license_obj, supporting_docs)
        return license_obj

    def update(self, instance, validated_data):
        supporting_docs = validated_data.pop('supporting_documents_upload', [])
        
        for attr, value in validated_data.items():
            setattr(instance, attr, value)
        instance.save()
        
        if supporting_docs:
            # Replace old supporting docs with new ones if any are uploaded
            instance.supporting_documents.all().delete()
            self._save_supporting_docs(instance, supporting_docs)
        return instance

    def _save_supporting_docs(self, license_obj, files):
        """Save list of uploaded files."""
        for file in files:
            SupportingDocument.objects.create(license=license_obj, file=file)

class DoctorAvailabilitySerializer(serializers.ModelSerializer):
    class Meta:
        model = DoctorAvailability
        fields = ['id', 'doctor', 'day_of_week', 'start_time', 'end_time', 'is_active']
        read_only_fields = ['id', 'doctor']

class AppointmentSerializer(serializers.ModelSerializer):
    patient_email = serializers.EmailField(source='patient.email', read_only=True)
    doctor_email = serializers.EmailField(source='doctor.email', read_only=True)
    doctor_full_name = serializers.SerializerMethodField()

    class Meta:
        model = Appointment
        fields = [
            'id', 'patient', 'patient_email', 'doctor', 'doctor_email', 'doctor_full_name',
            'appointment_date', 'start_time', 'end_time', 'status', 'notes',
            'payment_id', 'amount_paid', 'created_at'
        ]
        read_only_fields = ['id', 'patient', 'status', 'payment_id', 'amount_paid', 'created_at']

    def get_doctor_full_name(self, obj):
        return f"{obj.doctor.first_name} {obj.doctor.last_name}"
