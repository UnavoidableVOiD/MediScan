from rest_framework import serializers
from django.contrib.auth import get_user_model
from .models import DoctorPatientLink, DoctorComment, DoctorLicense
from reports.serializers import ReportSerializer

User = get_user_model()

class PatientUserSerializer(serializers.ModelSerializer):
 
    class Meta:
        model = User
        fields = ['id', 'email', 'first_name', 'last_name', 'phone_number']

class DoctorUserSerializer(serializers.ModelSerializer):
    
    class Meta:
        model = User
        fields = ['id', 'email', 'first_name', 'last_name']

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
        fields = ['id', 'patient', 'doctor', 'doctor_id', 'linked_at']
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

class DoctorLicenseSerializer(serializers.ModelSerializer):
    class Meta:
        model = DoctorLicense
        fields = [
            'id', 'license_number', 'license_file', 'other_certificates', 
            'status', 'rejection_reason', 'submitted_at', 'updated_at'
        ]
        read_only_fields = ['id', 'status', 'rejection_reason', 'submitted_at', 'updated_at']

    def validate(self, data):
        # Ensure only doctors can submit licenses
        user = self.context['request'].user
        if user.role != 'DOCTOR':
            raise serializers.ValidationError("Only doctors can submit license information.")
        return data
