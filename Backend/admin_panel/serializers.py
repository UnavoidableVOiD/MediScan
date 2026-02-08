from rest_framework import serializers
from doctor.models import DoctorLicense
from authentication.serializers import UserSerializer

class DoctorVerificationReviewSerializer(serializers.ModelSerializer):
    doctor_details = UserSerializer(source='doctor', read_only=True)
    
    class Meta:
        model = DoctorLicense
        fields = [
            'id', 'doctor', 'doctor_details', 'license_number', 
            'license_file', 'other_certificates', 'status', 
            'rejection_reason', 'submitted_at', 'updated_at'
        ]
        read_only_fields = ['id', 'doctor', 'license_number', 'license_file', 'other_certificates', 'submitted_at', 'updated_at']

    def validate_status(self, value):
        if value not in ['APPROVED', 'REJECTED']:
            raise serializers.ValidationError("Status must be either APPROVED or REJECTED.")
        return value

class AdminUserSerializer(serializers.ModelSerializer):
    class Meta:
        model = CustomUser
        fields = [
            'id', 'first_name', 'last_name', 'email', 'phone_number', 
            'role', 'specialization', 'is_verified', 'is_doctor_verified',
            'is_staff', 'is_superuser', 'created_at'
        ]
