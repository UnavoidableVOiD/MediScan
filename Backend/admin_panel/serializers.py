from rest_framework import serializers
from doctor.models import DoctorLicense, SupportingDocument
from doctor.serializers import SupportingDocumentSerializer
from authentication.serializers import UserSerializer
from authentication.models import CustomUser
from django.contrib.auth.hashers import make_password

class DoctorVerificationReviewSerializer(serializers.ModelSerializer):
    doctor_details = UserSerializer(source='doctor', read_only=True)
    license_file = serializers.SerializerMethodField()
    supporting_documents = serializers.SerializerMethodField()
    
    class Meta:
        model = DoctorLicense
        fields = [
            'id', 'doctor', 'doctor_details', 'license_number', 
            'license_file', 'supporting_documents', 'status', 
            'rejection_reason', 'submitted_at', 'updated_at'
        ]
        read_only_fields = ['id', 'doctor', 'license_number', 'license_file', 'supporting_documents', 'submitted_at', 'updated_at']

    def get_license_file(self, obj):
        request = self.context.get('request')
        if obj.license_file and request:
            return request.build_absolute_uri(obj.license_file.url)
        return obj.license_file.url if obj.license_file else None

    def get_supporting_documents(self, obj):
        request = self.context.get('request')
        return [
            request.build_absolute_uri(doc.file.url) if request else doc.file.url 
            for doc in obj.supporting_documents.all() if doc.file
        ]

    def validate_status(self, value):
        if value not in ['APPROVED', 'REJECTED']:
            raise serializers.ValidationError("Status must be either APPROVED or REJECTED.")
        return value

class AdminUserSerializer(serializers.ModelSerializer):
    license_info = serializers.SerializerMethodField()

    class Meta:
        model = CustomUser
        fields = [
            'id', 'first_name', 'last_name', 'email', 'phone_number', 
            'role', 'specialization', 'is_verified', 'doctor_status',
            'is_staff', 'is_superuser', 'is_active', 'created_at', 'license_info'
        ]

    def get_license_info(self, obj):
        if obj.role == 'DOCTOR':
            license = DoctorLicense.objects.filter(doctor=obj).first()
            if license:
                request = self.context.get('request')
                supporting_docs = license.supporting_documents.all()
                return {
                    'id': license.id,
                    'license_number': license.license_number,
                    'license_file': request.build_absolute_uri(license.license_file.url) if license.license_file and request else (license.license_file.url if license.license_file else None),
                    'supporting_documents': [request.build_absolute_uri(doc.file.url) if request else doc.file.url for doc in supporting_docs if doc.file],
                    'status': license.status,
                    'rejection_reason': license.rejection_reason
                }
        return None

class AdminCreateSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True)

    class Meta:
        model = CustomUser
        fields = ['first_name', 'last_name', 'email', 'phone_number', 'password']

    def create(self, validated_data):
        validated_data['password'] = make_password(validated_data['password'])
        return super().create(validated_data)
