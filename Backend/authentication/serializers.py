from rest_framework import serializers
from django.contrib.auth import get_user_model
from django.contrib.auth.hashers import make_password
from rest_framework_simplejwt.tokens import RefreshToken
from rest_framework_simplejwt.token_blacklist.models import BlacklistedToken, OutstandingToken
from phonenumber_field.serializerfields import PhoneNumberField
from django.utils.timezone import now
from datetime import timedelta
from django.core.cache import cache

from .models import CustomUser
from .otp import OTPHandler

User = get_user_model()

class RegisterSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True, min_length=8)
    confirm_password = serializers.CharField(write_only=True)
    phone_number = PhoneNumberField()

    class Meta:
        model = User
        fields = ['first_name', 'last_name', 'email', 'phone_number', 'password', 'confirm_password', 'role']

    def validate_role(self, value):
        role = value.upper()
        if role not in ['DOCTOR', 'PATIENT']:
            raise serializers.ValidationError("Role must be either 'DOCTOR' or 'PATIENT'.")
        return role

    def validate(self, data):
        if data['password'] != data['confirm_password']:
            raise serializers.ValidationError("Passwords do not match.")
        
        # Check uniqueness manually to fail fast before OTP if possible, 
        # though DB constraints exist, checking here gives better error messages without DB integrity error.
        if User.objects.filter(email=data['email']).exists():
             raise serializers.ValidationError({"email": "Email already exists."})
        if User.objects.filter(phone_number=data['phone_number']).exists():
             raise serializers.ValidationError({"phone_number": "Phone number already exists."})

        # Password complexity check (simple version, more logic can be added or rely on potential validators)
        password = data['password']
        if not any(char.isdigit() for char in password):
            raise serializers.ValidationError("Password must contain at least one digit.")
        if not any(char.isupper() for char in password):
            raise serializers.ValidationError("Password must contain at least one uppercase letter.")
        if not any(char.islower() for char in password):
             raise serializers.ValidationError("Password must contain at least one lowercase letter.")
        if not any(not char.isalnum() for char in password):
             raise serializers.ValidationError("Password must contain at least one special character.")

        return data

    def create(self, validated_data):
        # We DO NOT create the user here. We return the data to the view to handle OTP sending.
        # However, to be compliant with save(), we might need to override it or just use this for validation.
        # We will strip confirm_password and cache the registration data temporarily until OTP is verified.
        validated_data.pop('confirm_password')
        return validated_data


class VerifyOTPSerializer(serializers.Serializer):
    email = serializers.EmailField()
    otp = serializers.CharField(max_length=6)
    type = serializers.ChoiceField(choices=['register', 'login'])

    def validate(self, data):
        identifier = data['email']
        otp = data['otp']
        
        # Verify OTP
        # Note: logic for 'register' vs 'login' might differ in WHAT we verify against.
        # For registration, the user might not exist yet, so we verify against the temp cache key.
        # For login, user exists.
        
        is_valid, message = OTPHandler.verify_otp(identifier, otp)
        if not is_valid:
            raise serializers.ValidationError(message)
            
        return data


class LoginSerializer(serializers.Serializer):
    email = serializers.EmailField()
    password = serializers.CharField(write_only=True)
    role = serializers.CharField()

    def validate(self, data):
        email = data['email']
        password = data['password']

        try:
            user = User.objects.get(email=email)
        except User.DoesNotExist:
            raise serializers.ValidationError("Invalid credentials.")

        input_role = data.get('role', '').upper()
        if user.role != input_role:
             raise serializers.ValidationError("Invalid credentials.")

        if user.account_locked_until and user.account_locked_until > now():
            raise serializers.ValidationError("Account locked. Try again later.")

        if not user.check_password(password):
            user.failed_login_attempts += 1
            if user.failed_login_attempts >= 5:
                user.account_locked_until = now() + timedelta(minutes=15)
                user.failed_login_attempts = 0 # Reset or keep? Usually reset after lock or keep accumulating?
                # Requirement says "Lock account after 5 failures for 15 minutes".
                # It implies creating a lock.
                user.save()
                raise serializers.ValidationError("Account locked due to too many failed attempts.")
            user.save()
            raise serializers.ValidationError("Invalid credentials.")

        # Success reset failure count
        if user.failed_login_attempts > 0:
            user.failed_login_attempts = 0
            user.save()
            
        return data


class LogoutSerializer(serializers.Serializer):
    refresh = serializers.CharField()

    def validate(self, attrs):
        self.token = attrs['refresh']
        return attrs

    def save(self, **kwargs):
        try:
            RefreshToken(self.token).blacklist()
        except Exception:
            self.fail('bad_token')
