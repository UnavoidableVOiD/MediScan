from rest_framework import viewsets, status
from rest_framework.response import Response
from doctor.models import DoctorLicense
from .serializers import DoctorVerificationReviewSerializer, AdminUserSerializer
from .permissions import IsSystemAdmin
from rest_framework.views import APIView
from django.contrib.auth import authenticate
from rest_framework_simplejwt.tokens import RefreshToken
from django.conf import settings
from authentication.models import CustomUser

class DoctorVerificationViewSet(viewsets.ModelViewSet):
    """
    ViewSet for administrators to list and review doctor license submissions.
    """
    queryset = DoctorLicense.objects.all().order_by('-submitted_at')
    serializer_class = DoctorVerificationReviewSerializer
    permission_classes = [IsSystemAdmin]

    def perform_update(self, serializer):
        instance = serializer.save()
        
        # Sync the CustomUser field based on approval
        doctor = instance.doctor
        if instance.status == 'APPROVED':
            doctor.is_doctor_verified = True
        else:
            doctor.is_doctor_verified = False
        doctor.save()

class AdminLoginView(APIView):
    permission_classes = [] 

    def post(self, request):
        email = request.data.get('email')
        password = request.data.get('password')

        user = authenticate(email=email, password=password)

        if user and (user.is_staff or user.is_superuser):
            refresh = RefreshToken.for_user(user)
            response = Response({
                "success": True,
                "message": "Admin login successful.",
                "user": AdminUserSerializer(user).data
            }, status=status.HTTP_200_OK)

            response.set_cookie(
                key='access',
                value=str(refresh.access_token),
                httponly=True,
                secure=settings.DEBUG is False,
                samesite='Lax',
                path='/'
            )
            response.set_cookie(
                key='refresh',
                value=str(refresh),
                httponly=True,
                secure=settings.DEBUG is False,
                samesite='Lax',
                path='/'
            )
            return response

        return Response({"error": "Invalid admin credentials."}, status=status.HTTP_401_UNAUTHORIZED)

class AdminDoctorListView(viewsets.ReadOnlyModelViewSet):
    queryset = CustomUser.objects.filter(role='DOCTOR').order_by('-created_at')
    serializer_class = AdminUserSerializer
    permission_classes = [IsSystemAdmin]

class AdminPatientListView(viewsets.ReadOnlyModelViewSet):
    queryset = CustomUser.objects.filter(role='PATIENT').order_by('-created_at')
    serializer_class = AdminUserSerializer
    permission_classes = [IsSystemAdmin]
