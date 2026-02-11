from rest_framework import viewsets, status, generics, serializers
from rest_framework.decorators import action
from rest_framework.response import Response
from doctor.models import DoctorLicense
from .serializers import DoctorVerificationReviewSerializer, AdminUserSerializer, AdminCreateSerializer
from authentication.serializers import LoginSerializer
from .permissions import IsSystemAdmin, IsSuperAdmin
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

    def update(self, request, *args, **kwargs):
        partial = kwargs.pop('partial', False)
        instance = self.get_object()
        serializer = self.get_serializer(instance, data=request.data, partial=partial)
        serializer.is_valid(raise_exception=True)
        self.perform_update(serializer)

        if getattr(instance, '_prefetched_objects_cache', None):
            instance._prefetched_objects_cache = {}

        return Response(serializer.data)

    def perform_update(self, serializer):
        instance = serializer.save()
        
        # Sync the CustomUser field based on approval
        doctor = instance.doctor
        if instance.status == 'APPROVED':
            doctor.doctor_status = 'VERIFIED'
        elif instance.status == 'REJECTED':
            doctor.doctor_status = 'REJECTED'
        else:
            doctor.doctor_status = 'PENDING'
        doctor.save()

class AdminCreateView(generics.CreateAPIView):
    queryset = CustomUser.objects.all()
    serializer_class = AdminCreateSerializer
    permission_classes = [IsSystemAdmin]

    def perform_create(self, serializer):
        # Set is_staff=True for admin users
        serializer.save(is_staff=True, role='ADMIN')

class AdminLoginView(APIView):
    permission_classes = [] 
    serializer_class = LoginSerializer

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

class AdminDoctorListView(viewsets.ModelViewSet):
    serializer_class = AdminUserSerializer
    permission_classes = [IsSystemAdmin]

    def get_queryset(self):
        queryset = CustomUser.objects.filter(role='DOCTOR').order_by('-created_at')
        status_param = self.request.query_params.get('status')
        if status_param:
            # Map frontend status (tab name) to backend doctor_status
            status_map = {
                'pending': ['PENDING', 'UNVERIFIED'],
                'verified': ['VERIFIED'],
                'rejected': ['REJECTED']
            }
            backend_status = status_map.get(status_param.lower())
            if backend_status:
                queryset = queryset.filter(doctor_status__in=backend_status)
        return queryset

    @action(detail=True, methods=['post'])
    def unverify(self, request, pk=None):
        doctor = self.get_object()
        doctor.doctor_status = 'UNVERIFIED'
        doctor.save()
        
        # Delete the license document record as well
        DoctorLicense.objects.filter(doctor=doctor).delete()
        
        return Response({
            "success": True, 
            "message": f"Doctor {doctor.email} has been unverified.",
            "user": self.get_serializer(doctor).data
        })

    def perform_destroy(self, instance):
        if instance == self.request.user:
            raise serializers.ValidationError({"error": "You cannot delete your own account."})
        instance.delete()

class AdminPatientListView(viewsets.ModelViewSet):
    queryset = CustomUser.objects.filter(role='PATIENT').order_by('-created_at')
    serializer_class = AdminUserSerializer
    permission_classes = [IsSystemAdmin]

    def perform_destroy(self, instance):
        if instance == self.request.user:
            raise serializers.ValidationError({"error": "You cannot delete your own account."})
        instance.delete()
