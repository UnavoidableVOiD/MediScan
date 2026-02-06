from rest_framework import viewsets, generics, permissions, status
from rest_framework.decorators import action
from rest_framework.response import Response
from django.contrib.auth import get_user_model
from .models import DoctorPatientLink, DoctorComment
from .serializers import (
    DoctorPatientLinkSerializer, 
    DoctorCommentSerializer, 
    DoctorUserSerializer,
    PatientUserSerializer
)
from .permissions import IsDoctor, IsPatient
from reports.models import Report
from reports.serializers import ReportSerializer

User = get_user_model()

class DoctorListView(generics.ListAPIView):
    
    serializer_class = DoctorUserSerializer
    permission_classes = [permissions.IsAuthenticated]
    queryset = User.objects.filter(role='DOCTOR')

class MyDoctorView(generics.RetrieveAPIView):
    """
    API for patients to see their currently linked doctor.
    """
    serializer_class = DoctorPatientLinkSerializer
    permission_classes = [IsPatient]

    def get_object(self):
        return DoctorPatientLink.objects.filter(patient=self.request.user).first()

class LinkDoctorView(generics.CreateAPIView):
    """
    API for patients to link themselves to a doctor.
    """
    serializer_class = DoctorPatientLinkSerializer
    permission_classes = [IsPatient]

    def perform_create(self, serializer):
       
        if DoctorPatientLink.objects.filter(patient=self.request.user).exists():
          
            DoctorPatientLink.objects.filter(patient=self.request.user).delete()
        
        serializer.save(patient=self.request.user)

class MyPatientsViewSet(viewsets.ReadOnlyModelViewSet):
    """
    API for doctors to list their linked patients and view their reports.
    """
    serializer_class = PatientUserSerializer
    permission_classes = [IsDoctor]

    def get_queryset(self):
        links = DoctorPatientLink.objects.filter(doctor=self.request.user)
        patient_ids = links.values_list('patient_id', flat=True)
        return User.objects.filter(id__in=patient_ids)

    @action(detail=True, methods=['get'])
    def reports(self, request, pk=None):
        patient = self.get_object()
        reports = Report.objects.filter(user=patient).order_by('-uploaded_at')
        serializer = ReportSerializer(reports, many=True)
        return Response(serializer.data)

from rest_framework.decorators import action

class DoctorCommentViewSet(viewsets.ModelViewSet):
    """
    API for doctors to add/edit comments on patient reports.
    """
    serializer_class = DoctorCommentSerializer
    permission_classes = [IsDoctor]

    def get_queryset(self):
        return DoctorComment.objects.filter(doctor=self.request.user)

    def perform_create(self, serializer):
        serializer.save(doctor=self.request.user)
