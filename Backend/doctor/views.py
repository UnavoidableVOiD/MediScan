from rest_framework import viewsets, generics, permissions, status
from rest_framework.decorators import action
from rest_framework.response import Response
from django.contrib.auth import get_user_model
from .models import DoctorPatientLink, DoctorComment, DoctorLicense, DoctorAvailability, Appointment
from .serializers import (
    DoctorPatientLinkSerializer,
    DoctorCommentSerializer, 
    DoctorUserSerializer,
    PatientUserSerializer,
    DoctorLicenseSerializer,
    DoctorAvailabilitySerializer,
    AppointmentSerializer
)
from .permissions import IsDoctor, IsDoctorRole, IsPatient
from reports.models import Report
from reports.serializers import ReportSerializer

User = get_user_model()

class DoctorListView(generics.ListAPIView):
    """
    List of all doctors, with optional filtering by specialization.
    """
    serializer_class = DoctorUserSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        queryset = User.objects.filter(role='DOCTOR')
        spec = self.request.query_params.get('specialization')
        if spec:
            queryset = queryset.filter(specialization=spec)
        return queryset

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
    queryset = User.objects.filter(role='PATIENT')

    def get_queryset(self):
        links = DoctorPatientLink.objects.filter(doctor=self.request.user)
        patient_ids = links.values_list('patient_id', flat=True)
        return User.objects.filter(id__in=patient_ids)

    @action(detail=False, methods=['get'])
    def stats(self, request):
        from django.utils.timezone import now
        from datetime import timedelta
        
        doctor = request.user
        links = DoctorPatientLink.objects.filter(doctor=doctor)
        
        stats = {
            "total_patients": links.count(),
            "ongoing_patients": links.filter(status='ONGOING').count(),
            "completed_patients": links.filter(status='COMPLETED').count(),
            "new_patients_7_days": links.filter(linked_at__gte=now() - timedelta(days=7)).count(),
        }
        return Response(stats)

    @action(detail=True, methods=['get'])
    def reports(self, request, pk=None):
        patient = self.get_object()
        reports = Report.objects.filter(user=patient).order_by('-uploaded_at')
        
        # Build enriched report list with analysis summary
        data = []
        for report in reports:
            report_data = ReportSerializer(report).data
            # Include AI Summary if it exists
            if hasattr(report, 'result'):
                from reports.serializers import ReportResultSerializer
                report_data['ai_analysis'] = ReportResultSerializer(report.result).data
            data.append(report_data)
            
        return Response(data)
    @action(detail=True, methods=['post'])
    def update_notes(self, request, pk=None):
        patient = self.get_object()
        doctor = request.user
        link = DoctorPatientLink.objects.filter(doctor=doctor, patient=patient).first()
        
        if not link:
            return Response({"error": "No link found"}, status=status.HTTP_404_NOT_FOUND)
            
        notes = request.data.get('notes')
        link.notes = notes
        link.save()
        return Response({"status": "Notes updated", "notes": notes})

class DoctorCommentViewSet(viewsets.ModelViewSet):
    """
    API for doctors to add/edit comments on patient reports.
    """
    serializer_class = DoctorCommentSerializer
    permission_classes = [IsDoctor]
    queryset = DoctorComment.objects.all()

    def get_queryset(self):
        return DoctorComment.objects.filter(doctor=self.request.user)

    def perform_create(self, serializer):
        serializer.save(doctor=self.request.user)

class DoctorLicenseView(generics.RetrieveUpdateAPIView):
    """
    API for doctors to upload and view their license status.
    PUT with: license_number, license_file, and optional supporting_documents files.
    """
    serializer_class = DoctorLicenseSerializer
    permission_classes = [IsDoctorRole]

    def get_object(self):
        try:
            return DoctorLicense.objects.get(doctor=self.request.user)
        except DoctorLicense.DoesNotExist:
            return None

    def update(self, request, *args, **kwargs):
        instance = self.get_object()
        if instance is None:
            # First-time submission → create
            serializer = self.get_serializer(data=request.data)
            serializer.is_valid(raise_exception=True)
            serializer.save(doctor=request.user, status='PENDING')
            # Update User doctor_status
            request.user.doctor_status = 'PENDING'
            request.user.save()
            return Response(serializer.data, status=201)
        else:
            # Re-submission → update
            serializer = self.get_serializer(instance, data=request.data, partial=True)
            serializer.is_valid(raise_exception=True)
            serializer.save(status='PENDING')
            request.user.doctor_status = 'PENDING'
            request.user.save()
            return Response(serializer.data)
class DoctorAvailabilityViewSet(viewsets.ModelViewSet):
    """
    API for doctors to manage their availability slots.
    """
    serializer_class = DoctorAvailabilitySerializer
    permission_classes = [IsDoctor]

    def get_queryset(self):
        return DoctorAvailability.objects.filter(doctor=self.request.user)

    @action(detail=False, methods=['post'])
    def sync(self, request):
        """
        Synchronize all availability slots for the doctor.
        Expected data: list of slots [{day_of_week: 0, start_time: \"09:00\", end_time: \"17:00\", is_active: true}, ...]
        """
        slots_data = request.data
        if not isinstance(slots_data, list):
            return Response({"error": "Expected a list of slots"}, status=status.HTTP_400_BAD_REQUEST)

        # Atomic transaction to ensure consistency
        from django.db import transaction
        from datetime import datetime
        try:
            # Group slots by day for overlap checking
            slots_by_day = {}
            for slot in slots_data:
                if not slot.get('is_active', True): continue
                day = slot.get('day_of_week')
                if day not in slots_by_day: slots_by_day[day] = []
                slots_by_day[day].append(slot)

            # Check for overlaps within each day
            for day, day_slots in slots_by_day.items():
                # Sort slots by start time
                sorted_slots = sorted(day_slots, key=lambda x: x.get('start_time'))
                for i in range(len(sorted_slots) - 1):
                    current_end = sorted_slots[i].get('end_time')
                    next_start = sorted_slots[i+1].get('start_time')
                    if current_end > next_start:
                        day_name = dict(DoctorAvailability.DAY_CHOICES).get(int(day), f"Day {day}")
                        raise ValueError(f"Overlapping slots detected on {day_name}: {sorted_slots[i]['start_time']}-{current_end} and {next_start}-{sorted_slots[i+1]['end_time']}")

            with transaction.atomic():
                # Delete existing slots for this doctor
                DoctorAvailability.objects.filter(doctor=request.user).delete()
                
                created_slots = []
                for slot in slots_data:
                    if not slot.get('is_active', True):
                        continue
                        
                    serializer = self.get_serializer(data=slot)
                    if serializer.is_valid():
                        serializer.save(doctor=request.user)
                        created_slots.append(serializer.data)
                    else:
                        raise ValueError(serializer.errors)
                
                return Response(created_slots)
        except ValueError as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            return Response({"error": "An error occurred during synchronization", "detail": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    def perform_create(self, serializer):
        serializer.save(doctor=self.request.user)


class AppointmentViewSet(viewsets.ModelViewSet):
    """
    API for patients to book appointments and for doctors to view them.
    """
    serializer_class = AppointmentSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        user = self.request.user
        if user.role == 'DOCTOR':
            return Appointment.objects.filter(doctor=user)
        return Appointment.objects.filter(patient=user)

    def perform_create(self, serializer):
        # Default status is PENDING until payment is verified
        serializer.save(patient=self.request.user)

    @action(detail=True, methods=['post'])
    def verify_payment(self, request, pk=None):
        """
        Verify Khalti payment for an appointment.
        Expected data: { "token": "...", "amount": 1000 }
        """
        appointment = self.get_object()
        token = request.data.get('token')
        amount = request.data.get('amount')

        if not token or not amount:
            return Response({"error": "Token and amount are required"}, status=status.HTTP_400_BAD_REQUEST)

        # Integration with Khalti API
        import requests
        from django.conf import settings

        url = "https://khalti.com/api/v2/payment/verify/"
        payload = {
            "token": token,
            "amount": amount
        }
        headers = {
            "Authorization": f"Key {settings.KHALTI_SECRET_KEY}"
        }

        try:
            response = requests.post(url, payload, headers=headers)
            if response.status_code == 200:
                resp_data = response.json()
                # Verify amount matches (Khalti amount is in paisa)
                if int(amount) == int(float(appointment.doctor.consultation_fee) * 100):
                    appointment.status = 'PAID'
                    appointment.payment_id = resp_data.get('idx')
                    appointment.amount_paid = float(amount) / 100
                    appointment.save()
                    return Response({"status": "Payment verified and appointment confirmed"})
                else:
                    return Response({"error": "Amount mismatch"}, status=status.HTTP_400_BAD_REQUEST)
            else:
                return Response({"error": "Khalti verification failed", "detail": response.json()}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            return Response({"error": "Connection error", "detail": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
