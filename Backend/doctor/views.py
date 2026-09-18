from django.db import models
from rest_framework import viewsets, generics, permissions, status
from rest_framework.decorators import action
from rest_framework.response import Response
from django.contrib.auth import get_user_model
from .models import DoctorPatientLink, DoctorComment, DoctorLicense, DoctorAvailability, Appointment
from .serializers import (
    DoctorPatientLinkSerializer,
    DoctorCommentSerializer, 
    DoctorUserSerializer,
    DoctorUserDetailSerializer,
    PatientUserSerializer,
    DoctorLicenseSerializer,
    DoctorAvailabilitySerializer,
    AppointmentSerializer
)
from .permissions import IsDoctor, IsDoctorRole, IsPatient
from reports.models import Report
from reports.serializers import ReportSerializer

User = get_user_model()
class DoctorDetailView(generics.RetrieveAPIView):
    """
    API for anyone to view a doctor's public profile.
    """
    queryset = User.objects.filter(role='DOCTOR', doctor_status='VERIFIED')
    serializer_class = DoctorUserDetailSerializer
    permission_classes = [permissions.IsAuthenticated]

class DoctorListView(generics.ListAPIView):
    """
    List of all doctors, with optional filtering by specialization.
    """
    serializer_class = DoctorUserSerializer
    permission_classes = [permissions.AllowAny]

    def get_queryset(self):
        queryset = User.objects.filter(role='DOCTOR', doctor_status='VERIFIED')
        spec = self.request.query_params.get('specialization')
        risk_level = self.request.query_params.get('risk_level')
        
        if spec:
            # Handle comma-separated list of specializations
            spec_list = spec.split(',')
            queryset = queryset.filter(specialization__in=spec_list)
        
        # If everything is normal (Low risk), give one slot of recommended
        if risk_level == 'Low':
            return queryset[:1]
            
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
    permission_classes = [IsDoctorRole]
    queryset = User.objects.filter(role='PATIENT')

    def get_queryset(self):
        link_ids = DoctorPatientLink.objects.filter(doctor=self.request.user).values_list('patient_id', flat=True)
        appt_ids = Appointment.objects.filter(
            doctor=self.request.user, status__in=['PAID', 'COMPLETED']
        ).values_list('patient_id', flat=True)
        all_ids = set(link_ids) | set(appt_ids)
        return User.objects.filter(id__in=all_ids)

    @action(detail=False, methods=['get'])
    def stats(self, request):
        from django.utils.timezone import now
        from datetime import timedelta
        
        from django.db.models import Sum
        
        doctor = request.user
        
        # Unique patients
        link_ids = DoctorPatientLink.objects.filter(doctor=doctor).values_list('patient_id', flat=True)
        appt_ids = Appointment.objects.filter(
            doctor=doctor, 
            status__in=['PAID', 'COMPLETED']
        ).values_list('patient_id', flat=True)
        
        all_patient_ids = set(link_ids) | set(appt_ids)
        total_patients = len(all_patient_ids)
        
        ongoing_count = DoctorPatientLink.objects.filter(doctor=doctor, status='ONGOING').count()
        completed_count = DoctorPatientLink.objects.filter(doctor=doctor, status='COMPLETED').count()
        
        appt_only_ongoing = Appointment.objects.filter(
            doctor=doctor, status='PAID'
        ).exclude(patient_id__in=link_ids).values('patient_id').distinct().count()

        # Revenue statistics
        # Actual gross is what remains after refunds
        revenue_data = Appointment.objects.filter(
            doctor=doctor,
            status__in=['PAID', 'COMPLETED', 'CANCELLED']
        ).aggregate(
            total_paid=Sum('amount_paid'),
            total_refunded=Sum('refund_amount'),
            total_net_stored=Sum('doctor_revenue'),
            total_comm_stored=Sum('admin_revenue')
        )
        
        paid = float(revenue_data['total_paid'] or 0)
        refunded = float(revenue_data['total_refunded'] or 0)
        retained = paid - refunded
        
        # Fallback for historical data
        stored_net = float(revenue_data['total_net_stored'] or 0)
        final_net = stored_net if stored_net > 0 else retained * 0.75
        
        stored_comm = float(revenue_data['total_comm_stored'] or 0)
        final_comm = stored_comm if stored_comm > 0 else retained * 0.25
        
        stats = {
            "total_patients": total_patients,
            "ongoing_patients": ongoing_count + appt_only_ongoing,
            "completed_patients": completed_count,
            "new_patients_7_days": DoctorPatientLink.objects.filter(doctor=doctor, linked_at__gte=now() - timedelta(days=7)).count(),
            "revenue": {
                "total_gross": retained,
                "total_net": final_net,
                "total_commission": final_comm,
                "total_refunded": refunded,
            }
        }
        return Response(stats)

    @action(detail=True, methods=['get'])
    def reports(self, request, pk=None):
        doctor = request.user
        try:
            patient = User.objects.get(pk=pk, role='PATIENT')
        except User.DoesNotExist:
            return Response({"detail": "Patient not found."}, status=status.HTTP_404_NOT_FOUND)

        has_link = DoctorPatientLink.objects.filter(doctor=doctor, patient=patient).exists()
        has_appointment = Appointment.objects.filter(doctor=doctor, patient=patient).exists()

        if not has_link and not has_appointment:
            return Response({"detail": "You do not have access to this patient's reports."}, status=status.HTTP_403_FORBIDDEN)

        reports = Report.objects.filter(user=patient).order_by('-uploaded_at')
        return Response(ReportSerializer(reports, many=True, context={'request': request}).data)

    @action(detail=True, methods=['post'])
    def complete(self, request, pk=None):
        patient = self.get_object()
        doctor = request.user
        link = DoctorPatientLink.objects.filter(doctor=doctor, patient=patient).first()
        
        updated = False
        if link:
            link.status = 'COMPLETED'
            link.save()
            updated = True

        # Also mark any active PAID appointments for this patient and doctor as COMPLETED
        appt_count = Appointment.objects.filter(
            doctor=doctor, 
            patient=patient, 
            status='PAID'
        ).update(status='COMPLETED')
        
        if appt_count > 0:
            updated = True
            
        if not updated:
            return Response({"error": "No link or active appointments found to complete"}, status=status.HTTP_404_NOT_FOUND)

        return Response({"status": "Patient session marked as completed"})

    @action(detail=True, methods=['post'])
    def update_observations(self, request, pk=None):
        patient = self.get_object()
        doctor = request.user
        link = DoctorPatientLink.objects.filter(doctor=doctor, patient=patient).first()
        if not link:
            return Response({"error": "No link found"}, status=status.HTTP_404_NOT_FOUND)
        observations = request.data.get('clinical_observations')
        link.clinical_observations = observations
        link.save()
        return Response({"status": "Observations updated", "clinical_observations": observations})
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
    Patients can read comments on their own reports.
    """
    serializer_class = DoctorCommentSerializer
    queryset = DoctorComment.objects.all()

    def get_permissions(self):
        if self.action in ['list', 'retrieve']:
            return [permissions.IsAuthenticated()]
        return [IsDoctor()]

    def get_queryset(self):
        user = self.request.user
        if user.role == 'DOCTOR':
            return DoctorComment.objects.filter(doctor=user)
        # Patients only see comments on their own reports
        return DoctorComment.objects.filter(report__user=user)

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
    API for doctors to manage their availability slots and patients to view them.
    Admins can also manage slots for any doctor.
    """
    serializer_class = DoctorAvailabilitySerializer

    def get_permissions(self):
        if self.action in ['list', 'retrieve']:
            return [permissions.IsAuthenticated()]
        # Allow Doctor to manage own, or Admin to manage any
        if self.action in ['create', 'update', 'partial_update', 'destroy', 'sync']:
            return [permissions.IsAuthenticated()] 
        return [IsDoctorRole()]

    def get_queryset(self):
        queryset = DoctorAvailability.objects.all()
        doctor_id = self.request.query_params.get('doctor')
        
        if doctor_id:
            queryset = queryset.filter(doctor_id=doctor_id)
        elif self.request.user.is_authenticated and self.request.user.role == 'DOCTOR':
            queryset = queryset.filter(doctor=self.request.user)
        else:
            # Patients/Admins must provide a doctor_id to see slots (or Admins see all?)
            # For list view without params, maybe Admins want to see all? 
            # But usually we filter by doctor.
            if self.request.user.is_staff:
                return queryset
            queryset = queryset.none()
            
        return queryset.filter(is_active=True)

    @action(detail=False, methods=['post'])
    def sync(self, request):
        """
        Synchronize all availability slots for the doctor.
        Expected data: list of slots [{day_of_week: 0, start_time: "09:00", end_time: "17:00", is_active: true}, ...]
        Admins must provide 'doctor_id' in query param or body to specify which doctor.
        """
        slots_data = request.data
        if not isinstance(slots_data, list):
            return Response({"error": "Expected a list of slots"}, status=status.HTTP_400_BAD_REQUEST)

        # Determine target doctor
        target_doctor = request.user
        if request.user.is_staff:
            doc_id = request.query_params.get('doctor') or request.data[0].get('doctor') if len(slots_data) > 0 and 'doctor' in request.data[0] else None
            # If passed as a separate param in body (not standard for list), check query params.
            # Best is query param `?doctor=ID` for sync.
            if doc_id:
                try:
                    target_doctor = User.objects.get(id=doc_id, role='DOCTOR')
                except User.DoesNotExist:
                    return Response({"error": "Doctor not found"}, status=status.HTTP_404_NOT_FOUND)
            # If admin but no doc_id, fail or default to self? Fail is safer.
            elif not request.user.role == 'DOCTOR':
                 return Response({"error": "Doctor ID required for admin sync"}, status=status.HTTP_400_BAD_REQUEST)

        # Atomic transaction to ensure consistency
        from django.db import transaction
        from datetime import datetime
        try:
            # Key is strictly 'date:YYYY-MM-DD'
            slots_by_key = {}
            for slot in slots_data:
                if not slot.get('is_active', True): continue
                
                date_val = slot.get('date')
                if not date_val:
                    raise ValueError("Date is required for all slots. Weekly day-of-week slots are no longer supported.")
                
                key = f"date:{date_val}"
                if key not in slots_by_key: slots_by_key[key] = []
                slots_by_key[key].append(slot)

            # Check for overlaps within each group
            for key, group_slots in slots_by_key.items():
                sorted_slots = sorted(group_slots, key=lambda x: x.get('start_time'))
                for i in range(len(sorted_slots) - 1):
                    current_end = sorted_slots[i].get('end_time')
                    next_start = sorted_slots[i+1].get('start_time')
                    if current_end > next_start:
                        raise ValueError(f"Overlapping slots detected on {key.split(':')[1]}: {sorted_slots[i]['start_time']}-{current_end} and {next_start}-{sorted_slots[i+1]['end_time']}")

            with transaction.atomic():
                # Delete existing slots for this doctor
                DoctorAvailability.objects.filter(doctor=target_doctor).delete()
                
                created_slots = []
                for slot in slots_data:
                    if not slot.get('is_active', True):
                        continue
                        
                    # Remove id if present and ensure day_of_week is null
                    slot.pop('id', None)
                    slot['day_of_week'] = None 
                    
                    serializer = self.get_serializer(data=slot)
                    serializer.is_valid(raise_exception=True)
                    serializer.save(doctor=target_doctor)
                    created_slots.append(serializer.data)
                
                return Response(created_slots)
        except ValueError as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            import traceback
            traceback.print_exc()
            return Response({"error": "An error occurred during synchronization", "detail": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    def perform_create(self, serializer):
        user = self.request.user
        if not self.request.data.get('date'):
             return Response({"error": "Date is required"}, status=status.HTTP_400_BAD_REQUEST)
        
        if user.is_staff and 'doctor' in self.request.data:
             doc_id = self.request.data.get('doctor')
             doctor = User.objects.get(id=doc_id)
             serializer.save(doctor=doctor, day_of_week=None)
        else:
            serializer.save(doctor=user, day_of_week=None)


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

    def create(self, request, *args, **kwargs):
        doctor_id = request.data.get('doctor')
        date = request.data.get('appointment_date')
        start_time = request.data.get('start_time')

        # 1. Check to prevent the same patient from booking overlapping appointments
        if Appointment.objects.filter(
            patient=request.user,
            appointment_date=date,
            start_time=start_time,
            status__in=['PAID', 'PENDING']
        ).exists():
            return Response(
                {"error": "You already have an appointment or pending payment for this time slot."}, 
                status=status.HTTP_400_BAD_REQUEST
            )

        # 2. Check to prevent double-booking the same slot for the doctor
        if Appointment.objects.filter(
            doctor_id=doctor_id, 
            appointment_date=date, 
            start_time=start_time, 
            status__in=['PAID', 'PENDING']
        ).exists():
            return Response(
                {"error": "This time slot has already been booked."}, 
                status=status.HTTP_400_BAD_REQUEST
            )
            
        return super().create(request, *args, **kwargs)

    @action(detail=False, methods=['get'])
    def booked_slots(self, request):
        """Return booked time slots for a given doctor + date."""
        doctor_id = request.query_params.get('doctor')
        date = request.query_params.get('date')
        if not doctor_id or not date:
            return Response({"error": "doctor and date params required"}, status=status.HTTP_400_BAD_REQUEST)
        booked = Appointment.objects.filter(
            doctor_id=doctor_id,
            appointment_date=date,
            status__in=['PAID', 'PENDING']
        ).values('start_time', 'end_time')
        return Response(list(booked))

    def perform_create(self, serializer):
        # Default status is PENDING until payment is verified
        serializer.save(patient=self.request.user)

    @action(detail=True, methods=['post'])
    def complete(self, request, pk=None):
        appointment = self.get_object()
        if appointment.doctor != request.user:
            return Response({"error": "Permission denied"}, status=status.HTTP_403_FORBIDDEN)
        appointment.status = 'COMPLETED'
        appointment.save()
        return Response({"status": "Appointment marked as completed"})

    @action(detail=True, methods=['post'])
    def cancel(self, request, pk=None):
        appointment = self.get_object()
        
        # Security: Only the patient who booked it can cancel
        if appointment.patient != request.user:
            return Response({"error": "You can only cancel your own appointments."}, status=status.HTTP_403_FORBIDDEN)
            
        if appointment.status in ['CANCELLED', 'COMPLETED']:
            return Response({"error": "Appointment cannot be cancelled in its current state."}, status=status.HTTP_400_BAD_REQUEST)

        from django.utils import timezone
        from datetime import datetime, date, time
        
        # Calculate time until appointment
        appt_datetime = timezone.make_aware(datetime.combine(appointment.appointment_date, appointment.start_time))
        now = timezone.now()
        
        time_diff = appt_datetime - now
        
        # Refund logic: 60% if > 24 hours
        refund_eligible = time_diff.total_seconds() > 24 * 3600
        
        appointment.status = 'CANCELLED'
        if refund_eligible and appointment.amount_paid > 0:
            appointment.refund_amount = float(appointment.amount_paid) * 0.6
            # Recalculate revenue based on the retained 40%
            retained_amount = float(appointment.amount_paid) * 0.4
            appointment.admin_revenue = retained_amount * 0.25
            appointment.doctor_revenue = retained_amount * 0.75
        else:
            appointment.refund_amount = 0
            # If no refund, doctor and admin keep 100% of their split
            # (Revenue already calculated in verify_payment)
            
        appointment.save()
        
        msg = "Appointment cancelled."
        if refund_eligible:
            msg += f" A refund of {appointment.refund_amount} (60%) has been initiated."
        else:
            msg += " No refund is applicable as cancellation is within 24 hours."
            
        return Response({
            "status": "CANCELLED",
            "message": msg,
            "refund_amount": appointment.refund_amount
        })

    @action(detail=True, methods=['post'])
    def update_observations(self, request, pk=None):
        appointment = self.get_object()
        if appointment.doctor != request.user:
            return Response({"error": "Permission denied"}, status=status.HTTP_403_FORBIDDEN)
        observations = request.data.get('clinical_observations')
        appointment.clinical_observations = observations
        appointment.save()
        return Response({"status": "Observations updated", "clinical_observations": observations})

    @action(detail=True, methods=['post'])
    def verify_payment(self, request, pk=None):
        """
        Verify Khalti payment for an appointment and distribute revenue.
        Expected data: { "token": "...", "amount": 1000 }
        """
        appointment = self.get_object()
        token = request.data.get('token')
        amount = request.data.get('amount')

        if not token or not amount:
            return Response({"error": "Token and amount are required"}, status=status.HTTP_400_BAD_REQUEST)

        # Double check availability before confirming to prevent race conditions
        if Appointment.objects.filter(
            doctor=appointment.doctor,
            appointment_date=appointment.appointment_date,
            start_time=appointment.start_time,
            status='PAID'
        ).exclude(id=appointment.id).exists():
            return Response(
                {"error": "This slot was just booked by someone else. Please contact support for a refund."}, 
                status=status.HTTP_400_BAD_REQUEST
            )

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
                expected_amount = int(float(appointment.doctor.consultation_fee) * 100)
                if int(amount) == expected_amount:
                    total_paid = float(amount) / 100
                    appointment.status = 'PAID'
                    appointment.payment_id = resp_data.get('idx')
                    appointment.amount_paid = total_paid
                    
                    # 25% Admin Commission, 75% Doctor Revenue
                    appointment.admin_revenue = total_paid * 0.25
                    appointment.doctor_revenue = total_paid * 0.75
                    
                    appointment.save()
                    return Response({"status": "Payment verified and revenue distributed."})
                else:
                    return Response({
                        "error": "Amount mismatch", 
                        "expected": expected_amount, 
                        "received": amount
                    }, status=status.HTTP_400_BAD_REQUEST)
            else:
                return Response({"error": "Khalti verification failed", "detail": response.json()}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            return Response({"error": "Connection error", "detail": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    @action(detail=False, methods=['get'])
    def admin_financial_stats(self, request):
        """Financial overview for administrators."""
        if not request.user.is_staff:
            return Response({"error": "Admin access required"}, status=status.HTTP_403_FORBIDDEN)
            
        from django.db.models import Sum, Count
        
        # Overall totals
        overall_raw = Appointment.objects.aggregate(
            total_paid=Sum('amount_paid'),
            total_refunded=Sum('refund_amount'),
            total_admin_revenue=Sum('admin_revenue'),
            total_doctor_revenue=Sum('doctor_revenue'),
            paid_count=Count('id', filter=models.Q(status__in=['PAID', 'COMPLETED'])),
            cancelled_count=Count('id', filter=models.Q(status='CANCELLED'))
        )
        
        # Calculate totals
        paid = float(overall_raw['total_paid'] or 0)
        refunded = float(overall_raw['total_refunded'] or 0)
        retained = paid - refunded
        
        # Fallback for historical data: if stored revenue is zero but payments exist
        stored_admin = float(overall_raw['total_admin_revenue'] or 0)
        final_admin_revenue = stored_admin if stored_admin > 0 else retained * 0.25
        
        stored_doctor = float(overall_raw['total_doctor_revenue'] or 0)
        final_doctor_revenue = stored_doctor if stored_doctor > 0 else retained * 0.75

        overall = {
            "total_gross": retained,
            "total_paid": paid,
            "total_refunded": refunded,
            "total_admin_revenue": final_admin_revenue,
            "total_doctor_revenue": final_doctor_revenue,
            "paid_count": overall_raw['paid_count'],
            "cancelled_count": overall_raw['cancelled_count'],
        }
        
        # Breakdown by doctor
        doctor_breakdown = Appointment.objects.values(
            'doctor__id', 'doctor__first_name', 'doctor__last_name', 'doctor__email'
        ).annotate(
            paid=Sum('amount_paid'),
            refunded=Sum('refund_amount'),
            admin_share=Sum('admin_revenue'),
            doctor_share=Sum('doctor_revenue'),
            appt_count=Count('id')
        ).order_by('-admin_share')
        
        # Adjust per-doctor gross and calculate shares if missing
        breakdown_list = []
        for doc in doctor_breakdown:
            doc_paid = float(doc['paid'] or 0)
            doc_refunded = float(doc['refunded'] or 0)
            doc_retained = doc_paid - doc_refunded
            
            doc['gross'] = doc_retained
            
            # Fallback for missing/zero shares
            current_admin_share = float(doc['admin_share'] or 0)
            if current_admin_share == 0 and doc_retained > 0:
                doc['admin_share'] = doc_retained * 0.25
                doc['doctor_share'] = doc_retained * 0.75
            else:
                doc['admin_share'] = current_admin_share
                doc['doctor_share'] = float(doc['doctor_share'] or 0)
                
            breakdown_list.append(doc)
        
        return Response({
            "overall": overall,
            "doctor_breakdown": breakdown_list
        })
