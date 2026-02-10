from django.db import models
from django.conf import settings

class DoctorPatientLink(models.Model):
    STATUS_CHOICES = (
        ('ONGOING', 'Ongoing'),
        ('COMPLETED', 'Completed'),
    )

    patient = models.OneToOneField(
        settings.AUTH_USER_MODEL, 
        on_delete=models.CASCADE, 
        limit_choices_to={'role': 'PATIENT'},
        related_name='doctor_link'
    )
    doctor = models.ForeignKey(
        settings.AUTH_USER_MODEL, 
        on_delete=models.CASCADE, 
        limit_choices_to={'role': 'DOCTOR'},
        related_name='patient_links'
    )
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='ONGOING')
    notes = models.TextField(null=True, blank=True)
    linked_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = "Doctor-Patient Link"
        verbose_name_plural = "Doctor-Patient Links"

    def __str__(self):
        return f"P: {self.patient.email} -> D: {self.doctor.email}"


class DoctorComment(models.Model):
    report = models.OneToOneField(
        'reports.Report', 
        on_delete=models.CASCADE, 
        related_name='doctor_comment'
    )
    doctor = models.ForeignKey(
        settings.AUTH_USER_MODEL, 
        on_delete=models.CASCADE,
        limit_choices_to={'role': 'DOCTOR'}
    )
    comment = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Comment by {self.doctor.email} on Report {self.report.id}"


class DoctorLicense(models.Model):
    STATUS_CHOICES = (
        ('PENDING', 'Pending'),
        ('APPROVED', 'Approved'),
        ('REJECTED', 'Rejected'),
    )

    doctor = models.OneToOneField(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='license_info',
        limit_choices_to={'role': 'DOCTOR'}
    )
    license_number = models.CharField(max_length=100)
    license_file = models.FileField(upload_to='doctor_licenses/')
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='PENDING')
    rejection_reason = models.TextField(null=True, blank=True)
    
    submitted_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"License of {self.doctor.email} - {self.status}"


class SupportingDocument(models.Model):
    """Stores additional certificates/documents uploaded alongside a doctor license."""
    license = models.ForeignKey(
        DoctorLicense,
        on_delete=models.CASCADE,
        related_name='supporting_documents'
    )
    file = models.FileField(upload_to='doctor_certificates/')
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Document for {self.license.doctor.email}"


class DoctorAvailability(models.Model):
    DAY_CHOICES = (
        (0, 'Monday'),
        (1, 'Tuesday'),
        (2, 'Wednesday'),
        (3, 'Thursday'),
        (4, 'Friday'),
        (5, 'Saturday'),
        (6, 'Sunday'),
    )

    doctor = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='availability_slots',
        limit_choices_to={'role': 'DOCTOR'}
    )
    day_of_week = models.IntegerField(choices=DAY_CHOICES)
    start_time = models.TimeField()
    end_time = models.TimeField()
    is_active = models.BooleanField(default=True)

    class Meta:
        verbose_name = "Doctor Availability"
        verbose_name_plural = "Doctor Availabilities"
        unique_together = ('doctor', 'day_of_week', 'start_time', 'end_time')

    def __str__(self):
        return f"{self.doctor.email} - {self.get_day_of_week_display()} ({self.start_time}-{self.end_time})"


class Appointment(models.Model):
    STATUS_CHOICES = (
        ('PENDING', 'Pending Payment'),
        ('PAID', 'Paid / Confirmed'),
        ('CANCELLED', 'Cancelled'),
        ('COMPLETED', 'Completed'),
    )

    patient = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='appointments',
        limit_choices_to={'role': 'PATIENT'}
    )
    doctor = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='doctor_appointments',
        limit_choices_to={'role': 'DOCTOR'}
    )
    appointment_date = models.DateField()
    start_time = models.TimeField()
    end_time = models.TimeField()
    
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='PENDING')
    notes = models.TextField(null=True, blank=True)
    
    # Khalti / Payment tracking
    payment_id = models.CharField(max_length=100, null=True, blank=True, help_text="Transaction ID from Khalti")
    amount_paid = models.DecimalField(max_digits=10, decimal_places=2, default=0.00)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-appointment_date', '-start_time']

    def __str__(self):
        return f"Appt: {self.patient.email} with {self.doctor.email} on {self.appointment_date}"

