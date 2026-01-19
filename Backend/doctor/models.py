from django.db import models
from django.conf import settings

class DoctorPatientLink(models.Model):
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
