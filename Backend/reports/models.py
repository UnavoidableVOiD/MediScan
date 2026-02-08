from django.db import models
from django.contrib.auth import get_user_model
from django.utils.translation import gettext_lazy as _

User = get_user_model()

class Report(models.Model):
    STATUS_CHOICES = (
        ('PENDING', 'Pending'),
        ('PROCESSED', 'Processed'),
        ('FAILED', 'Failed'),
    )

    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='reports')
    file = models.FileField(upload_to='reports/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='PENDING')

    def __str__(self):
        return f"Report {self.id} - {self.user.email}"


class ExtractedReportData(models.Model):
    report = models.OneToOneField(Report, on_delete=models.CASCADE, related_name='extracted_data')
    raw_ocr_data = models.JSONField(help_text="Original data returned by OCR engine")
    final_data = models.JSONField(help_text="Editable data for frontend/backend usage", null=True, blank=True)
    is_corrected = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)


class ReportResult(models.Model):
    report = models.OneToOneField(Report, on_delete=models.CASCADE, related_name='result')
    summary = models.TextField(help_text="AI generated summary for the patient")
    doctor_summary = models.TextField(help_text="AI generated summary for the doctor", null=True, blank=True)
    key_findings = models.JSONField(help_text="List of key findings")
    conditions = models.JSONField(help_text="List of detected conditions")
    risk_level = models.CharField(max_length=20, choices=[('Low', 'Low'), ('Medium', 'Medium'), ('High', 'High')])
    confidence_score = models.FloatField(default=0.0)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Result for Report {self.report.id}"

