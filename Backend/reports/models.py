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

    def __str__(self):
        return f"Data for Report {self.report.id}"
