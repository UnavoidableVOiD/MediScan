from django.contrib import admin
from .models import Report, ExtractedReportData, ReportResult

admin.site.register(Report)
admin.site.register(ExtractedReportData)
admin.site.register(ReportResult)
