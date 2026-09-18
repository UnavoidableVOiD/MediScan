from django.contrib import admin
from .models import Report, ExtractedReportData, ReportResult

@admin.register(Report)
class ReportAdmin(admin.ModelAdmin):
    list_display = ('id', 'user', 'status', 'uploaded_at')
    list_filter = ('status', 'uploaded_at')
    search_fields = ('user__email',)

@admin.register(ExtractedReportData)
class ExtractedReportDataAdmin(admin.ModelAdmin):
    list_display = ('report', 'is_corrected', 'created_at')
    list_filter = ('is_corrected',)
    search_fields = ('report__user__email',)

@admin.register(ReportResult)
class ReportResultAdmin(admin.ModelAdmin):
    list_display = ('report', 'risk_level', 'confidence_score', 'created_at')
    list_filter = ('risk_level',)
    search_fields = ('report__user__email',)
