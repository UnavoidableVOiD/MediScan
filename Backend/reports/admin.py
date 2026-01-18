from django.contrib import admin
from .models import Report, ExtractedReportData

@admin.register(Report)
class ReportAdmin(admin.ModelAdmin):
    list_display = ('id', 'user', 'file', 'status', 'uploaded_at')
    list_filter = ('status', 'uploaded_at', 'user')
    search_fields = ('user__email', 'id')
    readonly_fields = ('uploaded_at',)

@admin.register(ExtractedReportData)
class ExtractedReportDataAdmin(admin.ModelAdmin):
    list_display = ('id', 'report', 'is_corrected', 'created_at', 'updated_at')
    list_filter = ('is_corrected', 'created_at', 'updated_at')
    search_fields = ('report__user__email', 'id')
    readonly_fields = ('created_at', 'updated_at')
