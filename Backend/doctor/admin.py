from django.contrib import admin
from .models import DoctorPatientLink, DoctorComment

@admin.register(DoctorPatientLink)
class DoctorPatientLinkAdmin(admin.ModelAdmin):
    list_display = ('patient', 'doctor', 'linked_at')
    search_fields = ('patient__email', 'doctor__email')

@admin.register(DoctorComment)
class DoctorCommentAdmin(admin.ModelAdmin):
    list_display = ('report', 'doctor', 'created_at')
    search_fields = ('report__id', 'doctor__email')
