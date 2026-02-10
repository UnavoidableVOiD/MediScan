from django.contrib import admin
from .models import DoctorPatientLink, DoctorComment, DoctorLicense, SupportingDocument, DoctorAvailability, Appointment

class SupportingDocumentInline(admin.TabularInline):
    model = SupportingDocument
    extra = 1

@admin.register(DoctorLicense)
class DoctorLicenseAdmin(admin.ModelAdmin):
    list_display = ('doctor', 'license_number', 'status', 'submitted_at', 'updated_at')
    list_filter = ('status',)
    search_fields = ('doctor__email', 'license_number')
    inlines = [SupportingDocumentInline]

@admin.register(DoctorPatientLink)
class DoctorPatientLinkAdmin(admin.ModelAdmin):
    list_display = ('patient', 'doctor', 'status', 'linked_at')
    list_filter = ('status',)
    search_fields = ('patient__email', 'doctor__email')

@admin.register(DoctorComment)
class DoctorCommentAdmin(admin.ModelAdmin):
    list_display = ('report', 'doctor', 'created_at')
    search_fields = ('report__id', 'doctor__email')

@admin.register(DoctorAvailability)
class DoctorAvailabilityAdmin(admin.ModelAdmin):
    list_display = ('doctor', 'day_of_week', 'start_time', 'end_time', 'is_active')
    list_filter = ('day_of_week', 'is_active')
    search_fields = ('doctor__email',)

@admin.register(Appointment)
class AppointmentAdmin(admin.ModelAdmin):
    list_display = ('patient', 'doctor', 'appointment_date', 'status', 'amount_paid')
    list_filter = ('status', 'appointment_date')
    search_fields = ('patient__email', 'doctor__email', 'payment_id')

@admin.register(SupportingDocument)
class SupportingDocumentAdmin(admin.ModelAdmin):
    list_display = ('license', 'uploaded_at')
    list_filter = ('uploaded_at',)
    search_fields = ('license__doctor__email',)
