from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from .models import CustomUser

class CustomUserAdmin(UserAdmin):
    ordering = ('email',)
    list_display = ('email', 'first_name', 'last_name', 'role', 'doctor_status', 'is_staff', 'is_active')
    list_filter = ('role', 'doctor_status', 'is_staff', 'is_active')
    search_fields = ('email', 'first_name', 'last_name')
    
    fieldsets = (
        (None, {'fields': ('email', 'password')}),
        ('Personal info', {'fields': ('first_name', 'last_name', 'phone_number', 'role', 'specialization', 'doctor_status')}),
        ('Permissions', {'fields': ('is_active', 'is_staff', 'is_superuser', 'groups', 'user_permissions')}),
    )
    add_fieldsets = (
        (None, {
            'classes': ('wide',),
            'fields': ('email', 'password', 'first_name', 'last_name', 'role', 'specialization', 'doctor_status', 'is_active'),
        }),
    )
    
    readonly_fields = ('created_at', 'updated_at', 'last_login')

admin.site.register(CustomUser, CustomUserAdmin)
