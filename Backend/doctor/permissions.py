from rest_framework import permissions

class IsDoctor(permissions.BasePermission):
    """
    Allows access only to users with the 'DOCTOR' role.
    """
    def has_permission(self, request, view):
        return bool(request.user and request.user.is_authenticated and request.user.role == 'DOCTOR')

class IsPatient(permissions.BasePermission):
    """
    Allows access only to users with the 'PATIENT' role.
    """
    def has_permission(self, request, view):
        return bool(request.user and request.user.is_authenticated and request.user.role == 'PATIENT')
