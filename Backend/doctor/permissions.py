from rest_framework import permissions


class IsDoctorRole(permissions.BasePermission):
    """
    Allows access to any authenticated user with the DOCTOR role,
    regardless of their clinical verification status.
    Use this for endpoints like license submission / verification.
    """
    def has_permission(self, request, view):
        return bool(
            request.user and
            request.user.is_authenticated and
            request.user.role == 'DOCTOR'
        )


class IsVerifiedDoctor(permissions.BasePermission):
    """
    Allows access only to doctors whose clinical profile is VERIFIED.
    Use this for clinical endpoints (patients, reports, comments).
    """
    def has_permission(self, request, view):
        return bool(
            request.user and
            request.user.is_authenticated and
            request.user.role == 'DOCTOR' and
            request.user.doctor_status == 'VERIFIED'
        )


# Keep backward-compatible alias
IsDoctor = IsVerifiedDoctor


class IsPatient(permissions.BasePermission):

    def has_permission(self, request, view):
        return bool(request.user and request.user.is_authenticated and request.user.role == 'PATIENT')
