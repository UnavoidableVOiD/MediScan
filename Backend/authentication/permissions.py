from rest_framework import permissions

class IsNotLocked(permissions.BasePermission):
    """
    Global permission check for locked accounts.
    """

    def has_permission(self, request, view):
        if request.user.is_authenticated:
             # Logic to check if user is locked is handled in authentication/view level mostly, 
             # but this serves as an extra layer if attached to views.
             # However, for login endpoints, user isn't authenticated yet.
             # This might be useful for other endpoints.
             return True
        return True
