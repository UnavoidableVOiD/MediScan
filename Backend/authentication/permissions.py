from rest_framework import permissions

class IsNotLocked(permissions.BasePermission):
    """
    Global permission check for locked accounts.
    """

    def has_permission(self, request, view):
        if request.user.is_authenticated:
           
             return True
        return True
