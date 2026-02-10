from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import DoctorVerificationViewSet, AdminLoginView, AdminDoctorListView, AdminPatientListView, AdminCreateView

router = DefaultRouter()
# Renamed from 'doctor-verifications' to 'verify-doctor'
router.register(r'verify-doctor', DoctorVerificationViewSet, basename='doctor-verifications')
router.register(r'doctors', AdminDoctorListView, basename='admin-doctors')
router.register(r'patients', AdminPatientListView, basename='admin-patients')

urlpatterns = [
    path('login/', AdminLoginView.as_view(), name='admin-login'),
    path('create-admin/', AdminCreateView.as_view(), name='admin-create'),
    path('', include(router.urls)),
]
