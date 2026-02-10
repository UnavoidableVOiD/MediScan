from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import (
    DoctorListView, 
    MyDoctorView, 
    LinkDoctorView, 
    MyPatientsViewSet, 
    DoctorCommentViewSet,
    DoctorLicenseView,
    DoctorAvailabilityViewSet,
    AppointmentViewSet
)

router = DefaultRouter()
router.register(r'my-patients', MyPatientsViewSet, basename='my-patients')
router.register(r'comments', DoctorCommentViewSet, basename='doctor-comments')
router.register(r'availability', DoctorAvailabilityViewSet, basename='doctor-availability')
router.register(r'appointments', AppointmentViewSet, basename='appointments')

urlpatterns = [
    path('', include(router.urls)),
    path('list/', DoctorListView.as_view(), name='doctor-list'),
    path('my-doctor/', MyDoctorView.as_view(), name='my-doctor'),
    path('link/', LinkDoctorView.as_view(), name='link-doctor'),
    path('verify/', DoctorLicenseView.as_view(), name='doctor-verify-upload'),
]
