from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import (
    DoctorListView, 
    MyDoctorView, 
    LinkDoctorView, 
    MyPatientsViewSet, 
    DoctorCommentViewSet
)

router = DefaultRouter()
router.register(r'my-patients', MyPatientsViewSet, basename='my-patients')
router.register(r'comments', DoctorCommentViewSet, basename='doctor-comments')

urlpatterns = [
    path('', include(router.urls)),
    path('list/', DoctorListView.as_view(), name='doctor-list'),
    path('my-doctor/', MyDoctorView.as_view(), name='my-doctor'),
    path('link/', LinkDoctorView.as_view(), name='link-doctor'),
]
