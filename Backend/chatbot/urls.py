from django.urls import path
from .views import ChatStreamView

urlpatterns = [
    path('chat/', ChatStreamView.as_view(), name='chat-stream'),
]
