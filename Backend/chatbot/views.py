import httpx
from django.http import StreamingHttpResponse
from django.conf import settings
from rest_framework.views import APIView
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

class ChatStreamView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request, *args, **kwargs):
        question = request.data.get('question')
        patient_context = request.data.get('patient_context')

        if not question:
            return Response({"error": "Question is required"}, status=400)

        ml_service_url = getattr(settings, 'ML_SERVICE_URL', 'http://127.0.0.1:8001') or 'http://127.0.0.1:8001'
        chat_stream_url = f"{ml_service_url}/chat_stream"

        payload = {
            "question": question,
            "patient_context": patient_context,
            "is_premium": True  # Default to true for now as we don't have a premium field yet
        }

        async def stream_generator():
            async with httpx.AsyncClient(timeout=60.0) as client:
                try:
                    async with client.stream("POST", chat_stream_url, json=payload) as response:
                        if response.status_code != 200:
                            yield b"Error from AI Service: " + str(response.status_code).encode()
                            return
                            
                        async for chunk in response.aiter_bytes():
                            yield chunk
                except Exception as e:
                    yield f"Error connecting to AI Service: {str(e)}".encode()

        response = StreamingHttpResponse(stream_generator(), content_type="text/plain")
        response['X-Accel-Buffering'] = 'no'  # Disable Nginx buffering
        response['Cache-Control'] = 'no-cache'
        return response
