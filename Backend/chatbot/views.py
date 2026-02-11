import httpx
from django.http import StreamingHttpResponse
from django.conf import settings
from rest_framework.views import APIView
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from reports.models import ExtractedReportData

class ChatStreamView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request, *args, **kwargs):
        question = request.data.get('question')
        report_id = request.data.get('report_id')
        patient_context = request.data.get('patient_context')

        # If report_id is provided, fetch all relevant data to use as context
        if report_id:
            try:
                # 1. Get Extracted Values
                report_data = ExtractedReportData.objects.filter(
                    report_id=report_id,
                    report__user=request.user
                ).first()
                
                # 2. Get AI Analysis Result
                from reports.models import ReportResult
                report_res = ReportResult.objects.filter(report_id=report_id).first()

                context_dict = {}
                
                if report_data:
                    context_dict['lab_values'] = report_data.final_data or report_data.raw_ocr_data
                
                if report_res:
                    context_dict['analysis'] = {
                        'summary': report_res.summary,
                        'conditions': report_res.conditions,
                        'risk_level': report_res.risk_level,
                        'key_findings': report_res.key_findings,
                        'specialization': report_res.suggested_specialization
                    }

                if context_dict:
                    patient_context = context_dict

            except Exception as e:
                # Log error but don't fail, use provided context if any
                print(f"Error fetching report context: {str(e)}")
        if not question:
            return Response({"error": "Question is required"}, status=400)

        ml_service_url = getattr(settings, 'ML_SERVICE_URL', 'http://127.0.0.1:8001') or 'http://127.0.0.1:8001'
        chat_stream_url = f"{ml_service_url}/chat_stream"

        # Ensure patient_context is a dict or None for FastAPI validation
        if not isinstance(patient_context, dict):
            patient_context = None

        payload = {
            "question": question,
            "patient_context": patient_context,
            "is_premium": True
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
