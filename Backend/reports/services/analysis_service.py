import logging
import requests
from django.conf import settings
from ..models import ReportResult

logger = logging.getLogger('security')

class AnalysisService:
    @staticmethod
    def run_analysis(report_instance, validated_data, user_id):
        """
        Calls the external FastAPI analysis service and saves the results.
        """
        try:
            # 1. Prepare data for FastAPI (ensure numeric values are sent as numbers)
            cleaned_data = {}
            for k, v in validated_data.items():
                try:
                    # Attempt to convert to float if it looks like a number
                    if isinstance(v, str) and (v.replace('.', '', 1).isdigit() or (v.startswith('-') and v[1:].replace('.', '', 1).isdigit())):
                        cleaned_data[k] = float(v)
                    else:
                        cleaned_data[k] = v
                except (ValueError, TypeError):
                    cleaned_data[k] = v

            analysis_payload = {
                "patient_data": cleaned_data,
                "user_id": str(user_id)
            }
            
            # 2. Call FastAPI Analysis Endpoint
            url = settings.ANALYSIS_API_URL
            if not url.startswith('http'):
                # Safety fallback if env var is relative
                base_url = getattr(settings, 'ML_SERVICE_URL', 'http://127.0.0.1:8001')
                url = f"{base_url.rstrip('/')}/{url.lstrip('/')}"
            
            print(f"DEBUG: Calling Analysis service at {url} with payload: {analysis_payload}")
            analysis_res = requests.post(url, json=analysis_payload)
            print(f"DEBUG: Analysis service response status: {analysis_res.status_code}")
            analysis_res.raise_for_status()
            analysis_data = analysis_res.json()
            
            # 3. Extract components from FastAPI response
            health_analysis = analysis_data.get('risk_assessment', {})
            critical_alerts = analysis_data.get('critical_alerts', [])
            summary_patient = analysis_data.get('summary_patient', '')
            summary_doctor = analysis_data.get('summary_doctor', '')
            
            # 4. Map health_analysis to key_findings and conditions
            key_findings = []
            conditions = []
            max_risk_score = 0
            
            for disease, result in health_analysis.items():
                if isinstance(result, dict) and 'prediction' in result:
                    pred = result['prediction']
                    score = result.get('risk_score', 0)
                    key_findings.append(f"{disease}: {pred} (Risk: {score}%)")
                    conditions.append({"name": disease, "details": pred})
                    if score > max_risk_score:
                        max_risk_score = score

            # 5. Determine Risk Level
            if critical_alerts or max_risk_score > 70:
                risk_level = 'High'
            elif max_risk_score > 30:
                risk_level = 'Medium'
            else:
                risk_level = 'Low'

            # 6. Save AI Result
            result_obj, created = ReportResult.objects.update_or_create(
                report=report_instance,
                defaults={
                    "summary": summary_patient,
                    "doctor_summary": summary_doctor,
                    "key_findings": key_findings,
                    "conditions": conditions,
                    "risk_level": risk_level,
                    "confidence_score": 95.0
                }
            )
            return True, result_obj

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"DEBUG: AI Analysis failed for report {report_instance.id}")
            print(f"DEBUG: Exception: {str(e)}")
            print(f"DEBUG: Traceback: {error_details}")
            logger.error(f"AI Analysis call failed for report {report_instance.id}: {str(e)}")
            return False, f"{str(e)} - {error_details[:200]}"
