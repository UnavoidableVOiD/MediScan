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
            # 1. Prepare data for FastAPI
            analysis_payload = {
                "patient_data": validated_data,
                "user_id": str(user_id)
            }
            
            # 2. Call FastAPI Analysis Endpoint
            analysis_res = requests.post(settings.ANALYSIS_API_URL, json=analysis_payload)
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
            logger.error(f"AI Analysis call failed for report {report_instance.id}: {str(e)}")
            return False, str(e)
