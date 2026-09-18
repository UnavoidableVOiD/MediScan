import httpx
import json
from django.conf import settings
from rest_framework.exceptions import APIException
from .models import Appointment

class KhaltiService:
    BASE_URL = "https://a.khalti.com/api/v2/epayment"
    
    @staticmethod
    def get_headers():
        return {
            "Authorization": f"Key df19d96325c548c09fdf0bf2aaf684b3",
            "Content-Type": "application/json",
        }

    @classmethod
    def initiate_payment(cls, appointment, return_url, website_url):
        url = f"{cls.BASE_URL}/initiate/"
        
        # Ensure amounts are integers (paisa)
        # Fallback to 1000 (Rs 10) if consultation_fee is missing/invalid for testing
        try:
            fee = float(appointment.doctor.consultation_fee)
        except (AttributeError, ValueError):
            # Try getting it from the appointment user profile or default
            # For now, let's assume a default if not found, or check if we can pass it
            fee = 100.0 # Default fallback
            
        amount_paisa = int(fee * 100)
        
        patient_name = "Guest"
        patient_email = "guest@example.com"
        patient_phone = ""

        if appointment.patient:
            patient_name = f"{appointment.patient.first_name} {appointment.patient.last_name}"
            patient_email = appointment.patient.email
            # Check for phone number
            if hasattr(appointment.patient, 'phone_number'):
                patient_phone = str(appointment.patient.phone_number)

        payload = {
            "return_url": return_url,
            "website_url": website_url,
            "amount": amount_paisa, 
            "purchase_order_id": str(appointment.id),
            "purchase_order_name": f"Appointment-{appointment.id}",
            "customer_info": {
                "name": patient_name,
                "email": patient_email,
                "phone": patient_phone
            }
        }
        
        try:
            response = httpx.post(url, headers=cls.get_headers(), json=payload, timeout=10.0)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            raise APIException(f"Khalti Init Failed: {e.response.text}")
        except Exception as e:
             raise APIException(f"Khalti Connection Failed: {str(e)}")

    @classmethod
    def verify_payment(cls, pidx):
        url = f"{cls.BASE_URL}/lookup/"
        payload = {"pidx": pidx}
        
        try:
            response = httpx.post(url, headers=cls.get_headers(), json=payload, timeout=10.0)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            raise APIException(f"Khalti Verification Failed: {e.response.text}")
        except Exception as e:
             raise APIException(f"Khalti Connection Failed: {str(e)}")
