from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status, permissions
from .models import Appointment
from .services import KhaltiService
import logging

logger = logging.getLogger(__name__)

class KhaltiInitView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request):
        appointment_id = request.data.get('appointmentId')
        return_url = request.data.get('returnUrl')
        website_url = request.data.get('websiteUrl')

        if not appointment_id or not return_url or not website_url:
            return Response({'error': 'Missing required fields (appointmentId, returnUrl, websiteUrl)'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            appointment = Appointment.objects.get(id=appointment_id)
        except Appointment.DoesNotExist:
            return Response({'error': 'Appointment not found'}, status=status.HTTP_404_NOT_FOUND)

        if appointment.patient != request.user:
             return Response({'error': 'Not authorized to pay for this appointment'}, status=status.HTTP_403_FORBIDDEN)

        try:
            payment_data = KhaltiService.initiate_payment(appointment, return_url, website_url)
            
            # Save pidx immediately to associate appointment with this payment attempt
            pidx = payment_data.get('pidx')
            if pidx:
                appointment.payment_id = pidx
                appointment.save()
                
            return Response(payment_data)
        except Exception as e:
            # Cleanup: Delete appointment if initiation fails
            appointment.delete()
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)

class KhaltiVerifyView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request):
        pidx = request.data.get('pidx')
        
        if not pidx:
             return Response({'error': 'Missing pidx'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            verification_data = KhaltiService.verify_payment(pidx)
            print(f"DEBUG: Verification Data Raw: {verification_data}") 
            
            appointment_id = verification_data.get('purchase_order_id')
            status_val = verification_data.get('status')
            
            appointment = None
            
            # Try finding appointment by purchase_order_id first
            if appointment_id:
                try:
                    appointment = Appointment.objects.get(id=appointment_id)
                except Appointment.DoesNotExist:
                    print(f"DEBUG: Appointment {appointment_id} NOT FOUND by ID.")
            
            # Fallback: Find by pidx (payment_id) if not found by ID
            if not appointment:
                print(f"DEBUG: Attempting lookup by payment_id (pidx): {pidx}")
                try:
                    appointment = Appointment.objects.get(payment_id=pidx)
                    print(f"DEBUG: Found appointment {appointment.id} via pidx")
                except Appointment.DoesNotExist:
                     print(f"DEBUG: Appointment with pidx {pidx} NOT FOUND.")
            
            if not appointment:
                return Response({'error': f'Appointment not found for pidx {pidx} or purchase_order_id {appointment_id}'}, status=status.HTTP_404_NOT_FOUND)

            if status_val == 'Completed':
                 if appointment.status != 'PAID':
                     appointment.status = 'PAID'
                     # payment_id is likely already set, but ensure it matches
                     appointment.payment_id = pidx
                     appointment.amount_paid = float(verification_data.get('total_amount', 0)) / 100
                     appointment.save()
            else:
                # Cleanup: Delete appointment if payment is not Completed
                print(f"DEBUG: Payment status {status_val}. Deleting appointment {appointment.id}")
                appointment.delete()
                return Response({'status': 'Failed', 'message': f'Payment failed with status: {status_val}. Appointment cancelled.'})
                 
            return Response(verification_data)

        except Exception as e:
            print(f"DEBUG: Exception in verify: {e}")
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)
