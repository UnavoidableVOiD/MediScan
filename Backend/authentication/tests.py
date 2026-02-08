from django.test import TestCase
from django.urls import reverse
from rest_framework import status
from rest_framework.test import APIClient
from unittest.mock import patch
from django.contrib.auth import get_user_model

User = get_user_model()

class GoogleLoginTest(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.url = reverse('google-login')

    @patch('google.oauth2.id_token.verify_oauth2_token')
    def test_google_login_new_user(self, mock_verify):
        mock_verify.return_value = {
            'email': 'newuser@gmail.com',
            'given_name': 'New',
            'family_name': 'User',
            'sub': '123456789'
        }

        response = self.client.post(self.url, {'token': 'header.payload.signature'}, format='json')

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['success'], True)
        self.assertIn('access', response.cookies)
        self.assertIn('refresh', response.cookies)
        
   
        user = User.objects.get(email='newuser@gmail.com')
        self.assertEqual(user.first_name, 'New')
        self.assertEqual(user.role, 'PATIENT')
        self.assertTrue(user.is_verified)

    @patch('google.oauth2.id_token.verify_oauth2_token')
    def test_google_login_existing_doctor_blocked(self, mock_verify):
        """Verify that existing doctors are blocked from Google Login."""
        User.objects.create_user(
            email='doctor@gmail.com',
            first_name='Doctor',
            last_name='Who',
            role='DOCTOR',
            specialization='CARDIOLOGY',
            password='Password123!',
            phone_number='+1234567890'
        )

        mock_verify.return_value = {
            'email': 'doctor@gmail.com',
            'given_name': 'Doctor',
            'family_name': 'Who',
            'sub': '987654321'
        }

        response = self.client.post(self.url, {'token': 'header.payload.signature'}, format='json')

        self.assertEqual(response.status_code, status.HTTP_403_FORBIDDEN)
        self.assertIn("Doctors cannot use Google Login", response.data['error'])

    def test_google_login_no_token(self):
        response = self.client.post(self.url, {}, format='json')
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

class DoctorRegistrationTest(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.url = reverse('register')

    def test_doctor_registration_missing_specialization(self):
        data = {
            "first_name": "Doctor",
            "last_name": "Strange",
            "email": "strange@example.com",
            "phone_number": "+9779812345678",
            "password": "Password123!",
            "confirm_password": "Password123!",
            "role": "DOCTOR"
        }
        response = self.client.post(self.url, data, format='json')
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn("specialization", response.data)

    def test_doctor_registration_with_specialization(self):
        data = {
            "first_name": "Doctor",
            "last_name": "House",
            "email": "house@example.com",
            "phone_number": "+9779812345679",
            "password": "Password123!",
            "confirm_password": "Password123!",
            "role": "DOCTOR",
            "specialization": "CARDIOLOGY"
        }
        response = self.client.post(self.url, data, format='json')
        # This view sends OTP, so 200 OK is expected
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['success'], True)
