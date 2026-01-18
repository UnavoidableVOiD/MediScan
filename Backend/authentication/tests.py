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

        response = self.client.post(self.url, {'token': 'valid-token'}, format='json')

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['success'], True)
        self.assertIn('access', response.data)
        
   
        user = User.objects.get(email='newuser@gmail.com')
        self.assertEqual(user.first_name, 'New')
        self.assertEqual(user.role, 'PATIENT')
        self.assertTrue(user.is_verified)

    @patch('google.oauth2.id_token.verify_oauth2_token')
    def test_google_login_existing_user(self, mock_verify):
       
        User.objects.create_user(
            email='existing@gmail.com',
            first_name='Existing',
            last_name='User',
            role='DOCTOR',
            password='Password123!',
            phone_number='+1234567890'
        )

        mock_verify.return_value = {
            'email': 'existing@gmail.com',
            'given_name': 'Existing',
            'family_name': 'User',
            'sub': '987654321'
        }

        response = self.client.post(self.url, {'token': 'valid-token'}, format='json')

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        user = User.objects.get(email='existing@gmail.com')
        self.assertEqual(user.role, 'DOCTOR') 

    def test_google_login_no_token(self):
        response = self.client.post(self.url, {}, format='json')
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
