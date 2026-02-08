from rest_framework import status, views, permissions
from rest_framework.response import Response
from django.contrib.auth import get_user_model
from django.core.cache import cache
from rest_framework_simplejwt.tokens import RefreshToken
from rest_framework_simplejwt.views import TokenRefreshView
from rest_framework_simplejwt.settings import api_settings as jwt_settings
import logging


from .serializers import (
    RegisterSerializer, 
    VerifyOTPSerializer, 
    LoginSerializer, 
    LogoutSerializer, 
    GoogleLoginSerializer,
    UserSerializer
)
from .otp import OTPHandler
import os
from django.conf import settings
from google.oauth2 import id_token
from google.auth.transport import requests
from drf_spectacular.utils import extend_schema

User = get_user_model()
logger = logging.getLogger('security')

class RegisterView(views.APIView):
    permission_classes = [permissions.AllowAny]
    serializer_class = RegisterSerializer
    throttle_scope = 'anon'

    def post(self, request):
        serializer = RegisterSerializer(data=request.data)
        if serializer.is_valid():
            email = serializer.validated_data['email']
            
          
            success, msg = OTPHandler.send_otp(email, email)
            if success:
                reg_data = serializer.validated_data
                cache.set(f"register_data:{email}", reg_data, timeout=300) 
                
                return Response({"success": True, "message": msg}, status=status.HTTP_200_OK)
            else:
                return Response({"success": False, "message": msg}, status=status.HTTP_400_BAD_REQUEST)
        
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class VerifyOTPView(views.APIView):
    permission_classes = [permissions.AllowAny]
    serializer_class = VerifyOTPSerializer

    def post(self, request):
        serializer = VerifyOTPSerializer(data=request.data)
        if serializer.is_valid():
            email = serializer.validated_data['email']
            flow_type = serializer.validated_data['type']
            
            if flow_type == 'register':
                # retrieve data
                reg_data = cache.get(f"register_data:{email}")
                if not reg_data:
                     return Response({"error": "Registration session expired. Please register again."}, status=status.HTTP_400_BAD_REQUEST)
                
                # Create User
                try:
                    user = User.objects.create_user(
                        email=email,
                        first_name=reg_data['first_name'],
                        last_name=reg_data['last_name'],
                        phone_number=reg_data['phone_number'],
                        password=reg_data['password'],
                        role=reg_data['role'],
                        specialization=reg_data.get('specialization')
                    )
                    user.is_verified = True
                    user.save()
                    
                    # Cleanup
                    cache.delete(f"register_data:{email}")
                    
                    return Response({"success": True, "message": "Registration completed."}, status=status.HTTP_201_CREATED)
                except Exception as e:
                    logger.error(f"User creation failed: {str(e)}")
                    return Response({"error": "User creation failed."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            elif flow_type == 'login':
                try:
                    user = User.objects.get(email=email)
                    user.is_verified = True 
                    user.save()

                    refresh = RefreshToken.for_user(user)
                    
                    response = Response({
                        "success": True,
                        "message": "Login completed.",
                        "user": {
                            "email": user.email,
                            "first_name": user.first_name,
                            "last_name": user.last_name,
                            "role": user.role
                        }
                    }, status=status.HTTP_200_OK)

                    response.set_cookie(
                        key='access',
                        value=str(refresh.access_token),
                        httponly=True,
                        secure=settings.DEBUG is False,
                        samesite='Lax',
                        path='/'
                    )
                    response.set_cookie(
                        key='refresh',
                        value=str(refresh),
                        httponly=True,
                        secure=settings.DEBUG is False,
                        samesite='Lax',
                        path='/'
                    )

                    return response
                except User.DoesNotExist:
                     return Response({"error": "User not found."}, status=status.HTTP_404_NOT_FOUND)
            
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class LoginView(views.APIView):
    permission_classes = [permissions.AllowAny]
    serializer_class = LoginSerializer

    def post(self, request):
        serializer = LoginSerializer(data=request.data)
        if serializer.is_valid():
            email = serializer.validated_data['email']
            
            # Send OTP
            success, msg = OTPHandler.send_otp(email, email)
            if success:
                return Response({"success": True, "message": msg}, status=status.HTTP_200_OK)
            else:
                return Response({"success": False, "message": msg}, status=status.HTTP_400_BAD_REQUEST)

        return Response(serializer.errors, status=status.HTTP_401_UNAUTHORIZED)


class LogoutView(views.APIView):
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request):
        response = Response({"success": True, "message": "Logged out."}, status=status.HTTP_200_OK)
        response.delete_cookie('access', path='/')
        response.delete_cookie('refresh', path='/')

        return response


class GoogleLoginView(views.APIView):
    permission_classes = [permissions.AllowAny]
    serializer_class = GoogleLoginSerializer

    @extend_schema(request=GoogleLoginSerializer)
    def post(self, request):
        serializer = GoogleLoginSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        token = serializer.validated_data.get('token')

        try:
            # Check if it's a JWT (ID Token) or Access Token
            if token.count('.') == 2:
                # Likely an ID Token (JWT)
                idinfo = id_token.verify_oauth2_token(token, requests.Request(), settings.GOOGLE_CLIENT_ID)
                email = idinfo['email']
                first_name = idinfo.get('given_name', '')
                last_name = idinfo.get('family_name', '')
            else:
                # Likely an Access Token (OAuth2)
                # Verify access token via Google userinfo endpoint
                import requests as http_requests
                userinfo_url = "https://www.googleapis.com/oauth2/v3/userinfo"
                response = http_requests.get(userinfo_url, params={'access_token': token})
                
                if not response.ok:
                    return Response({"error": "Failed to verify Access Token with Google"}, status=status.HTTP_400_BAD_REQUEST)
                
                idinfo = response.json()
                email = idinfo.get('email')
                first_name = idinfo.get('given_name', '')
                last_name = idinfo.get('family_name', '')

                if not email:
                    return Response({"error": "Email not provided by Google"}, status=status.HTTP_400_BAD_REQUEST)

            try:
                user = User.objects.get(email=email)
                # If user exists, check role
                if user.role == 'DOCTOR':
                    return Response({
                        "error": "Doctors cannot use Google Login. Please use professional email and OTP."
                    }, status=status.HTTP_403_FORBIDDEN)
            except User.DoesNotExist:
                # If user doesn't exist, create as PATIENT
                user = User.objects.create_user(
                    email=email,
                    first_name=first_name,
                    last_name=last_name,
                    role='PATIENT',
                    is_verified=True
                )
                created = True
            else:
                created = False

            if not created:
                user.is_verified = True
                user.save()

            refresh = RefreshToken.for_user(user)
            response = Response({
                "success": True,
                "message": "Login completed." if not created else "Registration and Login completed.",
                "user": {
                    "email": user.email,
                    "first_name": user.first_name,
                    "last_name": user.last_name,
                    "role": user.role
                }
            }, status=status.HTTP_200_OK)

            response.set_cookie(
                key='access',
                value=str(refresh.access_token),
                httponly=True,
                secure=settings.DEBUG is False,
                samesite='Lax',
                path='/'
            )
            response.set_cookie(
                key='refresh',
                value=str(refresh),
                httponly=True,
                secure=settings.DEBUG is False,
                samesite='Lax',
                path='/'
            )

            return response

        except ValueError as e:
            logger.error(f"Google Token Validation failed: {str(e)}")
            return Response({"error": f"Invalid token: {str(e)}"}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
             logger.error(f"Google Login failed: {str(e)}")
             return Response({"error": "Google Login failed"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class ProfileView(views.APIView):
    permission_classes = [permissions.IsAuthenticated]
    serializer_class = UserSerializer

    def get(self, request):
        serializer = UserSerializer(request.user)
        return Response(serializer.data)

    def patch(self, request):
        serializer = UserSerializer(request.user, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response({
                "success": True,
                "message": "Profile updated successfully.",
                "user": serializer.data
            })
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
class CookieTokenRefreshView(TokenRefreshView):
    def post(self, request, *args, **kwargs):
        refresh_token = request.COOKIES.get('refresh')
        if refresh_token:
            request.data['refresh'] = refresh_token
        
        response = super().post(request, *args, **kwargs)
        
        if response.status_code == 200:
            access_token = response.data.get('access')
            if access_token:
                response.set_cookie(
                    key='access',
                    value=access_token,
                    httponly=True,
                    secure=settings.DEBUG is False,
                    samesite='Lax',
                    path='/'
                )
                # Remove from body to keep it purely in cookies if desired, 
                # but often it's fine to leave it for frontend to confirm success.
                # del response.data['access'] 
        
        return response
