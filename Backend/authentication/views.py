from rest_framework import status, views, permissions
from rest_framework.response import Response
from django.contrib.auth import get_user_model
from django.core.cache import cache
from rest_framework_simplejwt.tokens import RefreshToken
import logging

from .serializers import RegisterSerializer, VerifyOTPSerializer, LoginSerializer, LogoutSerializer
from .otp import OTPHandler

User = get_user_model()
logger = logging.getLogger('security')

class RegisterView(views.APIView):
    permission_classes = [permissions.AllowAny]
    serializer_class = RegisterSerializer
    throttle_scope = 'anon' # Apply rate limiting settings if needed

    def post(self, request):
        serializer = RegisterSerializer(data=request.data)
        if serializer.is_valid():
            email = serializer.validated_data['email']
            
            # Send OTP
            success, msg = OTPHandler.send_otp(email, email) # identifier is email
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
                        role=reg_data['role']
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
                    user.is_verified = True # Ensure verified if not already
                    user.save()

                    # Issue Tokens
                    refresh = RefreshToken.for_user(user)
                    
                    return Response({
                        "success": True,
                        "message": "Login completed.",
                        "refresh": str(refresh),
                        "access": str(refresh.access_token),
                    }, status=status.HTTP_200_OK)
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
        serializer = LogoutSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response({"success": True, "message": "Logged out."}, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
