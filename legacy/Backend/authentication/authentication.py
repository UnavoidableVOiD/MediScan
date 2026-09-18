from rest_framework_simplejwt.authentication import JWTAuthentication
from django.conf import settings
from drf_spectacular.extensions import OpenApiAuthenticationExtension

class JWTCookieAuthentication(JWTAuthentication):
    def authenticate(self, request):
        header = self.get_header(request)
        
        if header is None:
            raw_token = request.COOKIES.get('access') or None
        else:
            raw_token = self.get_raw_token(header)

        if raw_token is None:
            return None

        validated_token = self.get_validated_token(raw_token)
        return self.get_user(validated_token), validated_token

class JWTCookieAuthenticationScheme(OpenApiAuthenticationExtension):
    target_class = 'authentication.authentication.JWTCookieAuthentication'
    name = 'JWTCookieAuthentication'

    def get_security_definition(self, auto_schema):
        return {
            'type': 'apiKey',
            'in': 'cookie',
            'name': 'access',
        }
