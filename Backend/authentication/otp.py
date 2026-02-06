import hashlib
import secrets
import logging
from django.core.cache import cache
from django.conf import settings
from datetime import timedelta

logger = logging.getLogger('security')

OTP_EXPIRY = 180 
OTP_RESEND_COOLDOWN = 60 
MAX_ATTEMPTS = 3

class OTPHandler:
    @staticmethod
    def _get_cache_key(identifier, kind='otp'):
        return f"otp:{kind}:{identifier}"

    @staticmethod
    def generate_otp():
        """Generates a secure 6-digit OTP."""
        return "".join([str(secrets.randbelow(10)) for _ in range(6)])

    @staticmethod
    def hash_otp(otp):
        """Hashes the OTP for secure storage."""
        return hashlib.sha256(otp.encode()).hexdigest()

    @staticmethod
    def send_otp(identifier, email):
        """
        Generates and 'sends' an OTP.
        Returns Tuple(success, message).
        """
      
        cooldown_key = OTPHandler._get_cache_key(identifier, 'cooldown')
        if cache.get(cooldown_key):
             logger.warning(f"OTP resend blocked for {identifier} due to cooldown.")
             return False, "Please wait before resending OTP."

       
        otp = OTPHandler.generate_otp()
        otp_hash = OTPHandler.hash_otp(otp)

        otp_key = OTPHandler._get_cache_key(identifier, 'data')
        cache.set(otp_key, {'hash': otp_hash, 'attempts': 0}, timeout=OTP_EXPIRY)

    
        cache.set(cooldown_key, True, timeout=OTP_RESEND_COOLDOWN)

       
        try:
            from django.core.mail import send_mail
            send_mail(
                'Your OTP Code',
                f'Your OTP is: {otp}',
                settings.EMAIL_HOST_USER,
                [email],
                fail_silently=False,
            )
           
            return True, "OTP sent successfully."
        except Exception as e:
            logger.error(f"Failed to send OTP to {email}: {str(e)}")
            return False, "Failed to send OTP."

    @staticmethod
    def verify_otp(identifier, input_otp):
        """
        Verifies the OTP.
        Returns Tuple(success, message).
        """
        otp_key = OTPHandler._get_cache_key(identifier, 'data')
        data = cache.get(otp_key)

        if not data:
            return False, "OTP expired or invalid."

       
        if data['attempts'] >= MAX_ATTEMPTS:
            cache.delete(otp_key)
            logger.warning(f"OTP max attempts reached for {identifier}")
            return False, "Max attempts reached. Request a new OTP."

     
        input_hash = OTPHandler.hash_otp(input_otp)
        if input_hash == data['hash']:
          
            cache.delete(otp_key)
            return True, "OTP verified."
        else:
         
            data['attempts'] += 1
            cache.set(otp_key, data, timeout=cache.ttl(otp_key))
            logger.warning(f"Invalid OTP attempt for {identifier}")
            return False, "Invalid OTP."
