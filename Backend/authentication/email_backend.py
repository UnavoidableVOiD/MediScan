"""
Custom email backend that handles SSL certificate verification issues on Windows.
This is necessary because Windows sometimes has issues with SSL certificate verification
when connecting to Gmail SMTP servers.
"""
import ssl
from django.core.mail.backends.smtp import EmailBackend as DjangoEmailBackend


class EmailBackend(DjangoEmailBackend):
    """
    Custom SMTP email backend that uses an unverified SSL context.
    This solves SSL certificate verification issues on Windows.
    """
    
    def open(self):
        """
        Ensure an open connection to the email server. Return whether or not a
        new connection was required (True or False) or None if an exception
        occurred.
        """
        if self.connection:
           
            return False

        connection_params = {'timeout': self.timeout} if self.timeout else {}
        try:
            self.connection = self.connection_class(
                self.host, self.port, **connection_params
            )

           
            if not self.use_ssl and self.use_tls:
              
                context = ssl._create_unverified_context()
                self.connection.starttls(context=context)
            
            if self.username and self.password:
                self.connection.login(self.username, self.password)
            return True
        except OSError:
            if not self.fail_silently:
                raise
