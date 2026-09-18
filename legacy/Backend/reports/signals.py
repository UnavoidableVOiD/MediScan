import os
from django.db.models.signals import post_delete
from django.dispatch import receiver
from .models import Report

@receiver(post_delete, sender=Report)
def auto_delete_file_on_delete(sender, instance, **kwargs):
    """
    Deletes file from filesystem when corresponding `Report` object is deleted.
    """
    if instance.file:
        if os.path.isfile(instance.file.path):
            os.remove(instance.file.path)
