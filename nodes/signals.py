from nodes.models import InstanceConfig
from django.dispatch import receiver
from django.db.models.signals import post_save

from pages.models import InstanceSiteContent

@receiver(post_save, sender=InstanceConfig)
def create_instance_site_content(sender, instance, created, **kwargs):
    if created:
        InstanceSiteContent.objects.create(instance=instance)
