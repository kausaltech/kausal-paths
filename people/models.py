from django.utils.translation import gettext_lazy as _
from kausal_common.people.models import BasePerson
from users.models import User

class Person(BasePerson):
    class Meta:
        verbose_name = _('Person')
        verbose_name_plural = _('People')

    def __str__(self):
        return f"{self.first_name} {self.last_name}"

    def download_avatar(self):
        # Since this is a base implementation, we'll return None
        # Subclasses can override this to implement actual avatar downloading
        return None

    def get_avatar_url(self, **kwargs) -> str | None:
        # Return the URL of the person's image if it exists
        if self.image:
            return self.image.url
        return None

    def create_corresponding_user(self):
        # Get or create a user based on the person's email
        if not self.email:
            return None

        user, created = User.objects.get_or_create(
            email__iexact=self.email,
            defaults={
                'email': self.email,
                'first_name': self.first_name,
                'last_name': self.last_name,
            }
        )
        return user

    def visible_for_user(self, user, **kwargs) -> bool:
        # By default, make the person visible to all authenticated users
        # and to the person themselves
        if not user.is_authenticated:
            return False

        # Person is always visible to themselves
        if user == self.user:
            return True

        # Person is visible to users in the same organization
        if user.person and user.person.organization == self.organization:
            return True

        return False
