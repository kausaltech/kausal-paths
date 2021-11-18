from django.contrib.auth import REDIRECT_FIELD_NAME
from django.contrib.auth.decorators import login_required
from django.http import HttpResponseRedirect, HttpResponseBadRequest
from django.urls import reverse
from django.utils.http import url_has_allowed_host_and_scheme
from django.views.decorators.http import require_http_methods


@require_http_methods(['POST', 'GET'])
@login_required
def change_admin_instance(request, instance_id):
    if request.method == 'POST':
        instance_id = request.POST.get('instance', None)

    if instance_id is None:
        return HttpResponseBadRequest("No instance given")
    try:
        instance_id = int(instance_id)
    except ValueError:
        return HttpResponseBadRequest("Invalid instance id")

    user = request.user
    adminable_instances = user.get_adminable_instances()
    try:
        user.selected_instance = next(i for i in adminable_instances if i.id == instance_id)
    except StopIteration:
        return HttpResponseBadRequest("Invalid instance ID")
    user.save(update_fields=['selected_instance'])

    redirect_to = request.POST.get(
        REDIRECT_FIELD_NAME,
        request.GET.get(REDIRECT_FIELD_NAME, '')
    )
    if redirect_to:
        url_is_safe = url_has_allowed_host_and_scheme(
            url=redirect_to,
            allowed_hosts=[request.get_host()],
            require_https=request.is_secure(),
        )
        if url_is_safe:
            return HttpResponseRedirect(redirect_to)

    return HttpResponseRedirect(reverse('wagtailadmin_home'))
    # If we decide to use the standard Django admin interface...
    # admin_type = request.GET.get('admin', 'django')
    # if admin_type == 'wagtail':
    #     return HttpResponseRedirect(reverse('wagtailadmin_home'))
    # else:
    #     return HttpResponseRedirect(reverse('admin:index'))
