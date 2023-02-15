from storages.backends.s3boto3 import S3Boto3Storage
from storages.utils import setting


class MediaFilesS3Storage(S3Boto3Storage):
    access_key_names = ['MEDIA_FILES_S3_ACCESS_KEY_ID']
    secret_key_names = ['MEDIA_FILES_S3_SECRET_ACCESS_KEY']
    security_token_names = ['MEDIA_FILES_S3_SESSION_TOKEN']

    def get_default_settings(self):
        defaults = super().get_default_settings()
        # Rename some settings to avoid conflicts with other S3 backends (e.g., dvc-pandas)
        defaults['access_key'] = setting('MEDIA_FILES_S3_ACCESS_KEY')
        defaults['secret_key'] = setting('MEDIA_FILES_S3_SECRET_ACCESS_KEY')
        defaults['bucket_name'] = setting('MEDIA_FILES_S3_BUCKET')
        defaults['endpoint_url'] = setting('MEDIA_FILES_S3_ENDPOINT')
        defaults['custom_domain'] = setting('MEDIA_FILES_S3_CUSTOM_DOMAIN')
        return defaults
