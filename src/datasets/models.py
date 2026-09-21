"""Rebuildable computation inputs, deliberately separate from authored datasets."""

from django.db import models


class DVCSourceManifest(models.Model):
    repository_url = models.CharField(max_length=500)
    revision = models.CharField(max_length=40)
    remote_name = models.CharField(max_length=100, blank=True)
    dataset_identifier = models.CharField(max_length=500)
    format_version = models.PositiveSmallIntegerField()
    content = models.JSONField()
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'paths_datasets_dvc_source_manifest__rebuildable'
        constraints = [
            models.UniqueConstraint(
                fields=['repository_url', 'revision', 'remote_name', 'dataset_identifier', 'format_version'],
                name='unique_dataset_source_manifest',
            ),
        ]

    def __str__(self) -> str:
        return f'{self.dataset_identifier}@{self.revision}'


class PreparedDataset(models.Model):
    key = models.CharField(max_length=64, primary_key=True)
    recipe = models.JSONField()
    frame_metadata = models.JSONField()
    payload = models.BinaryField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'paths_datasets_prepared_dataset__rebuildable'

    def __str__(self) -> str:
        return self.key
