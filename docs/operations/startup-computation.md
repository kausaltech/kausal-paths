# Startup computation

The shared image copies optional product scripts from `docker/pre-entry/` to
`/scripts/pre-entry/`, preserving executable permissions. The entrypoint runs
executable files in filename order before `gunicorn`, `uwsgi`, or `runserver`
(including the default server command). Migration commands, Celery workers and
other explicit commands do not run these hooks. Missing or empty directories
are allowed. Hooks inherit the server's user, environment and cache paths.

Paths supplies `10-compute-instances.sh`, which runs:

```sh
python manage.py compute_instances --in-customer-use --warm-dvc-cache
```

`InstanceConfig.in_customer_use` defaults to false. Set it explicitly for the
instances that should receive operational services such as startup computation.
The command always excludes inactive instances. No instances are automatically
marked as being in customer use by the migration.

For manual computation, supply one or more identifiers instead:

```sh
python manage.py compute_instances first-instance second-instance
```

The command uses the normal runtime loader, preferring published data when
available, and computes outcome nodes in the default and baseline scenarios.
With `--warm-dvc-cache`, the command first ensures all declared DVC inputs are
on disk, even when computation results are already in the external cache. Instances
without DVC inputs skip this step. Already-cached files are not downloaded again. Persisted source manifests allow
prefetching Parquet objects without decoding them or initializing Git. See
[Persistent dataset caches](dataset-caches.md) for metadata preparation and backups.

It leaves normal external caching enabled. Each instance is cleaned up before
moving on; failures are logged and remaining instances are attempted. The
command exits unsuccessfully if any selected instance failed.

The startup script continues to server startup after a command failure or timeout.
`COMPUTE_INSTANCES_TIMEOUT` sets the total warm-up timeout in seconds (default
120); a process that has not exited five seconds after termination is killed.
Configure deployment startup probes to allow this timeout plus normal server
startup. Readiness remains false while the server has not yet started.

Deploy the database migration before starting application pods. Begin with a
small selection of instances and compare deployment readiness time with the
first uncached request latency. Concurrent replicas can duplicate computation;
this initial implementation does not coordinate warm-up across pods.

On-disk DVC/Git/Parquet data, persisted prepared datasets and external computation
results survive the warm-up process. In-memory caches do not transfer to server workers. Cache reuse
requires the same source revision, model configuration, build and scenario inputs
as the subsequent request. The command does not warm custom user scenarios or
all disconnected nodes.
