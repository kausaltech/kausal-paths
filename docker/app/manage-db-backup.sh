#!/bin/bash

set -e
set -o pipefail

secret_path="${DB_BACKUP_SECRET_PATH:-/run/secrets/db-backup}"
required_vars="AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY RESTIC_PASSWORD S3_BUCKET S3_ENDPOINT"

if [ ! -d ${secret_path} ] ; then
    echo "Secrets not mounted at ${secret_path}, backups disabled."
    exit 0
fi


for fn in ${required_vars} ; do
  if [ ! -f ${secret_path}/${fn} ] ; then
    echo "Missing secret ${secret_path}/${fn}, aborting."
    exit 1
  fi
done

function get_secret() {
  cat ${secret_path}/$1
}

export AWS_ACCESS_KEY_ID=$(get_secret AWS_ACCESS_KEY_ID)
export AWS_SECRET_ACCESS_KEY=$(get_secret AWS_SECRET_ACCESS_KEY)
export RESTIC_PASSWORD_FILE=${secret_path}/RESTIC_PASSWORD
export RESTIC_REPOSITORY="s3:https://$(get_secret S3_ENDPOINT)/$(get_secret S3_BUCKET)"

if [ -z "$1" ] ; then
    echo "Usage: $0 {init|backup|restore|snapshots}"
    exit 2
fi

if [ "$1" == "init" ] ; then
    restic init
    exit 0
fi

if [ "$1" == "snapshots" ] ; then
    restic snapshots
    exit 0
fi

if [ "$1" == "backup" ] ; then
    if [ -z "$DATABASE_URL" ] ; then
        if [ -z "$PGPASSFILE" -o ! -r "$PGPASSFILE" ] ; then
            echo PGPASSFILE must be set and the file pointed by it readable. Alternatively, set DATABASE_URL.
            exit 1
        fi

        # Work around kubernetes secret mount permission limitations
        pgpasstmp=$(mktemp)
        cat $PGPASSFILE > $pgpasstmp
        export PGPASSFILE=$pgpasstmp

        database=$(cat $PGPASSFILE | cut -d ':' -f 3)
    else
        database="$DATABASE_URL"
    fi
    datatmp=$(mktemp)

    echo "Generating dump..."
    pg_dump -O "$database" > $datatmp
    echo "Uploading to restic..."
    cat $datatmp | restic backup --no-cache --stdin-filename database.sql --stdin
    echo "Pruning old backups..."
    restic forget --prune --keep-within-hourly 48h --keep-within-daily 30d --keep-within-weekly 1y --keep-monthly unlimited
    rm $datatmp
    if [ -z "$DATABASE_URL" ] ; then
        rm $pgpasstmp
    fi
    exit 0
fi

if [ "$1" == "restore" ] ; then
    echo "Restoring from backup..."
    restic dump latest database.sql | python manage.py dbshell
    exit 0
fi
