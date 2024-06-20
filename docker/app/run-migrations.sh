#!/bin/bash

set -e
set -o pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )


function restore_from_backup() {
    if [ -z "$DB_RESTORE_FROM_BACKUP" ] ; then
        return
    fi
    # Check we already have tables in the database
    tmpf=$(mktemp)
    echo "select count(*) from information_schema.tables where table_schema = 'public';" | ./manage.py dbshell -- -qtAX -o $tmpf
    nr_tables=$(cat $tmpf)
    rm $tmpf
    if [ ! "$nr_tables" -ge 0 -o ! "$nr_tables" -lt 10 ] ; then
        echo "Invalid number of tables found in database; expecting less than 10. Not restoring."
        return
    fi
    if [ -f $SCRIPT_DIR/manage-db-backup.sh ] ; then
        manage_cmd=$SCRIPT_DIR/manage-db-backup.sh
    else
        manage_cmd=manage-db-backup.sh
    fi
    $manage_cmd restore
}


restore_from_backup

exec python manage.py migrate --noinput
