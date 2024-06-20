#!/bin/bash

REPORT_PATH=/tmp/report.html
PYTEST_ARGS="--html=$REPORT_PATH --self-contained-html $@"
SHOULD_CREATE_DB=1

function import_test_db() {
    if [ -z "$BUILD_S3_BUCKET" -o -z "$BUILD_S3_ENDPOINT" ]; then
        echo "S3 env vars not configured."
        return
    fi
    if [ -z "$POSTGRES_DATABASE" ]; then
        echo "DB env vars not configured."
        return
    fi
    TEST_DB=test_${POSTGRES_DATABASE}
    url="https://$BUILD_S3_ENDPOINT/$BUILD_S3_BUCKET/test-database.sql.gz"
    echo "Attempting to download database dump..."
    curl --fail -s -o /tmp/database.sql.gz "$url"
    if [ "$?" -ne 0 ] ; then
        echo "No database dump found"
        return
    fi
    echo "Test database dump found; restoring from dump"
    set -eo pipefail
    echo "DROP DATABASE IF EXISTS $TEST_DB ; CREATE DATABASE $TEST_DB" | psql postgres
    cat /tmp/database.sql.gz | gunzip | psql $TEST_DB > /dev/null
    SHOULD_CREATE_DB=0
}

function upload_report() {
    if [ -z "$AWS_ACCESS_KEY_ID" -o -z "$AWS_SECRET_ACCESS_KEY" -o -z "$BUILD_S3_BUCKET" -o -z "$BUILD_S3_ENDPOINT" ]; then
        echo "S3 not configured; not uploading test report."
        return
    fi
    if [ -z "$BUILD_ID" ]; then
        echo "No build ID; not uploading test report."
        return
    fi
    report_loc=s3://$BUILD_S3_BUCKET/$BUILD_ID/pytest-report.html
    echo "Sending report to ${report_loc}..."
    s3cmd --host $BUILD_S3_ENDPOINT --host-bucket $BUILD_S3_ENDPOINT put $REPORT_PATH $report_loc
    if [ $? -ne 0 ] ; then
        echo "Upload failed."
        return
    fi
    if [ -z "$GITHUB_OUTPUT" -o -z "$GITHUB_STEP_SUMMARY" ] ; then
        echo "GitHub env vars not configured; not outputting summary."
        return
    fi

    export TEST_REPORT_URL="https://${BUILD_S3_ENDPOINT}/${BUILD_S3_BUCKET}/${BUILD_ID}/pytest-report.html"
    echo "test_report_url=${TEST_REPORT_URL}" >> $GITHUB_OUTPUT
    if [ $pytest_rc -eq 0 ] ; then
        echo "✅ Unit tests succeeded." >> $GITHUB_STEP_SUMMARY
    else
        echo "❌ Unit tests failed." >> $GITHUB_STEP_SUMMARY
    fi
    echo "🔗 [Test report](${TEST_REPORT_URL})" >> $GITHUB_STEP_SUMMARY
}

import_test_db

if [ $SHOULD_CREATE_DB -ne "1" ] ; then
    PYTEST_ARGS="--reuse-db $PYTEST_ARGS"
fi

set +e
echo "Running pytest with args: $PYTEST_ARGS"
python run_tests.py $PYTEST_ARGS
pytest_rc=$?
upload_report
exit $pytest_rc
