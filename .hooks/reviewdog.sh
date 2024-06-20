#!/bin/bash

#echo $PRE_COMMIT_FROM_REF
#echo $PRE_COMMIT_TO_REF
#git diff ${PRE_COMMIT_FROM_REF}..${PRE_COMMIT_TO_REF} > /tmp/diffi
exec reviewdog -reporter=local -fail-on-error -diff "git diff ${PRE_COMMIT_FROM_REF}..${PRE_COMMIT_TO_REF}"
