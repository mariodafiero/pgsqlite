#!/bin/bash
set -exu -o pipefail

EXAMPLE_FILES="example_dbs/*.sqlite"
for f in ${EXAMPLE_FILES} 
do
  echo $f
 python pgsqlite/pgsqlite.py -f "${f}" -p ${POSTGRES_CREDS_STRING} -d True --drop_tables_after_import True --drop_tables True
done

  
