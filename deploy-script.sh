#!/bin/sh
mlflow server --backend-store-uri $pg_backend_store_uri --default-artifact-root $default_artifact_root --host $remote_host