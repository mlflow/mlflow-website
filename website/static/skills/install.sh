#!/bin/sh
# Convenience entrypoint served at https://mlflow.org/skills/install.sh
# Fetches and runs the canonical MLflow Skills installer from
# https://github.com/mlflow/skills (single source of truth; see that repo for
# the script source, options, and issues).
set -eu
curl -fsSL https://raw.githubusercontent.com/mlflow/skills/main/install.sh | sh -s -- "$@"
