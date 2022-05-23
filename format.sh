set -e
black .
isort --profile black .
autoflake -r -i src
black .

