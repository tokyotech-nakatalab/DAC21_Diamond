FROM amd64/python:3.7.12-slim

RUN apt-get update && apt-get install -y \
  git sudo ssh openssh-client cmake wget vim curl wget

# Install dev libraries and cpu torch
RUN pip install flake8 pytest pyre-check pytest-cov black autoflake isort && \
  pip install torch==1.10.0+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html
