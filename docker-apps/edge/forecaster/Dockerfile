FROM python:3.7.10-slim-buster

ARG DOCKER_APP_PATH="docker-apps/edge/forecaster"
ARG CONTAINER_APP_PATH="/app"

# Copy code and install app specific requirements
WORKDIR ${CONTAINER_APP_PATH}
COPY ${DOCKER_APP_PATH}/app .
RUN apt-get update \
        # Upgrade installed packages to get latest security fixes if the base image does not contain them already.
        && apt-get upgrade -y --no-install-recommends \
        # add some packages required for the pip install
        && apt-get install -y --no-install-recommends \
        gcc \
        libc-dev \
        libxslt-dev \
        libxml2-dev \
        libffi-dev \
        libssl-dev \
        # cleanup package lists, they are not used anymore in this image
        && rm -rf /var/lib/apt/lists/* \
        && apt-cache search linux-headers-generic \
        # install additional python modules
        && pip install --no-cache-dir -r requirements.txt

# Copy and install common requirements
WORKDIR ${CONTAINER_APP_PATH}/common
COPY common/ .
RUN bash install-all.sh

WORKDIR ${CONTAINER_APP_PATH}
