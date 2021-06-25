FROM ibmfunctions/action-python-v3.7:master

ARG DOCKER_APP_PATH="docker-apps/faas/trainer-lstm"
ARG CONTAINER_APP_PATH="/app"

WORKDIR ${CONTAINER_APP_PATH}
COPY ${DOCKER_APP_PATH}/app .
# Install app specific requirements
RUN pip install --no-cache-dir -r requirements.txt

# Copy and install common requirements
WORKDIR ${CONTAINER_APP_PATH}/common
COPY common/ .
RUN bash install-all.sh

WORKDIR ${CONTAINER_APP_PATH}