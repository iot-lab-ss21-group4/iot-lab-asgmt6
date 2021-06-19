FROM ibmfunctions/action-python-v3.7:master

ARG docker_app_path="docker-apps/edge/forecaster"
ARG container_app_path="/app"

# Copy code and install app specific requirements
WORKDIR ${container_app_path}
COPY ${docker_app_path}/app .
RUN pip install --no-cache-dir -r requirements.txt

# Copy and install common requirements
WORKDIR ${container_app_path}/common
COPY common/ .
RUN bash install-all.sh

WORKDIR ${container_app_path}
