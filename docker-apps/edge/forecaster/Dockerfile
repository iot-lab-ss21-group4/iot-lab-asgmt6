FROM ibmfunctions/action-python-v3.7:latest

ARG docker_app_path="docker-apps/edge/forecaster"
ARG container_app_path="/app"

# Install app specific requirements
WORKDIR ${container_app_path}/requirements
COPY ${docker_app_path}/requirements .
RUN pip install --no-cache-dir -r requirements.txt

# Copy and install common requirements
WORKDIR ${container_app_path}/common
COPY common/ .
RUN bash install-all.sh

# Copy code
WORKDIR ${container_app_path}
COPY ${docker_app_path}/app .

