FROM ibmfunctions/action-python-v3.7:latest

COPY runtime_edge/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install own iotlab python modules
COPY common/ /iotlab-python-modules/common
COPY lr/ /iotlab-python-modules/lr
WORKDIR /iotlab-python-modules/common
RUN python setup.py install
WORKDIR /iotlab-python-modules/lr
RUN python setup.py install

# Copy edge directory
COPY edge/ /edge
# Copy edge main
COPY main_edge.py /

# create directory
WORKDIR /forecaster-configuration

WORKDIR /
