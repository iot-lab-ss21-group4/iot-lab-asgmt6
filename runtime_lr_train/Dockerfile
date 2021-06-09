FROM ibmfunctions/action-python-v3.7:latest

COPY runtime_lr_train/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install own iotlab python modules
COPY common/ /iotlab-python-modules/common
COPY lr/ /iotlab-python-modules/lr
WORKDIR /iotlab-python-modules/common
RUN python setup.py install
WORKDIR /iotlab-python-modules/lr
RUN python setup.py install

WORKDIR /
