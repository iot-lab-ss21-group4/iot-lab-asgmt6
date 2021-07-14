# Assignment 6


## **Installing Dependencies:**
* Install Python3 either system-wide, user-wide or as a virtual environment,
* Run `pip install pip-tools` command via the `pip` command associated with the installed Python
* Run `pip-compile requirements.in` to update package version information
* Run `pip-sync` inside the project root folder so that the packages will be installed
* Run `python setup.py install --old-and-unmanagable` inside each subfolder of the common folder
  This installs common util functions and the models inside the Lib folder of your chosen python interpreter
* If development changes are made, reinstalling of the previous commands are necessary. To make reinstalling faster we recommend to always run the script `install-iotlab-modules`. This is a powershell script. Linux and Mac users can install powershell on their system
* Note that running `pip-sync` deletes packages not included in `requirements.in`. That means also the common packages are deleted. You must then reinstall them again

## **FaaS:**
* Training the models will be done by FaaS functions
* faas folder contains FaaS code that is added to the IoT Platform
* Code can be executed in the respective faas python file for checking functionality
* Templates to indicate what JSON input is necessary

## **Edge:**
* On the edge two different services run
* Edge service for continous forecasting
* Offline service for doing periodic best offline forecasting
* Templates in forecaster-configuration for edge service and offline-configuration for offline service indicate necessary input for starting the services

## **Containerization:**
* Training models will run inside python environment on a docker container
* Edge components will run inside python environment on docker containers
* Images can be built for the forecaster container on the edge and the python-based runtimes for FaaS on the IoT platform
* Execute `buid-image` script in project root folder to build and push image to your docker repository. The script will ask for your Docker ID
* By default this script will build the image for the forecaster container. To build an image for one of the FaaS containers do the following for e.g. LSTM model: `build-image.ps1 -DeploymentType "faas" -AppName "trainer-lstm"`
* Necessary containers for the edge deployment are managed in docker-compose files
* If you set up a container network using one of the compose files specify the file by using `-f` in the docker-compose command
* There are several docker-compose files
* `docker-compose-edge.yml`: File for containers deployment on the edge. Note that _forecaster_ and _offline_ use both the edge-forecaster image
* `docker-compose-edge.dev.yml`: File to setup necessary containers for local development. Consider that the containers are then reachable over localhost and not by using their container name in the network.
* `docker-compose-edge.integration.yml`: File to setup necessary containers for integration test
* `docker-compose-edge.minio.yml`: File to setup a minio container
