# Assignment 6


## **Installing Dependencies:**
* Install Python3 either system-wide, user-wide or as a virtual environment,
* Run `pip install pip-tools` command via the `pip` command associated with the installed Python,
* Run `pip-sync` inside the project root folder.
* Run `python setup.py install --old-and-unmanagable` inside the common folder.
  This installs common util functions for the models inside the Lib folder of your chosen python interpreter.
* Run `python setup.py install --old-and-unmanagable` inside each model folder (lr, lstm, sarimax)
* If development changes are made, reinstalling of the previous commands are necessary

TODO: make installing of own modules via script and also remove egg, build etc.

## **FaaS:**
* Training the models will be done by FaaS functions
* faas folder contains FaaS code that is added to the IoT Platform
* Code can be executed in the respective faas python file for checking functionality


## **Containerization:**
* Models will run inside python environment on a docker container
* Execute script (.ps1/.sh) in project root folder to build and push container
* All source paths in a dockerfile should be relative to the project root folder

