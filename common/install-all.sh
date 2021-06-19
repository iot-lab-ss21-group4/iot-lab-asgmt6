#!/usr/bin/bash

for d in */
do
    cd $d
    if [ -f "setup.py" ]; then
        pip install .
    fi
    cd ..
done
