for d in */ ; do
    cd $d
    if [ -f "setup.py" ]; then
        python setup.py install
    fi
    cd ..
done
