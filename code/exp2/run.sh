for D in */ ; do
    echo $D
    pushd $D
    # cp ../run.py .
    ipython run.py > run.log 2>&1  &
    popd
done