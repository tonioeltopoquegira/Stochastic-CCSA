#!/bin/bash
set -e

# Set paths
CCSA_DIR=~/Desktop/Research/CCSA
NLPT_DIR=$CCSA_DIR/nlopt
PYTHON_EXEC=$(which python)   # Uses current active Python

echo "=== Cleaning old NLopt files ==="
rm -f $CCSA_DIR/_nlopt.so $CCSA_DIR/nlopt.py
rm -rf $NLPT_DIR/build

# Clone the correct branch if not exists
if [ ! -d "$NLPT_DIR/.git" ]; then
    echo "=== Cloning NLopt ccsa_inner_gradients branch ==="
    git clone -b ccsa_inner_gradients https://github.com/stevengj/nlopt.git $NLPT_DIR
else
    echo "=== Updating existing NLopt repo ==="
    cd $NLPT_DIR
    git fetch
    git checkout ccsa_inner_gradients
    git pull
fi

echo "=== Building NLopt with Python wrapper ==="
cd $NLPT_DIR
mkdir -p build && cd build
cmake .. -DBUILD_PYTHON=ON -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

echo "=== Copying Python wrapper and shared object ==="
cp src/swig/_nlopt.so $CCSA_DIR/
cp src/swig/CMakeFiles/nlopt_python.dir/nlopt.files/nlopt.py $CCSA_DIR/

echo "=== Verifying NLopt API ==="
export PYTHONPATH=$CCSA_DIR
$PYTHON_EXEC - <<'PYTHON'
import nlopt, _nlopt
print("nlopt loaded from:", nlopt.__file__)
print("_nlopt loaded from:", _nlopt.__file__)
print("Has LD_CCSAQ?:", hasattr(nlopt, 'LD_CCSAQ'))
params = nlopt.algorithm_specific_parameters() if hasattr(nlopt, 'algorithm_specific_parameters') else None
print("Algorithm-specific parameters:", params)
PYTHON

echo "=== NLopt CCSA build complete ==="
