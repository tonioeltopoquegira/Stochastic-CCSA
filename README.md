In order to reproduce the results do the following:
(1) Git clone locally:
git clone --recurse-submodules https://github.com/tonioeltopoquegira/Stochastic-CCSA.git
(2) Create environment so not to clash with nlopt (conda or pip with python 3.13)
(3) Build the nlopt package (this is how I do it)
"mkdir build
cd build  
cmake .. -DNLOPT_PYTHON=ON -DCMAKE_INSTALL_PREFIX=$HOME/.local
make -j$(nproc)
make install"
(4) install all remaining packages from requirements.txt
(5) to run experiments run python train.py with these possible flags:
--exp mnist_cnn
--dataset cifar10
--opt adam
--epochs 10
--batch-size 128
--lr 0.001
--weight-decay 0.0
--inner-gradients 0
--always-improve 0
--sigma-min 0.0
--maxeval 1e6
--max-inner-eval None
--verbose
--seed 42
--outdir ./runs
--plot-ylim None
--plot-eval-limit None


