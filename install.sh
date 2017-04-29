#!/bin/sh
set -e

clear
echo "==========================================================="
echo "===             install dependencies                    ==="
echo "==========================================================="
apt install -yf python-matplotlib python-pip python-tk ipython cython mercurial software-properties-common coinor-libipopt-dev
pip install krylov --force
echo "==========================================================="
echo "===                     success                         ==="
echo "==========================================================="

sleep 3
clear
echo "==========================================================="
echo "===             install fenics                          ==="
echo "===           This may take long time                   ==="
echo "==========================================================="
add-apt-repository -y ppa:fenics-packages/fenics
apt update -y
apt install -yf --no-install-recommends fenics
apt dist-upgrade -y
echo "==========================================================="
echo "===                     success                         ==="
echo "==========================================================="

sleep 3
clear
echo "==========================================================="
echo "===             install libadjoint                      ==="
echo "==========================================================="
sudo apt-add-repository -y ppa:libadjoint/ppa
sudo apt update -y
sudo apt install -yf python-libadjoint
echo "==========================================================="
echo "===                     success                         ==="
echo "==========================================================="

sleep 3
clear
echo "==========================================================="
echo "===             install moola                           ==="
echo "==========================================================="
git clone https://github.com/funsim/moola.git
cd moola
python setup.py build
python setup.py install
cd ..
rm -rf moola
hg clone https://bitbucket.org/amitibo/cyipopt
cd cyipopt
python setup.py build
python setup.py install
cd ..
rm -rf cyipopt
echo "==========================================================="
echo "===                     success                         ==="
echo "==========================================================="
