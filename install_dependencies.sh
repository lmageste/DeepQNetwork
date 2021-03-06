#!/usr/bin/env bash

######################################################################
# Torch install
######################################################################


TOPDIR=$PWD

# Prefix:
PREFIX=$PWD/torch
echo "Installing Torch into: $PREFIX"

if [[ `uname` != 'Linux' ]]; then
  echo 'Platform unsupported, only available for Linux'
  exit
fi
if [[ `which apt-get` == '' ]]; then
    echo 'apt-get not found, platform not supported'
    exit
fi

# Install dependencies for Torch:
#sudo apt-get update
#sudo apt-get install -qqy build-essential
#sudo apt-get install -qqy gcc g++
#sudo apt-get install -qqy cmake
#sudo apt-get install -qqy curl
#sudo apt-get install -qqy libreadline-dev
#sudo apt-get install -qqy git-core
#sudo apt-get install -qqy libjpeg-dev
#sudo apt-get install -qqy libpng-dev
#sudo apt-get install -qqy ncurses-dev
#sudo apt-get install -qqy imagemagick
#sudo apt-get install -qqy unzip
#sudo apt-get install -qqy libqt4-dev
#sudo apt-get install -qqy liblua5.1-0-dev
#sudo apt-get install -qqy libgd-dev
#sudo apt-get install -qqy scons
#sudo apt-get install -qqy libgtk2.0-dev
#sudo apt-get install -qqy libsdl-dev
#sudo apt-get update

echo "==> Torch7's dependencies have been installed"

# Build and install Torch7
cd /tmp
rm -rf luajit-rocks
git clone https://github.com/torch/luajit-rocks.git
cd luajit-rocks
mkdir -p build
cd build
git checkout master; git pull
rm -f CMakeCache.txt
cmake .. -DCMAKE_INSTALL_PREFIX=$PREFIX -DCMAKE_BUILD_TYPE=Release -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0
RET=$?; if [ $RET -ne 0 ]; then echo "Error. Exiting."; exit $RET; fi
make
RET=$?; if [ $RET -ne 0 ]; then echo "Error. Exiting."; exit $RET; fi
make install
RET=$?; if [ $RET -ne 0 ]; then echo "Error. Exiting."; exit $RET; fi


path_to_nvcc=$(which nvcc)
if [ -x "$path_to_nvcc" ]
then
    cutorch=ok
    cunn=ok
fi

# Install base packages:
$PREFIX/bin/luarocks install cwrap
$PREFIX/bin/luarocks install paths
$PREFIX/bin/luarocks install torch
$PREFIX/bin/luarocks install nn

[ -n "$cutorch" ] && \
($PREFIX/bin/luarocks install cutorch)
[ -n "$cunn" ] && \
($PREFIX/bin/luarocks install cunn)

$PREFIX/bin/luarocks install luafilesystem
$PREFIX/bin/luarocks install penlight
$PREFIX/bin/luarocks install sys
$PREFIX/bin/luarocks install xlua
$PREFIX/bin/luarocks install image
$PREFIX/bin/luarocks install env
$PREFIX/bin/luarocks install qtlua
$PREFIX/bin/luarocks install qttorch

echo ""
echo "=> Torch7 has been installed successfully"
echo ""


echo "Installing nngraph ... "
$PREFIX/bin/luarocks install nngraph
RET=$?; if [ $RET -ne 0 ]; then echo "Error. Exiting."; exit $RET; fi
echo "nngraph installation completed"

echo "Installing FCEUX_Learning_Environment ... "
cd /tmp
rm -rf FCEUX_Learning_Environment
git clone https://github.com/lmageste/FCEUX_Learning_Environment.git -b lm
cd FCEUX_Learning_Environment
$PREFIX/bin/luarocks make
RET=$?; if [ $RET -ne 0 ]; then echo "Error. Exiting."; exit $RET; fi
echo "FCEUX installation completed"

echo "Installing neswrap ... "
cd /tmp
rm -rf neswrap
git clone https://github.com/lmageste/neswrap.git -b lm
cd neswrap
$PREFIX/bin/luarocks make
RET=$?; if [ $RET -ne 0 ]; then echo "Error. Exiting."; exit $RET; fi
echo "neswrap installation completed"

echo "Installing Lua-GD ... "
mkdir $PREFIX/src
cd $PREFIX/src
rm -rf lua-gd
git clone https://github.com/ittner/lua-gd.git
cd lua-gd
sed -i "s/LUABIN=lua5.1/LUABIN=..\/..\/bin\/luajit/" Makefile
$PREFIX/bin/luarocks make
RET=$?; if [ $RET -ne 0 ]; then echo "Error. Exiting."; exit $RET; fi
echo "Lua-GD installation completed"

echo "Installing GPU dependencies..."
$PREFIX/bin/luarocks install cutorch
$PREFIX/bin/luarocks install cunn
echo "Done trying to install the GPU dependencies."

echo
echo "All done!"

