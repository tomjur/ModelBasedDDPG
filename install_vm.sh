#!/bin/sh
apt-get --assume-yes upgrade
apt-get --assume-yes update
apt-get --assume-yes install cmake g++ git ipython minizip python-dev python-h5py python-numpy python-scipy python-sympy qt4-dev-tools libassimp-dev libavcodec-dev libavformat-dev libavformat-dev libboost-all-dev libboost-date-time-dev libbullet-dev libfaac-dev libglew-dev libgsm1-dev liblapack-dev liblog4cxx-dev libmpfr-dev libode-dev libogg-dev libpcrecpp0v5 libpcre3-dev libqhull-dev libqt4-dev libsoqt-dev-common libsoqt4-dev libswscale-dev libswscale-dev libvorbis-dev libx264-dev libxml2-dev libxvidcore-dev libcairo2-dev libjasper-dev libpoppler-glib-dev libsdl2-dev libtiff5-dev libxrandr-dev libccd-dev

echo alias python=python2.7 >> ~/.bashrc
source ~/.bashrc

sudo -u ubuntu git clone https://github.com/rdiankov/collada-dom.git
cd collada-dom 
sudo -u ubuntu mkdir build
cd build
sudo -u ubuntu cmake ..
sudo -u ubuntu make -j4
make install

cd ~

sudo -u ubuntu git clone --branch OpenSceneGraph-3.4 https://github.com/openscenegraph/OpenSceneGraph.git
cd OpenSceneGraph 
sudo -u ubuntu mkdir build 
cd build
sudo -u ubuntu cmake .. -DDESIRED_QT_VERSION=4
sudo -u ubuntu make -j4
make install

cd ~

sudo -u ubuntu git clone https://github.com/flexible-collision-library/fcl.git
cd fcl
sudo -u ubuntu git checkout 0.5.0
sudo -u ubuntu mkdir build
cd build
sudo -u ubuntu cmake ..
sudo -u ubuntu make -j4
make install
ln -sf /usr/include/eigen3/Eigen /usr/include/Eigen

cd ~

sudo -u ubuntu git clone --branch latest_stable https://github.com/rdiankov/openrave.git
cd openrave
sudo -u ubuntu git checkout 9c79ea260e1c009b0a6f7c03ec34f59629ccbe2c
sudo -u ubuntu mkdir build 
cd build
sudo -u ubuntu cmake .. -DOSG_DIR=/usr/local/lib64/
sudo -u ubuntu make -j4
make install
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(openrave-config --python-dir)/openravepy/_openravepy_
export PYTHONPATH=$PYTHONPATH:$(openrave-config --python-dir)

cd ~

sudo -u ubuntu git clone https://github.com/tomjur/ModelBasedDDPG.git
apt --assume-yes install python-pip
sudo -u ubuntu pip install --upgrade --force-reinstall pip==9.0.3
pip install pyyaml tensorflow numpy shapely matplotlib descartes
apt-get --assume-yes install python-tk
