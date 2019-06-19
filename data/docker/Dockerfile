# syntax=docker/dockerfile:1.0.0-experimental
#
# Building this image requires Docker>=18.09, to build (takes about 30 min.):
# 1. turn on expermental features in docker->preferences->daemon
# 2. run `export DOCKER_BUILDKIT=1`
# 3. run `cd /path/to/folder/with/dockerfile`
# 4. run `docker build --secret id=id_rsa.pub,src=$HOME/.ssh/id_rsa.pub -t local-trajopt -f Dockerfile .`
#
# to create and run a container: 
#   docker run -p 2222:22 -d --name=local-trajopt local-trajopt
# 
# you can ssh into the machine: 
#   ssh -A -p 2222 root@127.0.0.1
# Note, for X forwarding, following the instructions below
# and ssh into the machine using:
#   ssh -A -p 2222 root@localtrajopt
# password is 'password' (you probably won't need to type it)
#
#
# the build process sets up a virtual environment, activate it with:
#   . venv2.7/bin/activate
#
# while the container is running it consumes memory and cpu, to stop the container:
#   docker stop local-trajopt
# restart with:
#   docker start local-trajopt
# to completely remove the container:
#   docker stop local-trajopt && docker rm local-trajopt
# note that you will lose any changes made to the container. 
# sometimes docker messes up the port mapping and you will not be able to ssh in.
# in that case, stop the container, restart docker and restart the container.
#
# you can use the deployment and remote shell functionality in pycharm as you would
# with a remote machine.
#
# for x-forwarding add the following to the host ~/.ssh/config:
#   Host localtrajopt
#       XAuthLocation /opt/X11/bin/xauth
#       ForwardX11 yes
#       ForwardX11Trusted yes
# run on the host:
#   defaults write org.macosforge.xquartz.X11 enable_iglx -bool true
# and install xquartz:
#   brew cask install xquartz
# now when you ssh into the container you should be able to run x apps; try to run 
# e.g. xeyes. pycharm doesn't have an option for x-forwarding. you can workaround 
# this limitation by creating a connection via the terminal this will create the 
# x-forwarding tunnel and then in pycharm add DISPLAY=:10 in the run/debug 
# configurations. another option is to use vnc (see below).
# EXPERIMENTAL: the ssh server will force DISPLAY=:10 on login so you can skip 
#               adding it in pycharm
#
# alternative: use vnc instead of x-forwarding
# use this if you use docker exec and need graphical output (you don't have 
# x-forwarding) or if you have opengl apps that have problems with xquartz. also 
# this is another workaround for pycharm's x-forwarding issue. to build with a vnc 
# server add "--build-arg USE_VNC=true" to the build command. you will need to add
# a port mapping for port 5901 when running the container:
#   docker run -p 2222:22 -p 5901:5901 -d --name=local-trajopt local-trajopt 
# when accessing the container with ssh, the ssh server will force the correct 
# DISPLAY environment variable, if you use docker exec you will need to set it on
# the run command:
#   docker run -p 5901:5901 --env=DISPLAY=:1 -d --name=local-trajopt local-trajopt 
# install a vnc client on the host:
#   brew cask install vnc-viewer
# use the vnc client to connect to 127.0.0.1::5901 to view graphical output.
#
# https://medium.com/@tonistiigi/build-secrets-and-ssh-forwarding-in-docker-18-09-ae8161d066
# https://www.visitusers.org/index.php?title=Re-enabling_INdirect_glx_on_your_X_server
# https://www.scm.com/doc/Installation/Remote_GUI.html
# https://askubuntu.com/a/747088
# https://docs.docker.com/engine/examples/running_ssh_service/
# https://stackoverflow.com/questions/41892039/how-to-enable-x11-forwarding-in-pycharm-ssh-session
# https://github.com/osrf/docker_images/issues/21
# https://stackoverflow.com/questions/48696021/pytest-error-could-not-load-path-to-conftest-py
#
FROM nvidia/cuda:9.1-runtime-ubuntu16.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
        build-essential \
        git \
        locales \
        # python-configobj \
        vim

RUN locale-gen en_US.UTF-8 \
 && update-locale LANG=en_US.UTF-8
ENV LANG=en_US.UTF-8

WORKDIR /root

# Install SSH
RUN apt-get update && apt-get install -y openssh-server net-tools 

# Configure SSH
RUN mkdir /var/run/sshd \
 && echo 'root:password' | chpasswd \
 && sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config \
 && echo "AcceptEnv *" >> /etc/ssh/sshd_config \
 && echo "PermitUserEnvironment yes" >> /etc/ssh/sshd_config \
 && echo "UseDNS no" >> /etc/ssh/sshd_config \
 && echo "AddressFamily inet" >> /etc/ssh/sshd_config \
 && sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

# Add hosts public key
RUN --mount=type=secret,id=id_rsa.pub \
    if [ -f /run/secrets/id_rsa.pub ]; then \
        mkdir -p ~/.ssh; \
        cat /run/secrets/id_rsa.pub >> ~/.ssh/authorized_keys; \
        chmod 600 ~/.ssh/authorized_keys; \
    fi

# Install and configure X Forwarding
ARG USE_X11=true

RUN if $USE_X11; then \
        apt-get update && apt-get install -y mesa-utils x11-apps; \
        mkdir -p ~/.ssh; \
        echo "DISPLAY=:10" >> ~/.ssh/environment; \
        echo "QT_X11_NO_MITSHM=1" >> ~/.ssh/environment; \
    fi

# Install and configure VNC
ARG USE_VNC=false

RUN if $USE_VNC; then \
        apt-get update && apt-get install -y lxpanel openbox xorg xfce4-terminal; \
        wget https://svwh.dl.sourceforge.net/project/turbovnc/2.2/turbovnc_2.2_amd64.deb; \
        dpkg -i *.deb; \
        rm *.deb; \
        mkdir -p .vnc; \
        echo '$geometry="1440x900";' > .vnc/turbovncserver.conf; \
        echo '$securityTypes="None";' >> .vnc/turbovncserver.conf; \
        echo "permitted-security-types = None" >> /etc/turbovncserver-security.conf; \
        mkdir -p .config/openbox; \
        echo 'lxpanel &' >> ~/.config/openbox/autostart; \
        mkdir -p ~/.ssh; \
        echo "DISPLAY=:1" > ~/.ssh/environment; \
    fi

# Create startup script
RUN echo "#!/bin/bash" > /startup.sh \
 && echo "term_handler() {" >> /startup.sh \
 && echo "    echo stopping" >> /startup.sh \
 && if $USE_VNC; then \
    echo "    /opt/TurboVNC/bin/vncserver -kill :1" >> /startup.sh; fi \
 && echo "    pkill sshd" >> /startup.sh \
 && echo "}" >> /startup.sh \
 && echo "trap \"term_handler\" SIGTERM SIGINT" >> /startup.sh \
 && if $USE_VNC; then \
    echo "/opt/TurboVNC/bin/vncserver :1" >> /startup.sh; fi \
 && echo "/usr/sbin/sshd -D &" >> /startup.sh \
 && echo "wait $!" >> /startup.sh \
 && chmod +x /startup.sh

CMD ["/startup.sh"]

# Install python2.7
RUN apt-get update \
 && apt-get install -y \
        python \
        python-pip \
        python-dev \
        virtualenv \
        python-tk

# # Install python3.5
# RUN apt-get update \
#  && apt-get install -y \
#         python3.5 \
#         python3.5-dev \
#         python3.5-venv \
#         python3.5-tk

# Download OSG, FCL, OpenRAVE, TrajOpt
RUN cd /tmp && git clone https://github.com/openscenegraph/OpenSceneGraph.git -b OpenSceneGraph-3.4.1 openscenegraph
RUN cd /tmp && git clone https://github.com/flexible-collision-library/fcl.git -b fcl-0.5
RUN cd /tmp && git clone https://github.com/EdsterG/openrave.git
RUN cd /tmp && git clone https://github.com/EdsterG/trajopt.git -b eddie-col

COPY trajopt_*.sh /scripts/

RUN bash /scripts/trajopt_deps_install.sh

RUN virtualenv -p python venv2.7 \
 && . venv2.7/bin/activate \
 && pip install h5py numpy scipy "sympy==0.7.1" six matplotlib \
 # ddpgmp dependencies
 pyyaml Pillow shapely descartes
# "tensorflow_gpu==1.5" pyyaml Pillow shapely descartes
 # "tensorflow==1.5" "pyyaml==5.1" "Pillow==6.0.0" "shapely==1.6.4.post2" "descartes==1.1.0"

# Install TensorFlow GPU version.
RUN virtualenv -p python venv2.7 \
 && . venv2.7/bin/activate \
 && pip --no-cache-dir install \
  http://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.5.0-cp27-none-linux_x86_64.whl

RUN . venv2.7/bin/activate \
 && export MAKEOPTS="-j4" \
 && bash /scripts/trajopt_build_and_install.sh
