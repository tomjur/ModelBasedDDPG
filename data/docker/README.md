### Build the Odin Docker Image

Building this image requires Docker>=18.09, to build:
  1) Turn on `expermental features` in the `daemon` tab in docker `preferences`
  2) Run:
```
./docker/build_dev_docker.sh
```
 
### Start the docker container
Create a container:
```
docker run -p 2222:22 -d --name=local-trajopt \
           local-trajopt
```

## Docker SHH
From the command line, run `ssh -A -p 2222 root@localtrajopt` to SSH, should look similar to below:
```
Joes-Computer:docker joe$ ssh -A -p 2222 root@localtrajopt
Welcome to Ubuntu 16.04.3 LTS (GNU/Linux 4.9.125-linuxkit x86_64)
 * Documentation:  https://help.ubuntu.com
 * Management:     https://landscape.canonical.com
 * Support:        https://ubuntu.com/advantage
Last login: Wed Nov 28 22:22:57 2018 from 172.17.0.1
root@ec7704a23cbe:~#
```

Once inside the container, activate the python2.7 environment via `. venv2.7/bin/activate`
