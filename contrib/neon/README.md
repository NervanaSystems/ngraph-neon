# Helper script to install neon python2 and python3 in virtualenv

install_cpu.sh is a helper script that creates two virtualenv for python2 and python3 and installs neon with nGraph and aeon.
The script is position independent and can be called from anywhere.

## Usage

For example, on Ubuntu 16.04. First checkout neon code from github.

```
git clone https://github.com/NervanaSystems/ngraph-neon.git
```

Then grab Aeon's dependencies:

### Ubuntu (release 16.04 LTS and later):

    apt-get install git clang libcurl4-openssl-dev libopencv-dev libsox-dev libboost-filesystem-dev libboost-system-dev libssl-dev

##### For Python 3.n

    apt-get install python3-dev python3-pip python3-numpy

### CentOS (release 7.2 and later):

    yum install epel-release
    yum install git clang gcc-c++ make cmake openssl-devel opencv-devel libcurl-devel sox-devel boost-devel boost-filesystem boost-system

##### For Python 2.7

    yum install python-pip python-devel

##### For Python 3.n

    yum install python-pip python34-pip python34-devel python34-opencv python34-numpy

### OSX:

    brew tap homebrew/science
    brew install opencv
    brew install sox
    brew install boost

Next, check the recommended versions for nGraph, aeon and neon in the version files

```
cat nGraph.version
cat aeon.version
cat VERSION
```

Next, create a work directory and change directory.
Then run the script passing the recommended versions as args

```
mkdir build && cd build
./ngraph-neon/contrib/ngraph/install_cpu.sh v0.3.0 v1.3.1 ngraph-v0.3.0
```

neon for Python2 is installed in virtualenv `.venv2` and neon for Python3 is installed in virtualenv `.venv3`.
Just activate the virtualenv and start using neon.

# Helper script to build neon python2 and python3 wheels

neon_cpu_wheels.sh is a helper script that builds python2 and python3 neon wheels called `ngraph-neon`.
The wheels includes embedded nGraph python wrapper with CPU backend enabled.
The script is position independent and can be called from anywhere.
The wheels will the placed in the current directory.

## Usage (Build neon wheels)

For example, on Ubuntu 16.04. First checkout neon code from github.

```
git clone https://github.com/NervanaSystems/ngraph-neon.git
```

Then create a build directory and create neon wheels in the build directory.

```
mkdir build && cd build
./ngraph-neon/contrib/ngraph/neon_cpu_wheels.sh
```

Then you will see two python wheel files(X.Y.Z is a version number) created.
```
ngraph_neon-X.Y.Z-cp27-cp27mu-linux_x86_64.whl
ngraph_neon-X.Y.Z-cp35-cp35m-linux_x86_64.whl
```

### Usage (Install neon from wheels)

On Ubuntu 16.04

For python 2.7

Create and activate virtualenv and Install neon
```
virtualenv .neon_py2 && . .neon_py2/bin/activate
pip install ngraph_neon-X.Y.Z-cp27-cp27mu-linux_x86_64.whl
```
or

For python 3.5

Create and activate virtualenv and Install neon
```
cd ..
python3 -m venv .neon_py3 && . .neon_py3/bin/activate
pip install ngraph_neon-X.Y.Z-cp35-cp35m-linux_x86_64.whl
```
