# Helper script to build neon python2 and python3 wheels


neon_cpu_wheels.sh is a helper script that builds python2 and python3 neon wheels called `ngraph-neon`.
The wheels includes embedded nGraph python wrapper with CPU backend enabled.
The script is position independent and can be called from anywhere.
The wheels will the placed in the current directory.

### Usage (Build neon wheels)

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
