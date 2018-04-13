# Helper script to build nGraph for neon developers


ngraph_cpu_wheels.sh is a helper script that builds nGraph python wrapper wheels with CPU backend enabled.
The script is position independent and can be called from anywhere.
The wheel will the placed in the current directory.

### Usage

For example, on Ubuntu 16.04 to get neon and create a nGraph wrapper wheels in a build directory parallel to neon, do

```
git clone https://github.com/NervanaSystems/ngraph-neon.git
mkdir build && cd build
../ngraph-neon/contrib/ngraph/ngraph_cpu_wheels.sh
```

Then you will see two files(X.Y.Z is a version number)

```
ngraph-X.Y.Z-cp27-cp27mu-linux_x86_64.whl
ngraph-X.Y.Z-cp35-cp35m-linux_x86_64.whl
```

