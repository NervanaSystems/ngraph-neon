# Helper script to build nGraph for neon developers


ngraph_cpu_wheels.sh is a helper script that builds nGraph python wrapper wheels with CPU backend enabled.
The script is position independent and can be called from anywhere.
`build` directory will be created under the current directory.
The wheel will the placed in the neon source root directory.

### Usage

For example, on Ubuntu 16.04 to get neon and create a nGraph wrapper build directory parallel to neon, do

```
git clone https://github.com/NervanaSystems/ngraph-neon.git
pushd ..
./ngraph-neon/contrib/ngraph/ngraph_cpu_wheels.sh
popd
ls
```

Then you will see two files,(X.Y.Z is a version number)
```
ngraph-X.Y.Z-cp27-cp27mu-linux_x86_64.whl
ngraph-X.Y.Z-cp35-cp35m-linux_x86_64.whl
```

For python 2.7

Create and activate virtualenv
```
cd ..
virtualenv .neon_py2 && . .neon_py2/bin/activate
cd ngraph-neon
```

Install neon and nGraph
```
make install
pip install ngraph-X.Y.Z-cp27-cp27mu-linux_x86_64.whl
```
or

For python 3.5

Create and activate virtualenv
```
cd ..
python3 -m venv .neon_py3 && . .neon_py3/bin/activate
cd ngraph-neon
```

Install neon and nGraph
```
make install
pip install ngraph-X.Y.Z-cp35-cp35m-linux_x86_64.whl
```
