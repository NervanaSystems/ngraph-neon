# Intel® Neon

## A Deep Learning framework powered by Intel® nGraph

Welcome to Intel® Neon™, an open source Deep Learning framwork powered by Intel® nGraph 


## Installation

### Prerequisites  

Installing within a virtual environment is the easiest option for most users. 
To prepare for a system installation, you may skip this step.  

Python3 
  To create and activate a Python 3 virtualenv:
     
```
$ python3 -m venv .venv
$ . .venv/bin/activate
```

Python 2.7
  To create and activate a Python 2 virtualenv:


```
  $ virtualenv -p python2.7 .venv
  $ . .venv/bin/activate
```

Download the source code
```
  $ git clone https://github.com/NervanaSystems/ngraph-neon.git
  $ cd ngraph-neon
```

### Install Neon
  
To build and install Intel Neon, keep the virtualenv activated and 
simply run ``make install`` from within the clone of the repo as follows:

```
$ make install
```

### Install nGraph (Required)

After completing the prerequisites and installation of the base Neon
Graph package, Intel® nGraph and the python wraooer needs to be installed.
Clone the source code from https://github.com/NervanaSystems/ngraph
Activate the virtualenv if you haven't already. And follow the instructions at
https://github.com/NervanaSystems/ngraph/blob/master/python/README.md
Once the python wheel is built

```
$ pip install <path-to-python-wheel>/<wheel-name>.whl
```

### Run Neon Unit Tests

After installing Intel Neon and Intel nGraph, keep the virtualenv activated and
simply run ``make test`` from within the clone of the repo as follows:

```
$ make test
```

