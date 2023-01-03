# DISCONTINUATION OF PROJECT #
This project will no longer be maintained by Intel.
This project has been identified as having known security escapes.
Intel has ceased development and contributions including, but not limited to, maintenance, bug fixes, new releases, or updates, to this project.
Intel no longer accepts patches to this project.
# Intel® Neon

## A Deep Learning framework powered by Intel® nGraph

Welcome to Intel® Neon, an open source Deep Learning framework powered by Intel® nGraph

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

### Install nGraph (Required)

See ngraph install docs for [instructions] to build and install ngraph

### Install nGraph python binding (Required)

After installating ngraph, follow these [steps] to install the pybind wrapper

### Install Neon

Download the source code
```
  $ git clone https://github.com/NervanaSystems/ngraph-neon.git
  $ cd ngraph-neon
```

To build and install Intel Neon, keep the virtualenv activated and
simply run ``make install`` from within the clone of the repo as follows:

```
$ make install
```

### Run Neon Unit Tests

After installing Intel Neon and Intel nGraph, keep the virtualenv activated and
simply run ``make test`` from within the clone of the repo as follows:

```
$ make test
```
[instructions]:http://ngraph.nervanasys.com/docs/latest/install.html
[steps]:https://github.com/NervanaSystems/ngraph/blob/master/python/README.md
