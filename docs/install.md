---
layout: default
title: Install
---

# Table of Contents
{: .no_toc .text-delta }

- TOC
{:toc}

# From source

Follow the 
[installation intructions](https://triqs.github.io/triqs/latest/install.html)
for TRIQS. Crucially, this project requires an older version `3.0.x`. If 
compiling from 
[source](https://triqs.github.io/triqs/latest/install.html#compiling-triqs-from-source-advanced)
you will need to specifify an older branch when you `git clone` with

```bash
git clone https://github.com/TRIQS/triqs --branch 3.0.x triqs.src
```

If compiling from source you have to be careful about library versions and 
compilers. We are not sure if higher than `gcc` `10.x` will work on this older 
version of TRIQS.

The below instructions are taken from the Docker files in each repository. 
Each repository requires TRIQS `3.0.x` to be installed, and will by default 
be installed into the same location as TRIQS.

Make sure that TRIQS is loaded into your environment with

```bash
source $INSTALL_PREFIX/share/triqs/triqsvars.sh
```

The number of cores to use for compiling should be set with

```bash
export NCORES=
```

## k-space integration kint

```bash
git clone https://github.com/thenoursehorse/kint kint.src
mkdir -p kint.build && cd kint.build
cmake ../kint.src/ -DTRIQS_ROOT=${TRIQS_ROOT}
```

## Embedding solver embedding_ed

### ARPACK-NG

Install [ARPACK-NG](https://github.com/opencollab/arpack-ng). 
We specify a version below that we know works correctly with our project.

```bash
export ARPACK_NG_ROOT=${HOME}/arpack-ng

git clone https://github.com/opencollab/arpack-ng --branch 3.6.0 arpack-ng.src
mkdir -p arpack-ng.build && cd arpack-ng.build 
cmake ../arpack-ng.src/ -DCMAKE_INSTALL_PREFIX=${ARPACK_NG_ROO} \
                        -DBUILD_SHARED_LIBS=ON
make -j${NCORES}
make test
make install
```

In order for the project to find the libraries at runtime you have to 
add their location to your environment. For example, they can be added as

```bash
export LD_LIBRARY_PATH=${ARPACK_NG_ROOT}/lib/:${LD_LIBRARY_PATH}
echo "export LD_LIBRARY_PATH=${ARPACK_NG_ROOT}/lib/:${LD_LIBRARY_PATH}" >> ${HOME}/.bashrc
```

### ezARPACK

Install [ezARPACK](https://github.com/krivenko/ezARPACK).
Currently, the version has to be 0.9 for our project.

```bash
export EZARPACK_ROOT=/home/${NB_USER}/ezARPACK

git clone https://github.com/krivenko/ezARPACK --branch 0.9 ezARPACK.src
mkdir -p ezARPACK.build && cd ezARPACK.build 
cmake ../ezARPACK.src/ -DCMAKE_INSTALL_PREFIX=${EZARPACK_ROOT} \
                       -DARPACK_NG_ROOT=${ARPACK_NG_ROOT}
```

As above, the project has to find these libraries at runtime. E.g., they are 
added to the environment as

```bash
export LD_LIBRARY_PATH=${EZARPACK_ROOT}/lib/:${LD_LIBRARY_PATH}
echo "export LD_LIBRARY_PATH=${EZARPACK_ROOT}/lib/:${LD_LIBRARY_PATH}" >> ${HOME}/.bashrc
```

### embedding_ed

```bash
git clone https://github.com/thenoursehorse/embedding_ed embedding_ed.src
mkdir -p embedding_ed.build && cd embedding_ed.build 
cmake ../embedding_ed.src/ -DTRIQS_ROOT=${TRIQS_ROOT} \
                           -DARPACK_NG_ROOT=${ARPACK_NG_ROOT} \
                           -DEZARPACK_ROOT=${EZARPACK_ROOT}
```

## Embedding solver embedding_dmrg

## risb

```bash
git clone https://github.com/thenoursehorse/risb risb.src
mkdir -p risb.build && cd risb.build
cmake ../risb.src/ -DTRIQS_ROOT=${TRIQS_ROOT}
make -j${NCORES}
make test
make install
```

# With Docker

[Docker](https://www.docker.com/) has the advantage of creating a container 
with the required libraries to install risb with user controlled separation 
from the operating system. Using a container gaurantees that risb will install 
and work without further testing, because it has been setup by the project 
maintainers.

There are Docker files in the risb repositories. They rely on a Docker file 
that sets up a container for TRIQS, located in the 
[thenoursehorse/docker](https://github.com/thenoursehorse/docker/) repository.

A complete container can be set up using 
[Compose](https://docs.docker.com/compose/), 
with the YAML file located at
[docker-compose.yml](https://github.com/thenoursehorse/docker/blob/main/risb_all/docker-compose.yml).
This installs TRIQS, dft-tools, cthyb, and risb. 

Unfortunately, Compose can be non-trivial to get working. It is difficult to 
give comprehensive instructions until more of the features of Compose 
are better ironed out. You can contact us for additional support and we will 
try our best to help.