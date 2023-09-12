---
title: Install
---

# Table of Contents
{: .no_toc .text-delta }

- TOC
{:toc}

# From source

## The main python package risb

1. Update packaging software
    ```
    python3 -m pip install --upgrade pip setuptools wheel
    ```

1. (Optional) Create a 
[virtual environment](https://packaging.python.org/en/latest/tutorials/installing-packages/#creating-virtual-environments).

1. Clone source
    ```
    git clone https://github.com/thenoursehorse/risb
    ```

1. Install from local (-e allows to develop code without reinstalling, omit if
not editing the source code)
    ```
    cd risb
    python3 -m pip install -e ./
    ```

## `TetrahedronKWeight` k-space integrator

```bash
git clone https://github.com/thenoursehorse/kint kint.src
mkdir -p kint.build && cd kint.build
cmake ../kint.src/ -DTRIQS_ROOT=${TRIQS_ROOT}
```

## `EmbeddingEd` embedding space solver

### ARPACK-NG

Install [ARPACK-NG](https://github.com/opencollab/arpack-ng). A specific 
version can be requested with `--branch <version>`. Minimum
version is `3.8.0`.

```bash
git clone https://github.com/opencollab/arpack-ng arpack-ng.src
mkdir -p arpack-ng.build && cd arpack-ng.build 
cmake ../arpack-ng.src/ -DBUILD_SHARED_LIBS=ON \
                        --DMPI=ON
make -j${NCORES}
make test
sudo make install
```

### ezARPACK

Install [ezARPACK](https://github.com/krivenko/ezARPACK). Minimum version 
is `1.0`.

```bash
git clone https://github.com/krivenko/ezARPACK ezARPACK.src
mkdir -p ezARPACK.build && cd ezARPACK.build 
cmake ../ezARPACK.src/
make -j${NCORES}
make test
sudo make install
```

### embedding_ed

Below assumes TRIQS is installed to system path. If it is not 
then `-DTRIQS_ROOT=/path/to/triqs` has to be passed to `cmake`.

```bash
git clone https://github.com/thenoursehorse/embedding_ed embedding_ed.src
mkdir -p embedding_ed.build && cd embedding_ed.build 
cmake ../embedding_ed.src/ -DEZARPACK_ROOT=/path/to/ezARPACK
make -j${NCORES}
make test
sudo make install
```

The `ezARPACK` path has to be given because `cmake` does not find 
the configuration correctly. By default it is `/usr/local/` if 
no `CMAKE_INSTALL_PREFIX` is specified.

# With Docker

[Docker](https://www.docker.com/) has the advantage of creating a container 
with the required libraries to install risb with user controlled separation 
from the operating system. Using a container gaurantees that risb will install 
and work without further testing, because it has been setup by the project 
maintainers.

A container can be set up using [Compose](https://docs.docker.com/compose/), 
with the YAML file [docker-compose.yml]. This uses the 
[official TRIQS docker image](https://hub.docker.com/r/flatironinstitute/triqs).

There is a complete container setup includeing TRIQS, dft-tools, cthyb, risb, 
kint, embedding_ed and a Jupyter notebook with this compose file
[docker-compose.yml](https://github.com/thenoursehorse/docker/blob/main/risb_all/docker-compose.yml).
THe file has to be edited where the comments suggest.

For any private repos, a very insecure way to get Compose to be able to clone 
them is to create an 
[access token](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token) 
for GitHub, and then change any of the `context:` lines that depend on private 
repos within the YAML file as

```bash
context: https://ACCESS-TOKEN@github.com/private/repo
```

{: .warning }
Anyone that has access to the Docker images will also have access to your 
access token, so be very careful. There are other ways to do the above that 
are more secure, just Google around.

Unfortunately, Compose can be non-trivial to get working. It is difficult to 
give comprehensive instructions until more of the features of Compose 
are better ironed out.