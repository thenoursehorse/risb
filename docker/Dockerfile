ARG TRIQSTAG
FROM flatironinstitute/triqs:${TRIQSTAG}
ARG APPNAME=risb

USER root

ENV SRC=/src \
    OMP_NUM_THREADS=4 \
    NCORES=4

COPY . $SRC/$APPNAME
WORKDIR $SRC/$APPNAME
RUN python3 -m pip install --no-warn-script-location --upgrade pip \
  && python3 -m pip install . \
  && rm -rf $SRC/$APPNAME
WORKDIR /home/triqs/

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
USER triqs
