ARG TRIQSTAG
FROM flatironinstitute/triqs:${TRIQSTAG}
ARG APPNAME=risb

USER root
RUN apt-get update \
  && apt-get -y install sudo \
  && adduser --disabled-password --gecos '' build \
  && adduser build sudo \
  && echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER build

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV SRC=/home/build/src \
    BUILD=/home/build \
    OMP_NUM_THREADS=4 \
    NCORES=4

COPY --chown=build requirements.txt $SRC/$APPNAME/requirements.txt
RUN pip3 install -r $SRC/$APPNAME/requirements.txt

COPY --chown=build . $SRC/$APPNAME
WORKDIR $BUILD/$APPNAME
RUN chown build .
RUN cmake $SRC/$APPNAME \
  && make -j$NCORES || make -j1 VERBOSE=1
#RUN sudo make install