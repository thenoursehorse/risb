ARG TRIQSTAG
FROM flatironinstitute/triqs:${TRIQSTAG}
ARG APPNAME=risb

USER root
RUN apt-get update && \
      apt-get -y install sudo
RUN adduser --disabled-password --gecos '' build
RUN adduser build sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER build

ENV SRC=/home/build/src \
    BUILD=/home/build \
    OMP_NUM_THREADS=4 \
    NCORES=4

COPY --chown=build requirements.txt $SRC/$APPNAME/requirements.txt
RUN pip3 install -r $SRC/$APPNAME/requirements.txt

COPY --chown=build . $SRC/$APPNAME
WORKDIR $BUILD/$APPNAME
RUN chown build .

# -DTRIQS_ROOT=${INSTALL_PREFIX}
#RUN cmake $SRC/$APPNAME
#RUN make -j$NCORES || make -j1 VERBOSE=1
#USER root
#RUN make install
#USER build