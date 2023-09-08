ARG TRIQSTAG
FROM flatironinstitute/triqs:${TRIQSTAG}
ARG APPNAME=risb

USER root
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN groupadd --gid $USER_GID build \
    && useradd --uid $USER_UID --gid $USER_GID -m build
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
#USER root
#RUN make install

USER triqs