ARG TRIQSTAG
FROM triqs:${TRIQSTAG}
ARG APPNAME=risb

RUN mkdir -p ${SRC_DIR}/${APPNAME}
COPY requirements.txt ${SRC_DIR}/${APPNAME}/requirements.txt
RUN pip3 install -r ${SRC_DIR}/${APPNAME}/requirements.txt

COPY . ${SRC_DIR}/${APPNAME}

RUN mkdir -p ${BUILD_DIR}/${APPNAME}
WORKDIR ${BUILD_DIR}/${APPNAME}

SHELL ["/bin/bash", "-cli"]
RUN cmake ${SRC_DIR}/${APPNAME} -DTRIQS_ROOT=${INSTALL_PREFIX} \
  && make -j${NCORES} \
  || make -j1 VERBOSE=1 \
  && make install