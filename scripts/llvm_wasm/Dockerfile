FROM ubuntu

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get -y install python-is-python3 ninja-build python3-pip vim git unzip software-properties-common wget
RUN wget https://github.com/Kitware/CMake/releases/download/v4.2.0/cmake-4.2.0-linux-aarch64.sh && bash cmake-4.2.0-linux-aarch64.sh --skip-license --prefix=/usr

RUN add-apt-repository -y ppa:deadsnakes/ppa && apt update && apt -y install python3.13 python3.13-venv
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.13 99

RUN python -m venv venv
RUN /venv/bin/python -m pip install pip-tools pyodide-build==0.30.9
RUN echo 'source "/venv/bin/activate"' >> $HOME/.bashrc

RUN git clone https://github.com/emscripten-core/emsdk.git
# EMSCRIPTEN_VERSION=$(pyodide config get emscripten_version)
RUN cd emsdk && \
    ./emsdk install 4.0.9 ccache-git-emscripten-64bit && \
    ./emsdk activate 4.0.9 ccache-git-emscripten-64bit && \
    cd ..
RUN echo 'source "/emsdk/emsdk_env.sh"' >> $HOME/.bashrc
