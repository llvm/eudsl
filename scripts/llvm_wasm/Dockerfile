FROM ubuntu

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get -y install cmake python-is-python3 ninja-build python3-pip vim git unzip software-properties-common wget ripgrep pkg-config autoconf libtool curl
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
RUN echo 'source "/root/.cargo/env"' >> $HOME/.bashrc

RUN add-apt-repository -y ppa:deadsnakes/ppa && apt update && apt -y install python3.13 python3.13-venv
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.13 99

RUN python -m venv venv
RUN echo 'source "/venv/bin/activate"' >> $HOME/.bashrc

RUN git clone --recursive https://github.com/pyodide/pyodide && \
    cd pyodide && \
    git checkout 0.29.0 && \
    git submodule update --init && \
    pip install -r requirements.txt && \
    touch src/core/jsproxy.h && \
    EXTRA_CFLAGS="-D DEBUG_F" EXTRA_LDFLAGS="-s ASSERTIONS=2" PYODIDE_SOURCEMAPS=1 make && \
    EXTRA_CFLAGS="-D DEBUG_F" EXTRA_LDFLAGS="-s ASSERTIONS=2" PYODIDE_SOURCEMAPS=1 make -C cpython rebuild-all && \
    EXTRA_CFLAGS="-D DEBUG_F" EXTRA_LDFLAGS="-s ASSERTIONS=2" PYODIDE_SOURCEMAPS=1 make all-but-packages && \
    cd ..

RUN . "/venv/bin/activate" && pip install /pyodide/pyodide-build

RUN /pyodide/emsdk/emsdk/emsdk install ccache-git-emscripten-64bit && \
    /pyodide/emsdk/emsdk/emsdk activate ccache-git-emscripten-64bit
RUN echo 'source "/pyodide/emsdk/emsdk/emsdk_env.sh"' >> $HOME/.bashrc

# has to be after emscripten ccache because that build for some reason fails with 4.2.0
RUN wget https://github.com/Kitware/CMake/releases/download/v4.2.0/cmake-4.2.0-linux-aarch64.sh && bash cmake-4.2.0-linux-aarch64.sh --skip-license --prefix=/usr
