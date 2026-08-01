FROM ubuntu

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get -y install python-is-python3 cmake ninja-build python3.12-venv python3-pip vim git unzip

RUN python -m venv venv
RUN /venv/bin/python -m pip install pip-tools "pyodide-build>=0.28.0"
RUN echo 'source "/venv/bin/activate"' >> $HOME/.bashrc

RUN git clone https://github.com/emscripten-core/emsdk.git
# Take the emscripten version from pyodide-build rather than hardcoding one.
# The wheel must be built with the toolchain pyodide expects, and CI already
# derives it this way (.github/actions/setup_base/action.yml). A pinned version
# here silently drifts from CI: this was stuck at 3.1.58 long after pyodide had
# moved to 4.0.9.
RUN cd emsdk && \
    EMSCRIPTEN_VERSION="$(/venv/bin/pyodide config get emscripten_version)" && \
    echo "pyodide wants emscripten $EMSCRIPTEN_VERSION" && \
    ./emsdk install "$EMSCRIPTEN_VERSION" ccache-git-emscripten-64bit && \
    ./emsdk activate "$EMSCRIPTEN_VERSION" ccache-git-emscripten-64bit && \
    cd ..
RUN echo 'source "/emsdk/emsdk_env.sh"' >> $HOME/.bashrc
