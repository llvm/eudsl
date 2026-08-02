FROM ubuntu

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get -y install python-is-python3 cmake ninja-build python3.12-venv python3-pip vim git unzip

RUN python -m venv venv
RUN /venv/bin/python -m pip install pip-tools "pyodide-build>=0.28.0"
RUN echo 'source "/venv/bin/activate"' >> $HOME/.bashrc

# Pin the cross-build environment; its python version becomes the ABI the distro
# and everything built against it must use. Unpinned, pyodide-build takes the
# latest compatible one, which drifts ahead of the Pyodide that actually runs the
# result. Keep in sync with .github/actions/setup_base/action.yml, and note the
# venv above must be python 3.12 for a 3.12 cross-build env to install at all.
ARG PYODIDE_VERSION=0.27.7
RUN /venv/bin/pyodide xbuildenv install "$PYODIDE_VERSION" && \
    echo "pyodide $PYODIDE_VERSION -> python $(/venv/bin/pyodide config get python_version)"

RUN git clone https://github.com/emscripten-core/emsdk.git
# Take the emscripten version from pyodide-build rather than hardcoding one. The
# wheel must be built with the toolchain pyodide expects, and CI derives it the
# same way (.github/actions/setup_base/action.yml). A pinned version here
# silently drifts: this was stuck at 3.1.58 long after CI had moved on.
RUN cd emsdk && \
    EMSCRIPTEN_VERSION="$(/venv/bin/pyodide config get emscripten_version)" && \
    echo "pyodide wants emscripten $EMSCRIPTEN_VERSION" && \
    ./emsdk install "$EMSCRIPTEN_VERSION" ccache-git-emscripten-64bit && \
    ./emsdk activate "$EMSCRIPTEN_VERSION" ccache-git-emscripten-64bit && \
    cd ..
RUN echo 'source "/emsdk/emsdk_env.sh"' >> $HOME/.bashrc
