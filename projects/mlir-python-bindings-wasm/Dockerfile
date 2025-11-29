FROM ubuntu

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get -y install python-is-python3 cmake ninja-build python3.12-venv python3-pip vim git

RUN python -m venv venv
RUN /venv/bin/python -m pip install pip-tools pyodide-build
RUN echo 'source "/venv/bin/activate"' >> $HOME/.bashrc

RUN git clone https://github.com/emscripten-core/emsdk.git
# EMSCRIPTEN_VERSION=$(pyodide config get emscripten_version)
RUN cd emsdk && ./emsdk install 3.1.58 && ./emsdk activate 3.1.58 && cd ..
RUN echo 'source "/emsdk/emsdk_env.sh"' >> $HOME/.bashrc
