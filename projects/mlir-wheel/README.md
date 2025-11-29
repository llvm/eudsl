# LLVM/MLIR distribution as (`manylinux`-compatible) wheel

What's this? It's a means to getting a distribution of MLIR like this:

```shell
$ pip install mlir_wheel -f https://llvm.github.io/eudsl
```

This will install a thing that will let you do this:

```python
import mlir_wheel
print(mlir_wheel.bin_dir())
>>> /home/mlevental/mambaforge/envs/eudsl/lib/python3.11/site-packages/mlir_wheel/bin
```

or this

```shell
$ python -m mlir_wheel
usage: mlir_wheel [-h] [--bin-dir] [--cmake-dir] [--include-dir] [--lib-dir] [--root-dir]

options:
  -h, --help     show this help message and exit
  --bin-dir      Print the path to the LLVM/MLIR distribution binary directory.
  --cmake-dir    Print the path to the LLVM/MLIR distribution CMake module directory.
  --include-dir  Print the path to the LLVM/MLIR distribution header directory.
  --lib-dir      Print the path to the LLVM/MLIR distribution library directory.
  --root-dir     Print the path to the LLVM/MLIR distribution root directory.
```

i.e., a full distribution of MLIR (and LLVM) that can be used to do out-of-tree builds i.e.,
```
-DCMAKE_PREFIX_PATH=$(python -m mlir_wheel --root-dir)
```
