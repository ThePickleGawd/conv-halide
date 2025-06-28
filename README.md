# Depthwise Separable Convolution with Halide

Discovering optimal scheduling algorithms for separable convolution on Compute In-Memory (CIM) architectures.

## Setup

```bash
# Install uv
pip install uv

# Init venv
uv venv
uv sync
```

## Usage

```bash
# Run inference
uv run inference.py

# 1) Generate Halide code for convolution
uv run sep_conv_generate.py

# 2) Compile (replace paths as necessary)
cd sep_conv
cl /LD /EHsc /std:c++17 sep_conv.py.cpp sep_conv.o ^
    /I C:\Users\dylan\Developer\tetramem\conv-halide\.venv\Lib\site-packages\halide\include ^
    /I C:\User\dylan\Developer\tetramem\conv-halide\.venv\Lib\site-packages\pybind11\include ^
    /I C:\Users\dylan\AppData\Roaming\uv\python\cpython-3.9.21-windows-x86_64-none\Include ^
    /link ^
    /LIBPATH:C:\Users\dylan\Developer\tetramem\conv-halide\.venv\Lib\site-packages\halide\lib ^
    /LIBPATH:C:\Users\dylan\AppData\Roaming\uv\python\cpython-3.9.21-windows-x86_64-none\libs ^
    python39.lib ^
    /OUT:sep_conv.pyd
```
