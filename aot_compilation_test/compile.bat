cl /LD /EHsc /std:c++17 lesson_10_halide.py.cpp lesson_10_halide.o ^
    /I C:\Users\dylan\Developer\tetramem\conv-halide\.venv\Lib\site-packages\halide\include ^
    /I C:\User\dylan\Developer\tetramem\conv-halide\.venv\Lib\site-packages\pybind11\include ^
    /I C:\Users\dylan\AppData\Roaming\uv\python\cpython-3.9.21-windows-x86_64-none\Include ^
    /link ^
    /LIBPATH:C:\Users\dylan\Developer\tetramem\conv-halide\.venv\Lib\site-packages\halide\lib ^
    /LIBPATH:C:\Users\dylan\AppData\Roaming\uv\python\cpython-3.9.21-windows-x86_64-none\libs ^
    python39.lib ^
    /OUT:lesson_10_halide.pyd