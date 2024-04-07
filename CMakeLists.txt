# 1. CMake最低版本、项目名称和所需语言:
cmake_minimum_required(VERSION 3.7)
project (MatplotlibCPP)
# 2. C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# 3. 寻找Python解释器。这是一个REQUIRED依赖:
find_program(Python_EXECUTABLE python NO_CMAKE_FIND_ROOT_PATH)
message(STATUS "Python_EXECUTABLE: ${Python_EXECUTABLE}")
# 5. 使用hello-embedded-python.c源文件，添加一个可执行目标:
add_executable(test test.cpp)
# 6. 可执行文件包含Python.h头文件。因此，这个目标的include目录必须包含Python的include目录，可以通过PYTHON_INCLUDE_DIRS变量进行指定:
target_include_directories(test PRIVATE ${PYTHON_INCLUDE_DIRS})
# 7. 最后，将可执行文件链接到Python库，通过PYTHON_LIBRARIES变量访问:
target_link_libraries(test PRIVATE ${PYTHON_LIBRARIES})



# et(PYTHON_INCLUDE_DIR "/home/zhulin/anaconda3/envs/torch/include/python3.8")
# set(NUMPY_INCLUDE_DIR "/home/zhulin/anaconda3/envs/torch/lib/python3.8/site-packages/numpy/core/include")
# set(PYTHON_LIB "/home/zhulin/anaconda3/envs/torch/lib")

# include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include ${PYTHON_INCLUDE_DIR} ${NUMPY_INCLUDE_DIR})
# link_directories(${PYTHON_LIB})

# set(LIB "libpython3.8.so", "libpthread.so")

# # include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
# # include_directories(${PYTHONHOME2}/include)
# # include_directories(${PYTHONHOME2}/Lib/site-packages/numpy/core/include)
# # link_directories(${PYTHONHOME2}/libs)

# add_definitions(-DMATPLOTLIBCPP_PYTHON_HEADER=Python.h)

# # message(STATUS "*** dump start cmake variables ***")
# # get_cmake_property(_variableNames VARIABLES)
# # foreach(_variableName ${_variableNames})
# #         message(STATUS "${_variableName}=${${_variableName}}")
# # endforeach()
# # message(STATUS "*** dump end ***")

# add_executable(main ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
# # add_executable(minimal ${CMAKE_CURRENT_SOURCE_DIR}/../examples/minimal.cpp)
# # add_executable(basic ${CMAKE_CURRENT_SOURCE_DIR}/../examples/basic.cpp)
# # add_executable(modern ${CMAKE_CURRENT_SOURCE_DIR}/../examples/modern.cpp)
# # add_executable(animation ${CMAKE_CURRENT_SOURCE_DIR}/../examples/animation.cpp)
# # add_executable(nonblock ${CMAKE_CURRENT_SOURCE_DIR}/../examples/nonblock.cpp)
# # add_executable(xkcd ${CMAKE_CURRENT_SOURCE_DIR}/../examples/xkcd.cpp)
# # add_executable(bar ${CMAKE_CURRENT_SOURCE_DIR}/../examples/bar.cpp)

# target_link_libraries(main ${LIB})