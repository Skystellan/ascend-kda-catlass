cmake_minimum_required(VERSION 3.16)
find_package(ASC REQUIRED)
project(ascend_kda_ops LANGUAGES ASC CXX)
set(CMAKE_CXX_STANDARD 17)

if(NOT DEFINED ENV{ASCEND_HOME_PATH} OR "$ENV{ASCEND_HOME_PATH}" STREQUAL "")
    message(FATAL_ERROR "ASCEND_HOME_PATH is not set. Please source CANN set_env.sh first.")
endif()
set(CANN $ENV{ASCEND_HOME_PATH})

if(NOT DEFINED CATLASS_ROOT OR "${CATLASS_ROOT}" STREQUAL "")
    if(DEFINED ENV{CATLASS_ROOT} AND NOT "$ENV{CATLASS_ROOT}" STREQUAL "")
        set(CATLASS_ROOT $ENV{CATLASS_ROOT})
    elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/catlass/include")
        set(CATLASS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/catlass")
    else()
        message(FATAL_ERROR "CATLASS_ROOT is not set. Set -DCATLASS_ROOT=<path/to/catlass> or clone https://gitcode.com/cann/catlass into third_party/catlass.")
    endif()
endif()

find_package(Python3 COMPONENTS Interpreter REQUIRED)
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import pathlib,torch,torch_npu;print(pathlib.Path(torch.__file__).parent);print(pathlib.Path(torch_npu.__file__).parent)"
    OUTPUT_VARIABLE PY_PATHS
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
)
string(REPLACE "\n" ";" PY_PATHS "${PY_PATHS}")
list(GET PY_PATHS 0 TORCH_ROOT)
list(GET PY_PATHS 1 TORCH_NPU_ROOT)
list(APPEND CMAKE_PREFIX_PATH ${TORCH_ROOT})
find_package(Torch REQUIRED)

set(ASC_SOURCES kernel/kda_chunk.asc)

add_library(ascend_kda_ops SHARED
    src/kda_ops_torch.cpp
    ${ASC_SOURCES}
)
set_source_files_properties(${ASC_SOURCES} PROPERTIES LANGUAGE ASC)
set_target_properties(ascend_kda_ops PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_definitions(ascend_kda_ops PRIVATE
    CATLASS_ARCH=2201
    TILING_KEY_VAR
    _GLIBCXX_USE_CXX11_ABI=1
)
target_include_directories(ascend_kda_ops PRIVATE
    include
    ${CATLASS_ROOT}/include
    ${CANN}/include
    ${TORCH_INCLUDE_DIRS}
    ${TORCH_NPU_ROOT}/include
)
target_compile_options(ascend_kda_ops PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-2201>
    $<$<COMPILE_LANGUAGE:ASC>:-Wno-macro-redefined>
    $<$<COMPILE_LANGUAGE:ASC>:-Wno-ignored-attributes>
    $<$<COMPILE_LANGUAGE:ASC>:-Wno-unused-value>
    "SHELL:$<$<COMPILE_LANGUAGE:ASC>:-mllvm -cce-aicore-record-overflow=true>"
    "SHELL:$<$<COMPILE_LANGUAGE:ASC>:-mllvm -cce-aicore-dcci-insert-for-scalar=false>"
)
target_link_directories(ascend_kda_ops PRIVATE
    ${CANN}/lib64
    ${CANN}/${CMAKE_SYSTEM_PROCESSOR}-linux/devlib
    ${TORCH_NPU_ROOT}/lib
)
target_link_libraries(ascend_kda_ops PRIVATE
    ${TORCH_LIBRARIES}
    torch_npu
    ascendc_runtime
    runtime
    profapi
    mmpa
    ascend_dump
    c_sec
    error_manager
    ascendcl
    tiling_api
    nnopbase
    platform
    dl
    ascend_hal
)
if(EXISTS "${CANN}/lib64/libascendalog.so")
    target_link_libraries(ascend_kda_ops PRIVATE ascendalog)
endif()
if(EXISTS "${CANN}/lib64/libunified_dlog.so")
    target_link_libraries(ascend_kda_ops PRIVATE unified_dlog)
endif()
