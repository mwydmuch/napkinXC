# Variable to hold the path to src folder to compute relative path of its descendants
set(PY_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})

find_package(Python ${PYTHON_VERSION} EXACT COMPONENTS Interpreter Development REQUIRED)

add_subdirectory("napkinxc")
