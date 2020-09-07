# Build external module
add_subdirectory("_napkinxc")

# Copy all the py modules to dist folder
file(RELATIVE_PATH PY_OUTPUT_DIR ${PY_SRC_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
file(GLOB PY_FILES LIST_DIRECTORIES false RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.py")
install(FILES ${PY_FILES} DESTINATION ${PY_OUTPUT_DIR} COMPONENT "PY") # Copy all the *.py files in dir
