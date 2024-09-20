
set(LIBSOPHON /usr/local/libsophon-0.4.9)
set(CMAKE_C_COMPILER $ENV{CROSS_TOOLCHAINS}/gcc-arm-10.3-2021.07-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-gcc)
file(GLOB_RECURSE DEVICE_SRCS
    ${PROJECT_SOURCE_DIR}/src/*.c
)
include_directories(
    ${PROJECT_SOURCE_DIR}/../include/device ${PROJECT_SOURCE_DIR}/../include/kernel ${PROJECT_SOURCE_DIR}/../include
)
# Add chip arch defination
# add_definitions(-D__bm$ENV{FIRMWARE_CHIPID}__)

# Set the library directories for the shared library
link_directories(${PROJECT_SOURCE_DIR}/lib/)
set(SHARED_LIBRARY_OUTPUT_FILE libbm1684x_kernel_module)
add_library(${SHARED_LIBRARY_OUTPUT_FILE} SHARED ${DEVICE_SRCS})
target_link_libraries(${SHARED_LIBRARY_OUTPUT_FILE} -Wl,--whole-archive libbm1684x.a -Wl,--no-whole-archive m)
set_target_properties(${SHARED_LIBRARY_OUTPUT_FILE} PROPERTIES PREFIX "" SUFFIX ".so" COMPILE_FLAGS "-fPIC" LINK_FLAGS "-shared")

set(INSTALL_PREFIX ${PROJECT_SOURCE_DIR}/../lib)


# # time md5sum
add_custom_command(
    TARGET ${SHARED_LIBRARY_OUTPUT_FILE} POST_BUILD
    COMMAND rm -rf ${INSTALL_PREFIX}/${SHARED_LIBRARY_OUTPUT_FILE}.so
    COMMAND date > ${INSTALL_PREFIX}/${SHARED_LIBRARY_OUTPUT_FILE}.md5
    COMMAND md5sum ${PROJECT_SOURCE_DIR}/build/${SHARED_LIBRARY_OUTPUT_FILE}.so >> ${INSTALL_PREFIX}/${SHARED_LIBRARY_OUTPUT_FILE}.md5
)

# install
install(TARGETS ${SHARED_LIBRARY_OUTPUT_FILE} DESTINATION ${INSTALL_PREFIX})