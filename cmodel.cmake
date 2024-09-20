# use libbmlib libbmcv
if(DEFINED ENV{CMODEL_REGRESSION})
include_directories($ENV{LIBSOPHON_TOP}/bmlib/include)
include_directories($ENV{LIBSOPHON_TOP}/bmlib/src)
endif()

if(NOT DEFINED ENV{CMODEL_REGRESSION})
find_package(libsophon REQUIRED)
include_directories(${LIBSOPHON_INCLUDE_DIRS})
endif()

set(PROJECT_SOURCE_DIR $ENV{UNTPU_TOP})

# include_directories(
#     ${PROJECT_SOURCE_DIR}/include/device ${PROJECT_SOURCE_DIR}/include/kernel ${PROJECT_SOURCE_DIR}/include
# )

link_directories($ENV{TPUKERNEL_TOP}/lib)
# Add chip arch defination
add_definitions(-D__bm$ENV{FIRMWARE_CHIPID}__)
aux_source_directory(src KERNEL_SRC_FILES)
add_library(firmware SHARED ${KERNEL_SRC_FILES})
target_include_directories(firmware PRIVATE
    ${PROJECT_SOURCE_DIR}/include/device ${PROJECT_SOURCE_DIR}/include/kernel ${PROJECT_SOURCE_DIR}/include
)
target_compile_definitions(firmware PRIVATE -DUSING_CMODEL)

target_link_libraries(firmware PRIVATE $ENV{BMLIB_CMODEL_PATH} m)
set_target_properties(firmware PROPERTIES OUTPUT_NAME cmodel)


# install(TARGETS firmware DESTINATION ${PROJECT_SOURCE_DIR}/device/lib)