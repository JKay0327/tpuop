

export UNTPU_TOP=$(cd $(dirname "${BASH_SOURCE[0]}") && pwd)

function build_cmodel_device_op()
{
    export TPUKERNEL_DEV_MODE="cmodel"
    export BMLIB_CMODEL_PATH=$UNTPU_TOP/device/lib/libcmodel_firmware.so
    export LD_LIBRARY_PATH=""
    mkdir -p $UNTPU_TOP/device/build
    pushd $UNTPU_TOP/device/build
    # rm -rf ./*
    cmake ..
    make -j8
    # make install
    popd
    set_cmodel
}

function set_cmodel()
{
    export TPUKERNEL_DEV_MODE="cmodel"
    export BMLIB_CMODEL_PATH=$UNTPU_TOP/device/lib/libcmodel_firmware.so
    export PS1="\[\e[1;35m\](${TPUKERNEL_DEV_MODE})\[\e[1;33m\]\w\[\e[1;34m\]\$ \[\e[0m\]"
    pushd $UNTPU_TOP/device/lib/ >/dev/null && {
        ln -sf libbmlib_cmodel.so libbmlib.so
        ln -sf libbmlib_cmodel.so libbmlib.so.0
        popd > /dev/null
    }
    # export TPUKERNEL_FIRMWARE_PATH=$UNTPU_TOP/device/lib/libcmodel.so
    export TPUKERNEL_FIRMWARE_PATH=$UNTPU_TOP/device/build/libcmodel.so
    export LD_LIBRARY_PATH=$UNTPU_TOP/device/lib/:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=$UNTPU_TOP/device/build/:$LD_LIBRARY_PATH
}