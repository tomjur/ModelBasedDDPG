#!/usr/bin/env bash
set -eo pipefail
set -xv

# defaults to -j (cpus)
: ${MAKEOPTS:="-j $(nproc)"}
export MAKEOPTS

if [ -z "${VIRTUAL_ENV}" ]; then echo "venv not detected, aborting.";  exit 1; else echo "using venv '$VIRTUAL_ENV'"; fi

DEPS_DIR=/tmp
: ${BUILD_DIR:=$VIRTUAL_ENV/build}

OGS_BUILD_DIR=$BUILD_DIR/openscenegraph
FCL_BUILD_DIR=$BUILD_DIR/fcl
OR_BUILD_DIR=$BUILD_DIR/openrave
TRAJOPT_BUILD_DIR=$BUILD_DIR/trajopt

PYTHON_LIB_DIR=$(python -c "import sys; print(sys.prefix)")
PYTHON_VERSION=$(python -c "import sys; print('{}.{}'.format(*sys.version_info[:2]))")

PYTHON_EXECUTABLE=$(which python)
if [ ${PYTHON_VERSION:0:1} -ge 3 ]; then
    PYTHON_INCLUDE_DIR=/usr/include/python${PYTHON_VERSION}m
    PYTHON_LIBRARY=/usr/lib/python${PYTHON_VERSION}/config-${PYTHON_VERSION}m-x86_64-linux-gnu/libpython${PYTHON_VERSION}m.so
else
    PYTHON_INCLUDE_DIR=/usr/include/python${PYTHON_VERSION}
    PYTHON_LIBRARY=/usr/lib/python${PYTHON_VERSION}/config-x86_64-linux-gnu/libpython${PYTHON_VERSION}.so
fi

if [ ! -d "$BUILD_DIR" ]; then
    mkdir $BUILD_DIR
fi

make() {
    # automake since 4.1 dngaf about MAKEOPTS.
    command make $MAKEOPTS "$@"
}

cleanup() {
    if [ -d "$OGS_BUILD_DIR" ] && [ ! -f "$OGS_BUILD_DIR/install_manifest.txt" ]; then
        rm -r $OGS_BUILD_DIR
    fi
    if [ -d "$FCL_BUILD_DIR" ] && [ ! -f "$FCL_BUILD_DIR/install_manifest.txt" ]; then
        rm -r $FCL_BUILD_DIR
    fi
    if [ -d "$OR_BUILD_DIR" ] && [ ! -f "$OR_BUILD_DIR/install_manifest.txt" ]; then
        rm -r $OR_BUILD_DIR
    fi
    if [ -d "$TRAJOPT_BUILD_DIR" ]; then
        # Remove hacky fix for osg renderer
        sed -i -e ':a;N;$!ba;s/return geode;\n  if (!geom/if (!geom/g' $DEPS_DIR/trajopt/src/osgviewer/osgviewer.cpp
        # Remove hacky fix to link OpenRAVE
        sed -i -e '/OpenRAVE_LIBRARY_DIRS/d' $DEPS_DIR/trajopt/CMakeLists.txt

        if [ ! -f "$VIRTUAL_ENV/lib/python${PYTHON_VERSION}/site-packages/trajopt.pth" ]; then
            rm -r $TRAJOPT_BUILD_DIR
        fi
    fi
}
trap cleanup EXIT

hack-clean-post-build() {
    if [[ -d "${VIRTUAL_ENV:?}/lib64" ]]; then
        echo "[forever unclean]" >&2
        rsync -avhPS "$VIRTUAL_ENV/lib64/" "$VIRTUAL_ENV/lib"
        rm -rf "$VIRTUAL_ENV/lib64"
        ln -sfvr "$VIRTUAL_ENV/lib" "$VIRTUAL_ENV/lib64"
    fi
}

openscenegraph() {
    if [ ! -d "$OGS_BUILD_DIR" ]; then
        mkdir $OGS_BUILD_DIR
        cd $OGS_BUILD_DIR
        cmake $DEPS_DIR/openscenegraph \
                -DDESIRED_QT_VERSION=4 \
                -DCMAKE_INSTALL_PREFIX=$VIRTUAL_ENV
        make
        make install
    fi
    hack-clean-post-build
}

fcl() {
    if [ ! -d "$FCL_BUILD_DIR" ]; then
        mkdir $FCL_BUILD_DIR
        cd $FCL_BUILD_DIR
        cmake $DEPS_DIR/fcl \
                -DCMAKE_INSTALL_PREFIX=$VIRTUAL_ENV \
                -DCMAKE_CXX_FLAGS="-w"
        make
        make install
    fi
    hack-clean-post-build
}

openrave() {
    if [ ! -d "$OR_BUILD_DIR" ]; then
        mkdir $OR_BUILD_DIR
        cd $OR_BUILD_DIR
        cmake $DEPS_DIR/openrave \
                -DPYTHON_EXECUTABLE=$PYTHON_EXECUTABLE \
                -DPYTHON_INCLUDE_DIR=$PYTHON_INCLUDE_DIR \
                -DPYTHON_LIBRARY=$PYTHON_LIBRARY \
                -DODE_USE_MULTITHREAD=ON \
                -DOPT_BULLET=OFF \
                -DOPT_MATLAB=OFF \
                -DOPT_OCTAVE=OFF \
                -DOPT_VIDEORECORDING=OFF \
                -DCMAKE_PREFIX_PATH=$VIRTUAL_ENV \
                -DCMAKE_INSTALL_PREFIX=$VIRTUAL_ENV \
                -DPKG_CONFIG_USE_CMAKE_PREFIX_PATH=TRUE \
                -DCMAKE_CXX_FLAGS="-w"
                # -DOPT_FCL_COLLISION=OFF
        make
        make install

        # Remove build artifacts from source directory
        rm -rf "$DEPS_DIR/openrave/__pycache__"
    fi
    hack-clean-post-build
}

trajopt() {
    if [ ! -d "$TRAJOPT_BUILD_DIR" ]; then
        mkdir $TRAJOPT_BUILD_DIR
        cd $TRAJOPT_BUILD_DIR
        # Add hacky fix for osg renderer
        sed -i -e 's/if (!geom/return geode;\n  if (!geom/g' $DEPS_DIR/trajopt/src/osgviewer/osgviewer.cpp
        # Add hacky fix to link OpenRAVE
        sed -i -e 's/find_package(OpenRAVE REQUIRED)/find_package(OpenRAVE REQUIRED)\nlink_directories(${OpenRAVE_LIBRARY_DIRS})/g' $DEPS_DIR/trajopt/CMakeLists.txt
        cmake $DEPS_DIR/trajopt \
                -DPYTHON_EXECUTABLE=$PYTHON_EXECUTABLE \
                -DPYTHON_INCLUDE_DIR=$PYTHON_INCLUDE_DIR \
                -DPYTHON_LIBRARY=$PYTHON_LIBRARY \
                -DCMAKE_PREFIX_PATH=$VIRTUAL_ENV \
                -DOSG_DIR=$VIRTUAL_ENV/lib64
        make

        if [ ! -f "$VIRTUAL_ENV/lib/python${PYTHON_VERSION}/site-packages/trajopt.pth" ]; then
            echo "$DEPS_DIR/trajopt" >> $VIRTUAL_ENV/lib/python${PYTHON_VERSION}/site-packages/trajopt.pth
            echo "$TRAJOPT_BUILD_DIR/lib" >> $VIRTUAL_ENV/lib/python${PYTHON_VERSION}/site-packages/trajopt.pth
        fi
    fi
    hack-clean-post-build
}

build-all() {
    openscenegraph
    fcl
    openrave
    trajopt
}

main() {
    local cmd="${0#*/}"

    case "$cmd" in
        openscenegraph|fcl|openrave|trajopt) ;;
        *) cmd=build-all ;;
    esac

    "$cmd" "$@"
}

main "$@"
