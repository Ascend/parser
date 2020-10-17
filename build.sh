#!/bin/bash
# Copyright 2019-2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

set -e
BASEPATH=$(cd "$(dirname $0)"; pwd)
OUTPUT_PATH="${BASEPATH}/output"
export BUILD_PATH="${BASEPATH}/build/"

# print usage message
usage()
{
  echo "Usage:"
  echo "sh build.sh [-j[n]] [-h] [-v] [-s] [-t] [-u] [-c]"
  echo ""
  echo "Options:"
  echo "    -h Print usage"
  echo "    -u Only compile ut, not execute"
  echo "    -s Build st"
  echo "    -j[n] Set the number of threads used for building GraphEngine, default is 8"
  echo "    -t Build and execute ut"
  echo "    -c Build ut with coverage tag"
  echo "    -v Display build command"
  echo "to be continued ..."
}

# parse and set options
checkopts()
{
  VERBOSE=""
  THREAD_NUM=8
  # ENABLE_GE_UT_ONLY_COMPILE="off"
  ENABLE_GE_UT="off"
  ENABLE_GE_ST="off"
  ENABLE_GE_COV="off"
  GE_ONLY="on"
  # Process the options
  while getopts 'ustchj:v' opt
  do
    OPTARG=$(echo ${OPTARG} | tr '[A-Z]' '[a-z]')
    case "${opt}" in
      u)
        # ENABLE_GE_UT_ONLY_COMPILE="on"
        ENABLE_GE_UT="on"
        GE_ONLY="off"
        ;;
      s)
        ENABLE_GE_ST="on"
        ;;
      t)
	      ENABLE_GE_UT="on"
	      GE_ONLY="off"
	      ;;
      c)
        ENABLE_GE_COV="on"
        GE_ONLY="off"
        ;;
      h)
        usage
        exit 0
        ;;
      j)
        THREAD_NUM=$OPTARG
        ;;
      v)
        VERBOSE="VERBOSE=1"
        ;;
      *)
        echo "Undefined option: ${opt}"
        usage
        exit 1
    esac
  done
}
checkopts "$@"

mk_dir() {
    local create_dir="$1"  # the target to make

    mkdir -pv "${create_dir}"
    echo "created ${create_dir}"
}

# Parser build start
echo "---------------- Parser build start ----------------"

# create build path
build_parser()
{
  echo "create build directory and build Parser";
  mk_dir "${BUILD_PATH}/parser"
  cd "${BUILD_PATH}/parser"
  CMAKE_ARGS="-DBUILD_PATH=$BUILD_PATH -DGE_ONLY=$GE_ONLY"

  if [[ "X$ENABLE_GE_COV" = "Xon" ]]; then
    CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_GE_COV=ON"
  fi

  if [[ "X$ENABLE_GE_UT" = "Xon" ]]; then
    CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_GE_UT=ON"
  fi


  if [[ "X$ENABLE_GE_ST" = "Xon" ]]; then
    CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_GE_ST=ON"
  fi

  CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_OPEN_SRC=True"
  echo "${CMAKE_ARGS}"
  cmake ${CMAKE_ARGS} ../..
  make ${VERBOSE} -j${THREAD_NUM}
  echo "Parser build success!"
}
g++ -v
build_parser
echo "---------------- Parser build finished ----------------"
mk_dir ${OUTPUT_PATH}
cp -rf "${BUILD_PATH}/parser/"*.so "${OUTPUT_PATH}"
rm -rf "${OUTPUT_PATH}/"libproto*
rm -f ${OUTPUT_PATH}/libgmock*.so
rm -f ${OUTPUT_PATH}/libgtest*.so
rm -f ${OUTPUT_PATH}/lib*_stub.so

chmod -R 750 ${OUTPUT_PATH}
find ${OUTPUT_PATH} -name "*.so*" -print0 | xargs -0 chmod 500

echo "---------------- Parser output generated ----------------"

# generate output package in tar form, including ut/st libraries/executables
generate_package()
{
  cd "${BASEPATH}"
  FWK_PATH="fwkacllib/lib64"
  ATC_PATH="atc/lib64"
  NNENGINE_PATH="plugin/nnengine/ge_config"
  OPSKERNEL_PATH="plugin/opskernel"

  ATC_LIB=("libc_sec.so" "libge_common.so" "libge_compiler.so" "libgraph.so")
  FWK_LIB=("libge_common.so" "libge_runner.so" "libgraph.so")

  rm -rf ${OUTPUT_PATH:?}/${FWK_PATH}/
  rm -rf ${OUTPUT_PATH:?}/${ATC_PATH}/
  mk_dir "${OUTPUT_PATH}/${FWK_PATH}/${NNENGINE_PATH}"
  mk_dir "${OUTPUT_PATH}/${FWK_PATH}/${OPSKERNEL_PATH}"
  mk_dir "${OUTPUT_PATH}/${ATC_PATH}/${NNENGINE_PATH}"
  mk_dir "${OUTPUT_PATH}/${ATC_PATH}/${OPSKERNEL_PATH}"

  find output/ -name graphengine_lib.tar -exec rm {} \;
  cp ge/engine_manager/engine_conf.json ${OUTPUT_PATH}/${FWK_PATH}/${NNENGINE_PATH}
  cp ge/engine_manager/engine_conf.json ${OUTPUT_PATH}/${ATC_PATH}/${NNENGINE_PATH}

  find output/ -maxdepth 1 -name libengine.so -exec cp -f {} ${OUTPUT_PATH}/${FWK_PATH}/${NNENGINE_PATH}/../ \;
  find output/ -maxdepth 1 -name libengine.so -exec cp -f {} ${OUTPUT_PATH}/${ATC_PATH}/${NNENGINE_PATH}/../ \;

  find output/ -maxdepth 1 -name libge_local_engine.so -exec cp -f {} ${OUTPUT_PATH}/${FWK_PATH}/${OPSKERNEL_PATH} \;
  find output/ -maxdepth 1 -name libge_local_engine.so -exec cp -f {} ${OUTPUT_PATH}/${ATC_PATH}/${OPSKERNEL_PATH} \;

  cd "${OUTPUT_PATH}"
  for lib in "${ATC_LIB[@]}";
  do
    cp "$lib" "${OUTPUT_PATH}/${ATC_PATH}"
  done

  for lib in "${FWK_LIB[@]}";
  do
    cp "$lib" "${OUTPUT_PATH}/${FWK_PATH}"
  done

  tar -cf graphengine_lib.tar fwkacllib/ atc/
}

if [[ "X$ENABLE_GE_UT" = "Xoff" ]]; then
  generate_package
fi
echo "---------------- GraphEngine package archive generated ----------------"
