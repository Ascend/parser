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
  echo "sh build.sh [-j[n]] [-h] [-v] [-s] [-t] [-u] [-c] [-S on|off]"
  echo ""
  echo "Options:"
  echo "    -h Print usage"
  echo "    -u Only compile ut, not execute"
  echo "    -s Build st"
  echo "    -j[n] Set the number of threads used for building Parser, default is 8"
  echo "    -t Build and execute ut"
  echo "    -c Build ut with coverage tag"
  echo "    -v Display build command"
  echo "    -S Enable enable download cmake compile dependency from gitee , default off"
  echo "to be continued ..."
}

# check value of input is 'on' or 'off'
# usage: check_on_off arg_value arg_name
check_on_off()
{
  if [[ "X$1" != "Xon" && "X$1" != "Xoff" ]]; then
    echo "Invalid value $1 for option -$2"
    usage
    exit 1
  fi
}

# parse and set options
checkopts()
{
  VERBOSE=""
  THREAD_NUM=8
  # ENABLE_PARSER_UT_ONLY_COMPILE="off"
  ENABLE_PARSER_UT="off"
  ENABLE_PARSER_ST="off"
  ENABLE_PARSER_COV="off"
  GE_ONLY="on"
  ENABLE_GITEE="off"
  # Process the options
  while getopts 'ustchj:vS:' opt
  do
    OPTARG=$(echo ${OPTARG} | tr '[A-Z]' '[a-z]')
    case "${opt}" in
      u)
        ENABLE_PARSER_UT="on"
        GE_ONLY="off"
        ;;
      s)
        ENABLE_PARSER_ST="on"
        ;;
      t)
        ENABLE_PARSER_UT="on"
        GE_ONLY="off"
        ;;
      c)
        ENABLE_PARSER_COV="on"
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
      S)
        check_on_off $OPTARG S
        ENABLE_GITEE="$OPTARG"
        echo "enable download from gitee"
        ;;
      *)
        echo "Undefined option: ${opt}"
        usage
        exit 1
    esac
  done
}
checkopts "$@"

git submodule update --init metadef

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
  mk_dir "${BUILD_PATH}"
  cd "${BUILD_PATH}"
  CMAKE_ARGS="-DBUILD_PATH=$BUILD_PATH -DGE_ONLY=$GE_ONLY"

  if [[ "X$ENABLE_PARSER_COV" = "Xon" ]]; then
    CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_PARSER_COV=ON"
  fi

  if [[ "X$ENABLE_PARSER_UT" = "Xon" ]]; then
    CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_PARSER_UT=ON"
  fi


  if [[ "X$ENABLE_PARSER_ST" = "Xon" ]]; then
    CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_PARSER_ST=ON"
  fi

  if [[ "X$ENABLE_GITEE" = "Xon" ]]; then
    CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_GITEE=ON"
  fi

  CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_OPEN_SRC=True -DCMAKE_INSTALL_PREFIX=${OUTPUT_PATH}"
  echo "${CMAKE_ARGS}"
  cmake ${CMAKE_ARGS} ..
  if [ 0 -ne $? ]
  then
    echo "execute command: cmake ${CMAKE_ARGS} .. failed."
    return 1
  fi

  if [ "X$ENABLE_PARSER_UT" = "Xon" ]; then
    make ut_parser -j8
  elif [ "X$ENABLE_PARSER_ST" = "Xon" ]; then
    make st_parser -j8
  else
    make ${VERBOSE} -j${THREAD_NUM} && make install
  fi

  if [ 0 -ne $? ]
  then
    echo "execute command: make ${VERBOSE} -j${THREAD_NUM} && make install failed."
    return 1
  fi
  echo "Parser build success!"
}
g++ -v
mk_dir ${OUTPUT_PATH}
build_parser || { echo "Parser build failed."; return; }
echo "---------------- Parser build finished ----------------"
rm -f ${OUTPUT_PATH}/libgmock*.so
rm -f ${OUTPUT_PATH}/libgtest*.so
rm -f ${OUTPUT_PATH}/lib*_stub.so

chmod -R 750 ${OUTPUT_PATH}
find ${OUTPUT_PATH} -name "*.so*" -print0 | xargs -0 chmod 500

echo "---------------- Parser output generated ----------------"

if [[ "X$ENABLE_PARSER_UT" = "Xon" || "X$ENABLE_PARSER_COV" = "Xon" ]]; then
    cp ${BUILD_PATH}/tests/ut/parser/ut_parser ${OUTPUT_PATH}

    RUN_TEST_CASE=${OUTPUT_PATH}/ut_parser && ${RUN_TEST_CASE}
    if [[ "$?" -ne 0 ]]; then
        echo "!!! UT FAILED, PLEASE CHECK YOUR CHANGES !!!"
        echo -e "\033[31m${RUN_TEST_CASE}\033[0m"
        exit 1;
    fi
    echo "Generating coverage statistics, please wait..."
    cd ${BASEPATH}
    rm -rf ${BASEPATH}/cov
    mkdir ${BASEPATH}/cov
    lcov -c -d build/tests/ut/parser -o cov/tmp.info
    lcov -r cov/tmp.info '*/output/*' '*/build/opensrc/*' '*/build/proto/*' '*/third_party/*' '*/tests/*' '/usr/local/*' '*/metadef/inc/*' -o cov/coverage.info
    cd ${BASEPATH}/cov
    genhtml coverage.info
fi

if [[ "X$ENABLE_PARSER_ST" = "Xon" ]]; then
    cp ${BUILD_PATH}/tests/st/st_parser ${OUTPUT_PATH}

    RUN_TEST_CASE=${OUTPUT_PATH}/st_parser && ${RUN_TEST_CASE}
    if [[ "$?" -ne 0 ]]; then
        echo "!!! ST FAILED, PLEASE CHECK YOUR CHANGES !!!"
        echo -e "\033[31m${RUN_TEST_CASE}\033[0m"
        exit 1;
    fi
    echo "Generating coverage statistics, please wait..."
    cd ${BASEPATH}
    rm -rf ${BASEPATH}/cov
    mkdir ${BASEPATH}/cov
    lcov -c -d build/tests/st -o cov/tmp.info
    lcov -r cov/tmp.info '*/output/*' '*/build/opensrc/*' '*/build/proto/*' '*/third_party/*' '*/tests/*' '/usr/local/*' '*/metadef/inc/*' -o cov/coverage.info
    cd ${BASEPATH}/cov
    genhtml coverage.info
fi

# generate output package in tar form, including ut/st libraries/executables for cann
generate_package()
{
  cd "${BASEPATH}"

  PARSER_LIB_PATH="lib"
  COMPILER_PATH="compiler/lib64"

  COMMON_LIB=("libgraph.so" "libregister.so" "liberror_manager.so")
  PARSER_LIB=("lib_caffe_parser.so" "libfmk_onnx_parser.so" "libfmk_parser.so" "libparser_common.so")

  rm -rf ${OUTPUT_PATH:?}/${COMPILER_PATH}/

  mk_dir "${OUTPUT_PATH}/${COMPILER_PATH}"

  find output/ -name parser_lib.tar -exec rm {} \;

  cd "${OUTPUT_PATH}"

  for lib in "${PARSER_LIB[@]}";
  do
    find ${OUTPUT_PATH}/${PARSER_LIB_PATH} -maxdepth 1 -name "$lib" -exec cp -f {} ${OUTPUT_PATH}/${COMPILER_PATH} \;
  done

  for lib in "${COMMON_LIB[@]}";
  do
    find ${OUTPUT_PATH}/${PARSER_LIB_PATH} -maxdepth 1 -name "$lib" -exec cp -f {} ${OUTPUT_PATH}/${COMPILER_PATH} \;
  done

  find ${OUTPUT_PATH}/${PARSER_LIB_PATH} -maxdepth 1 -name "libc_sec.so" -exec cp -f {} ${OUTPUT_PATH}/${COMPILER_PATH} \;

  tar -cf parser_lib.tar compiler
}

if [[ "X$ENABLE_PARSER_UT" = "Xoff" && "X$ENABLE_PARSER_ST" = "Xoff" ]]; then
  generate_package
fi
echo "---------------- Parser package archive generated ----------------"
