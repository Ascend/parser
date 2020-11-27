/**
 * Copyright 2020 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "common/op_map.h"

#include <map>
#include <string>
#include <vector>

#include "framework/omg/parser/parser_types.h"
#include "register/op_registry.h"

using std::map;
using std::string;
using std::vector;
using namespace ge::parser;

namespace ge {
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY std::map<std::string, std::string> caffe_op_map = {
    {"Input", DATA},
    {"DummyData", DATA},
    {"Reshape", RESHAPE},
    {"Dropout", DROPOUT},
    {"NetOutput", NETOUTPUT},
};

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY std::map<std::string, std::string> tensorflow_op_map = {
    {"BroadcastGradientArgs", BROADCASTGRADIENTARGS},
    {"StopGradient", STOPGRADIENT},
    {"ExpandDims", EXPANDDIMS},
    {"DestroyTemporaryVariable", DESTROYTEMPORARYVARIABLE},
    {"GuaranteeConst", GUARANTEECONST},
    {"BroadcastArgs", BROADCASTARGS},
    {"PreventGradient", PREVENTGRADIENT},
    {"Empty", EMPTY},
    {"Placeholder", DATA},
    {"ControlTrigger", CONTROLTRIGGER},
    {"_ParallelConcatStart", PARALLELCONCATSTART},
    {"Const", CONSTANT},
    {"FrameworkOp", FRAMEWORKOP},
    {"Reshape", RESHAPE},
    {"Squeeze", SQUEEZE},
    {"Enter", ENTER},
    {"RefEnter", REFENTER},
    {"Exit", EXIT},
    {"RefExit", REFEXIT},
    {"LoopCond", LOOPCOND},
    {"NextIteration", NEXTITERATION},
    {"RefNextIteration", REFNEXTITERATION},
    {"Identity", IDENTITY},
    {"IdentityN", IDENTITYN},
    {"PlaceholderWithDefault", PLACEHOLDERWITHDEFAULT},
    {"Size", SIZE},
    {"Shape", SHAPE},
    {"ShapeN", SHAPEN},
    {"Fill", FILL},
    {"Rank", RANK},
    {"Merge", MERGE},
    {"RefMerge", REFMERGE},
    {"Switch", SWITCH},
    {"RefSwitch", REFSWITCH},
    {"LayerNorm", LAYERNORM},
    {"RNN", RNN},
    {"_Arg", ARG},
    {"_Retval", FRAMEWORKOP},
    {"Bitcast", BITCAST},
    {"Snapshot", SNAPSHOT},
};

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY map<string, string> tensorflow_train_op_map = {
    {"BroadcastGradientArgs", BROADCASTGRADIENTARGS},
    {"StopGradient", STOPGRADIENT},
    {"ExpandDims", EXPANDDIMS},
    {"DestroyTemporaryVariable", DESTROYTEMPORARYVARIABLE},
    {"TemporaryVariable", TEMPORARYVARIABLE},
    {"GuaranteeConst", GUARANTEECONST},
    {"BroadcastArgs", BROADCASTARGS},
    {"PreventGradient", PREVENTGRADIENT},
    {"Empty", EMPTY},
    {"ControlTrigger", CONTROLTRIGGER},
    {"_Arg", ARG},
    {"_ParallelConcatStart", PARALLELCONCATSTART},
    {"Const", CONSTANTOP},
    {"VariableV2", VARIABLE},
    {"VarHandleOp", VARHANDLEOP},
    {"VarIsInitializedOp", VARISINITIALIZEDOP},
    {"IsVariableInitialized", ISVARIABLEINITIALIZED},
    {"ReadVariableOp", READVARIABLEOP},
    {"Reshape", RESHAPE},
    {"Squeeze", SQUEEZE},
    {"NoOp", NOOP},
    {"Enter", ENTER},
    {"RefEnter", REFENTER},
    {"Exit", EXIT},
    {"RefExit", REFEXIT},
    {"LoopCond", LOOPCOND},
    {"NextIteration", NEXTITERATION},
    {"RefNextIteration", REFNEXTITERATION},
    {"Identity", IDENTITY},
    {"IdentityN", IDENTITYN},
    {"PlaceholderWithDefault", PLACEHOLDERWITHDEFAULT},
    {"Size", SIZE},
    {"Shape", SHAPE},
    {"ShapeN", SHAPEN},
    {"Rank", RANK},
    {"Merge", MERGE},
    {"Switch", SWITCH},
    {"LayerNorm", LAYERNORM},
    {"LayerNormGrad", LAYERNORMGRAD},
    {"Dropout", DROPOUT},
    {"Bitcast", BITCAST},
    {"Snapshot", SNAPSHOT},
};

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY map<string, int32_t> op_output_tensor_num = {
    {SSDDETECTIONOUTPUT, 3},
    {REFINEDETDETECTIONOUTPUT, 3},
    {FSRDETECTIONOUTPUT, 2},
    {FASTERRCNNFIRSTSTAGEPOSTPROCESSOR, 4},
    {FASTERRCNNSECONDSTAGEPOSTPROCESSOR, 4},
    {YOLODETECTIONOUTPUT, 2},
    {FASTRCNNPREDICTIONS, 4},
    {RPNPROPOSALS, 3},
    {MAXPOOLWITHARGMAX, 2},
    {REGION, 3},
    {TOPKV2, 2},
    {LogTimeStamp, 0},
    /* training op */
    {MAXPOOLWITHARGMAX, 2},
    {FUSEDBATCHNORM, 5},
    {FUSEDBATCHNORMGRAD, 3},
    {SHAPEN, 0},
    {SSDPOSTPROCESSOR, 4},
    {LAYERNORM, 3},
    {LAYERNORMGRAD, 3},
    {SPARSESOFTMAXCROSSENTROPYWITHLOGITS, 2},
};

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY vector<string> local_framework_op_vec = {
    "TensorDataset", "QueueDataset", "DeviceQueueDataset", "ParallelMapDataset", "BatchDatasetV2",
    "IteratorV2",    "MakeIterator", "IteratorGetNext",    "FilterDataset",      "MapAndBatchDatasetV2"};

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY vector<string> is_dataset_op_vec = {
    "TensorDataset", "QueueDataset", "DeviceQueueDataset", "ParallelMapDataset", "BatchDatasetV2",
    "IteratorV2",    "MakeIterator", "IteratorGetNext",    "FilterDataset",      "MapAndBatchDatasetV2"};
}  //  namespace ge
