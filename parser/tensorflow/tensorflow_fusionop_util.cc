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

#include "parser/tensorflow/tensorflow_fusionop_util.h"
#include "common/util/error_manager/error_manager.h"
#include "framework/common/debug/ge_log.h"
#include "parser/common/util.h"
#include "parser/tensorflow/tensorflow_parser.h"
#include "framework/omg/parser/parser_types.h"

#include <iostream>
#include <cstdlib>
#include <memory>

using domi::tensorflow::NodeDef;

namespace ge {
// constraint: At present, only a few fixed fusion operators are supported,
// and forward matching method is used for recognition
// eg: in the MaskRCNN network,
// clip_boxes are treated as fusion operators but generate_rpn_proposals/clip_boxes is also fused
// considered to be a child operator of generate_rpn_proposals.
// clip_boxes
// fastrcnn_predictions
// decode_bbox_target
// generate_rpn_proposals
// roi_align
// cond_1/roi_align
namespace {
const char *const kLstmCellKernelFw = "fw/basic_lstm_cell/kernel";
const char *const kLstmCellKernelBw = "bw/basic_lstm_cell/kernel";
const char *const kLstmCellBiasFw = "fw/basic_lstm_cell/bias";
const char *const kLstmCellBiasBw = "bw/basic_lstm_cell/bias";
const char *const kAttentionDecoderEmbeeding = "embedding_attention_decoder/embedding";
const char *const kAttentionDecoderAttenW0 = "embedding_attention_decoder/attention_decoder/AttnW_0";
const char *const kAttentionDecoderAttenVa = "embedding_attention_decoder/attention_decoder/AttnV_0";
const char *const kAttentionDecoderAttentionDecoderKernel = "embedding_attention_decoder/attention_decoder/kernel";
const char *const kAttentionDecoderAtteBias = "embedding_attention_decoder/attention_decoder/bias";
const char *const kAttentionDecoderCell0GatesKernel =
    "embedding_attention_decoder/attention_decoder/multi_rnn_cell/cell_0/gru_cell/gates/kernel";
const char *const kAttentionDecoderCell0GatesBias =
    "embedding_attention_decoder/attention_decoder/multi_rnn_cell/cell_0/gru_cell/gates/bias";
const char *const kAttentionDecoderCell0CandidateKernel =
    "embedding_attention_decoder/attention_decoder/multi_rnn_cell/cell_0/gru_cell/candidate/kernel";
const char *const kAttentionDecoderCell0CandidateBias =
    "embedding_attention_decoder/attention_decoder/multi_rnn_cell/cell_0/gru_cell/candidate/bias";
const char *const kAttentionDecoderCell1GatesKernel =
    "embedding_attention_decoder/attention_decoder/multi_rnn_cell/cell_1/gru_cell/gates/kernel";
const char *const kAttentionDecoderCell1GatesBias =
    "embedding_attention_decoder/attention_decoder/multi_rnn_cell/cell_1/gru_cell/gates/bias";
const char *const kAttentionDecoderCell1CandidateKernel =
    "embedding_attention_decoder/attention_decoder/multi_rnn_cell/cell_1/gru_cell/candidate/kernel";
const char *const kAttentionDecoderCell1CandidateBias =
    "embedding_attention_decoder/attention_decoder/multi_rnn_cell/cell_1/gru_cell/candidate/bias";
const char *const kAttentionDecoderAttention0Kernel =
    "embedding_attention_decoder/attention_decoder/Attention_0/kernel";
const char *const kAttentionDecoderAttention0Bias = "embedding_attention_decoder/attention_decoder/Attention_0/bias";
const char *const kAttentionDecoderAttnOutputProjectionKernel =
    "embedding_attention_decoder/attention_decoder/AttnOutputProjection/kernel";
const char *const kAttentionDecoderAttnOutputProjectionBias =
    "embedding_attention_decoder/attention_decoder/AttnOutputProjection/bias";
const char *const kHuberLossFill = "gradients/Fill";
const char *const kHuberLossConst = "huber_loss/Const";
const char *const kHuberLossMul2X = "huber_loss/Mul_2/x";
const char *const kSparseSoftmaxConst = "sparse_softmax_cross_entropy_loss/Const";
const char *const kDeeplabV3ConfusionMatrix = "Select";
const char *const kDeeplabV3ConfusionMatrix1 = "ToFloat_1";
const char *const kConstantFoldingSuffix = "ConstantFolding/";
}  // namespace
vector<string> const_op_update_vec = {kLstmCellKernelFw,
                                      kLstmCellKernelBw,
                                      kLstmCellBiasFw,
                                      kLstmCellBiasBw,
                                      kAttentionDecoderAttenW0,
                                      kAttentionDecoderAttention0Kernel,
                                      kAttentionDecoderAttnOutputProjectionKernel,
                                      kAttentionDecoderAttentionDecoderKernel,
                                      kAttentionDecoderCell0GatesKernel,
                                      kAttentionDecoderCell0CandidateKernel,
                                      kAttentionDecoderCell1GatesKernel,
                                      kAttentionDecoderCell1CandidateKernel,
                                      kAttentionDecoderAttention0Bias,
                                      kAttentionDecoderAttnOutputProjectionBias,
                                      kAttentionDecoderAtteBias,
                                      kAttentionDecoderCell0GatesBias,
                                      kAttentionDecoderCell0CandidateBias,
                                      kAttentionDecoderCell1GatesBias,
                                      kAttentionDecoderCell1CandidateBias,
                                      kAttentionDecoderEmbeeding,
                                      kAttentionDecoderAttenVa,
                                      kHuberLossFill,
                                      kHuberLossConst,
                                      kHuberLossMul2X,
                                      kSparseSoftmaxConst,
                                      kDeeplabV3ConfusionMatrix,
                                      kDeeplabV3ConfusionMatrix1};

static map<string, string> tensorflow_fusionop_map = {
};

// <Types of fusion operators, Number of children operators>
static map<string, vector<int>> tensorflow_fusionop_children_nums_map = {
    {ge::parser::CLIPBOXES, {8}},
    {ge::parser::FASTRCNNPREDICTIONS, {118, 119, 120, 123, 125}},
    {ge::parser::RPNPROPOSALS, {75, 85, 97}},
    {ge::parser::DECODEBBOX, {24, 28}},
    {ge::parser::ROIALIGN, {82, 83, 84}},
    {ge::parser::FUSIONBATCHNORM, {8}},
    {ge::parser::GETSPAN, {81, 71, 91}},  // The pbtxt only has 62 nodes when test GetSpan sub net. However the
    {ge::parser::HUBERLOSSGRAD, {8, 9, 10, 20, 21}},
};

// <Types of fusion operators, Name of children operators(Remove the prefixes and/)>
static map<string, vector<string>> tensorflow_fusionop_children_names_map = {
    {ge::parser::FUSIONBATCHNORM, {"add/y", "add", "Rsqrt", "mul", "mul_1", "mul_2", "sub", "add_1"}},
    {ge::parser::GETSPAN, {}},
    {ge::parser::HUBERLOSSGRAD, {}},
};

// ----------------------------Index table of input and output of fusion operator--------------
// The specific operator is the input and output of the whole fusion operator, and the index number is specified
// Considering that an operator may have multiple inputs / outputs, vector is used to save
// search method: new_index=vector(old_index),
// Generally, the old index is 0. If the new index value is kFusionDisableIndex, the edge can be ignored.
// If it is control edge input, the index is graph::kControlSlot(-1).
static map<string, vector<std::pair<string, vector<int32_t>>>> tensorflow_fusionop_inputs_map = {
    {ge::parser::FUSIONBATCHNORM,
     {{"mul_1", {0, kFusionDisableIndex}},
      {"mul", {1, 1}},
      {"sub", {2, kFusionDisableIndex}},
      {"mul_2", {3, kFusionDisableIndex}},
      {"add", {4, kFusionDisableIndex}}}},
    {ge::parser::GETSPAN, {{"transpose", {0}}, {"TensorArray", {1}}, {"transpose_1", {2}}}},
    {ge::parser::HUBERLOSSGRAD, {{"Sub_1_grad/Neg", {1}}, {"Abs_grad/Sign", {0}}}},
};

static map<string, vector<std::pair<string, vector<int32_t>>>> tensorflow_fusionop_outputs_map = {
    {ge::parser::FUSIONBATCHNORM, {{"add_1", {0}}}},
    {ge::parser::GETSPAN, {{"while/Exit_1", {0}}, {"while/Exit_2", {1}}}},
    {ge::parser::HUBERLOSSGRAD, {{"Abs_grad/mul", {0}}}},
};
map<string, vector<std::pair<string, uint32_t>>> tensorflow_fusionop_input_const_weight_index_map = {
    {ge::parser::FUSIONBATCHNORM, {{"mul", 0}, {"sub", 1}, {"mul_2", 2}, {"add", 3}}},
};

// Can a string be converted to an integer
bool TensorFlowFunsionOPUtil::IsIntegerStr(const string &index_str) {
  try {
    if (std::stoi(index_str) > 0) {
      return true;
    }
  } catch (std::invalid_argument &) {
    GELOGE(FAILED, "index_str:%s is invalid", index_str.c_str());
  } catch (std::out_of_range &) {
    GELOGE(FAILED, "index_str:%s is out of range", index_str.c_str());
  } catch (...) {
    GELOGE(FAILED, "index_str:%s cannot change to int s", index_str.c_str());
  }
  return false;
}

// Get child node name of fusion operator.
// eg: input: fastrcnn_predictions/map/TensorArray_2 output: map/TensorArray_2
string TensorFlowFunsionOPUtil::GetChildName(const string &node_name, const string &fusion_node_name) {
  GE_CHK_BOOL_EXEC_NOLOG(
      (node_name.length() - fusion_node_name.length()) > 0, GELOGW("fusion_node_name length not valid."); return "";);

  string child_name;
  string sub_name;

  // node_name begin with "ConstantFolding/"
  if (node_name.find(kConstantFoldingSuffix) == 0) {
    auto length = strlen(kConstantFoldingSuffix);
    sub_name =
        node_name.substr(fusion_node_name.length() + length, node_name.length() - fusion_node_name.length() - length);
  } else {
    sub_name = node_name.substr(fusion_node_name.length(), node_name.length() - fusion_node_name.length());
  }

  auto index = sub_name.find('/');
  if (index != string::npos) {
    child_name = sub_name.substr(index + 1, sub_name.length() - index - 1);
  }

  return child_name;
}

// Check whether the operator node name can be a fusion operator
bool TensorFlowFunsionOPUtil::MaybeFusionOp(const string &node_name, ScopeFusionOpInfo *info) {
  GE_CHK_BOOL_EXEC(info != nullptr, return false, "info is null.");
  info->node_name = node_name;
  // Direct forward matching
  for (auto iter = tensorflow_fusionop_map.begin(); iter != tensorflow_fusionop_map.end(); ++iter) {
    const string fop_name = iter->first;

    string node_name_tmp = node_name;
    // begin with "ConstantFolding/"
    if (node_name_tmp.find(kConstantFoldingSuffix) == 0) {
      auto length = strlen(kConstantFoldingSuffix);
      node_name_tmp = node_name.substr(length, node_name.length() - length);
    }

    // not match
    if (node_name_tmp.find(fop_name) != 0) {
      continue;
    }

    // match,"FusionName/" scene:
    if (node_name_tmp.substr(fop_name.length(), 1) == string("/")) {
      info->fusion_node_name = fop_name;
      info->fusion_op_type = tensorflow_fusionop_map[fop_name];
      info->description = "";
      info->scope_pass = false;
      return true;
    }

    // match "FusionName_Index/" scene:
    // special characters need unified definition
    string sub_name = node_name_tmp.substr(fop_name.length(), node_name_tmp.length() - fop_name.length());
    auto index = sub_name.find('/');
    if ((sub_name.substr(0, 1) == string("_")) && (index > 1) && IsIntegerStr(sub_name.substr(1, index - 1))) {
      info->fusion_node_name = fop_name + sub_name.substr(0, index);
      info->fusion_op_type = tensorflow_fusionop_map[fop_name];
      info->description = "";
      info->scope_pass = false;
      return true;
    }
  }

  return false;
}

// Confirm whether it is a fusion operator
bool TensorFlowFunsionOPUtil::IsFusionOp(const domi::tensorflow::NodeDef *node_def) {
  GE_CHK_BOOL_EXEC(node_def != nullptr, return false, "node_def is null.");
  string type = node_def->op();
  auto iter = tensorflow_fusionop_children_nums_map.find(type);
  return iter != tensorflow_fusionop_children_nums_map.end();
}

// Check the validity of fusion operator (all child nodes)
Status TensorFlowFunsionOPUtil::CheckFusionOpChildren(const string &fusion_node_name,
                                                      const vector<const domi::tensorflow::NodeDef *> &nodedef_list,
                                                      const string &funsion_op_type) {
  // Number matching of fusion operators
  auto iter_children_nums = tensorflow_fusionop_children_nums_map.find(funsion_op_type);
  if (iter_children_nums == tensorflow_fusionop_children_nums_map.end()) {
    REPORT_INNER_ERROR("E19999", "Op[%s]'s optype[%s] not a Fusion OP, check invalid",
                       fusion_node_name.c_str(), funsion_op_type.c_str());
    GELOGE(domi::INTERNAL_ERROR,
        "Op[%s]'s optype[%s] not a Fusion OP!", fusion_node_name.c_str(), funsion_op_type.c_str());
    return domi::INTERNAL_ERROR;
  }

  vector<int> children_nums = iter_children_nums->second;
  bool find = false;
  int children_num = nodedef_list.size();
  for (uint32_t i = 0; i < children_nums.size(); i++) {
    if (children_nums[i] == children_num) {
      find = true;
      break;
    }
  }

  if (!find) {
    REPORT_INNER_ERROR("E19999", "CheckFusionOp op[%s]'s optype[%s] children_nums[%d] is not the same for define",
                       fusion_node_name.c_str(), funsion_op_type.c_str(), children_num);
    GELOGE(domi::INTERNAL_ERROR,
           "Op[%s]'s optype[%s] children_nums:%d is not the same for define.",
           fusion_node_name.c_str(),
           funsion_op_type.c_str(),
           children_num);
    return domi::INTERNAL_ERROR;
  }

  // Key children operators matching
  auto iter_children_names = tensorflow_fusionop_children_names_map.find(funsion_op_type);
  if (iter_children_names != tensorflow_fusionop_children_names_map.end()) {
    vector<string> children_names = iter_children_names->second;
    if (!children_names.empty()) {
      uint32_t count = 0;
      for (uint32_t i = 0; i < children_names.size(); i++) {
        for (uint32_t j = 0; j < nodedef_list.size(); j++) {
          const domi::tensorflow::NodeDef *node_def = nodedef_list[j];
          GE_CHECK_NOTNULL(node_def);
          string node_name = node_def->name();
          string child_name = GetChildName(node_name, fusion_node_name);
          if (children_names[i] == child_name) {
            count++;
            break;
          }
        }
      }

      GE_IF_BOOL_EXEC(count != children_names.size(),
          REPORT_INNER_ERROR("E19999", "Op[%s]'s optype[%s] has no enough importance child.", fusion_node_name.c_str(),
              funsion_op_type.c_str());
          GELOGE(domi::INTERNAL_ERROR, "Op[%s]'s optype[%s] has no enough importance child.", fusion_node_name.c_str(),
              funsion_op_type.c_str());
          return domi::INTERNAL_ERROR;);
    }
  }

  return SUCCESS;
}

// Get the child node of the fusion operator as the input / output index number of the whole fusion operator
Status TensorFlowFunsionOPUtil::GetNodeindex(
    const ScopeFusionOpInfo &info, const int32_t old_index, int32_t &new_index,
    const map<string, vector<std::pair<string, vector<int32_t>>>> &fusionop_context_map) {
  auto iter = fusionop_context_map.find(info.fusion_op_type);

  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(iter == fusionop_context_map.end(),
      return domi::INTERNAL_ERROR,
      "Op[%s] could not find item of optype[%s] in fusionop_context_map",
      info.node_name.c_str(), info.fusion_op_type.c_str());

  vector<std::pair<string, vector<int32_t>>> pairs = iter->second;

  string child_name = GetChildName(info.node_name, info.fusion_node_name);

  GELOGI("GetNodeindex: info.node_name:%s, old_index:%d", info.node_name.c_str(), old_index);
  for (const auto &pair : pairs) {
    if (pair.first == child_name) {
      vector<int32_t> indexs = pair.second;
      if (static_cast<int32_t>(indexs.size()) < (old_index + 1)) {
        new_index = kFusionDisableIndex;
        return SUCCESS;
      }

      if (old_index != -1) {
        new_index = indexs[old_index];
        return SUCCESS;
      }
    }
  }

  new_index = kFusionDisableIndex;
  return SUCCESS;
}

// Get the input index of the fusion operator
Status TensorFlowFunsionOPUtil::GetInPutIndex(const ScopeFusionOpInfo &info, const int32_t old_index,
                                              int32_t &new_index) {
  return GetNodeindex(info, old_index, new_index, tensorflow_fusionop_inputs_map);
}

// Get the output index of the fusion operator
Status TensorFlowFunsionOPUtil::GetOutPutIndex(const ScopeFusionOpInfo &info, const int32_t old_index,
                                               int32_t &new_index) {
  return GetNodeindex(info, old_index, new_index, tensorflow_fusionop_outputs_map);
}

bool TensorFlowFunsionOPUtil::FusionOpChildIgnore(const ScopeFusionOpInfo &info) {
  // If the small operator is not in the input and output index table of the fusion operator,
  // it is unnecessary to establish the edge relationship and can be ignored
  int32_t old_index = 0;
  int32_t in_new_index = 0;
  int32_t out_new_index = 0;
  GE_CHK_STATUS(GetInPutIndex(info, old_index, in_new_index), "GetInPutIndex failed");
  GE_CHK_STATUS(GetOutPutIndex(info, old_index, out_new_index), "GetOutPutIndex failed");

  return (in_new_index == kFusionDisableIndex) && (out_new_index == kFusionDisableIndex);
}
}  // namespace ge
