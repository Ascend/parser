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

#ifndef GE_COMMON_OP_MAP_H_
#define GE_COMMON_OP_MAP_H_

#include <map>
#include <string>
#include <vector>

/*lint -e1073*/
namespace ge {
// the operator type mapping table of caffe and  mindspore
extern std::map<std::string, std::string> caffe_op_map;

// the operator type mapping table of TensorFlow and  mindspore
extern std::map<std::string, std::string> tensorflow_op_map;

// the network training operator type mapping table of TensorFlow and  mindspore
extern std::map<std::string, std::string> tensorflow_train_op_map;

// local framework op vec
extern std::vector<std::string> local_framework_op_vec;

// dataset op vec
extern std::vector<std::string> is_dataset_op_vec;

// output tensor num
extern std::map<std::string, int32_t> op_output_tensor_num;
}  // namespace ge
/*lint +e1073*/
#endif  // GE_COMMON_OP_MAP_H_
