#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
#-------------------------------------------------------------------
# Purpose:
# Copyright 2021 Huawei Technologies Co., Ltd. All rights reserved.
#-------------------------------------------------------------------

# Given a bool scalar input cond.
# return constant tensor x if cond is True, otherwise return constant tensor y.
import numpy as np
import onnx
from onnx import helper
from onnx import numpy_helper
from onnx import AttributeProto, TensorProto, GraphProto

then_out = onnx.helper.make_tensor_value_info('then_out', onnx.TensorProto.FLOAT, [5])
else_out = onnx.helper.make_tensor_value_info('else_out', onnx.TensorProto.FLOAT, [5])
then_in = onnx.helper.make_tensor_value_info('then_in', onnx.TensorProto.FLOAT, [5])
else_in = onnx.helper.make_tensor_value_info('else_in', onnx.TensorProto.FLOAT, [5])
cond = onnx.helper.make_tensor_value_info('cond', onnx.TensorProto.FLOAT, [])
res = onnx.helper.make_tensor_value_info('res', onnx.TensorProto.FLOAT, [5])

x = np.array([1, 2, 3, 4, 5]).astype(np.float32)
y = np.array([5, 4, 3, 2, 1]).astype(np.float32)

add_out_node = onnx.helper.make_node(
    'Add',
    inputs=['then_in', 'else_in'],
    outputs=['add_out'],
)

then_identity_node = onnx.helper.make_node(
    'Identity',
    inputs=['add_out'],
    outputs=['then_out'],
)

else_identity_node = onnx.helper.make_node(
    'Identity',
    inputs=['add_out'],
    outputs=['else_out'],
)

then_body = onnx.helper.make_graph(
    [then_identity_node],
    'then_body',
    [],
    [then_out]
)

else_body = onnx.helper.make_graph(
    [else_identity_node],
    'else_body',
    [],
    [else_out]
)

if_node = onnx.helper.make_node(
    'If',
    inputs=['cond'],
    outputs=['res'],
    then_branch=then_body,
    else_branch=else_body
)

add_if_node = onnx.helper.make_node(
    'Add',
    inputs=['add_out', 'res'],
    outputs=['res'],
)

graph_def = helper.make_graph(
        [add_out_node, if_node, add_if_node],
        'test_if',
        [cond, else_in, then_in],
        [res],
)
model_def = helper.make_model(graph_def, producer_name='if-onnx')
model_def.opset_import[0].version = 11
onnx.save(model_def, "./if.onnx")
