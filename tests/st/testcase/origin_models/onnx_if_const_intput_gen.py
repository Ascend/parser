#!/usr/bin/env python3
# -*- coding utf-8 -*-
# Copyright Huawei Technologies Co., Ltd 2019-2022. All rights reserved.

import os
import numpy as np
import onnx

def gen_onnx():
    X = onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [5])
    Y = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [5])
    then_out = onnx.helper.make_tensor_value_info("then_out", onnx.TensorProto.FLOAT, [5])
    else_out = onnx.helper.make_tensor_value_info("else_out", onnx.TensorProto.FLOAT, [5])

    const_out_node = onnx.helper.make_node("Constant", inputs=[], outputs=["Y"])
    
    then_const_node = onnx.helper.make_node("Constant", inputs=[], outputs=["then_out"])
    else_const_node = onnx.helper.make_node("Constant", inputs=[], outputs=["else_out"])

    then_body = onnx.helper.make_graph(
        [then_const_node],
        "then_body",
        [],
        [then_out]
    )

    else_body = onnx.helper.make_graph(
        [else_const_node],
        "else_body",
        [],
        [else_out]
    )

    if_node = onnx.helper.make_node("If", inputs=["X"], outputs=[], then_branch=then_body, else_branch=else_body)

    graph_def = onnx.helper.make_graph(
        [if_node, const_out_node],
        "if_model",
        [X],
        [Y]
    )

    model_def = onnx.helper.make_model(graph_def)
    model_def.opset_import[0].version=11
    onnx.save(model_def, "onnx_if_const_intput.onnx")
    print(model_def)

if __name__ == "__main__":
    gen_onnx()
