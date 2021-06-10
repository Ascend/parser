#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#-------------------------------------------------------------------
# Purpose:
# Copyright 2020 Huawei Technologies Co., Ltd. All rights reserved.
#-------------------------------------------------------------------

import os
import sys
import getopt

from google.protobuf import text_format
import tensorflow as tf
from tensorflow.python.framework import function_def_to_graph
from tensorflow.python.framework.errors_impl import NotFoundError
from tensorflow.python.platform import gfile

from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import tensor_shape_pb2
from tensorflow.core.framework import types_pb2
from tensorflow.core.framework import versions_pb2
from tensorflow.python.eager import context
from tensorflow.python.framework import importer
from tensorflow.python.framework import ops
from tensorflow.python.framework import versions

sys.path.append(os.path.join(os.path.split(os.path.realpath(__file__))[0], "util"))

import graph_library_pb2


def _get_num_args(arg_def, node_def):
    if arg_def.number_attr:
        return node_def.attr[arg_def.number_attr].i
    elif arg_def.type_list_attr:
        return len(node_def.attr[arg_def.type_list_attr].list.type)
    elif arg_def.type_attr or arg_def.type != types_pb2.DT_INVALID:
        return 1
    else:
        raise ValueError("Invalid arg_def:\n\n{}".format(str(arg_def)))


def is_function(fname):
    """Checks for a function definition with `fname` in the current context."""
    if context.executing_eagerly():
        return context.context().has_function(fname)
    else:
        return ops.get_default_graph()._is_function(fname)

def create_arg_for_input_nodes(fdef, graph_def, input_shapes):
    for i, arg_def in enumerate(fdef.signature.input_arg):
        node_def = graph_def.node.add()
        node_def.name = arg_def.name
        node_def.op = "_Arg"
        node_def.attr["T"].type = arg_def.type
        node_def.attr["index"].i = i
        if input_shapes and input_shapes[i] is not None:
            input_shape = input_shapes[i]
            if not isinstance(input_shape, tensor_shape_pb2.TensorShapeProto):
                input_shape = input_shape.as_proto()
            node_def.attr["shape"].shape.CopyFrom(input_shape)
        arg_attrs = fdef.arg_attr[i].attr
        for k in arg_attrs:
            # Only copy internal attributes. Normal attributes for nodes cannot be
            # applied to these Arg nodes.
            if k.startswith("_"):
                node_def.attr[k].CopyFrom(arg_attrs[k])
    return

def create_retval_for_output_nodes(fdef, graph_def, nested_to_flat_tensor_name):
    for i, arg_def in enumerate(fdef.signature.output_arg):
        node_def = graph_def.node.add()
        node_def.name = '{}_Retval'.format(arg_def.name)
        node_def.op = "_Retval"
        node_def.attr["T"].type = arg_def.type
        node_def.attr["index"].i = i
        node_def.attr["op_def"].s = ops.get_default_graph()._get_op_def(node_def.op).SerializeToString()

        ret_name = fdef.ret[arg_def.name]
        node_def.input.append(nested_to_flat_tensor_name[ret_name])
    return

def updat_input_index(node_def, op_def, nested_to_flat_tensor_name):
    flattened_index = 0
    for arg_def in op_def.output_arg:
        num_args = _get_num_args(arg_def, node_def)
        for i in range(num_args):
            # Map tensor names from "node_name:output_arg_name:index" to
            # "node_name:flattened_index".
            nested_name = "{}:{}:{}".format(node_def.name, arg_def.name, i)
            if flattened_index == 0:
                flat_name = node_def.name
            else:
                flat_name = "{}:{}".format(node_def.name, flattened_index)
            nested_to_flat_tensor_name[nested_name] = flat_name
            flattened_index += 1
    control_name = "^" + node_def.name
    nested_to_flat_tensor_name[control_name] = control_name
    return

def build_tensor_name(fdef, default_graph):
    nested_to_flat_tensor_name = {}
    for arg_def in fdef.signature.input_arg:
        nested_to_flat_tensor_name[arg_def.name] = arg_def.name
        control_name = '^{}'.format(arg_def.name)
        nested_to_flat_tensor_name[control_name] = control_name

    global op_def
    for node_def in fdef.node_def:
        f = default_graph._functions.get(node_def.op, None)
        if f is not None and hasattr(f, "signature"):
            op_def = f.signature
            if node_def.op not in copied_functions:
                # Since this function is referenced as an op type, we have no choice but
                # to copy it into the GraphDef if we want downstream tools to process
                # it.
                graph_def.library.function.add().CopyFrom(f.definition)
                copied_functions.add(node_def.op)
        else:
            op_def = ops.get_default_graph()._get_op_def(node_def.op)

        for attr in op_def.attr:
            if attr.type == "func":
                fname = node_def.attr[attr.name].func.name
                if not is_function(fname):
                    raise ValueError("%s function not found." % fname)
            elif attr.type == "list(func)":
                for fn in node_def.attr[attr.name].list.func:
                    fname = fn.name
                    if not is_function(fname):
                        raise ValueError("%s function not found." % fname)

        # Iterate over output_args in op_def to build the map.
        # Index of the output tensor in the flattened list of *all* output
        # tensors of the op.
        updat_input_index(node_def, op_def, nested_to_flat_tensor_name)
    return  nested_to_flat_tensor_name

def convert_function_def_to_graph_def(fdef, input_shapes=None, copy_functions=True):
    graph_def = graph_pb2.GraphDef()
    graph_def.versions.CopyFrom(
        versions_pb2.VersionDef(
            producer=versions.GRAPH_DEF_VERSION,
            min_consumer=versions.GRAPH_DEF_VERSION_MIN_CONSUMER))

    default_graph = ops.get_default_graph()

    copied_functions = set()

    # Copy *all* functions from outer graph to `graph_def` so that both direct
    # and indirect references are safely handled.
    if copy_functions:
        default_graph._copy_functions_to_graph_def(graph_def, 0)
        for function_name in default_graph._functions.keys():
            copied_functions.add(function_name)

    if input_shapes and len(input_shapes) != len(fdef.signature.input_arg):
        raise ValueError("Length of input_shapes must match the number of " +
                         "input_args. len(input_shapes): {} len(input_arg): {}".
                         format(len(input_shapes), len(fdef.signature.input_arg)))

    # 1. Create _Arg for input nodes.
    create_arg_for_input_nodes(fdef, graph_def, input_shapes)

    # 2. Copy all body NodeDefs to the GraphDef.
    graph_def.node.extend(fdef.node_def)

    # 3. Perform the renaming.

    # Build the tensor name mapping then flatten the tensor names.
    # See comment on `FunctionDef.node_def` on how the tensor naming in
    # FunctionDefs is different from GraphDefs.
    nested_to_flat_tensor_name = build_tensor_name(fdef, default_graph)

    # Update inputs of all nodes in graph.
    for node_def in graph_def.node:
        for i in range(len(node_def.input)):
            node_def.input[i] = nested_to_flat_tensor_name[node_def.input[i]]

    # Create _Retval for output nodes.
    create_retval_for_output_nodes(fdef, graph_def, nested_to_flat_tensor_name)

    return graph_def, nested_to_flat_tensor_name


def convert_graphs(filename):
    try:
        with tf.io.gfile.GFile(filename, 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
            if len(graph_def.library.function) == 0:
                print("INFO: The input model does not contain a functionDef and does not require conversion.")
                return
            try:
                convert_subgraphs(graph_def, filename)
            except Exception as e:
                print("ERROR: Convert subgraphs failed.", e)
                return
            print("INFO: Convert to subgraphs successfully.")
    except NotFoundError:
        print('ERROR: model file {} does not exist'.format(filename))
    return


def convert_subgraphs(graph_def, filename):
    graph_def_library = graph_library_pb2.GraphDefLibrary()
    for i, fdef in enumerate(graph_def.library.function):
        sub_graph, nested_to_flat_tensor_name = convert_function_def_to_graph_def(fdef, copy_functions=False)
        print("INFO: Convert FunctionDef, index:{}, name:{}".format(str(i), fdef.signature.name))
        sub_graph_name = '{}.pb'.format(fdef.signature.name)
        result_path = '{}/results'.format(os.path.dirname(os.path.abspath(filename)))
        tf.io.write_graph(sub_graph, result_path, sub_graph_name, as_text=False)
        data = sub_graph.SerializeToString()
        ge_graph_def = graph_library_pb2.GeGraphDef()
        ge_graph_def.name = fdef.signature.name
        ge_graph_def.graph.ParseFromString(data)
        graph_def_library.graph_def.append(ge_graph_def)
        print(graph_def_library.graph_def[i])

    # Write to prototxt
    try:
        graph_def_file = '{}/graph_def_library.pbtxt'.format(os.path.dirname(os.path.abspath(filename)))
        print("graph_def_file: ", graph_def_file)
        with open(graph_def_file, "w") as f:
            print(graph_def_library, file=f)
    except IOError:
        print("Could not open file. Creating a new one.")


def usage():
    print(
        '''
        Based on tensorflow 1.15 or later, Python 3

        Convert the tensorflow functionDefs in the input model file to single GraphDefs,
        and save the result to the "results" directory and graph_def_library.pbtxt in
        the input file directory.
        The name of the sub graph is same as the name of the corresponding functionDef.

        Usage: func2grpah.py <command>

        Available commands:
          model (-m)              Input model file.
          version (-v)            Prints the version of this software.
          help (-h)               Prints help for commands.
        '''
    )


if __name__ == '__main__':
    model = ''
    try:
        opts, args = getopt.getopt(sys.argv[1:], '-v-h-m:', ['version', 'help', 'model='])
        for opt_name, opt_value in opts:
            if opt_name in ('-m', '--model'):
                model = opt_value
                print("INFO: Input model file is", model)
                convert_graphs(model)
            elif opt_name in ('-h', '--help'):
                usage()
                break
            elif opt_name in ('-v', '--version'):
                print("version 1.0.0")
                break
    except getopt.GetoptError:
        print("ERROR: Input parameters is invalid, use '--help' to view the help.")
    if (len(sys.argv) == 1):
        print("INFO: Please specify the input parameters, and use '--help' to view the help.")
