import onnx
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto


def make_clip_V9():
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [3, 4, 5])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [3, 4, 5])
    node_def = helper.make_node('Clip',
                                inputs=['X'],
                                outputs=['Y'],
                                max = 1.0,
                                min = -1.0,
                                )
    graph = helper.make_graph(
        [node_def],
        "test_clip_case_V9",
        [X],
        [Y],
    )

    model = helper.make_model(graph, producer_name="onnx-mul_test")
    model.opset_import[0].version = 9
    onnx.save(model, "./onnx_clip_v9.onnx")


if __name__ == '__main__':
    make_clip_V9()
