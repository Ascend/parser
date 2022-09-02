/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.
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
#include "framework/omg/parser/parser_types.h"


namespace ge {
namespace parser {
const char *DATA = "Data";
const char *AIPPDATA = "AippData";
const char *CONVOLUTION = "Convolution";
const char *CORRELATION = "Correlation";
const char *CORRELATIONV2 = "Correlation_V2";
const char *DECONVOLUTION = "Deconvolution";
const char *POOLING = "Pooling";
const char *ELTWISE = "Eltwise";
const char *RELU = "ReLU";
const char *RELU6 = "ReLU6";
const char *SIGMOID = "Sigmoid";
const char *ABSVAL = "AbsVal";
const char *TANH = "TanH";
const char *PRELU = "PReLU";
const char *BATCHNORM = "BatchNorm";
const char *FUSIONBATCHNORM = "FusionBatchNorm";
const char *SCALE = "Scale";
const char *FULL_CONNECTION = "FullConnection";
const char *SOFTMAX = "Softmax";
const char *PLUS = "Plus";
const char *ACTIVATION = "Activation";
const char *FLATTEN = "Flatten";
const char *ADD = "Add";
const char *SUB = "Sub";
const char *MUL = "Mul";
const char *MATMUL = "MatMul";
const char *RSQRT = "Rsqrt";
const char *BIASADD = "BiasAdd";
const char *RESHAPE = "Reshape";
const char *REFORMAT = "ReFormat";
const char *DEPCONVOLUTION = "ConvolutionDepthwise";
const char *DROPOUT = "Dropout";
const char *DROPOUTGENMASK = "DropOutGenMask";
const char *DROPOUTDOMASK = "DropOutDoMask";
const char *CONCAT = "Concat";
const char *ROIPOOLING = "ROIPooling";
const char *PROPOSAL = "Proposal";
const char *FSRDETECTIONOUTPUT = "FSRDetectionOutput";
const char *DETECTIONPOSTPROCESS = "Detectpostprocess";
const char *LRN = "LRN";
const char *TRANSDATA = "TransData";
const char *PERMUTE = "Permute";
const char *SSDNORMALIZE = "SSDNormalize";
const char *SSDPRIORBOX = "SSDPriorBox";
const char *NETOUTPUT = "NetOutput";
const char *SSDDETECTIONOUTPUT = "SSDDetectionOutput";
const char *REFINEDETDETECTIONOUTPUT = "RefinedetDetectionOutput";
const char *CHANNELAXPY = "ChannelAxpy";
const char *PSROIPOOLING = "PSROIPooling";
const char *POWER = "Power";
const char *POW = "Pow";
const char *ROIALIGN = "ROIAlign";
const char *PYTHON = "Python";
const char *FREESPACEEXTRACT = "FreespaceExtract";
const char *SPATIALTF = "SpatialTransform";
const char *SHAPE = "Shape";
const char *SHAPEN = "ShapeN";
const char *ARGMAX = "ArgMax";
const char *GATHERND = "GatherNd";
const char *GATHER = "Gather";
const char *REALDIV = "RealDiv";
const char *PACK = "Pack";
const char *SLICE = "Slice";
const char *SLICED = "SliceD";
const char *FLOORDIV = "FloorDiv";
const char *SQUEEZE = "Squeeze";
const char *UNSQUEEZE = "Unsqueeze";
const char *STRIDEDSLICE = "StridedSlice";
const char *RANGE = "Range";
const char *RPNPROPOSALS = "RpnProposals";
const char *DECODEBBOX = "DecodeBbox";
const char *PAD = "Pad";
const char *PADV2 = "PadV2";
const char *MIRRORPAD = "MirrorPad";
const char *TILE = "Tile";
const char *SIZE = "Size";
const char *CLIPBOXES = "ClipBoxes";
const char *FASTRCNNPREDICTIONS = "FastrcnnPredictions";
const char *SPLIT = "Split";
const char *SPLITV = "SplitV";
const char *EXPANDDIMS = "ExpandDims";
const char *EMPTY = "Empty";
const char *MEAN = "Mean";
const char *GREATER = "Greater";
const char *SWITCH = "Switch";
const char *SWITCHN = "SwitchN";
const char *MERGE = "Merge";
const char *SYMBOLICGRADIENT = "SymbolicGradient";
const char *REMOTECALL = "RemoteCall";
const char *_IF = "_If";
const char *STATELESSIF = "StatelessIf";
const char *IF = "If";
const char *CASE = "Case";
const char *_WHILE = "_While";
const char *WHILE = "While";
const char *STATELESSWHILE = "StatelessWhile";
const char *FOR = "For";
const char *PARTITIONEDCALL = "PartitionedCall";
const char *STATEFULPARTITIONEDCALL = "StatefulPartitionedCall";
const char *FAKEPARAM = "FakeParam";
const char *TRANSPOSE = "Transpose";
const char *TRANSPOSED = "TransposeD";
const char *CAST = "Cast";
const char *REGION = "Region";
const char *YOLO = "Yolo";
const char *YOLODETECTIONOUTPUT = "YoloDetectionOutput";
const char *FILL = "Fill";
const char *REVERSE = "Reverse";
const char *UNPACK = "Unpack";
const char *YOLO2REORG = "Yolo2Reorg";
const char *REDUCESUM = "ReduceSum";
const char *SUM = "Sum";
const char *CONSTANT = "Const";
const char *FILECONSTANT = "FileConstant";
const char *RESIZEBILINEAR = "ResizeBilinear";
const char *RESIZEBILINEARGRAD = "ResizeBilinearGrad";
const char *MAXIMUM = "Maximum";
const char *FRAMEWORKOP = "FrameworkOp";
const char *ARG = "_Arg";
const char *FUSEDBATCHNORMGRAD = "FusedBatchNormGrad";
const char *LSTM = "LSTM";
const char *HIGHWAY = "HighWay";
const char *RNN = "RNN";
const char *ATTENTIONDECODER = "AttentionDecoder";
const char *LOGICAL_NOT = "LogicalNot";
const char *LOGICAL_AND = "LogicalAnd";
const char *LOGICAL_OR = "LogicalOr";
const char *EQUAL = "Equal";
const char *NOTEQUAL = "NotEqual";
const char *INTERP = "Interp";
const char *SHUFFLECHANNEL = "ShuffleChannel";
const char *AIPP = "Aipp";
const char *MULTISHAPE = "MultiShape";
const char *RECIPROCAL = "Reciprocal";
const char *SELU = "Selu";
const char *ELU = "Elu";
const char *ACOSH = "Acosh";
const char *ASINH = "Asinh";
const char *MINIMUM = "Minimum";
const char *CLIP = "Clip";
const char *L2NORMALIZE = "L2Normalize";
const char *CROPANDRESIZE = "CropAndResize";
const char *UNUSEDCONST = "UnusedConst";
const char *SPARSETODENSE = "SparseToDense";
const char *NONMAXSUPPRESSION = "NonMaxSuppression";
const char *TOPKV2 = "TopKV2";
const char *INVERTPERMUTATION = "InvertPermutation";
const char *MULTINOMIAL = "Multinomial";
const char *REVERSESEQUENCE = "ReverseSequence";
const char *REDUCEPROD = "ReduceProd";
const char *REDUCEMAX = "ReduceMax";
const char *REDUCEMIN = "ReduceMin";
const char *EXTRACTIMAGEPATCHES = "ExtractImagePatches";
const char *SQRT = "Sqrt";
const char *REDUCEALL = "ReduceAll";
const char *RESIZENEARESTNEIGHBOR = "ResizeNearestNeighbor";
const char *SPACETOBATCHND = "SpaceToBatchND";
const char *BATCHTOSPACEND = "BatchToSpaceND";
const char *ASSERT = "Assert";
const char *GREATEREQUAL = "GreaterEqual";
const char *FLOOR = "Floor";
const char *RANDOMUNIFORM = "RandomUniform";
const char *BATCHMATMUL = "BatchMatMul";
const char *SPACETODEPTH = "SpaceToDepth";
const char *DEPTHTOSPACE = "DepthToSpace";
const char *RINT = "Rint";
const char *ATAN = "Atan";
const char *ATAN2 = "Atan2";
const char *ATANH = "Atanh";
const char *ACOS = "Acos";
const char *ASIN = "Asin";
const char *NEG = "Neg";
const char *LOG = "Log";
const char *TAN = "Tan";
const char *ROUND = "Round";
const char *UPSAMPLE = "Upsample";
const char *FLOORMOD = "FloorMod";
const char *LESS = "Less";
const char *LESSEQUAL = "LessEqual";
const char *ONEHOT = "OneHot";
const char *REFSWITCH = "RefSwitch";
const char *REFMERGE = "RefMerge";
const char *ENTER = "Enter";
const char *REFENTER = "RefEnter";
const char *LOOPCOND = "LoopCond";
const char *NEXTITERATION = "NextIteration";
const char *REFNEXTITERATION = "RefNextIteration";
const char *EXIT = "Exit";
const char *REFEXIT = "RefExit";
const char *CONTROLTRIGGER = "ControlTrigger";
const char *ZEROSLIKE = "ZerosLike";
const char *EXP = "Exp";
const char *WHERE = "Where";
const char *FAKEQUANTWITHMINMAXVARS = "FakeQuantWithMinMaxVars";
const char *SOFTPLUS = "Softplus";
const char *SOFTSIGN = "Softsign";
const char *COSH = "Cosh";
const char *SINH = "Sinh";
const char *SQUAREDDIFFERENCE = "SquaredDifference";
const char *REQUIREDSPACETOBATCHPADDINGS = "RequiredSpaceToBatchPaddings";  // for retinanet scope fusion
const char *SSDPOSTPROCESSOR = "SSDPostProcessor";
const char *RETINANETBOXES = "RetinanetBoxes";
const char *RETINAMULTIANCHORS = "RetinaMultiAnchor";
const char *RETINANETCLIPPEDBOXES = "RetinanetClippedBoxes";
const char *RETINANETFILTEREDDETECTIONS = "RetinanetFilteredDetections";
const char *RETINANETPOSTPROCESSOR = "RetinanetPostProcessor";
const char *RETINANETANCHORS = "RetinanetAnchors";
const char *FASTERRCNNMAP = "FasterRCNNMap";
const char *FASTERRCNNMAP1 = "FasterRCNNMap1";
const char *FASTERRCNNSECONDSTAGEPOSTPROCESSOR = "FasterRCNNSecondStagePostprocessor";
const char *FASTERRCNNROIINTERPOOLING = "FasterRCNNROIInterPooling";
const char *FASTERRCNNFIRSTSTAGEPOSTPROCESSOR = "FasterRCNNFirstStagePostprocessor";
const char *FASTERRCNNGRIDANCHORGENERATOR = "FasterRCNNGridAnchorGenerator";
const char *ROIINTERPOOLING = "ROIInterPooling";
const char *FASTERRCNNCLIPTOWINDOW = "FasterRCNNClipToWindow";
const char *EMBEDLOOKUP = "EmbedLookup";
const char *HASHLOOKUP = "HashLookup";
const char *LSH_PROJ = "LshProject";
const char *SVDF = "SVDF";
const char *SSDANCHORGENERATOR = "SSDAnchorGenerator";
const char *IDENTITY = "Identity";
const char *IDENTITYN = "IdentityN";
const char *PLACEHOLDERWITHDEFAULT = "PlaceholderWithDefault";
const char *SELECT = "Select";
const char *GETSPAN = "GetSpan";
const char *STOPGRADIENT = "StopGradient";
const char *PREVENTGRADIENT = "PreventGradient";
const char *GUARANTEECONST = "GuaranteeConst";
const char *BROADCASTGRADIENTARGS = "BroadcastGradientArgs";
const char *BROADCASTARGS = "BroadcastArgs";
const char *CONFUSIONMATRIX = "ConfusionMatrix";
const char *RANK = "Rank";
const char *PLACEHOLDER = "PlaceHolder";
const char *END = "End";
const char *BASICLSTMCELL = "BasicLSTMCell";
const char *GETNEXT = "GetNext";
const char *INITDATA = "InitData";
const char *REFIDENTITY = "RefIdentity";
const char *BITCAST = "Bitcast";

/***************Ann special operator*************************/
const char *ANN_MEAN = "AnnMean";
const char *ANN_CONVOLUTION = "AnnConvolution";
const char *ANN_DEPCONVOLUTION = "AnnDepthConv";
const char *ANN_FULLCONNECTION = "AnnFullConnection";
const char *ANN_NETOUTPUT = "AnnNetOutput";
const char *ANN_DATA = "AnnData";
const char *ANN_RESHAPE = "AnnReshape";
const char *ANN_ADD = "AnnAdd";
const char *ANN_MUL = "AnnMul";
const char *ANN_SUB = "AnnSub";
const char *ANN_DIV = "AnnDiv";
const char *ANN_DEQUANTIZE = "AnnDequant";
const char *ANN_QUANTIZE = "AnnQuant";
const char *ANN_PAD = "AnnPad";
const char *ANN_RESIZE_BILINEAR = "AnnResizeBilinear";

/***************************************************/
/******************Training operator*************************/
const char *GATHERV2 = "GatherV2";
const char *CONVGRADFILTER = "Conv2DBackpropFilter";
const char *CONV2D = "Conv2D";
const char *CONV2DBACKPROPINPUT = "Conv2DBackpropInput";
const char *FUSEDBATCHNORM = "FusedBatchNorm";
const char *BIASADDGRAD = "BiasAddGrad";
const char *ACTIVATIONGRAD = "ReluGrad";
const char *MAXPOOLWITHARGMAX = "MaxPoolWithArgmax";
const char *MAXPOOLGRADWITHARGMAX = "MaxPoolGradWithArgmax";
const char *SPARSESOFTMAXCROSSENTROPYWITHLOGITS = "SparseSoftmaxCrossEntropyWithLogits";
const char *SNAPSHOT = "Snapshot";
const char *VAR = "Var";
const char *MEANGRAD = "MeanGrad";
const char *TRANSLATE = "Translate";
const char *ADDN = "AddN";
const char *L2LOSS = "L2Loss";
const char *MULTIPLY = "Multiply";
const char *HUBERLOSSGRAD = "HuberLossGrad";
const char *HUBERLOSS = "HuberLoss";
const char *NEGATIVE = "Negative";
const char *SSDCAST = "SSDCast";
const char *SPARSESOFTMAXCROSSENTROPY = "SsdSparseSoftmaxCrossEntropy";
const char *SPARSESOFTMAXCROSSENTROPYGRAD = "SsdSparseSoftmaxCrossEntropyGrad";
const char *SSDSQUEEZEFUSION = "SsdSqueezeFusion";
const char *CONCATFOUR2FIVE = "ConcatFour2Five";
const char *CONCATFIVE2FOUR = "ConcatFive2Four";
const char *SSDREALDIVTILEMUL = "SSDRealdivTileMul";
const char *SSDSUMMULREALDIVMEAN = "SSDSumMulRealdivMean";

const char *VARIABLEV2 = "VariableV2";
const char *VARHANDLEOP = "VarHandleOp";
const char *TEMPORARYVARIABLE = "TemporaryVariable";
const char *DESTROYTEMPORARYVARIABLE = "DestroyTemporaryVariable";
const char *VARIABLE = "Variable";
const char *ASSIGN = "Assign";
const char *ASSIGNVARIABLEOP = "AssignVariableOp";
const char *ASSIGNADD = "AssignAdd";
const char *ASSIGNADDVARIABLEOP = "AssignAddVariableOp";
const char *ASSIGNSUB = "AssignSub";
const char *ASSIGNSUBVARIABLEOP = "AssignSubVariableOp";
const char *APPLYMOMENTUM = "ApplyMomentum";
const char *RESOURCEAPPLYMOMENTUM = "ResourceApplyMomentum";
const char *SGD = "SGD";
const char *NOOP = "NoOp";
const char *READVARIABLEOP = "ReadVariableOp";
const char *PARALLELCONCATSTART = "_ParallelConcatStart";
const char *CONSTANTOP = "Constant";
const char *DEPTHWISECONV2DBACKPROPFILTER = "DepthwiseConv2dNativeBackpropFilter";
const char *DEPTHWISECONV2DBACKPORPINPUT = "DepthwiseConv2dNativeBackpropInput";
const char *DEPTHWISECONV2DFORWARDNATIVE = "DepthwiseConv2dNative";
const char *DROPOUTGRAD = "DropOutGrad";
const char *APPLYRMSPROPMIXEDPRECISION = "apply_rms_prop_mixed_precision";
const char *APPLYRMSPROP = "ApplyRMSProp";
const char *RELU6GRAD = "Relu6Grad";
const char *AVGPOOLGRAD = "AvgPoolGrad";
const char *CONCATV2 = "ConcatV2";
const char *CONCATOFFSET = "ConcatOffset";
const char *LAYERNORMGRAD = "LayerNormGrad";
const char *LAYERNORM = "LayerNorm";
const char *LARS = "Lars";
const char *DYNAMICSTITCH = "DynamicStitch";

/***************************************************/
const char *SQUARE = "Square";
const char *HCOMBROADCAST = "HcomBroadcast";
const char *HCOMALLGATHER = "HcomAllGather";
const char *HCOMALLREDUCE = "HcomAllReduce";
const char *HCOMREDUCESCATTER = "HcomReduceScatter";
const char *HCOMSEND = "HcomSend";
const char *HCOMRECEIVE = "HcomReceive";
const char *HCOMREMOTEREAD = "HcomRemoteRead";
const char *HCOMREMOTEREFREAD = "HcomRemoteRefRead";
const char *HCOMREMOTEWRITE = "HcomRemoteWrite";
const char *HCOMREMOTESCATTERWRITE = "HcomRemoteScatterWrite";

const char *VARASSIGN = "VarAssign";
const char *VARISINITIALIZEDOP = "VarIsInitializedOp";
const char *LogTimeStamp = "LogTimeStamp";
const char *ISVARIABLEINITIALIZED = "IsVariableInitialized";
const char *STREAMSWITCH = "StreamSwitch";
const char *STREAMSWITCHN = "StreamSwitchN";
const char *STREAMACTIVE = "StreamActive";
const char *MEMCPYASYNC = "MemcpyAsync";
const char *MEMCPYADDRASYNC = "MemcpyAddrAsync";
const char *STREAMMERGE = "StreamMerge";
const char *ENDGRAPH = "EndGraph";
const char *SEND = "Send";
const char *RECV = "Recv";
const char *ENDOFSEQUENCE = "EndOfSequence";

const char *LABELSET = "LabelSet";
const char *LABELGOTO = "LabelGoto";
const char *LABELGOTOEX = "LabelGotoEx";
const char *LABELSWITCH = "LabelSwitch";
const char *LABELSWITCHBYINDEX = "LabelSwitchByIndex";

const char *ATOMICADDRCLEAN = "AtomicAddrClean";

const char *ABS_GRAD = "AbsGrad";
const char *ACCUMULATE_N_V2 = "AccumulateNV2";
const char *ACOS_GRAD = "AcosGrad";
const char *ACOSH_GRAD = "AcoshGrad";
const char *ANY = "Any";
const char *APPROXIMATE_EQUAL = "ApproximateEqual";
const char *ASIN_GRAD = "AsinGrad";
const char *ASINH_GRAD = "AsinhGrad";
const char *ATAN_GRAD = "AtanGrad";
const char *BROADCAST_TO = "BroadcastTo";
const char *ELU_GRAD = "EluGrad";
const char *ADD_V2 = "AddV2";
const char *DATAFORMATDIMMAP = "DataFormatDimMap";
const char *DATAFORMATVECPERMUTE = "DataFormatVecPermute";
const char *BESSELI0E = "BesselI0e";
const char *BESSELI1E = "BesselI1e";
const char *APPLYADADELTA = "ApplyAdadelta";
const char *APPLYADAGRAD = "ApplyAdagrad";
const char *APPLYADAGRADDA = "ApplyAdagradDA";
const char *APPLYADAM = "ApplyAdam";
const char *APPLYADAMAX = "ApplyAdaMax";
const char *APPLYADDSIGN = "ApplyAddSign";
const char *APPLYCENTEREDRMSPROP = "ApplyCenteredRMSProp";
const char *APPLYFTRL = "ApplyFtrl";
const char *APPLYFTRLV2 = "ApplyFtrlV2";
const char *APPLYGRADIENTDESCENT = "ApplyGradientDescent";
const char *APPLYPOWERSIGN = "ApplyPowerSign";
const char *APPLYPROXIMALADAGRAD = "ApplyProximalAdagrad";
const char *APPLYPROXIMALGRADIENTDESCENT = "ApplyProximalGradientDescent";
const char *DEQUANTIZE = "Dequantize";

const char *FOCAL_LOSS = "FocalLoss";
const char *FOCAL_LOSS_GRAD = "FocalLossGrad";
const char *SMOOTHL1_LOSS = "SmoothL1Loss";
const char *SMOOTHL1_LOSS_grad = "SmoothL1LossGrad";
const char *REDUCEMEAN = "ReduceMean";
const char *CONCAT_V2 = "ConcatV2";
const char *ONEHOT_V2 = "OneHotV2";
const char *SLICE_V2 = "SliceV2";
const char *TILE_V2 = "TileV2";
const char *SUM_V2 = "SumV2";
// Common type when the operator has the same name
const char *DETECTIONOUTPUT = "DetectionOutput";
// Custom operator
const char *CUSTOMOP = "CustomOp";
const char *CUSTOMOP_NCHW = "CustomOpNchw";
const char *CUSTOMOP_NHWC = "CustomOpNhwc";
const char *CUSTOMOP_NC1HWC0 = "CustomOpNc1hwc0";

// Depthwise 4d_2_6d,6d_2_4d
const char *DEPTHWISEWEIGHT4D26D = "depthwise_weight_4d_2_6d";
const char *DEPTHWISEWEIGHT6D24D = "depthwise_weight_6d_2_4d";

const char *SQRTGRAD = "SqrtGrad";
const char *SIGMOIDGRAD = "SigmoidGrad";

const char *TRANSSHAPE = "TransShape";

// Horovod operator
const char *HVDCALLBACKALLREDUCE = "HorovodAllreduce";
const char *HVDCALLBACKALLGATHER = "HorovodAllgather";
const char *HVDCALLBACKBROADCAST = "HorovodBroadcast";
const char *HVDWAIT = "HorovodWait";

///
/// @brief Magic number of model file
///
const uint32_t MODEL_FILE_MAGIC_NUM = 0x444F4D49;  // magic number

///
/// @brief Model head length
///
const uint32_t MODEL_FILE_HEAD_LEN = 256;

const uint32_t MODEL_VERSION = 0x10000000; ///< Model version 1.0///

///
/// @ingroup domi_omg
/// @brief alpha default value
///
const float ALPHA_DEFAULT_VALUE = 1.0;

///
/// @ingroup domi_omg
/// @brief beta default value
///
const float BETA_DEFAULT_VALUE = 0.0;

///
/// @ingroup domi_omg
/// @brief Input node type
///
const std::string INPUT_TYPE = "Input";
const std::string DUMMY_DATA = "DummyData";

// for fusion op plugin
const std::string ATTR_NAME_FUSIONOP_ORIGINAL_TYPE = "_fusionop_original_type";

const std::string ATTR_NAME_INPUT_TENSOR_DESC = "input_tensor_desc";
const std::string ATTR_NAME_OUTPUT_TENSOR_DESC = "output_tensor_desc";

///
/// @ingroup domi_omg
/// @brief DATA node type
///
const std::string DATA_TYPE = "Data";

///
/// @ingroup domi_omg
/// @brief Frame operator type
///
const std::string FRAMEWORK_OP_TYPE = "FrameworkOp";

///
/// @ingroup domi_omg
/// @brief Convolution node type
///
const std::string NODE_NAME_NET_OUTPUT = "Node_Output";
}  // namespace parser
}  // namespace ge
