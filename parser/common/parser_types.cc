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
const char_t * const DATA = "Data";
const char_t * const AIPPDATA = "AippData";
const char_t * const CONVOLUTION = "Convolution";
const char_t * const CORRELATION = "Correlation";
const char_t * const CORRELATIONV2 = "Correlation_V2";
const char_t * const DECONVOLUTION = "Deconvolution";
const char_t * const POOLING = "Pooling";
const char_t * const ELTWISE = "Eltwise";
const char_t * const RELU = "ReLU";
const char_t * const RELU6 = "ReLU6";
const char_t * const SIGMOID = "Sigmoid";
const char_t * const ABSVAL = "AbsVal";
const char_t * const TANH = "TanH";
const char_t * const PRELU = "PReLU";
const char_t * const BATCHNORM = "BatchNorm";
const char_t * const FUSIONBATCHNORM = "FusionBatchNorm";
const char_t * const SCALE = "Scale";
const char_t * const FULL_CONNECTION = "FullConnection";
const char_t * const SOFTMAX = "Softmax";
const char_t * const PLUS = "Plus";
const char_t * const ACTIVATION = "Activation";
const char_t * const FLATTEN = "Flatten";
const char_t * const ADD = "Add";
const char_t * const SUB = "Sub";
const char_t * const MUL = "Mul";
const char_t * const MATMUL = "MatMul";
const char_t * const RSQRT = "Rsqrt";
const char_t * const BIASADD = "BiasAdd";
const char_t * const RESHAPE = "Reshape";
const char_t * const REFORMAT = "ReFormat";
const char_t * const DEPCONVOLUTION = "ConvolutionDepthwise";
const char_t * const DROPOUT = "Dropout";
const char_t * const DROPOUTGENMASK = "DropOutGenMask";
const char_t * const DROPOUTDOMASK = "DropOutDoMask";
const char_t * const CONCAT = "Concat";
const char_t * const ROIPOOLING = "ROIPooling";
const char_t * const PROPOSAL = "Proposal";
const char_t * const FSRDETECTIONOUTPUT = "FSRDetectionOutput";
const char_t * const DETECTIONPOSTPROCESS = "Detectpostprocess";
const char_t * const LRN = "LRN";
const char_t * const TRANSDATA = "TransData";
const char_t * const PERMUTE = "Permute";
const char_t * const SSDNORMALIZE = "SSDNormalize";
const char_t * const SSDPRIORBOX = "SSDPriorBox";
const char_t * const NETOUTPUT = "NetOutput";
const char_t * const SSDDETECTIONOUTPUT = "SSDDetectionOutput";
const char_t * const REFINEDETDETECTIONOUTPUT = "RefinedetDetectionOutput";
const char_t * const CHANNELAXPY = "ChannelAxpy";
const char_t * const PSROIPOOLING = "PSROIPooling";
const char_t * const POWER = "Power";
const char_t * const POW = "Pow";
const char_t * const ROIALIGN = "ROIAlign";
const char_t * const PYTHON = "Python";
const char_t * const FREESPACEEXTRACT = "FreespaceExtract";
const char_t * const SPATIALTF = "SpatialTransform";
const char_t * const SHAPE = "Shape";
const char_t * const SHAPEN = "ShapeN";
const char_t * const ARGMAX = "ArgMax";
const char_t * const GATHERND = "GatherNd";
const char_t * const GATHER = "Gather";
const char_t * const REALDIV = "RealDiv";
const char_t * const PACK = "Pack";
const char_t * const SLICE = "Slice";
const char_t * const SLICED = "SliceD";
const char_t * const FLOORDIV = "FloorDiv";
const char_t * const SQUEEZE = "Squeeze";
const char_t * const UNSQUEEZE = "Unsqueeze";
const char_t * const STRIDEDSLICE = "StridedSlice";
const char_t * const RANGE = "Range";
const char_t * const RPNPROPOSALS = "RpnProposals";
const char_t * const DECODEBBOX = "DecodeBbox";
const char_t * const PAD = "Pad";
const char_t * const PADV2 = "PadV2";
const char_t * const MIRRORPAD = "MirrorPad";
const char_t * const TILE = "Tile";
const char_t * const SIZE = "Size";
const char_t * const CLIPBOXES = "ClipBoxes";
const char_t * const FASTRCNNPREDICTIONS = "FastrcnnPredictions";
const char_t * const SPLIT = "Split";
const char_t * const SPLITV = "SplitV";
const char_t * const EXPANDDIMS = "ExpandDims";
const char_t * const EMPTY = "Empty";
const char_t * const MEAN = "Mean";
const char_t * const GREATER = "Greater";
const char_t * const SWITCH = "Switch";
const char_t * const SWITCHN = "SwitchN";
const char_t * const MERGE = "Merge";
const char_t * const SYMBOLICGRADIENT = "SymbolicGradient";
const char_t * const REMOTECALL = "RemoteCall";
const char_t * const _IF = "_If";
const char_t * const STATELESSIF = "StatelessIf";
const char_t * const IF = "If";
const char_t * const CASE = "Case";
const char_t * const _WHILE = "_While";
const char_t * const WHILE = "While";
const char_t * const STATELESSWHILE = "StatelessWhile";
const char_t * const FOR = "For";
const char_t * const PARTITIONEDCALL = "PartitionedCall";
const char_t * const STATEFULPARTITIONEDCALL = "StatefulPartitionedCall";
const char_t * const FAKEPARAM = "FakeParam";
const char_t * const TRANSPOSE = "Transpose";
const char_t * const TRANSPOSED = "TransposeD";
const char_t * const CAST = "Cast";
const char_t * const REGION = "Region";
const char_t * const YOLO = "Yolo";
const char_t * const YOLODETECTIONOUTPUT = "YoloDetectionOutput";
const char_t * const FILL = "Fill";
const char_t * const REVERSE = "Reverse";
const char_t * const UNPACK = "Unpack";
const char_t * const YOLO2REORG = "Yolo2Reorg";
const char_t * const REDUCESUM = "ReduceSum";
const char_t * const SUM = "Sum";
const char_t * const CONSTANT = "Const";
const char_t * const FILECONSTANT = "FileConstant";
const char_t * const RESIZEBILINEAR = "ResizeBilinear";
const char_t * const RESIZEBILINEARGRAD = "ResizeBilinearGrad";
const char_t * const MAXIMUM = "Maximum";
const char_t * const FRAMEWORKOP = "FrameworkOp";
const char_t * const ARG = "_Arg";
const char_t * const FUSEDBATCHNORMGRAD = "FusedBatchNormGrad";
const char_t * const LSTM = "LSTM";
const char_t * const HIGHWAY = "HighWay";
const char_t * const RNN = "RNN";
const char_t * const ATTENTIONDECODER = "AttentionDecoder";
const char_t * const LOGICAL_NOT = "LogicalNot";
const char_t * const LOGICAL_AND = "LogicalAnd";
const char_t * const LOGICAL_OR = "LogicalOr";
const char_t * const EQUAL = "Equal";
const char_t * const NOTEQUAL = "NotEqual";
const char_t * const INTERP = "Interp";
const char_t * const SHUFFLECHANNEL = "ShuffleChannel";
const char_t * const AIPP = "Aipp";
const char_t * const MULTISHAPE = "MultiShape";
const char_t * const RECIPROCAL = "Reciprocal";
const char_t * const SELU = "Selu";
const char_t * const ELU = "Elu";
const char_t * const ACOSH = "Acosh";
const char_t * const ASINH = "Asinh";
const char_t * const MINIMUM = "Minimum";
const char_t * const CLIP = "Clip";
const char_t * const L2NORMALIZE = "L2Normalize";
const char_t * const CROPANDRESIZE = "CropAndResize";
const char_t * const UNUSEDCONST = "UnusedConst";
const char_t * const SPARSETODENSE = "SparseToDense";
const char_t * const NONMAXSUPPRESSION = "NonMaxSuppression";
const char_t * const TOPKV2 = "TopKV2";
const char_t * const INVERTPERMUTATION = "InvertPermutation";
const char_t * const MULTINOMIAL = "Multinomial";
const char_t * const REVERSESEQUENCE = "ReverseSequence";
const char_t * const REDUCEPROD = "ReduceProd";
const char_t * const REDUCEMAX = "ReduceMax";
const char_t * const REDUCEMIN = "ReduceMin";
const char_t * const EXTRACTIMAGEPATCHES = "ExtractImagePatches";
const char_t * const SQRT = "Sqrt";
const char_t * const REDUCEALL = "ReduceAll";
const char_t * const RESIZENEARESTNEIGHBOR = "ResizeNearestNeighbor";
const char_t * const SPACETOBATCHND = "SpaceToBatchND";
const char_t * const BATCHTOSPACEND = "BatchToSpaceND";
const char_t * const ASSERT = "Assert";
const char_t * const GREATEREQUAL = "GreaterEqual";
const char_t * const FLOOR = "Floor";
const char_t * const RANDOMUNIFORM = "RandomUniform";
const char_t * const BATCHMATMUL = "BatchMatMul";
const char_t * const SPACETODEPTH = "SpaceToDepth";
const char_t * const DEPTHTOSPACE = "DepthToSpace";
const char_t * const RINT = "Rint";
const char_t * const ATAN = "Atan";
const char_t * const ATAN2 = "Atan2";
const char_t * const ATANH = "Atanh";
const char_t * const ACOS = "Acos";
const char_t * const ASIN = "Asin";
const char_t * const NEG = "Neg";
const char_t * const LOG = "Log";
const char_t * const TAN = "Tan";
const char_t * const ROUND = "Round";
const char_t * const UPSAMPLE = "Upsample";
const char_t * const FLOORMOD = "FloorMod";
const char_t * const LESS = "Less";
const char_t * const LESSEQUAL = "LessEqual";
const char_t * const ONEHOT = "OneHot";
const char_t * const REFSWITCH = "RefSwitch";
const char_t * const REFMERGE = "RefMerge";
const char_t * const ENTER = "Enter";
const char_t * const REFENTER = "RefEnter";
const char_t * const LOOPCOND = "LoopCond";
const char_t * const NEXTITERATION = "NextIteration";
const char_t * const REFNEXTITERATION = "RefNextIteration";
const char_t * const EXIT = "Exit";
const char_t * const REFEXIT = "RefExit";
const char_t * const CONTROLTRIGGER = "ControlTrigger";
const char_t * const ZEROSLIKE = "ZerosLike";
const char_t * const EXP = "Exp";
const char_t * const WHERE = "Where";
const char_t * const FAKEQUANTWITHMINMAXVARS = "FakeQuantWithMinMaxVars";
const char_t * const SOFTPLUS = "Softplus";
const char_t * const SOFTSIGN = "Softsign";
const char_t * const COSH = "Cosh";
const char_t * const SINH = "Sinh";
const char_t * const SQUAREDDIFFERENCE = "SquaredDifference";
const char_t * const REQUIREDSPACETOBATCHPADDINGS = "RequiredSpaceToBatchPaddings";  // for retinanet scope fusion
const char_t * const SSDPOSTPROCESSOR = "SSDPostProcessor";
const char_t * const RETINANETBOXES = "RetinanetBoxes";
const char_t * const RETINAMULTIANCHORS = "RetinaMultiAnchor";
const char_t * const RETINANETCLIPPEDBOXES = "RetinanetClippedBoxes";
const char_t * const RETINANETFILTEREDDETECTIONS = "RetinanetFilteredDetections";
const char_t * const RETINANETPOSTPROCESSOR = "RetinanetPostProcessor";
const char_t * const RETINANETANCHORS = "RetinanetAnchors";
const char_t * const FASTERRCNNMAP = "FasterRCNNMap";
const char_t * const FASTERRCNNMAP1 = "FasterRCNNMap1";
const char_t * const FASTERRCNNSECONDSTAGEPOSTPROCESSOR = "FasterRCNNSecondStagePostprocessor";
const char_t * const FASTERRCNNROIINTERPOOLING = "FasterRCNNROIInterPooling";
const char_t * const FASTERRCNNFIRSTSTAGEPOSTPROCESSOR = "FasterRCNNFirstStagePostprocessor";
const char_t * const FASTERRCNNGRIDANCHORGENERATOR = "FasterRCNNGridAnchorGenerator";
const char_t * const ROIINTERPOOLING = "ROIInterPooling";
const char_t * const FASTERRCNNCLIPTOWINDOW = "FasterRCNNClipToWindow";
const char_t * const EMBEDLOOKUP = "EmbedLookup";
const char_t * const HASHLOOKUP = "HashLookup";
const char_t * const LSH_PROJ = "LshProject";
const char_t * const SVDF = "SVDF";
const char_t * const SSDANCHORGENERATOR = "SSDAnchorGenerator";
const char_t * const IDENTITY = "Identity";
const char_t * const IDENTITYN = "IdentityN";
const char_t * const PLACEHOLDERWITHDEFAULT = "PlaceholderWithDefault";
const char_t * const SELECT = "Select";
const char_t * const GETSPAN = "GetSpan";
const char_t * const STOPGRADIENT = "StopGradient";
const char_t * const PREVENTGRADIENT = "PreventGradient";
const char_t * const GUARANTEECONST = "GuaranteeConst";
const char_t * const BROADCASTGRADIENTARGS = "BroadcastGradientArgs";
const char_t * const BROADCASTARGS = "BroadcastArgs";
const char_t * const CONFUSIONMATRIX = "ConfusionMatrix";
const char_t * const RANK = "Rank";
const char_t * const PLACEHOLDER = "PlaceHolder";
const char_t * const END = "End";
const char_t * const BASICLSTMCELL = "BasicLSTMCell";
const char_t * const GETNEXT = "GetNext";
const char_t * const INITDATA = "InitData";
const char_t * const REFIDENTITY = "RefIdentity";
const char_t * const BITCAST = "Bitcast";

/***************Ann special operator*************************/
const char_t * const ANN_MEAN = "AnnMean";
const char_t * const ANN_CONVOLUTION = "AnnConvolution";
const char_t * const ANN_DEPCONVOLUTION = "AnnDepthConv";
const char_t * const ANN_FULLCONNECTION = "AnnFullConnection";
const char_t * const ANN_NETOUTPUT = "AnnNetOutput";
const char_t * const ANN_DATA = "AnnData";
const char_t * const ANN_RESHAPE = "AnnReshape";
const char_t * const ANN_ADD = "AnnAdd";
const char_t * const ANN_MUL = "AnnMul";
const char_t * const ANN_SUB = "AnnSub";
const char_t * const ANN_DIV = "AnnDiv";
const char_t * const ANN_DEQUANTIZE = "AnnDequant";
const char_t * const ANN_QUANTIZE = "AnnQuant";
const char_t * const ANN_PAD = "AnnPad";
const char_t * const ANN_RESIZE_BILINEAR = "AnnResizeBilinear";

/***************************************************/
/******************Training operator*************************/
const char_t * const GATHERV2 = "GatherV2";
const char_t * const CONVGRADFILTER = "Conv2DBackpropFilter";
const char_t * const CONV2D = "Conv2D";
const char_t * const CONV2DBACKPROPINPUT = "Conv2DBackpropInput";
const char_t * const FUSEDBATCHNORM = "FusedBatchNorm";
const char_t * const BIASADDGRAD = "BiasAddGrad";
const char_t * const ACTIVATIONGRAD = "ReluGrad";
const char_t * const MAXPOOLWITHARGMAX = "MaxPoolWithArgmax";
const char_t * const MAXPOOLGRADWITHARGMAX = "MaxPoolGradWithArgmax";
const char_t * const SPARSESOFTMAXCROSSENTROPYWITHLOGITS = "SparseSoftmaxCrossEntropyWithLogits";
const char_t * const SNAPSHOT = "Snapshot";
const char_t * const VAR = "Var";
const char_t * const MEANGRAD = "MeanGrad";
const char_t * const TRANSLATE = "Translate";
const char_t * const ADDN = "AddN";
const char_t * const L2LOSS = "L2Loss";
const char_t * const MULTIPLY = "Multiply";
const char_t * const HUBERLOSSGRAD = "HuberLossGrad";
const char_t * const HUBERLOSS = "HuberLoss";
const char_t * const NEGATIVE = "Negative";
const char_t * const SSDCAST = "SSDCast";
const char_t * const SPARSESOFTMAXCROSSENTROPY = "SsdSparseSoftmaxCrossEntropy";
const char_t * const SPARSESOFTMAXCROSSENTROPYGRAD = "SsdSparseSoftmaxCrossEntropyGrad";
const char_t * const SSDSQUEEZEFUSION = "SsdSqueezeFusion";
const char_t * const CONCATFOUR2FIVE = "ConcatFour2Five";
const char_t * const CONCATFIVE2FOUR = "ConcatFive2Four";
const char_t * const SSDREALDIVTILEMUL = "SSDRealdivTileMul";
const char_t * const SSDSUMMULREALDIVMEAN = "SSDSumMulRealdivMean";

const char_t * const VARIABLEV2 = "VariableV2";
const char_t * const VARHANDLEOP = "VarHandleOp";
const char_t * const TEMPORARYVARIABLE = "TemporaryVariable";
const char_t * const DESTROYTEMPORARYVARIABLE = "DestroyTemporaryVariable";
const char_t * const VARIABLE = "Variable";
const char_t * const ASSIGN = "Assign";
const char_t * const ASSIGNVARIABLEOP = "AssignVariableOp";
const char_t * const ASSIGNADD = "AssignAdd";
const char_t * const ASSIGNADDVARIABLEOP = "AssignAddVariableOp";
const char_t * const ASSIGNSUB = "AssignSub";
const char_t * const ASSIGNSUBVARIABLEOP = "AssignSubVariableOp";
const char_t * const APPLYMOMENTUM = "ApplyMomentum";
const char_t * const RESOURCEAPPLYMOMENTUM = "ResourceApplyMomentum";
const char_t * const SGD = "SGD";
const char_t * const NOOP = "NoOp";
const char_t * const READVARIABLEOP = "ReadVariableOp";
const char_t * const PARALLELCONCATSTART = "_ParallelConcatStart";
const char_t * const CONSTANTOP = "Constant";
const char_t * const DEPTHWISECONV2DBACKPROPFILTER = "DepthwiseConv2dNativeBackpropFilter";
const char_t * const DEPTHWISECONV2DBACKPORPINPUT = "DepthwiseConv2dNativeBackpropInput";
const char_t * const DEPTHWISECONV2DFORWARDNATIVE = "DepthwiseConv2dNative";
const char_t * const DROPOUTGRAD = "DropOutGrad";
const char_t * const APPLYRMSPROPMIXEDPRECISION = "apply_rms_prop_mixed_precision";
const char_t * const APPLYRMSPROP = "ApplyRMSProp";
const char_t * const RELU6GRAD = "Relu6Grad";
const char_t * const AVGPOOLGRAD = "AvgPoolGrad";
const char_t * const CONCATV2 = "ConcatV2";
const char_t * const CONCATOFFSET = "ConcatOffset";
const char_t * const LAYERNORMGRAD = "LayerNormGrad";
const char_t * const LAYERNORM = "LayerNorm";
const char_t * const LARS = "Lars";
const char_t * const DYNAMICSTITCH = "DynamicStitch";

/***************************************************/
const char_t * const SQUARE = "Square";
const char_t * const HCOMBROADCAST = "HcomBroadcast";
const char_t * const HCOMALLGATHER = "HcomAllGather";
const char_t * const HCOMALLREDUCE = "HcomAllReduce";
const char_t * const HCOMREDUCESCATTER = "HcomReduceScatter";
const char_t * const HCOMSEND = "HcomSend";
const char_t * const HCOMRECEIVE = "HcomReceive";
const char_t * const HCOMREMOTEREAD = "HcomRemoteRead";
const char_t * const HCOMREMOTEREFREAD = "HcomRemoteRefRead";
const char_t * const HCOMREMOTEWRITE = "HcomRemoteWrite";
const char_t * const HCOMREMOTESCATTERWRITE = "HcomRemoteScatterWrite";

const char_t * const VARASSIGN = "VarAssign";
const char_t * const VARISINITIALIZEDOP = "VarIsInitializedOp";
const char_t * const LogTimeStamp = "LogTimeStamp";
const char_t * const ISVARIABLEINITIALIZED = "IsVariableInitialized";
const char_t * const STREAMSWITCH = "StreamSwitch";
const char_t * const STREAMSWITCHN = "StreamSwitchN";
const char_t * const STREAMACTIVE = "StreamActive";
const char_t * const MEMCPYASYNC = "MemcpyAsync";
const char_t * const MEMCPYADDRASYNC = "MemcpyAddrAsync";
const char_t * const STREAMMERGE = "StreamMerge";
const char_t * const ENDGRAPH = "EndGraph";
const char_t * const SEND = "Send";
const char_t * const RECV = "Recv";
const char_t * const ENDOFSEQUENCE = "EndOfSequence";

const char_t * const LABELSET = "LabelSet";
const char_t * const LABELGOTO = "LabelGoto";
const char_t * const LABELGOTOEX = "LabelGotoEx";
const char_t * const LABELSWITCH = "LabelSwitch";
const char_t * const LABELSWITCHBYINDEX = "LabelSwitchByIndex";

const char_t * const ATOMICADDRCLEAN = "AtomicAddrClean";

const char_t * const ABS_GRAD = "AbsGrad";
const char_t * const ACCUMULATE_N_V2 = "AccumulateNV2";
const char_t * const ACOS_GRAD = "AcosGrad";
const char_t * const ACOSH_GRAD = "AcoshGrad";
const char_t * const ANY = "Any";
const char_t * const APPROXIMATE_EQUAL = "ApproximateEqual";
const char_t * const ASIN_GRAD = "AsinGrad";
const char_t * const ASINH_GRAD = "AsinhGrad";
const char_t * const ATAN_GRAD = "AtanGrad";
const char_t * const BROADCAST_TO = "BroadcastTo";
const char_t * const ELU_GRAD = "EluGrad";
const char_t * const ADD_V2 = "AddV2";
const char_t * const DATAFORMATDIMMAP = "DataFormatDimMap";
const char_t * const DATAFORMATVECPERMUTE = "DataFormatVecPermute";
const char_t * const BESSELI0E = "BesselI0e";
const char_t * const BESSELI1E = "BesselI1e";
const char_t * const APPLYADADELTA = "ApplyAdadelta";
const char_t * const APPLYADAGRAD = "ApplyAdagrad";
const char_t * const APPLYADAGRADDA = "ApplyAdagradDA";
const char_t * const APPLYADAM = "ApplyAdam";
const char_t * const APPLYADAMAX = "ApplyAdaMax";
const char_t * const APPLYADDSIGN = "ApplyAddSign";
const char_t * const APPLYCENTEREDRMSPROP = "ApplyCenteredRMSProp";
const char_t * const APPLYFTRL = "ApplyFtrl";
const char_t * const APPLYFTRLV2 = "ApplyFtrlV2";
const char_t * const APPLYGRADIENTDESCENT = "ApplyGradientDescent";
const char_t * const APPLYPOWERSIGN = "ApplyPowerSign";
const char_t * const APPLYPROXIMALADAGRAD = "ApplyProximalAdagrad";
const char_t * const APPLYPROXIMALGRADIENTDESCENT = "ApplyProximalGradientDescent";
const char_t * const DEQUANTIZE = "Dequantize";

const char_t * const FOCAL_LOSS = "FocalLoss";
const char_t * const FOCAL_LOSS_GRAD = "FocalLossGrad";
const char_t * const SMOOTHL1_LOSS = "SmoothL1Loss";
const char_t * const SMOOTHL1_LOSS_grad = "SmoothL1LossGrad";
const char_t * const REDUCEMEAN = "ReduceMean";
const char_t * const CONCAT_V2 = "ConcatV2";
const char_t * const ONEHOT_V2 = "OneHotV2";
const char_t * const SLICE_V2 = "SliceV2";
const char_t * const TILE_V2 = "TileV2";
const char_t * const SUM_V2 = "SumV2";
// Common type when the operator has the same name
const char_t * const DETECTIONOUTPUT = "DetectionOutput";
// Custom operator
const char_t * const CUSTOMOP = "CustomOp";
const char_t * const CUSTOMOP_NCHW = "CustomOpNchw";
const char_t * const CUSTOMOP_NHWC = "CustomOpNhwc";
const char_t * const CUSTOMOP_NC1HWC0 = "CustomOpNc1hwc0";

// Depthwise 4d_2_6d,6d_2_4d
const char_t * const DEPTHWISEWEIGHT4D26D = "depthwise_weight_4d_2_6d";
const char_t * const DEPTHWISEWEIGHT6D24D = "depthwise_weight_6d_2_4d";

const char_t * const SQRTGRAD = "SqrtGrad";
const char_t * const SIGMOIDGRAD = "SigmoidGrad";

const char_t * const TRANSSHAPE = "TransShape";

// Horovod operator
const char_t * const HVDCALLBACKALLREDUCE = "HorovodAllreduce";
const char_t * const HVDCALLBACKALLGATHER = "HorovodAllgather";
const char_t * const HVDCALLBACKBROADCAST = "HorovodBroadcast";
const char_t * const HVDWAIT = "HorovodWait";

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
