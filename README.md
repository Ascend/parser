# Ascend CANN Parser

Ascend CANN Parser（简称parser）配合TF_Adapter、 ATC工具、IR构图等使用，开发者通过以上工具，借助parser能方便地将第三方框架的算法表示转换成Ascend IR，充分利用昇腾AI处理器卓越的运算能力。
![parser系统框架](https://images.gitee.com/uploads/images/2020/1015/151426_71a73e7e_7876749.png "parser系统框架.PNG")

## 安装

parser以动态库的方式被调用。

### 源码安装

Parser支持由源码编译，进行源码编译前，首先确保你有昇腾910 AI处理器的环境进行源码编译前，确保系统满足以下要求：

- GCC >= 7.3.0
- CMake >= 3.14.0
- Autoconf >= 2.64
- Libtool >= 2.4.6
- Automake >= 1.15.1

#### 下载源码

```
git clone https://gitee.com/ascend/parser.git
cd parser
git submodule init && git submodule update
```

#### 源码编译

在parser根目录执行以下命令编译：
```
export ASCEND_CUSTOM_PATH=昇腾910基础安装包的安装路径
bash build.sh
```

## 贡献

欢迎参与贡献。

## 路标

以下将展示graphenine/parser近期的计划，我们会根据用户的反馈诉求，持续调整计划的优先级。

总体而言，我们会努力在以下几个方面不断改进。

    1、完备性：Cast/ConcatV2算子支持输入数据类型为int64的常量折叠；

    2、完备性：onnx parser支持一对多映射；

    3、架构优化：ATC解耦并迁移至parser；

    4、易用性：提供tensorflow训练的checkpoint文件转pb文件的一键式转化工具；

    5、易用性：提供一键式本地编译环境构建工具；

    6、可维测：ATC转换生成的om模型包含框架信息、cann版本信息和芯片信息等；

热忱希望各位在用户社区加入讨论，并贡献您的建议。

## Release Notes

Release Notes请参考[RELEASE](RELEASE.md)。

## 许可证

[Apache License 2.0](LICENSE)
