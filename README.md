# Ascend CANN Parser

Ascend CANN Parser（简称parser）配合TF_Adapter、 ATC工具、IR构图等使用，开发者通过以上工具，借助parser能方便地将第三方框架的算法表示转换成Ascend IR，充分利用昇腾AI处理器卓越的运算能力。
![parser系统框架](https://images.gitee.com/uploads/images/2020/1015/151426_71a73e7e_7876749.png "parser系统框架.PNG")

## 安装

parser以动态库的方式被调用。

### 源码安装

parser支持源码编译，进行源码编译前，首先确保你有昇腾910AI处理器的环境，同时确保系统满足以下要求：

- GCC >= 7.3.0
- CMake >= 3.14.0
- Autoconf >= 2.64
- Libtool >= 2.4.6
- Automake >= 1.15.1

#### 下载源码

```
git clone --recursive 
https://gitee.com/ascend/parser.git -b development
cd parser
chmod +x build.sh
```

#### 源码编译

在parser根目录执行以下命令编译：
```
bash build.sh

编译完成之后，相应的动态库文件会生成在output文件夹中
```

## 贡献

欢迎参与贡献。

## Release Notes

Release Notes请参考[RELEASE](RELEASE.md)。

## 许可证

[Apache License 2.0](LICENSE)
