更新日志：

2017年01月23日17:31:04

更新了OS X的正确安装方式。



第一次下载

cmake -DCMAKE_CXX_COMPILER=g++-6首先，根据建议[installation Guide](https://github.com/Microsoft/LightGBM/wiki/Installation-Guide)，  进行安装。

### Windows

LightGBM use Visual Studio (2013 or higher) to build in Windows.

Clone or download latest source code.
Open ./windows/LightGBM.sln by Visual Studio.
Set configuration to Release and x64 .
Press Ctrl+Shift+B to build.
The exe file is in ./windows/x64/Release/ after built.
### Linux

LightGBM use cmake to build in Unix. Run following:

```

git clone --recursive https://github.com/Microsoft/LightGBM ; cd LightGBM 

mkdir build ; cd build 

cmake .. 

make -j

```

**一切都没有问题, //但是，OS X很不友好//**
### OSX

OSX

LightGBM depends on OpenMP for compiling, which isn't supported by Apple Clang.

Please use gcc/g++ instead.

Run following:

在安装gcc前，最好采取如下措施。如果不这样做，直接按照官网的安装方式，基本失败。（=_=|||惨痛）

1. 检查gcc的版本。几种方式。gcc -v 检查默认gcc的版本，一般都是clang。 gcc-6 -v 指定版本的gcc的版本。由于后续会指定安装版本，因此，检查 `gcc-6 -v`会比较合适。结果如下，其中--without-multilib是安装支持openmp的gcc版本，默认是不支持的：

```

Thread model: posix

gcc version 6.3.0 (Homebrew GCC 6.3.0_1 --without-multilib) 

```

如果发现gcc不是6.X版本的或者不支持OpenMP（没有 --without-multilib）， 那么需要采取如下措施：

1. 更新Homebrew， 之后重新安装

```shell

brew update

brew reinstall gcc6 --without-multilib

```

这样能简单地安装gcc6，但要编译很长时间，make bootstrap大概在60-80min（mac pro 128G乞丐版）。之后执行。

```shell

git clone --recursive https://github.com/Microsoft/LightGBM ; cd LightGBM 

mkdir build ; cd build 

cmake -DCMAKE_CXX_COMPILER=g++-6 -DCMAKE_C_COMPILER=gcc-6 .. 

 make -j

```

#### 可能出现的其他问题

一般上述命令当然不会成功，os 对gcc6的支持极差。

然后就会开始报错。

```shell

fatal error: 'omp.h' file not found
```

"Please use gcc/g++ instead." 是指，在非 Clang的环境下， 命令改为 "cmake -DCMAKE_CXX_COMPILER=g++" 。教程中的是在指定gcc的版本进行安装。因为Clang的llvm和xcode不支持OpenMP

有几种解决办法：

1. 在OSX中安装gcc 6. [安装教程 更新日期9.22](https://solarianprogrammer.com/2016/05/10/compiling-gcc-6-mac-os-x/)

很复杂。

因此有了第二种办法。在安装[homebrew](http://brew.sh/)的情况下进行安装

>Homebrew 是管理mac os中缺失的包的强大工具。

```

brew install clang-omp

```

参考该文档[OpenMP®/Clang](http://clang-omp.github.io/)

继续刚刚的步骤，快乐地使用LightGBM

如果还不能用，试试下面的命令

```shell

cmake -DCMAKE_CXX_COMPILER=clang-omp++ -DCMAKE_C_COMPILER=clang-omp .. 

```

然而可能还没有用，我没有别的办法了，咧嘴笑。😁