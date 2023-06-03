---
layout: post
title:  在WSL2上安装CUDA和Pytorch
date:   2023-05-21 20:40:24 +0800
categories: cuda wsl2 pytorch
---

### 安装CUDA
1. 在nvidia官网[下载](https://www.nvidia.com/Download/index.aspx)对应电脑显卡的驱动并安装。注意，只需在windows上安装nvidia驱动，它会自动在wsl2中安装一个libcuda.so文件，作为wsl2的cuda驱动。这与直接在linux主机上安装nvidia驱动有所不同：在linux主机上安装cuda toolkit时，toolkit默认包含了驱动，只要装好toolkit，驱动也会同时装好。如果在wsl2上直接安装linux版本的cuda toolkit，会导致windows驱动自带的libcuda.so被覆盖。所以，nvidia专门提供了wsl版本的cuda toolkit，这个toolkit不包含驱动。
2. 进入wsl ubuntu，下载并安装cuda toolkit。这里下载安装cuda toolkit 11.7。依次执行如下命令：

```shell
  sudo apt-key del 7fa2af8
  wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
  sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
  wget https://developer.download.nvidia.com/compute/cuda/11.7.1/local_installers/cuda-repo-wsl-ubuntu-11-7-local_11.7.1-1_amd64.deb
  sudo dpkg -i cuda-repo-wsl-ubuntu-11-7-local_11.7.1-1_amd64.deb
  sudo cp /var/cuda-repo-wsl-ubuntu-11-7-local/cuda-*-keyring.gpg /usr/share/keyrings/
  sudo apt-get update
  sudo apt-get -y install cuda
```
至此，完成cuda在wsl2上的安装。详情参考nvidia官网[指南](https://docs.nvidia.com/cuda/wsl-user-guide/index.html#getting-started-with-cuda-on-wsl-2)。

### 安装pytorch
1. 安装python3、pip3和virtualenv，如果已安装，可以跳过此步：
```shell
  sudo apt update
  sudo apt install python3-dev python3-pip python3-venv
```
2. 创建虚拟环境并激活，在虚拟环境中安装pytorch。这里安装的pytorch版本是2.0.1+cu117。
```shell
  python3 -m venv --system-site-packages ./pytorch
  source pytorch/bin/activate
  pip3 install torch torchvision torchaudio torchtext
```
安装完成后进行简单测试：
```python
  import torch
  x = torch.rand(5, 3)
  print(x)
  torch.cuda.is_available()
```
3. pytorch实际占用磁盘空间只有约4.4G，但是安装之后会发现磁盘空间至少被占用了10G。这时需要压缩虚拟磁盘文件以节省磁盘空间。在windows powershell中打开windows自带的diskpart程序：
```shell
  wsl --shutdown
  diskpart
```
要压缩的虚拟磁盘文件通常位于"C:\Users\用户名\AppData\Local\Packages"目录下，以字符串"CanonicalGroupLimited"开头的目录中，例如，本人的虚拟磁盘文件路径为"C:\Users\yanwo\AppData\Local\Packages\CanonicalGroupLimited.Ubuntu20.04LTS_79rhkp1fndgsc\LocalState\ext4.vhdx"。在diskpart中执行如下命令：
```shell
  select vdisk file="C:\Users\yanwo\AppData\Local\Packages\CanonicalGroupLimited.Ubuntu20.04LTS_79rhkp1fndgsc\LocalState\ext4.vhdx"
  attach vdisk readonly
  compact vdisk
  detach vdisk
  exit
```
完成压缩后，磁盘空间多了6G以上。

