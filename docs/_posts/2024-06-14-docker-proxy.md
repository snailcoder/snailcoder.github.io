---
layout: post
title:  设置docker代理
date:   2024-06-14 10:33:41 +0800
tags:   docker wsl2
---

最近很多国内docker镜像站接到上级通知要求关闭，据说是为了配合监管。一切发生地静悄悄，莫名其妙就没了。中科大、南大、上海交大，无一幸免。不得已，只能自己配置docker代理。

参考[官方文档](https://docs.docker.com/config/daemon/systemd/)：

> If you're behind an HTTP or HTTPS proxy server, for example in corporate settings, the daemon proxy configurations must be specified in the systemd service file, not in the daemon.json file or using environment variables.

1. 创建目录：
```
sudo mkdir -p /etc/systemd/system/docker.service.d
```
2. 在上面目录中创建配置文件http-proxy.conf，在文件中设置代理：
```
[Service]
Environment="HTTP_PROXY=http://proxy.example.com:3128"
Environment="HTTPS_PROXY=https://proxy.example.com:3129"
Environment="NO_PROXY=localhost,127.0.0.1,docker-registry.example.com,.corp"
```
3. 重启docker
```
sudo systemctl daemon-reload
sudo systemctl restart docker
```

注意：在wsl下使用systemctl时有个小问题，直接执行上面的命令会报错：

> System has not been booted with systemd as init system (PID 1). Can't operate. Failed to connect to bus: Host is down

需要修改wsl配置文件才能正常使用systemctl命令，具体地，在/etc/wsl.conf文件中添加如下配置：

```
[boot]
systemd=true
```

然后在powershell中重启wsl即可：

```
wsl --shutddown
wsl
```

