# 桌面端与后端的平台设计

一套 Electron App 和桌宠源代码，只发布 Windows/macOS 客户端；Linux 只运行后端。
操作系统差异放在启动适配层，不分叉 Web UI、记忆、ASR、Router 或 TTS 业务逻辑。

## 平台分工

| 用户电脑 | App / 桌宠 / 麦克风 / 播放 | 本机推理后端 | 发布物 |
| --- | --- | --- | --- |
| Windows x64 + NVIDIA | Windows 原生 Electron | WSL2 中的 Linux CUDA；优先复用 Docker 镜像 | 一个含桌宠的 EXE 安装包 |
| Apple Silicon Mac | macOS 原生 Electron | macOS 原生 Python + MLX/Metal | 一个含桌宠的 DMG |
| Intel Mac 或无兼容 GPU 的 Windows | 同一个桌面客户端 | 连接远程 Linux/NVIDIA 服务 | 对应架构的桌面包 |
| Linux / WSL | 不安装 App，不自动启动桌宠 | CUDA Docker 或已有原生 Python 环境 | 后端镜像 / 源码 |

Windows 不运行 Linux App，也不通过 WSLg 运行桌宠。客户端发送音频给后端，
桌宠只观察 `/ws-pet`；两者均使用现有协议，不要求将麦克风设备透传给 WSL 或容器。

Windows 访问 WSL 内的服务优先使用 `http://127.0.0.1:8787`。
Microsoft 支持由 Windows 通过 localhost 访问 WSL 服务，但 VPN、防火墙及 WSL 配置仍可能影响连通性；
连接失败应诊断转发，不自动开放所有网卡或改防火墙。
参见 [WSL 网络说明](https://learn.microsoft.com/en-us/windows/wsl/networking)。

Windows 的 Docker 路线需要已运行的 Docker Desktop、启用 WSL2 后端，以及支持 WSL 的 NVIDIA 驱动。
这不是为 Windows 另做一套 CUDA 镜像，仍复用 Linux 镜像。
参见 [Docker Desktop GPU 条件](https://docs.docker.com/desktop/features/gpu/)。

## 配置页面

远程连接页与对话风格选择页共用浅色卡片视觉和 680×430 紧凑窗口；菜单中统一使用“连接设置”，主对话页面继续复用 Web。
源码启动提供两个入口：

- `npm start`：Mac 使用原生 MLX；Windows 检查 Docker Desktop、引导 API 配置、构建缺失的镜像并启动 WSL2/CUDA 容器，待后端就绪直接打开 App。Windows 无需手动安装 Ubuntu Python 或 Breeze。
- `npm run start:remote`：只配置服务地址，不要求本机具备 CUDA、MLX、WSL 或 Docker。

Windows 和 Mac 共用配置页面；平台差异由启动适配层处理。
模型文件存在且通过完整性检查就复用；缺失时先显示下载内容和进度，不覆盖已有模型。
本机模型路径必须属于实际运行后端的文件系统：Windows、WSL 和容器路径不是同一个命名空间，不能直接混用。
远程模式的模型和 API Key 留在服务器，不能拿客户端目录代替服务器配置。

Windows Docker 配置使用项目 `.env`（明文、Git 忽略），分别记录记忆和对话 provider/Key，保留已有字段和注释；已配置的角色不重复询问。容器继续由 Docker 管理，退出 App 不停止容器。组件设置只读，用户在 `.env` 中修改服务后重新运行入口。

Mac 首次 `npm start` 在终端分别选择 VoiceMem 记忆 API 与 Studio Agent 对话 API，分别读取需要的密钥。托管后端启动后，“设置 → 组件”
可分别修改记忆和回复的服务商、OpenAI-compatible 地址、模型名与 Key，并由 App 受控重启。
成功配置使用系统安全存储加密，下次启动直接复用；远程连接仍只读，由服务器管理配置和重启。

## 源码 App 启动流程

```text
npm start
  ├─ Apple Silicon Mac：记忆 API → 准备记忆模型 → 对话 API → MLX 后端预热
  └─ Windows：检查 Docker → 配置缺失的记忆/对话 API → 构建缺失的镜像
              → 启动 Compose CUDA 后端 → 跟踪模型下载/预热日志
  → 后端就绪 → 风格选择页 → 主界面 + 桌宠

npm run start:remote
  → 不检查或启动本机后端 → 空地址配置页 → 输入地址并连接已有服务
```

准备和等待期间只使用终端，不创建本地连接/状态窗口。终端准备失败时显示具体错误并退出；Electron 启动后的失败使用原生错误框。
受管后端首次下载和预热默认最多等待 30 分钟；连接已有服务仍使用较短的三分钟就绪等待。

自动识别平台不等于自动安装系统组件。缺少 WSL、驱动、Docker 或 MLX 环境时应准确提示，
不擅自提权、修改系统配置或新建第二套模型目录。

只有由 App 新创建并明确拥有的原生后端进程才随 App 退出；已运行的共享服务、Docker 容器和远程服务不自动停止。
Mac App 的启动适配传入 `STUDIO_DESKTOP_PET=0`，桌宠由 App 管理；Linux/WSL 后端不自动拉起桌宠。

## 当前实现与待验证边界

- 已实现：共享 App + 桌宠、配置文案、Windows/macOS 打包配置、连接现有服务、Linux/WSL 禁止自动拉起桌宠。
- 已实现代码并做模拟测试：`npm start` 的记忆/回复独立 provider 与密钥步骤、UI-only 必填地址、组件内模型服务配置、安全持久化与失败回退、VoiceMem 模型准备、Mac MLX 受控进程和 Windows `wsl.exe` CUDA 受控进程；App 退出时结束自己启动的进程。
- 已实现代码并做模拟测试：Windows Docker CLI 本机 named-pipe 检查、首次配置、镜像缺失时构建、Compose 启动和容器失败检测。
  图形连接页只启动已有服务。源码 `npm start` 可构建缺失的 Studio 镜像并更新配置有变化的容器，显示日志并最多等待 30 分钟；不自动安装或启动 Docker Desktop，不删除卷。
- 尚未实现：安装包内置 Python/模型环境；本地 ASR、TTS、感知和声纹模型继续由后端 profile 管理。
- 尚未实机验收：Windows 安装包、WSL/CUDA 联调、Mac 打包签名和麦克风权限。
  Linux 的虚拟桌面测试只验证共享客户端代码，不等于支持 Linux 桌面发布，也不能代替 Windows/macOS 验收。
