# TRELLIS.2 Image-to-3D Generator

输入一张图片，生成带 PBR 材质的 3D GLB 模型。

## 硬件要求

- NVIDIA GPU，**建议 40GB+ 显存**（A100 40GB 可跑 512 模式）
- 最低 16GB 显存（仅 512 模式，可能 OOM）
- 1024 模式需 80GB 显存

## 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/YOUR_USERNAME/trellis2-3d-generator.git
cd trellis2-3d-generator
```

### 2. 安装依赖

```bash
# 一键安装（推荐 CUDA 12.4）
bash setup.sh

# 或手动安装
pip install -r requirements.txt
```

### 3. 下载模型权重

```bash
python download_weights.py
```

首次运行会下载约 20GB 模型文件到 `~/.cache/trellis2/`。

部分模型需要 HuggingFace 授权：
- 访问 https://huggingface.co/briaai/RMBG-2.0 点击 "Agree and access repository"
- 创建 HF Token: https://huggingface.co/settings/tokens
- 设置环境变量: `export HF_TOKEN=your_token_here`

### 4. 生成 3D 模型

```bash
python generate.py 你的图片.jpg -o 输出.glb
```

**选项**：
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-o, --output` | 输出 GLB 路径 | `output.glb` |
| `-r, --resolution` | 管线分辨率 (512/1024) | 512 |
| `--steps` | 采样步数（越少越快） | 12 |
| `--no-pbr` | 不使用 PBR 材质 | False |

### 5. 后处理（可选）

色彩校准：让生成模型的颜色更接近原图。

```bash
python post_process.py 输出.glb --calibrate --original 原图.jpg --color-strength 0.3
```

## 示例

```bash
# 生成
python generate.py examples/chair.jpg -o chair.glb

# 校色
python post_process.py chair.glb --calibrate --original examples/chair.jpg
```

## 文件结构

```
├── generate.py           # 主推理脚本
├── post_process.py       # 后处理管线（色彩校准）
├── download_weights.py   # 模型权重下载
├── setup.sh              # 一键环境配置
├── requirements.txt      # Python 依赖
└── examples/             # 示例图片
```

## 常见问题

**Q: CUDA out of memory?**
A: 用 `-r 512` 降低分辨率，或减少 `--steps` 到 8。

**Q: 生成的模型没有颜色？**
A: 用支持 PBR 的查看器打开（Windows 3D Viewer、Blender、https://gltf-viewer.donmccurdy.com/）。

**Q: 颜色偏暗？**
A: 用 `post_process.py --calibrate` 做色彩校准。

## 致谢

基于 Microsoft [TRELLIS.2](https://github.com/microsoft/TRELLIS.2) 项目。
