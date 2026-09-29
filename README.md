# Image Processing Toolkit · 图像处理工具箱

基于 PyTorch 与 Flask 的图像处理项目，提供图像分类、图像去噪与相似图像检索的 Web 界面。

## 功能

- **图像分类**：识别上身衣服、鞋、包、下身衣服与手表五类商品。
- **图像去噪**：展示加噪图像与模型去噪后的结果。
- **相似图像检索**：基于编码器向量和余弦距离检索相似图片。

## 项目结构

- `web/`：Flask 服务、页面模板和界面图片。
- `image_classification/`：分类模型、训练与推理代码。
- `image_denoising/`：去噪模型、训练与推理代码。
- `image_similarity/`：相似度模型、预训练权重与图片向量。
- `common/`：通用工具、图片数据集及标签。
- `test/`：实验脚本与 Notebook。

本仓库包含现有模型权重、图片数据集及向量文件。首次下载的体积较大。

## 本地运行

在项目根目录创建并激活 Python 虚拟环境，安装依赖：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install flask torch torchvision numpy pillow scikit-learn matplotlib tqdm pandas ipython
```

由于现有程序使用相对路径，请进入 `web` 目录启动，并把项目根目录加入 Python 模块搜索路径：

```bash
cd web
PYTHONPATH=.. python web_app.py
```

浏览器打开 http://127.0.0.1:9000 。运行环境与依赖版本请根据实际机器调整，以上为根据现有代码整理的启动方式，尚未在全新环境验证。

## 关于 GitHub 页面

GitHub 仓库页面用于查看和管理代码。图像处理交互界面需要在本地或另行部署的服务器运行；上传仓库本身不会自动部署 Flask 服务。
