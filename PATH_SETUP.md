# 路径配置快速设置指南

## 🚀 3 步快速开始

### 第 1 步：验证项目根目录

打开 `configs/paths.yaml` 检查：

```yaml
project_root: null  # ← 保持为 null 让它自动检测
```

或手动指定（Windows 示例）：

```yaml
project_root: C:\Users\YourName\model_deployment_runtime_lab
```

### 第 2 步：验证路径配置

```powershell
python verify_paths.py
```

你会看到：

```
Model Deployment Runtime Lab - Path Verification

Project Root: ...
Critical directories:
  ✅ configs
  ✅ src
  ✅ README.md
  ⚠️ outputs  planned
  ⚠️ samples  planned
```

### 第 3 步：查看完整配置

```powershell
python verify_paths.py --show-config
```

## 📁 核心配置文件

### `configs/paths.yaml`

所有路径都定义在这里。示例结构：

```yaml
project_root: null

artifacts:
  root: outputs
  onnx: outputs/onnx
  quantized: outputs/quantized
  reports: outputs/reports

configs:
  root: configs
  paths: configs/paths.yaml
  model: configs/model.yaml
  runtime: configs/runtime.yaml
```

## 🔧 Python 脚本中使用

```python
from src.utils.path_manager import paths

# 获取单个路径
onnx_dir = paths.get("artifacts.onnx")
print(onnx_dir)  # PosixPath('.../outputs/onnx')

# 确保目录存在
paths.ensure_dir("artifacts.onnx")

# 获取所有路径
all_artifacts = paths.get_all("artifacts")
```

## ❓ 常见问题

**Q: 自动检测没有工作？**
A: 检查 project_root 设置，或手动指定绝对路径

**Q: 在 WSL 中运行需要什么？**
A: 使用 `/mnt/c/...` 来访问 Windows 路径

**Q: 我的 artifacts 不在 outputs/ 目录？**
A: 在 `configs/paths.yaml` 中编辑路径即可

## 验证脚本

```powershell
# 基本验证
python verify_paths.py

# 显示完整配置
python verify_paths.py --show-config
```

---

**所有路径都集中在一个地方。** 🎉
