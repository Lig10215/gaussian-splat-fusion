# gaussian-splat-fusion · 高斯泼溅空地融合

Align, fuse, repair and verify existing 3D Gaussian Splatting (3DGS) PLY models, especially scenes captured from both aerial and ground platforms.

以用户指定的底稿为坐标与编辑基准，把航拍、地面等多份高斯模型的可靠部分融合成可追溯的新 PLY，并做配准、局部修复与同机位 / 遮挡验证。全部工具为纯 CPU 实现（AMD / NVIDIA 显卡均可，无需 CUDA），但这**不代表**已实现 GPU 加速。

## 能做什么 / 不能做什么

| 支持 | 不支持 |
|---|---|
| 基于 SfM 轨迹或实测对应点的鲁棒 Sim(3) 配准、尺度锁定 ICP 微调 | 从照片训练 / 重建缺失几何 |
| 按三维区域显式划分「单一所有权」的模型融合，保留来源行映射 | 把多个模型直接拼接成叠层外墙 |
| 漂浮物、重复墙 / 台阶、透明屋顶 / 楼板、局部文字与亮度的诊断修复 | 自动猜测软混合带、自动鱼眼重训练 |
| CPU 参考渲染、像素贡献 trace、全场景遮挡审计、未修改区域保留验证 | 替代用户在真实查看器里的目视验收 |

输入必须是含完整 Gaussian 属性的 PLY（SH degree 0–3、log scale、logit opacity、wxyz 四元数）；普通彩色点云、ASCII PLY、mesh、packed/compressed GS 会被拒绝。原始输入只读，输出始终写入新的递增版本路径。

## 安装与使用

把本仓库克隆到 Agent 的用户级 skill 目录即可，例如 WorkBuddy：

```bash
git clone <your-repo-url> ~/.workbuddy/skills/gaussian-splat-fusion
```

依赖 Python 3.11+ 与 `scripts/requirements.txt`（NumPy、SciPy、Pillow、Numba），建议使用独立虚拟环境：

```bash
python -m pip install -r scripts/requirements.txt
python scripts/selfcheck.py          # 13 项行为自检，在临时目录运行，不改动用户模型
python scripts/gs_cli.py --help      # 查看 15 个子命令
```

典型流程：

```bash
python scripts/gs_cli.py inspect  "<source.ply>"
python scripts/gs_cli.py validate "<source.ply>"
python scripts/gs_cli.py fuse --manifest "<job.json>"
```

配置模板见 [`assets/fusion.example.json`](assets/fusion.example.json)。模板中所有长度仅为演示值，且带 `example_only` 标记会被拒绝执行——复制到任务目录、按真实坐标校准、移除该标记后再用。配置内的相对路径以配置文件所在目录为准。

## 目录结构

```
SKILL.md                 技能入口：触发条件、执行顺序、完成条件
agents/openai.yaml       可选的 Codex 展示元数据（WorkBuddy 不读取）
assets/                  4 个 JSON 配置模板：融合 / 相机 / 审计 / 支撑面
references/
  workflow.md            配准、融合、修复、交付的完整决策流程
  interfaces.md          PLY 字段约定、CLI 参数、JSON/NPY 数据接口
  verification.md        验收清单、反向遮挡测试、交付证据
  troubleshooting.md     模糊、重影、透地、导入异常的定位路径
  case-studies.md        两个实际场景的经验记录
  validation-record.md   已执行测试及其边界
scripts/                 13 个 Python 模块 + CLI + 自检，纯 CPU
```

## 关键约定

- **坐标**：不要假设全局 Z 是高度。必须确认 up 方向、单位、导出缩放和编辑器额外变换 `M = E_aerial @ T @ inverse(E_ground)`。
- **变换完整性**：`p'=sRp+t` 时，位置、尺度、四元数、SH 系数和相机必须同步变换，只改 XYZ 会造成视角颜色错误。
- **单一所有权**：每个实体只保留一份结构，按神龛、走廊、纪念碑等实体逐项恢复地面细节，不用大范围「低处归航拍」掩膜抹掉建筑。
- **可追溯**：输出附带 `.provenance.npz` 与 `.report.json`，所有局部索引与掩膜绑定模型 SHA256。
- **证据优先**：没有控制点或可用观测时停在证据缺口，不伪造完整结构；CPU 参考渲染不能冒称真实查看器验证。

完整命令与参数见 [`references/interfaces.md`](references/interfaces.md)。

## 来源与许可

本技能由本地 Codex skill 导入，并针对 WorkBuddy 补充了中文触发词与元数据；算法、脚本与参考文档保持原样。`scripts/selfcheck.py` 在 Python 3.13 + NumPy 2.5 / SciPy 1.18 / Numba 0.68 环境下 13 项检查全部通过。

仓库当前未附带 LICENSE。若原技能来自第三方，公开发布前请先确认其授权条款。
