# 工具、格式与接口

## 运行环境与文件接口

Python 3.11+、NumPy、SciPy、Pillow、Numba；运行 `python -m pip install -r "<skill-root>/scripts/requirements.txt"`，建议独立虚拟环境。仅在任务需要且允许时安装依赖。无需 CUDA、Torch、Open3D、AMD 驱动 SDK、网络服务或硬编码临时目录。首次 CPU Numba 渲染会编译缓存；全场景检查内存随点数及视图尺寸增长，先用合理分辨率定位，再在交付分辨率复核。未做完整性能/跨平台矩阵保证。

文件操作可用 PowerShell/Python。带空格和中文的路径作为独立参数；不要把路径拼进可执行 shell 文本。所有输出使用新路径。PLY reader 忽略注释中的命令；不执行配置内的字符串。CLI 的 JSON/NPY 文件是数据。

本工具支持含完整 Gaussian 属性的标量 binary_little_endian / binary_big_endian PLY，SH degree 0–3。属性按名字读取，可以重排，输出统一为 little-endian float32。要求：
- x/y/z；scale_0..2；opacity；rot_0..3；f_dc_0..2；
- f_rest_* 连续编号，数量为 0、9、24、45，按 RGB channel-major 存储各自非 DC 系数；
- 默认 scale=log、opacity=logit、quaternion=wxyz，必须从导出器确认而非自动猜测。

nx/ny/nz 可在规范化时丢弃，报告中记录；其他字段默认拒绝，只有确认可丢弃才显式 allow_extras。非 float32 核心字段默认拒绝，可显式 allow_cast，且不能再宣称原始字节精确保留。alpha 输入的 0/1 会夹至 1e-7/1-1e-7 后转 logit。拒绝 ASCII、mesh、packed/compressed GS、普通 RGB 点云和未知 SH 阶数；需使用可信转换器另存并验证，不修改头部冒充兼容。

规范行数组为 float32 NxD：
`xyz[0:3], log_scale[3:6], opacity_logit[6], quaternion_wxyz[7:11], DC[11:14], SHrest[14:]`。
D=14/23/38/59。原生 PLY 属性顺序不影响规范数组。跨阶数提升按每个颜色通道填零，不直接给扁平数组末尾补零。保留验证针对这些规范字段，不包括被明确丢弃的辅助属性。

## CLI

`python "<skill-root>/scripts/gs_cli.py" <command> --help` 可查精确参数。以下输入/输出均可为绝对路径，配置内相对路径以配置所在目录为准。成功 exit=0；校验失败 exit=2，需检查输出与异常。失败时不得发布残留的孤立 PLY 或不完整 sidecar。

| 命令 | 关键参数 | 结果与注意事项 |
|---|---|---|
| inspect | INPUT | 点数、字节、字段、SH、采样范围；不是全量验证 |
| validate | INPUT [--report JSON] | 全量 finite、四元数/尺度、字节数、SHA256 |
| normalize | INPUT OUTPUT | 规范化格式；不执行空间配准 |
| fit-pairs | --pairs JSON --threshold FLOAT --output JSON [--iterations INT] | 鲁棒 source→target Sim(3)、残差与独立控制点统计 |
| icp | --source PLY --target PLY --initial JSON --voxel FLOAT --max-distance FLOAT --output JSON [--source-mask NPY --target-mask NPY] | 已初始化、尺度锁定，输出矩阵与迭代报告 |
| transform | INPUT OUTPUT --matrix JSON | 同步变换所有 GS 参数，保持 SH 方向一致 |
| fuse | --manifest JSON | 新 PLY、同名 .provenance.npz 与 .report.json |
| shell | --config JSON --output PLY | 显式测量面生成的 SH0 支撑，附 .support.json |
| material | INPUT OUTPUT --indices NPY [--anchor-camera JSON --gain R G B --alpha-exponent FLOAT --expected-sha HASH] | 只改所选点 opacity/DC/SH，附 .material.json |
| render | INPUT --camera JSON --output PNG | CPU 参考图与相机/hash metadata |
| trace | INPUT --camera JSON --pixel-mask PNG [--point-mask NPY] --out-prefix PREFIX [--min-score FLOAT] | .scores.npy、.indices.npy、.trace.json |
| audit | --config JSON --output-dir DIR | 全场景基线/染色或探针/ROI 图与 audit.json |
| preserve | --baseline PLY --model PLY --provenance NPZ --report JSON [--source-id INT --changed-rows NPY --allowed-columns INT... --required-rows NPY] | 未修改规范字段及必须保留行验证 |
| colmap-camera | --sparse DIR --image NAME [--transform JSON] --output JSON | 从 COLMAP 相机导出正确变换后的世界位姿 |
| match-colmap | --source-sparse DIR --source-db DB --target-sparse DIR --target-db DB [--every INT --ratio FLOAT] --output JSON | 只读 SIFT/SfM 候选匹配，不是真值 |

inspect/validate/normalize/transform 支持：
`--opacity logit|alpha --scale log|linear --quaternion wxyz|xyzw --allow-cast --allow-extras`。
fuse 的各 input 对象支持对应同名字段，其余操作假定已规范化。

fit-pairs 的 JSON：
`{"source":[[x,y,z],...],"target":[[x,y,z],...],"trusted_holdout_indices":[...部分行索引...]}`。
至少 3 个非共线训练点；独立控制点不能参与拟合。输出 matrix 是 4x4 source→target，列向量表达，JSON 可直接为 4x4 或含 matrix 对象。accepted=false 的 fit 文件禁止融合；accepted=null 表示未提供真正独立控制点，不能自动当验收通过。矩阵需正尺度、无反射/剪切的 Sim(3)。

indices NPY 为一维 int64；点 mask 可为 N 个 bool 或行号列表；像素 PNG 的非零区域为选区，尺寸需与相机一致。preserve 的 changed-rows 和 required-rows 是 baseline 行号，allowed-columns 是上述规范数组列号，只豁免所声明行的所声明列。它不验证变换过的 baseline，应先在同一坐标建立可比基准。required-rows 必须给出，才能证实关键原始细节没有被删；仅比较保留行无法证明未丢点。

## 融合配置

复制 [fusion.example.json](../assets/fusion.example.json) 为实际任务配置。schema_version=1；example_only=true 时拒绝执行。核心字段：

| 字段 | 意义 |
|---|---|
| inputs[] | id 唯一；path；transform 矩阵或矩阵文件；可选 sha256、opacity/scale/quaternion、allow_cast/allow_extras |
| inputs[].opacity_weight | [0,1] 物理 alpha 权重，默认1；不是 logit 直接乘权重 |
| inputs[].optional | 默认 false；必须来源选中零点时报错，避免“恢复细节”却没包含地面模型 |
| output | 新输出路径，现有模型/sidecar 不覆盖 |
| frame | origin 与 proper orthonormal basis；basis 列为查询坐标轴 |
| default_owner | 未覆盖空间来源，默认 inputs[0] |
| ownership[] | owner、priority、bbox 或 polygon_xy+z_min+(z_max/ceiling)；有限三维区域 |
| ownership[].ceiling | planes 的下包络 z=min(a*x+b*y+c)，margin；可选 minimum_sigma |
| bound_covariance | 在所选有 ceiling 的来源内限制协方差，默认 false |
| inputs[].select_region | 附加裁剪，用同样区域语法；不会给缺失点补结构 |
| inputs[].selection | 默认单一所有权；append 需同时 generated_support=true，仅用于生成支撑 |
| up / chunk_size | 可选输出 up；默认优先 frame 的 Z，再源文件 up 经变换。流式批量默认32768 |

bbox=[[xmin,ymin,zmin],[xmax,ymax,zmax]]，含边界；polygon_xy 为严格凸点集，可顺/逆时针；可以分区覆盖但不允许同优先级不同 owner 冲突。区域 frame 可覆盖全局 query frame。平面和高度单位使用该正交坐标，不允许隐式非均匀缩放。

provenance NPZ：source_id 为 manifest inputs 零起始编号，source_row 为该来源原始行号。输出混合 SH 时保持逐通道基函数一致。模型局部删改后同时更新/继承映射并绑定新 SHA；material 不改顺序，sidecar 需显式保留并记录父模型。

## 相机与全场景遮挡配置

[camera.example.json](../assets/camera.example.json)：
model=PINHOLE/SIMPLE_PINHOLE，position、width/height、f 或 fx/fy/cx/cy；orientation 可为世界→相机 rotation 3x3，或 target+up。相机 +Z 前方、像素 X 右/Y 下；near 与模型坐标同单位。坐标和视场必须完全一致才可比较。

COLMAP 接口读取 cameras.bin/images.bin/points3D.bin 与匹配的 database.db，数据库 mode=ro。相机导出保留实际相机模型，不自动去畸变。参考 renderer 只接受去畸变针孔；鱼眼/ERP 应先使用已知校准生成透视面及正确相机，再投影，不能改 model 字符串冒充。project_points 还支持 SIMPLE_RADIAL/RADIAL/OPENCV 供重投影检查，其他模型显式拒绝。

[audit.example.json](../assets/audit.example.json)：
- model、可选 provenance；surfaces[] 含 id、frame、polygon_xy、planes、interior_margin。
- 每面 frame 的 +Z 指向要检查的外部可见侧。顶面可用建筑 up；竖墙应设置墙法线为该面的 +Z。不能直接把竖墙塞进建筑高度平面方程。
- recolor 需 hidden_sources（provenance 的数值编号）或 hidden_indices NPY；hidden_below 限定真实表面后的内容，可用 hidden_region 进一步限定。
- tests[] 含 id、surface、完整 camera、mode=recolor/probe、min_pixels、tolerance（RGB 0..1）、allowed_changed_pixels、可选 mask_erosion_pixels。
- probe 需 surface.probe_step/probe_depth，生成诊断点只存在测试内存，不写入交付模型。
- 同一表面多机位重复测试；只改隐藏内容的颜色，位置/alpha 保持，因此泄漏会在真实全场景排序中显现。

[shell.example.json](../assets/shell.example.json)：
frame、凸 polygon_xy、planes、step、normal_sigma、layer_offsets=[[dx,dy,dz],...]、measured_color=[0..1]、measurement_evidence、可选 opacity_logit/max_points。薄层间距、覆盖面、材料颜色均须实测。双坡面使用平面下包络；需避开真实开口。生成颜色为常量下衬，不包含摄影纹理。

## Python 接口与宿主工具

需要自定义流程时将 skill/scripts 加入 sys.path；所有依赖均为同目录模块或 requirements 内包。

| 模块 | 主要入口 |
|---|---|
| gs_core | SplatFile(path,encoding options).chunks()/rows()/info()/close()；sha256；validate_rows；promote；similarity；transform_rows；colors；attenuate；write_atomic；fit_pairs；umeyama |
| gs_regions | frame；region_mask；owners；bound_to_ceiling |
| gs_pipeline | fuse_manifest；transform_file；material_file；shell_file |
| gs_registration | refine_icp |
| colmap_io | read_cameras/read_images/read_points；camera_json；project_points；point_features；match_features |
| gs_render | camera；load_model；render_array；render_image；trace_visibility；contact_sheet |
| gs_audit | surface_roi；probe_cloud；run_audit |
| gs_verify | verify_preservation |
| selfcheck | 可直接运行的独立合成行为验证，不读取项目私有文件 |

close 必须在 Windows 重命名之前完成；用 with SplatFile(...) 管理 memmap。脚本已实现输出 partial→回读→rename、输入 hash 前后检查和禁止覆盖源模型。融合的 sidecar 写入不是跨文件事务，发布前检查三个文件都完整。

查看器通过用户指定应用的“导入 PLY”操作，无已知私有 HTTP API。历史应用为 F:/OOOSplat/AMD-v16/OOOSplat-AMD.exe，仅为案例路径，运行时需发现实际安装。可优先使用宿主文件/进程工具打开已授权的新文件，需交互检查时遵守宿主 computer-use 接口；用户自行导入时不要占用其当前编辑会话。截图不能取代保存相机/模型 hash，成功导入也不能取代几何验收。

