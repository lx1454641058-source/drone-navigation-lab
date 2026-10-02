# 来源与许可证记录

## 2026-09-17：TUM RGB-D 实测空间输入

从 [TUM 官方源](https://cvg.cit.tum.de/data/datasets/rgbd-dataset)取得 `freiburg3_walking_xyz` 的固定六时刻 ROS bag 子集与动捕位姿，数据 CC BY 4.0。署名 J. Sturm、N. Engelhard、F. Endres、W. Burgard、D. Cremers，IROS 2012。原始 HTTP 范围、消息、来源及改制说明保存在 `D:/DroneNavTools/tum-rgbd-walking/bag-subset/` 和 `work/tum-rgbd-bag-input-01/`。RGB 消息无损转 PNG，深度保留 float32；没有下载完整 bag，也没有与 TGZ 做逐字节等价检查。完整 TGZ 下载超时残件保留但未用于实验。

解析器、适配、验证、页面均为本项目原创；仅阅读 ROS 官方 BSD 格式实现和消息定义，未复制或执行其代码、未安装 ROS。复用现有 Python/Pillow/Node/ONNX 与 TinyFormer 及原组合许可，没有引入 NumPy 或新依赖。详见 [实验记录](TUM_RGBD_EXPERIMENT.md)，未上传或发布。

核对日期：2026-09-16。早期仅阅读官方资料；v0.9 引入物理核心，v0.12 引入自然视觉资源，分节记录如下。

## 2026-09-17 TinyFormer-S 航拍适配候选

- [发布仓库](https://huggingface.co/LibreYOLO/LibreTinyFormers-visdrone)，固定版本 `9f05e33763dc8292e9b1b4cf7276672be491f2d8`；ONNX 45,697,518 字节，SHA-256 `0cd550196b1a69fd68a0993aa8df62e6f52346dbb970735dd57957151917d509`，与 LFS 摘要一致。
- 保存的 LICENSE 包含 Apache-2.0（TinyFormer/DEIMv2）和 DINOv3 参数许可；两者同时适用，不能只称 Apache-2.0。保留 AICVlab/NYCU、Intellindust、Meta 的 NOTICE 署名，论文使用需致谢 DINOv3。模型卡指向 TinyFormer 上游提交及官方检查点镜像，但本项目未独立重现训练或证明导出等价。
- `D:/DroneNavTools/tinyformer-probe/` 保存模型、模型卡、许可、来源记录；`reference/` 保存公开预处理/后处理说明代码的阅读副本与摘要。未安装或执行 LibreYOLO，不加载 `.pt`；仅复用既有 Pillow 和 ONNX Runtime。项目适配、测试及页面原创。
- VisDrone 数据和叠加展示继续保留 CC BY-NC-SA 3.0 署名，与模型许可分开。仅本地研究，未上传或发布。详情见 [TINYFORMER_EXPERIMENT.md](TINYFORMER_EXPERIMENT.md)。
- PP-YOLOE-SOD/Paddle2ONNX 仅作下载检查，转换扩展缺 DLL，Paddle Windows wheel 附带的 GNU 运行库许可未核清，因此未安装 Paddle 或执行其推理。下载留在 `D:/DroneNavTools/sod-probe/` 供检查，不属于项目运行依赖。不能以仓库 Apache 声明替代每个打包组件的许可。

2026-09-17 漏检诊断及成果整理：仅使用已有 VisDrone 图像/标注、YOLOX 原始输出与本地运行库，没有引入新第三方资源。新诊断逻辑、测试和页面为项目原创；数据叠加展示延续原署名和 CC BY-NC-SA 3.0，模型许可不覆盖数据。详见 [DETECTION_FAILURE_ANALYSIS.md](DETECTION_FAILURE_ANALYSIS.md)。

| 项目 | 官方来源 | 用途 / 结论 | 引入状态 |
|---|---|---|---|
| PX4 仿真 | https://docs.px4.io/main/en/simulation/ | 核对后续飞控与仿真路线；具体版本组合待环境试验 | 仅参考 |
| PX4 Windows 环境 | https://docs.px4.io/main/en/dev_setup/dev_env_windows_wsl | Windows 下 WSL2 开发路线；尚未安装 | 仅参考 |
| PX4 许可证 | https://github.com/PX4/PX4-Autopilot/blob/main/LICENSE | 核心仓库采用 BSD 3-Clause；未来引入时还要核对具体版本、子模块和资产 | 未引入 |
| OpenCV 许可证 | https://github.com/opencv/opencv/blob/4.x/LICENSE | 4.x 当前官方许可文本为 Apache 2.0；拟作图像处理候选，后续固定版本再核对 | 未引入 |
| 高斯朴素贝叶斯 | https://scikit-learn.org/stable/modules/naive_bayes.html | 参考通用数学公式；本项目自行实现，不使用 scikit-learn 代码或软件包；输出不能直接当校准概率 | 仅算法参考 |
| 针孔相机投影 | https://docs.opencv.org/4.13.0/d9/d0c/group__calib3d.html | 2026-09-16 从官方 4.x 页面跳转至此；核对内参、世界/相机坐标与投影公式。射线求交和逆投影由本项目原创实现 | 仅公式参考，未复制代码或插图 |
| 体素射线遍历 | https://www.eecs.yorku.ca/~amana/research/grid.pdf | Amanatides / Woo，1987，作者网站论文；参考按格边界增量遍历的通用思想，本项目自行实现有限线段、并列轴推进和占用处理 | 仅算法思想参考，未复制第三方代码或图表 |
| NumPy 许可证 | https://numpy.org/doc/stable/license.html | 同时检查本机 2.3.5 发行包许可证：其二进制所含组件列有 GPL-3.0-or-later WITH GCC-exception-3.1；为遵守用户偏好，本项目不使用该包 | 已有环境只读检查，未引入 |
| 高德接口概览 | https://developer.amap.com/api/ | 前期已阅读：地图展示与地面路线接口不能直接替代空中导航 | 未接入，使用条款待选型时核对 |
| 美团配送案例 | https://www.meituan.com/news/NN241023053008001 | 前期已阅读：固定取餐点是课题场景参考，不代表合作或获得真实运营数据 | 仅事实参考，不复制照片 |

项目 Python 实现、HTML 界面、测试、合成场景、RGB-D 图像、合成像素标签与颜色训练模型来源：本毕设项目新写，未复制第三方实现。A*、高斯朴素贝叶斯、平面最小二乘拟合为通用算法思想。v0.12 另引入下述自然图像与预训练权重，其许可独立；没有引入字体或第三方三维模型，物理包附带资源未使用。

v0.1–v0.8 运行仅使用已安装 Python 标准库和浏览器标准能力，没有新增软件包。v0.9 的可选物理入口另使用下列固定版本 MuJoCo 核心。项目暂不设置对外开源授权；由用户在公开前决定。

v0.5 新增探索历史策略、分段匀加速/制动公式、秒级观测时效、对照实验、重放器与页面均为本项目原创。使用基础运动学关系 `v = at`、`s = vt`、`s = v²/(2b)` 自行实现，没有复制第三方运动库，也没有新增外部模型、数据或素材。新文件头已标注来源；此版本未新增需要核对许可证的第三方资源。

v0.6 的水平平移误差模型、误差范围检查、线段/长方体分段距离最小值、四布局矩阵、压缩重放器和页面均为原创。gzip 为已有 Python 标准库，未下载第三方定位库或模型；新文件头均标注来源。此版本不新增外部资源及其授权条款。

v0.7 的可停止方向搜索、配对条件核对、新旧实验重放与页面扩展均为原创。广度优先搜索为通用算法思想，使用 Python 标准库 deque 自行实现，没有复制外部规划器代码、引入新依赖或下载资源。项目对外分发许可证仍由用户决定。

v0.8 的透视末端检查、共享平面解法、任务门控、12 场景、同帧几何对照、重放器和页面均为原创。复用现有原创颜色模型和 Python 标准库，没有下载或引入第三方代码、模型权重、数据、字体、素材或依赖；每个新代码文件注明来源。对外分发许可仍由用户决定。

未来准入要求：每项代码、模型权重、数据、字体、地图/三维资产都记录精确版本、来源、许可证、署名要求、修改与再分发限制；不确定的资源先不引入。模型训练数据权利无法确认时不能宣称全部权利已核清。

## v0.9 新增外部核心库（已实际使用）

- MuJoCo 3.3.7 Windows x86_64：[官方固定发布](https://github.com/google-deepmind/mujoco/releases/tag/3.3.7)，主许可 [Apache-2.0](https://raw.githubusercontent.com/google-deepmind/mujoco/3.3.7/LICENSE)。完整读取该包第三方声明，保留于 D 盘运行目录；包/DLL/主许可摘要及各许可类别见 `PHYSICS_ENVIRONMENT.md`。
- 仅使用核心 DLL 与公开 C API，保留完整声明，不修改第三方代码；没有使用 Python mujoco/NumPy、示例无人机模型、第三方字体或图形资产。机体 XML、控制器、适配器、测试和页面均为本项目原创。
- 检查过 3.13.0 的完整包，发现 GPL 3 with special bison exception 声明，未启用该版本；压缩包仅作为检查缓存，不属于运行依赖。3.3.7 notices 中 LLVM 许可提及 GPL 兼容例外不等同于其本身为 GPL。GLM 的 [MIT 选项](https://raw.githubusercontent.com/g-truc/glm/1.0.1/copying.txt)另已核对。
- API 依据：[3.3.7 公开函数](https://mujoco.readthedocs.io/en/3.3.7/APIreference/APIfunctions.html)、[XML 模型定义](https://mujoco.readthedocs.io/en/3.3.7/XMLreference.html)及同包头文件。核心版本固定，不依据最新文档猜测结构体布局。
- 本轮没有发布或重新分发库；未来打包时须保留实际所需 Apache/BSD/MIT/zlib/Qhull/LLVM 等许可和来源声明，不能仅写“全部原创”。

## v0.10 沿用既有依赖

逐帧位姿建图、导航与物理执行适配、到达回执、六场景实验、重放器、测试和页面均为本项目原创，新源文件注明来源。沿用上述已核验 MuJoCo 3.3.7 核心、原创机体 XML 和颜色模型，没有新增或下载第三方代码、软件包、权重、图片、字体、地图或三维素材。没有改变既有许可证或发布代码。

## v0.11 沿用既有依赖

视觉下降、完整通道证据时效、局部视野检查、接地/停桨确认、九场景故障注入、重放、测试及页面均为本项目原创；已注明来源。复用现有针孔投影、平面拟合与颜色模型，没有新增第三方算法代码或资源。物理仍使用已核验 3.3.7 核心公开 API，原机体模型与许可声明未改。没有下载、安装、升级、上传或公开发布。

## v0.12 自然视觉资源（已实际使用）

- 明细、固定版本、署名和限制见 [REALVISION_EXPERIMENT.md](REALVISION_EXPERIMENT.md) 的来源表；每个下载地址、文件大小和 SHA-256 在 [vision_assets.json](../tools/vision_assets.json)。记录主包许可与第三方通知，不能只凭包名判断许可。
- ONNX Runtime Node/common 1.22.0：MIT 主许可及完整第三方通知；pngjs 7.0.0：MIT。仅抽取 Windows x64 CPU 运行内容。没有安装 npm 的构建/安装依赖，未执行包内安装脚本。
- NVIDIA SegFormer B0 ADE20K，经 Xenova 固定版本导出 ONNX：上游非商业研究/评估条款，完整许可留在 `D:/DroneNavTools/vision-v12/licenses/SegFormer-LICENSE`。
- UAVid 原作者授权 CC BY-NC-SA 4.0，九张图像及标注通过明确同许可的非官方镜像获取。原站 TLS 验证失败，未绕过证书检查。镜像已将标签转换为单通道，未证实与原档逐字节一致；页面和报告标明镜像及改动。
- 本机已有 Pillow 12.3.0 只用于一次缩放对照：主许可 MIT-CMU，FreeType 选择 FTL，liblzma 为 0BSD；未使用其 GPL 脚本或构建工具，未安装/升级包，也不成为应用依赖。许可文本保留在该安装包的 `dist-info/licenses/LICENSE`。
- 新预处理、评价、映射、准备脚本、测试、核验器和界面均为本项目原创。未引入 GPL/AGPL 实现、Ultralytics、NumPy、PyTorch、UAVidToolKit 代码或训练程序；模型训练集全部底层权利并未因此被证明，当前仅按模型作者给出的研究许可使用，不扩大为商用授权。

## v0.13 上游参数核对与原创分块

- 新增参考文件来自 [NVIDIA 官方模型固定版本](https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512/tree/489d5cd81a0b59fab9b7ea758d3548ebe99677da)：`model.safetensors`，15,036,944 字节，SHA-256 为 `6ae39addd01de6b1b8bde2cf677d43a5cd733424b8d186de3f95d1c51fee23f9`；同版 README、config 与预处理配置只作参考。存于 `D:/DroneNavTools/work/vision-v13/`。沿用 NVIDIA 非商业研究/评估条款，不新增商用权利。
- 格式依据：[ONNX v1.18.0 schema](https://raw.githubusercontent.com/onnx/onnx/v1.18.0/onnx/onnx.proto)、[Safetensors 格式说明](https://github.com/safetensors/safetensors)。解析器由本项目按公开格式原创实现，仅限这里的固定文件；未引入上述仓库实现或 Python 依赖。
- 融合公式核对参考 [Transformers v4.46.3 SegFormer 解码头](https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/models/segformer/modeling_segformer.py) 中卷积与批归一化结构；未复制或执行该模块。204 组参数核对不等于完整图计算等价。
- 分块、插值优化、拼接、重放、测试、报告和界面均为原创，复用 v0.12 CPU 运行库及数据，没有新增运行软件包或 GPL/AGPL 资源。UAVid 图像和改制图继续保留作者署名、CC BY-NC-SA 4.0 与镜像说明；不公开发布或上传。

## 2026-09-17 Cityscapes 候选开发探针

新增 NVIDIA SegFormer B0 Cityscapes 的 Xenova ONNX 转换；固定版本、摘要、来源和实验结论见 [CITYSCAPES_PROBE.md](CITYSCAPES_PROBE.md)。模型卡明确指向 NVIDIA SegFormer 非商业研究/评估许可，完整许可随本地模型保留。沿用既有 ONNX Runtime 和 pngjs，不新增运行依赖；新增探针和测试原创。UNetFormer/GeoSeg 的 GPL-3.0 不符合用户要求，仅阅读许可和说明，未引入实现或权重。

后续输入核对复用本机既有、已核验许可的 Pillow 12.3.0；许可选项同 v0.12 缩放对照。仅用于独立读取及缩放验证，不成为应用依赖，没有下载或升级包。新增核对脚本原创，结果和范围见 `VISION_IO_AUDIT.md`。

## 2026-09-17 YOLOX-S 目标检测开发实验

- 官方 YOLOX-S 0.1.1rc0 ONNX 模型，35,858,002 字节，SHA-256 `c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063`；模型、固定版本 Apache-2.0 全文和官方使用说明存于 `D:/DroneNavTools/yolox-probe/`。
- 模型下载与许可地址分别固定在 [probe_detection.py](../tools/probe_detection.py) 的资源表和每次归档的 `sources.json`；具体来源与限制见 [DETECTION_PROBE.md](DETECTION_PROBE.md)。初次 SHA 为本地记录，不是发布者签名。
- 复用现有 ONNX Runtime CPU 运行库、pngjs 和 UAVid 两张开发帧。没有新增运行软件包、PyTorch、Ultralytics 或 COCO 原始图片。模型训练数据所有底层权利仍未全部核清，不能把 Apache-2.0 扩大到 UAVid 数据或整个项目。
- 原创实现输入插值、框解码、去重、像素覆盖评价、只读核验、测试和页面；没有复制官方示例实现。页面保留 UAVid 作者署名、镜像及 CC BY-NC-SA 4.0。未公开发布或上传。

## 2026-09-17 VisDrone 物体级开发评价

数据来自 AISKYEYE / 天津大学的 VisDrone2019-DET 验证集，按官方学术用途和 CC BY-NC-SA 3.0 条款处理。官方下载配额超限后使用 Ultralytics assets 的原始 JPG/TXT 镜像；只读取数据，未复制或执行其 AGPL 软件、转换脚本或模型。固定地址、81,638,851 字节包摘要、声明与失败下载记录见 [OBJECT_EVALUATION.md](OBJECT_EVALUATION.md)；镜像与官方原包逐字节等价性未验证。

从 548 图的文件名目录预选 12 图，保留原 JPG、八字段 TXT、选择依据和所有忽略区。用既有 Pillow 12.3.0 转 PNG（许可同此前记录），本轮将其用于数据准备，不升级运行依赖；运行推理仍为已有 ONNX Runtime 和 pngjs。图像与叠加展示保留作者署名和原数据许可。新增解析器、固定规则评价器、实验脚本、测试及页面均为原创，未复制官方 Matlab/COCO 评测实现，也不宣称官方 mAP 等价。

政策/下载页由网页工具读取成功；本地 HTTP 存档请求返回 403，source-records/sources.json 保留失败状态，未声称完整法律文本已本地保存。原图、标注及改制物的许可与项目原创代码分开。没有向外上传图像或发布成果。


## 2026-09-17：整图去重与保留图片

复用原 VisDrone 镜像 SHA-256 `abeea063037e5d20398837deb11084e652402a34ddf4f207bdf541a6f2a35ef9`、已登记 YOLOX-S/TinyFormer-S 固定权重及运行库；无新第三方下载或依赖安装。新增十二图来自同包固定分组顺序第 13–24 项，保存原始 JPG/TXT、转换 PNG 和选择记录，仍按 CC BY-NC-SA 3.0 及原作者署名处理。模型各自许可不变，新归档保留候选的 LICENSE/NOTICE/模型卡/来源清单。原创新增编排、去重关系记录、评价页面与测试，未对外发布；详见 [实验记录](WHOLE_DEDUP_EXPERIMENT.md)。


## 2026-09-17：位置候选合并与第三批图片

同 query 合并算法、编排、页面与测试为项目原创。无第三方新下载、安装或升级；沿用原固定模型、运行库与 VisDrone 包。第三批按既定文件名前缀代表图哈希排序第 25–36 项选择，保留 JPG/TXT 和转换 PNG，数据署名与 CC BY-NC-SA 3.0 不变。各新归档保存 TinyFormer 模型卡、LICENSE/NOTICE/来源清单，模型的 Apache-2.0 与 DINOv3 组合许可不变。记录见 [三批实验](QUERY_DEDUP_EXPERIMENT.md)，没有上传或发布。


## 2026-09-17：检测到空间观测接口

新增内部适配、连通区域编排、页面和测试为项目原创，复用已有针孔相机、渲染器及颜色模型，无新增依赖或第三方下载。十二张真实图与 TinyFormer 输出来自 `work/query-dedup-holdout-01/`，保留原署名、VisDrone CC BY-NC-SA 3.0 及模型组合许可；新归档附 LICENSE/NOTICE/模型卡/来源记录。不上传、不公开发布。详见 [接口实验](DETECTION_BRIDGE_EXPERIMENT.md)。


## 2026-09-17：观测通道与新几何解除

观测缓存、路线视图、解析角域深度下界生成器、测试和页面为项目原创，无新依赖、下载或许可变更。`work/observation-channel-02/` 复用前轮真实图，沿用 VisDrone 许可/署名与模型来源；`work/reobservation-01/` 只使用原创合成观测和解析夹具，没有复制真实照片或权重。旧来源档案保留，无上传或发布。几何契约仅在受控解析场景成立，详见 [解除实验](REOBSERVATION_EXPERIMENT.md)。


## 2026-09-17：分辨率与多视角采样覆盖

新增固定世界目标、采样计数、矩阵编排、测试和展示为项目原创，复用原 `raycast.render`、针孔相机、材质和 PNG 编码。`work/sampling-coverage-01/` 的全部 864 帧是原创合成 RGB-D，不含第三方照片或模型权重。没有新下载、依赖安装、许可变更、上传或发布。详见 [采样实验](SAMPLING_COVERAGE_EXPERIMENT.md)。


## 2026-09-17：采样与运动检查联调

新增组合检查、原始帧编排、测试及演示为项目原创；复用原机体/定位误差/制动空间检查、运动学模拟、RGB-D 渲染和体素建图。`work/sampling-motion-01/` 只保存原创合成输入及轨迹，不含新第三方资源。没有依赖升级、下载、许可变更、上传或外部发布。详见 [实验记录](SAMPLING_MOTION_EXPERIMENT.md)。


## 2026-09-17：物理复查增补

新增 `physical_rescan.py`、实验脚本、测试和页面均为本项目原创；复用已经核验的 MuJoCo 3.3.7 核心 DLL、原创四旋翼模型、控制器和 RGB-D 渲染器。来源与 Apache-2.0 / 第三方声明沿用 PHYSICS_ENVIRONMENT.md；没有新下载、模型、数据或依赖。`work/physical-rescan-02/report.json` 保存源码、引擎和物理模型摘要。宽视角、20 秒静态有效期和处理时间均是明确的开发假设，不是第三方实测数据或推荐飞行参数。


## 2026-09-17：持续反馈恢复增补

新增恢复模块、实验脚本、测试和页面为本项目原创。继续使用原 MuJoCo 3.3.7、原创机体和控制器；父归档为 physical-rescan-02，没有第三方新增资源、软件安装或许可证变化。阈值是开发参数，不能作为实机安全标准引用。
