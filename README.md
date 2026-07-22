# MXMoE-Adapt

面向沐曦曦云 C500 与 MXMACA 的路由负载感知 MoE 融合算子协同优化与自适应派发系统。

> 当前状态：`0.1.0` 已完成 C500 M0 真机里程碑。项目在固定的 FlagGems v5.0.2 提交上完成一个真实 Decode Shape 的 FP16/BF16 正确性与多随机种子复验；完整模型层、更多路由分布和上游合入仍待完成。

当前实现复用 FlagGems 的 Fused MoE Kernel，通过 C500 专用配置注入、搜索、验证与置信派发完成适配和调优；它不是一个已经被 FlagGems 官方收纳的全新 Kernel。FlagGems 上游 Issue/PR 尚未提交，只有未来 PR 被官方主仓合并后才能称为上游贡献。

## 项目解决什么问题

通用 Fused MoE 配置通常只使用 token 数、专家数和矩阵 Shape 选择 Tile，但真实 MoE 推理还受到专家负载偏斜、空专家比例、对齐填充以及软件栈版本的影响。MXMoE-Adapt 将这些因素放进同一个闭环：

1. 采集真实 Top-K 路由并计算负载特征；
2. 联合搜索路由对齐块与专家 GEMM 参数；
3. 在 C500 上离线实测并生成带环境指纹的配置数据库；
4. 运行时进行置信派发，超出实测覆盖范围时安全回退 FlagGems；
5. 当 MXMACA、mcTriton、FlagGems 或驱动变化时自动标记配置过期。

## 核心创新

- **路由感知**：使用路由熵、变异系数、空专家比例和 padding ratio，而不只使用平均 token/expert。
- **联合优化**：把 `align_block_size` 与 `BLOCK_SIZE_M/N/K`、warps、stages 共同搜索。
- **置信回退**：仅派发到同 dtype、相近 Shape、相近路由类型且经过 C500 验证的配置。
- **漂移治理**：环境指纹或守护 Shape 性能漂移超过阈值时触发重新调优。
- **反事实路由回放**：保存同一批 Top-K IDs，在不同Kernel间复用，排除路由随机性干扰。
- **多保真搜索**：资源代理过滤、编译验证、合成路由、真实路由、模型层逐级升级。
- **证据化发布**：每个性能结论同时保存环境、Shape、路由、随机种子、正确性和计时方法。

## 仓库结构

```text
mxmoe-adapt/
├─ src/mxmoe_adapt/       # 路由特征、联合搜索、派发、环境与漂移检测
├─ benchmarks/             # C500 Fused MoE 正确性和延迟基准
├─ configs/                # 工作负载、环境及配置数据库示例
├─ tests/                  # 不依赖 GPU 的单元测试
├─ docs/                   # 申报书、技术方案、实测手册和项目管理
└─ results/                # 经脱敏的 C500 M0 证据；其他临时结果默认忽略
```

## 快速开始

项目命令统一使用 `python3`。

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e .
python3 -m unittest discover -s tests -v
```

在 C500/MXMACA 环境先采集只读信息：

```bash
python3 -m mxmoe_adapt.environment --output results/c500-environment.json
python3 -m mxmoe_adapt.search_space --output results/search-space.json
```

尚未开通实例时，先按 [C500租赁与镜像选择](docs/C500租赁与镜像选择.md) 选择硬件和官方镜像。
需要提交镜像选项或开通后的命令回显时，填写 [C500开通信息与环境日志表](docs/C500开通信息与环境日志表.md)。

将已脱敏的 Top-K ID 列表转换为聚合路由特征：

```bash
python3 -m mxmoe_adapt.trace results/topk-ids.json \
  --experts 64 --output results/route-features.json
```

运行 FlagGems 基线小 Shape 冒烟测试：

```bash
python3 benchmarks/benchmark_fused_moe.py \
  --candidate mxmoe_adapt.adapters.flaggems:run \
  --device cuda --dtype fp16 --tokens 4 --experts 8 \
  --hidden-size 256 --intermediate-size 512 --top-k 2 \
  --output results/flaggems-smoke-fp16.json
```

完成多组实测后生成不隐藏失败记录的汇总：

```bash
python3 -m mxmoe_adapt.report results --output results/summary.json
```

真实大 Shape 必须在确认显存需求后逐步放大，参见 [C500 实测手册](docs/C500实测手册.md)。

## 正确性和性能口径

- 正确性基准：清晰的 PyTorch Reference。
- 性能基线：固定 FlagGems v5.0.2 提交上的 C500 可运行安全锚点；不可编译的上游默认配置不用于计算加速比。
- 优化对象：现阶段为 C500 专用联合配置；独立 MetaX Fused MoE Kernel 属于后续工作。
- 计时：预热后同步设备，报告中位数/高分位延迟；首版脚本提供基础均值计时。
- 发布门槛：只有 `verified=true` 且环境指纹匹配的配置才能自动派发。

## C500 M0 实测结果

- 环境：单卡 MetaX C500 64GB、MXMACA 3.5.3.20、PyTorch `2.8.0+metax3.5.3.9`、FlagGems v5.0.2。
- Shape：`tokens=4, E=8, H=4096, I=14336, top-k=2`。
- 上游通用默认配置需要 73,728B 共享内存，超过 C500 的 65,536B 上限，编译失败。
- C500 安全锚点：`M16/N128/K64/warps4/stages2`。
- 当前最优配置：`M16/N32/K64/warps4/stages2`。
- FP16：3 个随机种子中位加速 `1.0283x`，最大绝对误差 `0.0009765625`。
- BF16：3 个随机种子中位加速 `1.0295x`，最大绝对误差 `0.0078125`。
- CPU 单元测试：18/18 通过。

完整成功与失败记录见 [`results/c500-e928d862973e-20260722`](results/c500-e928d862973e-20260722)，汇总结论见 [`c500-m0-milestone.json`](results/c500-e928d862973e-20260722/c500-m0-milestone.json)。这些结果仅支持上述单一 Shape 的 M0 结论，不外推到完整模型或所有 MoE 工作负载。

## 首期边界

首期聚焦单卡曦云 C500、FP16/BF16、MoE 专家计算主链路。多卡 Expert Parallel、All-to-All、FP8/W8A8 和训练场景只进入后续路线图。

## 开源与申报

项目采用 Apache-2.0 许可证，公开仓库为 [chionglee14/mxmoe-adapt](https://github.com/chionglee14/mxmoe-adapt)。后续可按青年开源专项基金要求同步镜像至 Gitee/GitLink。申报材料见 [项目申报书](docs/项目申报书.md)。
