# CEBS 核心防御代码

本仓库用于论文中的代码链接，只保留核心防御：证据计算、可信校准、最终掩码、受害模型训练，以及 FEDformer 和 TimesNet 预测骨干、必要配置和数据说明。SimpleTM 可通过上游代码单独接入。

不放消融、基线对照、多种子、投毒率、自适应搜索、计算成本测量、历史审计报告和云端运维代码；这些内容另存于私有实验仓库。

## 论文补充材料

[英文补充材料（PDF）](supplementary/CEBS_Supplementary_Material.pdf)包含论文的算法细节、实验配置与完整补充结果。

使用示例配置时，基础分支会在运行时重新选择历史区间与预测步的配对。论文的 PEMS03–BackTime 评估中，基础分支使用评估前冻结的六对，详见补充材料 S1 节。

入口是 `defense_cebs.py`：

```bash
python -m pip install -r requirements.txt
python defense_cebs.py --config configs/pems03_backtime.yaml --stage check
python defense_cebs.py --config configs/pems03_backtime.yaml --stage all
```

`check` 只检查输入，`detect` 重新检测并冻结掩码，`train` 用已完成检测的最终掩码训练，`all` 依次完成检测与训练。参数、数据格式与模型设置见 [英文 README](README.md) 和 [数据说明](data/README.md)。

数据、感染代理及模型权重不直接放入 Git；用户需按格式和来源说明准备。只读测试与 CPU 软件检查不代表本次已重新完成 GPU 端到端实验。原研究工程、正式掩码和论文不作修改。

如需使用 SimpleTM，请单独获取 [SimpleTM 上游代码](https://github.com/vsingh-group/SimpleTM)，将其仓库根目录加入 `PYTHONPATH`，并在配置中设置 `victim: SimpleTM`。参阅 [英文 README](README.md#optional-simpletm-predictor) 中的已测试上游版本。
