# Contributing

感谢参与 MXMoE-Adapt。项目优先接受可复现的 C500/MXMACA 贡献。

## 提交前

```bash
python3 -m unittest discover -s tests -v
python3 -m compileall -q src tests benchmarks
```

## 性能贡献必须包含

- MetaX C500设备信息和环境指纹；
- Driver、MXMACA、mcPytorch、mcTriton/FlagTree、FlagGems版本；
- baseline和candidate提交；
- Shape、dtype、Top-K及路由统计；
- 正确性误差、预热、重复次数和原始JSON；
- 失败配置和过滤原因。

不接受其他GPU结果冒充C500结果，也不接受只提供最快数字而缺少原始记录的性能PR。

## 代码要求

- 新功能附带CPU单元测试；
- C500专用代码通过适配器或后端目录隔离；
- 不提交密钥、账号、内网地址或用户请求数据；
- 保留清晰的PyTorch Reference和原始FlagGems基线。
