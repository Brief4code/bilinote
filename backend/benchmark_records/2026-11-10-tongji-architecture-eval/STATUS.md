# Tongji 四组架构评测（2026-11-10）

已按用户要求清理上一轮额度受限实验，工作区只保留当前实验；可恢复副本及清理范围见 [CLEANUP.md](CLEANUP.md)。2026-11-05 的历史对照与原始生产材料保留。

- 供应商：Cline 当前配置，`https://agi.tongji.edu.cn/v1`。
- 回答及裁判：`DeepSeek-V4.1-Flash`；大模型评分属于同模型自评，不是独立模型验收。
- 矩阵：40 题 × 全文/RAG/纯 Agent/RAG + Agent × tight/standard 两档，共 320 条；每组每题一次。
- 运行：`backend/data/agent_eval/2026-11-10-tongji-main/`。
- 裁判：答题结束后使用 `backend/data/agent_eval/2026-11-10-tongji-judge/`，单独评分，输出上限 8,192 token。
- 离线验证：评测相关 38 项测试通过（历史 11 项、新增 27 项）。

答题阶段已完成并通过完整性校验：320 条记录中，272 条预算内有效回答、43 条超预算、5 条 HTTP 429；这些失败均保留在分母中。正式 8,192-token 裁判已完成 272 次实际评分，所有 320 条判定已通过完整性校验，未知评分为 0。预算、评分及复现说明见 [协议](../../ARCHITECTURE_EVALUATION.md)。

运行初期并发答题/评分出现 429，部分裁判的 2,048-token 输出额度被推理耗尽，因此改为先答题后评分，正式裁判统一使用 8,192-token 输出额度。早期裁判已移出工作区，恢复位置见 [CLEANUP.md](CLEANUP.md)，不与正式评分混合。已发生的答题失败不补跑替换。

最小接口验证见 [tongji-probe.json](tongji-probe.json)，仅包含合成测试数据，不混入正式 320 条记录。

## 最终交付

- [Standard 中文完整报告](STANDARD_EVALUATION_REPORT.md)：仅标准预算，含结果、答题/评审消耗、方法与 40 题附录。
- [完整报告](REPORT.md)
- [结果解读与评分分歧](FINDINGS.md)
- [逐题答案、工具调用、usage 与判分](answers.jsonl)
- [汇总与输入/源码哈希](summary.json)

正式回答和正式裁判均完成；保留 5 条限流失败，未替换。旧失败运行和 2,048-token 诊断的清理位置见 [CLEANUP.md](CLEANUP.md)。

- [40 道题完整题干、参考答案与评分口径](QUESTIONS_AND_SCORING.md)
