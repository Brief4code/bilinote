"""Validate and report paired four-architecture runs with separate judge costs."""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics

from evaluate_architectures import digest, load_run, record_key, write_json
from evaluate_agent import summarize
from judge_architectures import (DIMENSIONS, complete_partial_judgments,
                                 load_judge_metadata, validate_judgments)

LABELS = {'full_context': '全文 + LLM', 'rag': 'RAG', 'agent_only': '纯 Agent', 'rag_agent': 'RAG + Agent'}
CATEGORY_LABELS = {'metadata': '元信息', 'summary': '概括', 'fact': '事实', 'quote': '原文',
                   'timestamp': '时间定位', 'cross_segment': '跨片段', 'unanswerable': '无答案'}


def build_summary(metadata, dataset, rows, judgments=None):
    from evaluate_architectures import validate_records
    validate_records(metadata, dataset, rows)
    if judgments is not None:
        validate_judgments(rows, judgments)
    judge_index = {record_key(j): j for j in judgments or []}
    result = dict(metadata=metadata, budgets={}, paired_rule_changes={},
                  judge_calibration='Not human-calibrated; model judgments are secondary, not ground truth')
    for budget in metadata['budgets']:
        selected = [r for r in rows if r['budget_id'] == budget]
        summary = summarize(selected)
        for mode, values in summary['modes'].items():
            subset = [r for r in selected if r['mode'] == mode]
            values['budget_valid'] = sum(r['budget_valid'] for r in subset)
            values['error_types'] = dict(Counter(r['error']['type'] for r in subset if r.get('error')))
            values['actual_models'] = sorted({e['model'] for r in subset for e in r['trace']
                                              if e['kind'] == 'completion' and e.get('model')})
            values['tokens_per_question_known'] = round(values['total_tokens'] / len(subset), 2)
            values['tokens_per_rule_pass_known'] = round(values['total_tokens'] / values['passed'], 2) if values['passed'] else None
            if judgments is not None:
                js = [judge_index[record_key(r)] for r in subset]
                scored = [j for j in js if j['status'] == 'judged']
                values['judge'] = dict(passed=sum(j['passed'] is True for j in js), total=len(js),
                    # Unknown judgments are not silently converted into content errors.
                    unknown=sum(j['passed'] is None for j in js), status=dict(Counter(j['status'] for j in js)),
                    known_outcomes=sum(j['passed'] is not None for j in js),
                    pass_rate_lower_bound=sum(j['passed'] is True for j in js) / len(js),
                    pass_rate_upper_bound=sum(j['passed'] is not False for j in js) / len(js),
                    rule_disagreements=[r['case_id'] for r, j in zip(subset, js)
                                        if j['passed'] is not None and r['score']['passed'] != j['passed']],
                    mean_scores={d: round(statistics.mean(j['judgment'][d] for j in scored), 3)
                                 if scored else None for d in DIMENSIONS})
                for category, category_values in values['categories'].items():
                    category_js = [judge_index[record_key(r)] for r in subset if r['category'] == category]
                    category_values.update(judge_passed=sum(j['passed'] is True for j in category_js),
                                           judge_unknown=sum(j['passed'] is None for j in category_js))
            values['videos'] = {}
            for video in sorted({r['video'] for r in subset}):
                vs = [r for r in subset if r['video'] == video]
                values['videos'][video] = dict(total=len(vs), rule_passed=sum(r['score']['passed'] for r in vs))
                if judgments is not None:
                    vs_j = [judge_index[record_key(r)] for r in vs]
                    values['videos'][video].update(judge_passed=sum(j['passed'] is True for j in vs_j),
                                                   judge_unknown=sum(j['passed'] is None for j in vs_j))
        result['budgets'][budget] = summary['modes']
        result['paired_rule_changes'][budget] = summary['paired']
    if judgments is not None:
        calls = [j['call'] for j in judgments if j.get('call')]
        result['judge_accounting'] = dict(requests=len(calls), usage_missing=sum(c.get('usage') is None for c in calls),
            total_tokens_known=sum((c.get('usage') or {}).get('total_tokens', 0) or 0 for c in calls),
            elapsed_seconds_sum=round(sum(c['elapsed_seconds'] for c in calls), 4),
            actual_models=sorted({c['model'] for c in calls if c.get('model')}))
    return result


def markdown_report(summary):
    meta = summary['metadata']
    provider_failures = sum(v['error_types'].get('PermissionDeniedError', 0)
                            for modes in summary['budgets'].values() for v in modes.values())
    lines = ['# 四组架构与统一预算评测', '']
    if provider_failures:
        lines += [f'**本轮受供应商访问/额度错误影响：{provider_failures} 条回答为 PermissionDeniedError。',
                  '以下通过数保留失败分母，用于审阅运行情况，不能据此对架构质量排名。**', '']
    if summary.get('partial_judge'):
        lines += ['**裁判运行未完成：保留已取得评分，未执行及裁判失败的评分标为未知。**', '']
    judge_meta = summary.get('judge_metadata') or {}
    if judge_meta.get('judge_relationship') == 'same_model_self_evaluation':
        lines += ['**大模型评分为同模型自评：回答与裁判均使用同一模型，不是独立模型验收。**', '']
    lines += [
             f"回答模型：`{meta['model_requested']}`；每组 {len(meta['case_ids'])} 题 × {meta['repeats']} 次。",
             f"端点：`{meta.get('api_base_url', meta.get('provider_id', '未记录'))}`；裁判：`{judge_meta.get('model_requested', '未运行')}`。",
             '诊断子集，不是完整 40 题结果。' if meta['diagnostic'] else '使用完整固定 40 题集；没有独立留出集。', '',
             '四组共用回答提示、冻结材料与评分规则。全文组不截断；纯 Agent 无初始检索；两个 RAG 组使用相同检索实现。',
             '这是统一编排的受控实验，Agent 组复用生产工具，但不是历史生产 chat() 配置的原样重跑。', '',
             '总 token 是供应商 usage 返回后的预算合规条件，包含每次重复输入与输出；超限计失败。',
             '输入 token 无法在提交前精确获知，最后一个请求可能造成实际开销超过额度；这里不承诺账单硬上限。',
             '累计输出、模型请求数、工具执行数在调用前限制；缺 usage、异常、截断和超时均保留在分母中。', '']
    for budget, modes in summary['budgets'].items():
        limit = meta['budgets'][budget]
        lines += [f'## {budget}', '',
                  f"每题额度：总 token {limit['total_tokens']:,}，累计输出 {limit['output_tokens']:,}，"
                  f"单次输出 {limit['per_call_tokens']:,}，{limit['seconds']} 秒，"
                  f"模型最多 {limit['model_calls']} 次，工具最多 {limit['tool_calls']} 次/{limit['tool_rounds']} 轮。", '',
                  '| 配置 | 规则通过 | 大模型通过（未知） | 预算可核验且合规 | 已知 token/题 | P50/P95 秒 | 工具调用 |',
                  '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
        for mode in meta['modes']:
            v = modes[mode]
            judge = v.get('judge')
            jtext = f"{judge['passed']}/{judge['total']}（{judge['unknown']}）" if judge else '未运行'
            lines.append(f"| {LABELS[mode]} | {v['passed']}/{v['total']} ({v['pass_rate']:.1%}) | {jtext} | "
                         f"{v['budget_valid']}/{v['total']} | {v['tokens_per_question_known']:,.2f} | "
                         f"{v['latency_p50_seconds']:.2f}/{v['latency_p95_seconds']:.2f} | {v['tool_calls']} |")
        lines += ['', '错误分类：' + '; '.join(f"{LABELS[m]}={v['error_types']}" for m, v in modes.items()), '']
        lines += ['### 分题型结果', '', '单元格为规则通过数 / 大模型通过数；未知评分另标。', '',
                  '| 类型 | 题数 | ' + ' | '.join(LABELS[m] for m in meta['modes']) + ' |',
                  '| --- | ---: | ' + ' | '.join('---:' for _ in meta['modes']) + ' |']
        categories = sorted({c for v in modes.values() for c in v['categories']})
        for category in categories:
            subsets = [modes[m]['categories'][category] for m in meta['modes']]
            cells = []
            for v in subsets:
                text = str(v['passed']) + ' / ' + str(v.get('judge_passed', '未评分'))
                if v.get('judge_unknown'):
                    text += f"（未知 {v['judge_unknown']}）"
                cells.append(text)
            lines.append('| ' + CATEGORY_LABELS.get(category, category) + ' | ' + str(subsets[0]['total']) + ' | ' + ' | '.join(cells) + ' |')
        lines.append('')
    lines += ['## 解释与审阅', '',
              '规则通过率与大模型评分分别报告。大模型核验参考答案和完整源材料，隐藏架构及规则结果；',
              '核心正确性必须 4/4，完整度和证据支持至少 3/4，且没有关键错误才通过。',
              '裁判异常或材料无法裁定标为未知，单列数量和通过率上下界，不当作模型内容错误。',
              '尚未进行人工裁判校准，不将大模型评分称为真实准确率；两类评分不一致的答案优先人工复核。',
              '参考材料仍来自笔记及 ASR/OCR，未逐字人工核验原视频；单次运行和小样本尾延迟不支持线上 SLA 或统计显著性结论。', '']
    if 'judge_accounting' in summary:
        a = summary['judge_accounting']
        lines += [f"裁判阶段实际请求 {a['requests']} 次，已知 token {a['total_tokens_known']:,}，缺 usage {a['usage_missing']} 次；"
                  '这些消耗不计入回答方案的预算或效率指标。', '']
    lines += ['逐题记录、规则/裁判分歧、分题型结果及响应中的实际模型名见同目录 JSON 产物。', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--judge-run', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-partial-judge', action='store_true', help='Explicitly report an incomplete judge run with unknowns')
    args = parser.parse_args()
    args.run, args.output = args.run.resolve(), args.output.resolve()
    metadata, dataset, rows = load_run(args.run)
    judgments = None
    judge_meta = None
    if args.judge_run:
        judge_meta = load_judge_metadata(args.judge_run, args.run, metadata, allow_partial=args.allow_partial_judge)
        judgments = [json.loads(line) for line in (args.judge_run / 'judgments.jsonl').read_text(encoding='utf-8').splitlines()]
        if judge_meta.get('complete') is False:
            judgments = complete_partial_judgments(rows, judgments)
    summary = build_summary(metadata, dataset, rows, judgments)
    summary.update(results_sha256=digest(args.run / 'results.jsonl'), judge_metadata=judge_meta,
                   partial_judge=bool(judge_meta and judge_meta.get('complete') is False))
    args.output.mkdir(parents=True, exist_ok=True)
    if any((args.output / name).exists() for name in ('summary.json', 'REPORT.md', 'answers.jsonl')):
        parser.error('Report artifacts already exist; use a fresh output directory')
    write_json(args.output / 'summary.json', summary)
    (args.output / 'REPORT.md').write_text(markdown_report(summary), encoding='utf-8')
    index = {record_key(j): j for j in judgments or []}
    with (args.output / 'answers.jsonl').open('x', encoding='utf-8') as output:
        for row in sorted(rows, key=record_key):
            compact = {k: v for k, v in row.items() if k != 'trace'}
            compact['completion_calls'] = [e for e in row['trace'] if e['kind'] == 'completion']
            compact['tool_calls'] = [{k: v for k, v in e.items() if k != 'result'}
                                      for e in row['trace'] if e['kind'] in ('tool', 'tool_denied')]
            compact['judge'] = index.get(record_key(row))
            output.write(json.dumps(compact, ensure_ascii=False) + '\n')
    print(json.dumps({'report': str(args.output / 'REPORT.md'), 'answers': len(rows)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
