"""Offline protocol, budget and blind-judging tests; no optional chat dependencies."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'


with patch.object(sys, 'path', [str(SCRIPTS)] + sys.path):
    import evaluate_architectures as ev
    import judge_architectures as judge
    import report_architectures as report


class ArchitectureEvaluationTests(unittest.TestCase):
    def test_cline_config_is_read_without_copying_credentials_to_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ev.write_json(root / 'globalState.json', dict(actModeApiProvider='openai',
                          openAiBaseUrl='https://agi.tongji.edu.cn/v1', actModeOpenAiModelId='DeepSeek-V4.1-Flash'))
            args = NS(provider_source='cline', cline_dir=root, model=None)
            config, model, identity = ev.resolve_provider(args, read_secret=False)
            self.assertIsNone(config['api_key'])
            self.assertEqual((model, identity), ('DeepSeek-V4.1-Flash', 'cline:agi.tongji.edu.cn'))
            ev.write_json(root / 'secrets.json', {'openAiApiKey': 'test-credential-only'})
            config, _, _ = ev.resolve_provider(args)
            self.assertEqual(config['api_key'], 'test-credential-only')
            self.assertEqual(config['base_url'], 'https://agi.tongji.edu.cn/v1')
            (root / 'secrets.json').unlink()
            with self.assertRaises(FileNotFoundError):
                ev.resolve_provider(args)

    def test_same_model_judge_requires_explicit_labeling(self):
        metadata = {'model_requested': 'same'}
        with self.assertRaisesRegex(ValueError, 'allow-self-judge'):
            judge.judge_relationship(metadata, 'same', False)
        self.assertEqual(judge.judge_relationship(metadata, 'same', True), 'same_model_self_evaluation')
        self.assertEqual(judge.judge_relationship(metadata, 'different', False), 'different_model')

    def setUp(self):
        self.case = dict(id='q', video='v', category='fact', question='用了哪些模式？',
                         reference_answer='策略模式和工厂模式。', answer_groups=[['策略'], ['工厂']],
                         evidence_groups=[['策略模式和工厂模式']])
        self.video = dict(id='v', task_id='task')
        self.note = dict(markdown='策略模式和工厂模式', transcript={'segments': []},
                         audio_meta={'title': '测试视频'})
        self.service = NS(VectorStoreManager=Mock(), _build_context=lambda c: c[0]['text'],
                          TOOLS=[{'type': 'function', 'function': {'name': 'lookup_transcript'}}],
                          execute_tool=Mock(return_value='{"text":"策略模式和工厂模式"}'))
        self.service.VectorStoreManager.return_value.query.return_value = [dict(text='策略模式和工厂模式', metadata={})]
        self.limits = dict(ev.BUDGETS['tight'])

    def response(self, answer='策略模式和工厂模式。', calls=None, usage=True, finish=None, prompt=100, output=20):
        return NS(model='actual-model', usage=NS(prompt_tokens=prompt, completion_tokens=output,
                                                total_tokens=prompt + output) if usage else None,
                  choices=[NS(finish_reason=finish or ('tool_calls' if calls else 'stop'),
                              message=NS(content=answer, tool_calls=calls))])

    def call(self, name='lookup_transcript', arguments='{}', ident='call'):
        return NS(id=ident, function=NS(name=name, arguments=arguments))

    def run_mode(self, mode, responses, **kwargs):
        create = Mock(side_effect=responses)
        row = ev.run_answer(self.case, self.video, mode, 0, 'tight', self.limits,
                            self.note, self.service, create, 'answer-model', **kwargs)
        return row, create

    def test_four_arms_control_initial_evidence_and_tool_access(self):
        for mode in ev.MODES:
            self.service.VectorStoreManager.reset_mock()
            row, create = self.run_mode(mode, [self.response()])
            self.assertTrue(row['score']['passed'])
            request = create.call_args.kwargs
            self.assertEqual('tools' in request, mode in ('agent_only', 'rag_agent'))
            self.assertEqual(self.service.VectorStoreManager.called, mode in ('rag', 'rag_agent'))
            context = request['messages'][0]['content']
            self.assertEqual('测试视频' in context, mode == 'full_context')
            if mode == 'agent_only':
                self.assertNotIn('策略模式和工厂模式', context)

    def test_full_context_is_not_silently_truncated(self):
        self.note['markdown'] = 'x' * 100000 + '结尾证据'
        row, create = self.run_mode('full_context', [self.response()])
        self.assertIn('结尾证据', create.call_args.kwargs['messages'][0]['content'])
        self.assertIn('结尾证据', row['trace'][0]['chunks'][0]['text'])

    def test_pure_agent_retrieves_real_tool_evidence_without_initial_query(self):
        row, _ = self.run_mode('agent_only', [self.response(calls=[self.call()]), self.response()])
        self.service.VectorStoreManager.assert_not_called()
        self.service.execute_tool.assert_called_once_with('task', 'lookup_transcript', {})
        self.assertTrue(row['score']['evidence_available'])
        self.assertEqual(row['accounting']['known_total_tokens'], 240)

    def test_last_request_closes_tools_and_respects_round_budget(self):
        row, create = self.run_mode('rag_agent', [self.response(calls=[self.call()])] * 3 + [self.response()])
        self.assertEqual(self.service.execute_tool.call_count, 3)
        self.assertEqual(create.call_count, 4)
        self.assertEqual(create.call_args.kwargs['tool_choice'], 'none')
        self.assertTrue(row['score']['passed'])

    def test_many_calls_in_one_round_do_not_bypass_tool_allowance(self):
        self.limits['tool_calls'] = 2
        row, create = self.run_mode('agent_only', [self.response(calls=[self.call(ident=str(i)) for i in range(5)]), self.response()])
        self.assertEqual(self.service.execute_tool.call_count, 2)
        self.assertEqual(row['accounting']['tool_calls'], 2)
        self.assertEqual(sum(e['kind'] == 'tool_denied' for e in row['trace']), 3)
        self.assertEqual(create.call_args.kwargs['tool_choice'], 'none')

    def test_cumulative_output_budget_reduces_subsequent_max_tokens(self):
        self.limits.update(output_tokens=30, per_call_tokens=25)
        row, create = self.run_mode('agent_only', [self.response(calls=[self.call()], output=20), self.response(output=10)])
        self.assertEqual([c.kwargs['max_tokens'] for c in create.call_args_list], [25, 10])
        self.assertTrue(row['budget_valid'])

    def test_actual_total_token_overrun_is_failure_even_with_correct_text(self):
        self.limits['total_tokens'] = 110
        row, create = self.run_mode('rag', [self.response()])
        self.assertEqual(row['error']['type'], 'BudgetExceeded')
        self.assertEqual(row['discarded_answer'], '策略模式和工厂模式。')
        self.assertFalse(row['budget_valid'])
        self.assertFalse(row['score']['passed'])
        self.assertEqual(create.call_count, 1)

    def test_missing_usage_is_uncertifiable_and_stops_agent(self):
        row, create = self.run_mode('agent_only', [self.response(calls=[self.call()], usage=False)])
        self.assertEqual(row['error']['type'], 'UsageUnavailable')
        self.assertFalse(row['budget_valid'])
        self.service.execute_tool.assert_not_called()
        self.assertEqual(create.call_count, 1)

    def test_invalid_usage_cannot_create_extra_budget(self):
        for usage in [NS(prompt_tokens=-1, completion_tokens=5, total_tokens=4),
                      NS(prompt_tokens=100, completion_tokens=20, total_tokens=10)]:
            with self.subTest(usage=usage), self.assertRaises(ev.UsageUnavailable):
                ev.usage_counts(usage)

    def test_late_response_fails_wall_time_budget(self):
        clock = Mock(side_effect=[0, 0, 0, 0, 91, 91, 91])
        row, _ = self.run_mode('agent_only', [self.response()], clock=clock)
        self.assertEqual(row['error']['type'], 'BudgetExceeded')
        self.assertFalse(row['budget_valid'])

    def test_bad_tool_arguments_are_returned_as_error_then_answer_can_recover(self):
        for arguments in ['[]', 'not-json']:
            self.service.execute_tool.reset_mock()
            row, create = self.run_mode('agent_only', [self.response(calls=[self.call(arguments=arguments)]), self.response()])
            self.service.execute_tool.assert_not_called()
            self.assertFalse(next(e for e in row['trace'] if e['kind'] == 'tool')['success'])
            self.assertTrue(row['score']['passed'])

    def test_empty_truncated_and_final_tool_responses_are_failures(self):
        for response in [self.response(answer=' '), self.response(finish='length')]:
            row, _ = self.run_mode('rag', [response])
            self.assertFalse(row['score']['passed'])
        self.limits['model_calls'] = 1
        row, _ = self.run_mode('agent_only', [self.response(calls=[self.call()])])
        self.assertEqual(row['error']['type'], 'InvalidResponse')

    def test_provider_exception_has_no_retries_or_sensitive_message(self):
        row, create = self.run_mode('rag', [RuntimeError('PRIVATE PROVIDER RESPONSE')])
        self.assertEqual(create.call_count, 1)
        self.assertNotIn('PRIVATE', json.dumps(row))
        self.assertFalse(row['accounting']['usage_complete'])

    def test_quota_machine_code_is_retained_without_provider_body(self):
        exc = RuntimeError('PRIVATE RESPONSE')
        exc.status_code = 403
        exc.body = {'error': {'code': 'insufficient_quota', 'message': 'PRIVATE BODY'}}
        row, _ = self.run_mode('rag', [exc])
        self.assertTrue(ev.quota_exhausted(row))
        self.assertNotIn('PRIVATE', json.dumps(row))

    def dataset_and_rows(self):
        rows = [self.run_mode(mode, [self.response()])[0] for mode in ev.MODES]
        dataset = dict(version='test', videos=[self.video], cases=[self.case])
        metadata = dict(case_ids=['q'], repeats=1, modes=list(ev.MODES), budgets={'tight': self.limits},
                        model_requested='test-model', diagnostic=True)
        return dataset, metadata, rows

    def test_report_rejects_missing_duplicate_or_changed_pairs(self):
        dataset, metadata, rows = self.dataset_and_rows()
        ev.validate_records(metadata, dataset, rows)
        for changed in [rows[:-1], rows + [rows[0]]]:
            with self.assertRaisesRegex(ValueError, 'Incomplete or duplicate'):
                ev.validate_records(metadata, dataset, changed)
        changed = copy.deepcopy(rows)
        changed[0]['limits']['total_tokens'] += 1
        with self.assertRaisesRegex(ValueError, 'budget changed'):
            ev.validate_records(metadata, dataset, changed)

    def test_report_rejects_mismatched_initial_retrieval(self):
        dataset, metadata, rows = self.dataset_and_rows()
        row = next(r for r in rows if r['mode'] == 'rag_agent')
        row['trace'][0]['chunks'] = [{'text': '不同证据', 'metadata': {}}]
        row['score'] = ev.score(self.case, row)
        with self.assertRaisesRegex(ValueError, 'different initial evidence'):
            ev.validate_records(metadata, dataset, rows)

    def test_report_recomputes_budget_compliance_instead_of_trusting_flag(self):
        dataset, metadata, rows = self.dataset_and_rows()
        rows[0]['accounting']['known_total_tokens'] = 0
        with self.assertRaisesRegex(ValueError, 'accounting does not match trace'):
            ev.validate_records(metadata, dataset, rows)
        dataset, metadata, rows = self.dataset_and_rows()
        rows[0]['budget_valid'] = False
        with self.assertRaisesRegex(ValueError, 'compliance does not match trace'):
            ev.validate_records(metadata, dataset, rows)

    def test_frozen_run_rejects_dataset_and_source_tampering(self):
        dataset, metadata, rows = self.dataset_and_rows()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'notes').mkdir()
            ev.write_json(root / 'notes/task.json', self.note)
            manifest = [dict(video='v', task_id='task', snapshot_sha256=ev.digest(root / 'notes/task.json'))]
            ev.write_json(root / 'dataset.json', dataset)
            ev.write_json(root / 'corpus_manifest.json', manifest)
            metadata.update(dataset_sha256=ev.digest(root / 'dataset.json'),
                            manifest_sha256=ev.digest(root / 'corpus_manifest.json'))
            ev.write_json(root / 'metadata.json', metadata)
            (root / 'results.jsonl').write_text('\n'.join(ev.json_text(r) for r in rows) + '\n')
            ev.load_run(root)
            original = (root / 'notes/task.json').read_text()
            (root / 'notes/task.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'corpus hash changed'):
                ev.load_run(root)
            (root / 'notes/task.json').write_text(original)
            (root / 'dataset.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'dataset hash changed'):
                ev.load_run(root)

    def test_following_judge_finishes_complete_matrix_without_duplicate_rows(self):
        dataset, metadata, rows = self.dataset_and_rows()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'notes').mkdir()
            ev.write_json(root / 'notes/task.json', self.note)
            ev.write_json(root / 'dataset.json', dataset)
            ev.write_json(root / 'corpus_manifest.json', [dict(video='v', task_id='task',
                          snapshot_sha256=ev.digest(root / 'notes/task.json'))])
            metadata.update(dataset_sha256=ev.digest(root / 'dataset.json'),
                            manifest_sha256=ev.digest(root / 'corpus_manifest.json'))
            ev.write_json(root / 'metadata.json', metadata)
            (root / 'results.jsonl').write_text('\n'.join(ev.json_text(r) for r in rows) + '\n')
            def execute(row):
                return judge.judge_one(row, self.case, self.note,
                                      Mock(return_value=self.good_judge_response()), 'judge')
            with patch('builtins.print'):
                judge.follow_answers(root, root / 'judgments.jsonl', execute, 2, 5, metadata)
            judgments = [json.loads(l) for l in (root / 'judgments.jsonl').read_text().splitlines()]
            judge.validate_judgments(rows, judgments)
            self.assertEqual(len(judgments), 4)
            stopped = threading.Event()
            create = Mock()
            exc = RuntimeError('quota exhausted')
            exc.status_code = 403
            exc.body = {'code': 'insufficient_quota'}
            create.side_effect = exc
            def failing_execute(row):
                if stopped.is_set():
                    return None
                result = judge.judge_one(row, self.case, self.note, create, 'judge')
                if ev.quota_exhausted(result):
                    stopped.set()
                return result
            with patch('builtins.print'):
                finished = judge.follow_answers(root, root / 'stopped.jsonl', failing_execute, 1, 5, metadata)
            self.assertFalse(finished)
            self.assertEqual(create.call_count, 1)

    def good_judge_response(self, **changes):
        value = dict(correctness=4, completeness=4, grounding=4,
                     insufficient_evidence=False, critical_errors=[], reason='材料支持这两个模式。')
        value.update(changes)
        response = self.response(answer=json.dumps(value, ensure_ascii=False))
        response.model = 'independent-judge'
        return response

    def test_judge_payload_is_blind_and_contains_complete_shared_material(self):
        row, _ = self.run_mode('rag_agent', [self.response()])
        create = Mock(return_value=self.good_judge_response())
        result = judge.judge_one(row, self.case, self.note, create, 'judge')
        self.assertTrue(result['passed'])
        payload = json.loads(create.call_args.kwargs['messages'][1]['content'])
        self.assertEqual(payload['full_source_material'], self.note)
        self.assertEqual(payload['reference_answer'], self.case['reference_answer'])
        for key in ('mode', 'budget_id', 'score', 'trace', 'elapsed_seconds'):
            self.assertNotIn(key, payload)

    def test_judge_rejects_invalid_schema_and_derives_pass_from_dimensions(self):
        row, _ = self.run_mode('rag', [self.response()])
        for response in [self.response(answer='not-json'), self.good_judge_response(correctness=True),
                         self.good_judge_response(extra='field')]:
            result = judge.judge_one(row, self.case, self.note, Mock(return_value=response), 'judge')
            self.assertEqual(result['status'], 'judge_error')
            self.assertIsNone(result['passed'])
        result = judge.judge_one(row, self.case, self.note, Mock(return_value=self.good_judge_response(correctness=3)), 'judge')
        self.assertFalse(result['passed'])

    def test_invalid_answer_and_uncertain_judgment_are_distinct(self):
        row, _ = self.run_mode('rag', [self.response()])
        uncertain = judge.judge_one(row, self.case, self.note, Mock(return_value=self.good_judge_response(insufficient_evidence=True)), 'judge')
        self.assertIsNone(uncertain['passed'])
        self.assertEqual(uncertain['status'], 'insufficient_evidence')
        row['error'] = {'type': 'BudgetExceeded'}
        create = Mock()
        invalid = judge.judge_one(row, self.case, self.note, create, 'judge')
        create.assert_not_called()
        self.assertIs(invalid['passed'], False)

    def test_report_preserves_unknown_judge_coverage_and_separate_usage(self):
        dataset, metadata, rows = self.dataset_and_rows()
        js = [judge.judge_one(r, self.case, self.note, Mock(return_value=self.good_judge_response()), 'judge') for r in rows]
        js[0] = judge.judge_one(rows[0], self.case, self.note, Mock(side_effect=TimeoutError()), 'judge')
        summary = report.build_summary(metadata, dataset, rows, js)
        full = summary['budgets']['tight']['full_context']
        self.assertEqual(full['judge']['unknown'], 1)
        self.assertEqual(full['judge']['pass_rate_lower_bound'], 0)
        self.assertEqual(full['judge']['pass_rate_upper_bound'], 1)
        self.assertEqual(full['total_tokens'], 120)
        self.assertEqual(summary['judge_accounting']['total_tokens_known'], 360)

    def test_judgment_cannot_be_reused_for_modified_answer(self):
        row, _ = self.run_mode('rag', [self.response()])
        result = judge.judge_one(row, self.case, self.note, Mock(return_value=self.good_judge_response()), 'judge')
        row['answer'] += '不同答案'
        with self.assertRaisesRegex(ValueError, 'does not match answer'):
            judge.validate_judgments([row], [result])

    def test_partial_judge_reporting_never_imputes_success(self):
        dataset, metadata, rows = self.dataset_and_rows()
        one = judge.judge_one(rows[0], self.case, self.note, Mock(return_value=self.good_judge_response()), 'judge')
        judgments = judge.complete_partial_judgments(rows, [one])
        self.assertEqual(sum(j['passed'] is True for j in judgments), 1)
        self.assertEqual(sum(j['passed'] is None for j in judgments), 3)
        summary = report.build_summary(metadata, dataset, rows, judgments)
        self.assertEqual(summary['budgets']['tight']['rag']['judge']['unknown'], 1)


if __name__ == '__main__':
    unittest.main()
