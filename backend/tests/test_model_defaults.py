"""Providers that reject sampling overrides must work on the default path."""
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from app.gpt.universal_gpt import UniversalGPT
from app.gpt.provider.OpenAI_compatible_provider import OpenAICompatibleProvider
from app.models.gpt_model import GPTSource
from app.models.transcriber_model import TranscriptSegment


class ModelDefaultTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.env = patch.dict('os.environ', {'NOTE_OUTPUT_DIR': self.temp.name})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='笔记'))])
        self.create = Mock(return_value=self.response)
        self.client = SimpleNamespace(base_url='https://example.invalid', chat=SimpleNamespace(
            completions=SimpleNamespace(create=self.create)))
        self.messages = [{'role': 'user', 'content': '测试字幕'}]

    def test_default_request_omits_sampling_and_thinking_overrides(self):
        gpt = UniversalGPT(self.client, 'kimi-k3')
        self.assertIs(gpt._do_create_unlimited(self.messages), self.response)
        self.assertEqual(self.create.call_args.kwargs, {'model': 'kimi-k3', 'messages': self.messages})

    def test_explicit_zero_temperature_is_preserved(self):
        gpt = UniversalGPT(self.client, 'test-model', temperature=0)
        gpt._do_create_unlimited(self.messages)
        self.assertEqual(self.create.call_args.kwargs['temperature'], 0)

    def test_explicit_unsupported_temperature_is_removed_once(self):
        self.create.side_effect = [ValueError("Parameter 'temperature'=0.7 is not supported for kimi-k3 model."), self.response]
        gpt = UniversalGPT(self.client, 'kimi-k3', temperature=.7)
        self.assertIs(gpt._do_create_unlimited(self.messages), self.response)
        self.assertEqual(self.create.call_count, 2)
        self.assertEqual(self.create.call_args_list[0].kwargs['temperature'], .7)
        self.assertNotIn('temperature', self.create.call_args_list[1].kwargs)

    def test_unrelated_invalid_request_is_not_retried_without_temperature(self):
        self.create.side_effect = ValueError('invalid messages')
        gpt = UniversalGPT(self.client, 'kimi-k3')
        with self.assertRaisesRegex(ValueError, 'invalid messages'):
            gpt._do_create_unlimited(self.messages)
        self.assertEqual(self.create.call_count, 1)

    def test_provider_rejection_is_not_repeated_when_no_override_was_sent(self):
        self.create.side_effect = ValueError('temperature is not supported')
        gpt = UniversalGPT(self.client, 'kimi-k3')
        with self.assertRaisesRegex(ValueError, 'not supported'):
            gpt._do_create_unlimited(self.messages)
        self.assertEqual(self.create.call_count, 1)

    def test_sampling_default_change_invalidates_old_summary_cache_signature(self):
        source = GPTSource(title='视频', tags=[], segment=[TranscriptSegment(0, 1, '字幕')])
        old = UniversalGPT(self.client, 'test-model', temperature=.7)
        current = UniversalGPT(self.client, 'test-model')
        self.assertNotEqual(old._build_source_signature(source), current._build_source_signature(source))

    def test_connection_probe_uses_provider_sampling_defaults(self):
        with patch('app.gpt.provider.OpenAI_compatible_provider.build_openai_client', return_value=self.client):
            self.assertTrue(OpenAICompatibleProvider.test_connection('test-key', 'https://example.invalid', 'kimi-k3'))
        self.assertEqual(self.create.call_args.kwargs, {'model': 'kimi-k3',
            'messages': [{'role': 'user', 'content': 'ping'}], 'max_tokens': 1})


if __name__ == '__main__':
    unittest.main()
