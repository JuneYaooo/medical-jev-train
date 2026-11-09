"""CPU tests for scripts/jevtrain with a tiny random Qwen2 model (no medical content).

Model-dependent tests are skipped when torch/transformers are unavailable.
"""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from jevtrain.data import (isolate, option_keys, read_rows, stratified_split, target_distribution, view,  # noqa: E402
                           view_offsets, views)

import importlib.util

HAVE_TORCH = all(importlib.util.find_spec(m) for m in ('torch', 'transformers', 'safetensors', 'tokenizers'))

COLORS = {'甲': 'a', '乙': 'b', '丙': 'c', '丁': 'd'}


def row(uid, word, task='color', group=None, gold=None):
    return {'id': uid, 'group_id': group or 'g:' + uid, 'task': task, 'gold': gold or COLORS[word],
            'request': {'state': {'clinical_question': '标记' + word + uid}, 'questions': {'decision': {
                'type': 'choice', 'instructions': '选择对应项',
                'criteria': {'a': '红色', 'b': '绿色', 'c': '蓝色', 'd': '黄色'}}}}}


def fixed_order(rows):
    # Position-dependent text keeps the displayed order fixed, so a tiny random
    # model can fit the labels; option-order robustness is not under test here.
    for r in rows:
        r['request']['questions']['decision']['criteria']['d'] = '以上均不对'
    return rows


def dataset(n, task='color', prefix='r'):
    words = list(COLORS)
    return [row(f'{prefix}{task}{i:03d}', words[i % 4], task) for i in range(n)]


class DataTests(unittest.TestCase):
    def test_views_keep_semantics(self):
        r = row('x1', '乙')
        self.assertEqual(view_offsets(r, 4), [0, 1, 2, 3])
        shown = views(r, 4)
        self.assertEqual({tuple(option_keys(v)) for v in shown}.__len__(), 4)
        for v in shown:
            self.assertEqual(v['gold'], 'b')
            self.assertEqual(v['request']['questions']['decision']['criteria']['b'], '绿色')
        self.assertEqual(r['request']['questions']['decision']['criteria'], shown[0]['request']['questions']['decision']['criteria'])

    def test_position_dependent_rows_are_not_permuted(self):
        r = row('x1', '甲')
        r['request']['questions']['decision']['criteria']['d'] = '以上都是'
        self.assertEqual(view_offsets(r, 4), [0])

    def test_target_smoothing(self):
        r = row('x1', '甲')
        keys = option_keys(view(r, 1))
        hard = target_distribution(r, keys, {'label_smoothing': 0.0})
        self.assertEqual(hard[keys.index('a')], 1.0)
        smooth = target_distribution(r, keys, {'label_smoothing': 0.2})
        self.assertAlmostEqual(smooth[keys.index('a')], 0.85)
        self.assertAlmostEqual(sum(smooth), 1.0)
        hard_task = target_distribution(r, keys, {'label_smoothing': 0.2, 'hard_tasks': ['color']})
        self.assertEqual(hard_task[keys.index('a')], 1.0)

    def test_stratified_split_isolates_groups(self):
        rows = dataset(40) + dataset(40, 'shape')
        rows[1]['group_id'] = rows[0]['group_id']
        sets, pool, report = stratified_split(rows, 5, ('development', 'test'), max_fraction=0.3)
        seen = {}
        for name, values in [*sets.items(), ('pool', pool)]:
            for r in values:
                self.assertNotIn(r['group_id'], {g for n, gs in seen.items() if n != name for g in gs})
            seen[name] = {r['group_id'] for r in values}
        self.assertEqual(sum(len(v) for v in sets.values()) + len(pool), 80)
        self.assertLessEqual(report['by_task']['color']['development'], 6)
        kept, dropped = isolate(rows, [sets['development']])
        self.assertEqual(dropped, len(sets['development']))

    def test_per_task_cap(self):
        from jevtrain.data import write_rows
        from jevtrain.train import load_training_rows
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'pool.jsonl'
            write_rows(path, dataset(30) + dataset(5, 'shape') + dataset(40, 'big'))
            rows = load_training_rows([{'path': str(path), 'exclude_tasks': ['big'], 'per_task_max': 10},
                                       {'path': str(path), 'tasks': ['big'], 'max_rows': 7}])
            counts = {t: sum(r['task'] == t for r in rows) for t in ('color', 'shape', 'big')}
            self.assertEqual(counts, {'color': 10, 'shape': 5, 'big': 7})


def build_tiny_model(directory):
    from tokenizers import Tokenizer, Regex, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM
    from jevtrain.prompt import SYMBOLS, SYSTEM
    chars = set(SYSTEM + '任务材料选项标记甲乙丙丁红绿蓝黄色选择对应项形状' + ''.join(chr(i) for i in range(32, 127)))
    chars |= set(SYMBOLS) | {'\n'}
    vocab = {'[UNK]': 0, '[PAD]': 1, '[EOS]': 2}
    for c in sorted(chars):
        vocab.setdefault(c, len(vocab))
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token='[UNK]'))
    tok.pre_tokenizer = pre_tokenizers.Split(Regex('.|\n'), behavior='isolated')
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token='[UNK]', pad_token='[PAD]', eos_token='[EOS]')
    fast.chat_template = ("{% for m in messages %}<{{ m.role }}>{{ m.content }}\n{% endfor %}"
                          "{% if add_generation_prompt %}<assistant>{% endif %}")
    fast.save_pretrained(directory)
    config = Qwen2Config(vocab_size=len(vocab), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=1024,
                         tie_word_embeddings=False, pad_token_id=1, eos_token_id=2, initializer_range=0.3)
    import torch
    torch.manual_seed(0)
    Qwen2ForCausalLM(config).save_pretrained(directory, safe_serialization=True)


@unittest.skipUnless(HAVE_TORCH, 'torch and transformers are required')
class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import PreTrainedTokenizerFast
        cls.tmp = tempfile.TemporaryDirectory()
        cls.model_dir = Path(cls.tmp.name) / 'tiny'
        build_tiny_model(cls.model_dir)
        cls.spec = {'model_path': str(cls.model_dir), 'device': 'cpu', 'dtype': 'fp32', 'attn_implementation': 'eager',
                    'max_tokens': 512, 'max_options': 10, 'lora': {'layers': 'all', 'rank': 4, 'alpha': 8}}
        # Transformers 5 AutoTokenizer selects Qwen2's byte-level tokenizer from
        # the backbone config, replacing this fixture's character tokenizer and
        # dropping Chinese characters absent from its byte vocabulary. Load the
        # saved fixture backend explicitly; real model tokenizers use GPU verify.
        cls.tokenizer_loader = patch('transformers.AutoTokenizer.from_pretrained',
                                     side_effect=PreTrainedTokenizerFast.from_pretrained)
        cls.tokenizer_loader.start()

    @classmethod
    def tearDownClass(cls):
        cls.tokenizer_loader.stop()
        cls.tmp.cleanup()

    def model(self):
        from jevtrain.model import ChoiceModel
        return ChoiceModel(self.spec, seed=3)

    def test_supported_sample_matches_model_limits(self):
        import contextlib
        import io
        from jev import main
        from jevtrain.data import write_rows
        root = Path(self.tmp.name) / 'sample'
        valid, wide, long = dataset(3)
        wide['request']['questions']['decision']['criteria'] = {str(i): '选项' for i in range(30)}
        long['request']['state']['clinical_question'] += '标记' * 2000
        write_rows(root / 'input.jsonl', [valid, wide, long])
        config = {'model': self.spec, 'data': {'train': [{'path': str(root / 'input.jsonl')}]}}
        (root / 'config.json').write_text(json.dumps(config))
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            main(['sample', '--config', str(root / 'config.json'), '--output', str(root / 'output.jsonl'),
                  '--supported-only'])
        self.assertEqual([r['id'] for r in read_rows(root / 'output.jsonl')], [valid['id']])
        self.assertEqual(json.loads(output.getvalue())['rejected'], {'options': 1, 'length': 1})

    def test_padding_readout_and_save_load(self):
        import torch
        model = self.model()
        rows = dataset(6)
        rows[0]['request']['state']['clinical_question'] += '很长' * 20
        single = [model.probabilities([r], batch_size=1)[0] for r in rows]
        batched = model.probabilities(rows, batch_size=6)
        diff = max(abs(a[k] - b[k]) for a, b in zip(single, batched) for k in a)
        self.assertLess(diff, 1e-5)
        model.set_adapter_enabled(False)
        plain = model.probabilities(rows)
        model.set_adapter_enabled(True)
        self.assertLess(max(abs(a[k] - b[k]) for a, b in zip(batched, plain) for k in a), 1e-7)
        with torch.no_grad():
            for p in model.trainable():
                p.normal_(0, 0.1)
        changed = model.probabilities(rows)
        path = Path(self.tmp.name) / 'adapter.safetensors'
        model.save(path)
        from jevtrain.model import ChoiceModel
        again = ChoiceModel(self.spec, checkpoint=path).probabilities(rows)
        self.assertLess(max(abs(a[k] - b[k]) for a, b in zip(changed, again) for k in a), 1e-6)

    def test_training_without_development_keeps_last_weights(self):
        from jevtrain.train import run_training
        rows = fixed_order(dataset(16))
        for i, r in enumerate(rows):
            r['task'] = 'guarded'
        config = {'model': self.spec, 'seed': 5, 'eval_every': 10**6, 'save_every': 10**6, 'log_every': 1000,
                  'objective': {'label_smoothing': 0.0},
                  'optim': {'lr': 1e-3, 'epochs': 1, 'batch_size': 8, 'grad_accum': 1, 'scheduler': 'constant'}}
        # verify's tiny fit trains without a development set: nothing to select, only the last weights.
        out = Path(self.tmp.name) / 'nodev'
        report, _ = run_training(config, out, new_rows=rows, development=[], model=self.model(), log=lambda *_: None)
        self.assertTrue((out / 'last.safetensors').exists())
        self.assertFalse((out / 'best.safetensors').exists())
        self.assertIsNone(report['best'])

    def test_training_fits_and_resumes(self):
        import torch
        from jevtrain.evaluate import predict, summary
        from jevtrain.train import run_training
        rows = fixed_order(dataset(16))
        config = {'model': self.spec, 'seed': 5, 'eval_every': 40, 'save_every': 10, 'log_every': 1000,
                  'objective': {'label_smoothing': 0.0},
                  'optim': {'lr': 3e-3, 'epochs': 25, 'batch_size': 8, 'grad_accum': 1, 'warmup_ratio': 0.05,
                            'scheduler': 'constant'}}
        out = Path(self.tmp.name) / 'fit'
        model = self.model()
        before = summary(predict(model, rows)[0])['accuracy']
        report, model = run_training(config, out, new_rows=rows, development=rows,
                                     model=model, log=lambda *_: None)
        model.load(report['selected'])
        after = summary(predict(model, rows)[0])
        self.assertGreater(after['accuracy'], max(before, 0.6))
        losses = [h['loss'] for h in read_rows(out / 'history.jsonl') if 'loss' in h]
        self.assertLess(sum(losses[-5:]) / 5, sum(losses[:5]) / 5)

        # An interrupted run resumes to exactly the uninterrupted parameters.
        short = {**config, 'optim': {**config['optim'], 'epochs': 6, 'grad_accum': 2}, 'eval_every': 10**6,
                 'save_every': 1, 'log_every': 1}
        from jevtrain.model import ChoiceModel
        def dropout_model():
            return ChoiceModel({**self.spec, 'lora': {**self.spec['lora'], 'dropout': 0.1}}, seed=3)
        _, reference = run_training(short, Path(self.tmp.name) / 'whole', new_rows=rows,
                                    development=None, model=dropout_model(), log=lambda *_: None)
        expected = reference.state()

        class Stop(Exception):
            pass

        def interrupt(message):
            if json.loads(message).get('step') == 3:
                raise Stop
        target = Path(self.tmp.name) / 'resumed'
        with self.assertRaises(Stop):
            run_training(short, target, new_rows=rows, development=None,
                         model=dropout_model(), log=interrupt)
        self.assertTrue((target / 'progress.pt').exists())
        changed = [{**rows[0], 'gold': 'b'}, *rows[1:]]
        with self.assertRaisesRegex(ValueError, 'training inputs changed'):
            run_training(short, target, new_rows=changed, development=None,
                         model=dropout_model(), log=lambda *_: None)
        _, resumed = run_training(short, target, new_rows=rows, development=None,
                                  model=dropout_model(), log=lambda *_: None)
        for key, value in resumed.state().items():
            self.assertTrue(torch.allclose(value, expected[key], atol=1e-6), key)


if __name__ == '__main__':
    unittest.main()
