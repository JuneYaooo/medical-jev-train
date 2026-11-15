"""CPU regressions for exact split isolation and the separate candidate audit."""
import copy
import json
import importlib.util
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from jevtrain import data
from jevtrain.metrics import input_hash


def row(uid, text=None, group=None, task='test'):
    return {'id': uid, 'group_id': group or 'g:' + uid, 'task': task, 'gold': 'A',
            'request': {'state': {'clinical_question': text or 'Question ' + uid},
                        'questions': {'decision': {'type': 'choice', 'instructions': 'Choose one.',
                                                   'criteria': {'A': 'first', 'B': 'second'}}}}}


def location(sets, pool):
    return {r['id']: name for name, values in [*sets.items(), ('pool', pool)] for r in values}


class ExactIsolationTests(unittest.TestCase):
    def test_different_groups_with_identical_request_never_cross_sets(self):
        rows = [row(str(i), 'Same question') for i in range(20)]
        sets, pool, report = data.stratified_split(rows, 3)
        self.assertEqual(len(set(location(sets, pool).values())), 1)
        self.assertEqual(sum(report['sizes'].values()), 20)

    def test_transitive_group_request_chain_and_exclusion(self):
        rows = [row('a', 'X', 'one'), row('b', 'X', 'two'),
                row('c', 'Y', 'two'), row('d', 'Y', 'three'), row('keep')]
        heldout = row('held', 'X', 'outside')
        g, i = data.exclusion_keys([heldout])
        sets, pool, report = data.stratified_split(rows, 1, exclude_groups=g, exclude_inputs=i)
        self.assertEqual(set(location(sets, pool)), {'keep'})
        self.assertEqual(report['dropped_by_exclusion'], 4)
        isolated, dropped = data.isolate(rows, [[heldout]])
        self.assertEqual(isolated, [rows[-1]])
        self.assertEqual(dropped, 4)
        # Matching a group instead of an input also removes the entire chain.
        isolated, dropped = data.isolate(rows, [[row('held2', 'Unrelated', 'three')]])
        self.assertEqual([r['id'] for r in isolated], ['keep'])
        self.assertEqual(dropped, 4)

    def test_mixed_task_component_stays_whole_and_reports_actual_tasks(self):
        rows = [row('a', 'same', task='alpha'), row('b', 'same', task='beta')]
        rows += [row('other' + str(i), task='alpha') for i in range(12)]
        sets, pool, report = data.stratified_split(rows, 3)
        where = location(sets, pool)
        self.assertEqual(where['a'], where['b'])
        for name, values in [*sets.items(), ('pool', pool)]:
            for task in ('alpha', 'beta'):
                self.assertEqual(report['by_task'][task][name], sum(r['task'] == task for r in values))
        self.assertEqual(report['components']['mixed_task'], 1)
        self.assertEqual(report['components']['total'], 13)

    def test_input_order_does_not_change_outputs_or_report(self):
        rows = [row(str(i), task='alpha' if i % 2 else 'beta') for i in range(40)]
        rows[5]['request'] = copy.deepcopy(rows[2]['request'])
        rows[7]['group_id'] = rows[5]['group_id']
        expected = data.stratified_split(rows, 3, seed=17)
        shuffled = copy.deepcopy(rows)
        random.Random(123).shuffle(shuffled)
        self.assertEqual(data.stratified_split(shuffled, 3, seed=17), expected)

    def test_different_content_and_option_order_do_not_gain_automatic_edges(self):
        original = row('base', 'Give 1.0 mg; do not repeat.')
        variants = []
        for uid, text in [('dose', 'Give 10 mg; do not repeat.'),
                          ('negation', 'Give 1.0 mg; repeat.'),
                          ('punctuation', 'Give 1.0 mg; do not repeat.:')]:
            variants.append(row(uid, text))
        instruction = copy.deepcopy(original)
        instruction.update(id='instructions', group_id='instruction-group')
        instruction['request']['questions']['decision']['instructions'] = 'Choose the incorrect answer.'
        order = copy.deepcopy(original)
        order.update(id='order', group_id='order-group')
        order['request']['questions']['decision']['criteria'] = {'B': 'second', 'A': 'first'}
        rows = [original, *variants, instruction, order]
        _, _, report = data.stratified_split(rows, 1)
        self.assertEqual(report['components']['total'], len(rows))
        self.assertNotEqual(input_hash(original['request']), input_hash(order['request']))

    def test_soft_target_small_tasks_and_legacy_positional_arguments(self):
        rows = [row(str(i), group='group' + str(i // 3)) for i in range(12)]
        sets, pool, report = data.stratified_split(rows, 1, ('development', 'test'), 17, 0.3, (), ())
        self.assertEqual([len(v) for v in sets.values()], [3, 3])
        self.assertEqual(len(pool), 6)
        tiny = [row('one'), row('two')]
        sets, pool, _ = data.stratified_split(tiny, 1)
        self.assertTrue(all(not v for v in sets.values()))
        self.assertEqual(len(pool), 2)

    def test_validate_split_rejects_group_input_and_exclusion_collisions(self):
        validate = data.validate_split
        a, b = row('a'), row('b')
        validate({'development': [a]}, [b])
        with self.assertRaisesRegex(ValueError, 'a.*b|b.*a'):
            validate({'development': [a]}, [row('b', group=a['group_id'])])
        with self.assertRaisesRegex(ValueError, 'a.*b|b.*a'):
            validate({'development': [a]}, [row('b', text='Question a')])
        with self.assertRaisesRegex(ValueError, 'a'):
            validate({'development': [a]}, [], exclude_groups=[a['group_id']])
        with self.assertRaisesRegex(ValueError, 'a'):
            validate({}, [a], exclude_inputs=[input_hash(a['request'])])

    def test_cli_checks_split_before_creating_any_output(self):
        from jevtrain.cli import main
        a, b = row('a', 'same'), row('b', 'same')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / 'source.jsonl', root / 'output'
            source.write_text(json.dumps(a) + '\n', encoding='utf-8')
            with patch('jevtrain.cli.stratified_split', return_value=({'development': [a]}, [b], {})):
                with self.assertRaisesRegex(ValueError, 'split isolation failed'):
                    main(['split', '--input', str(source), '--per-task', '1', '--output', str(output)])
            self.assertFalse(output.exists())


class CandidateAuditTests(unittest.TestCase):
    def audit_module(self):
        path = Path(__file__).resolve().parents[1] / 'scripts/audit_duplicates.py'
        spec = importlib.util.spec_from_file_location('audit_duplicates', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_audit_only_compares_eligible_option_and_answer_matches(self):
        audit = self.audit_module().audit_candidates
        train = row('train', 'A sufficiently long shared medical question asks for a treatment.')
        matching = copy.deepcopy(train)
        matching.update(id='evaluation', group_id='evaluation')
        matching['gold'] = 'B'
        matching['request']['questions']['decision']['criteria'] = {'B': 'first', 'A': 'second'}
        different_answer = copy.deepcopy(train)
        different_answer.update(id='other-answer', gold='B')
        different_options = copy.deepcopy(train)
        different_options['id'] = 'other-options'
        different_options['request']['questions']['decision']['criteria']['B'] = 'third'
        structured = copy.deepcopy(train)
        structured['id'] = 'structured'
        structured['request']['state'] = {'patient': 'This has no independent question field.'}
        report = audit([train], [matching, different_answer, different_options, structured])
        self.assertEqual(report['rows']['eligible_evaluation'], 3)
        self.assertEqual([c['evaluation_id'] for c in report['candidates']], ['evaluation'])
        self.assertTrue(report['candidates'][0]['option_order_or_keys_changed'])
        with self.assertRaisesRegex(ValueError, 'duplicate evaluation id'):
            audit([train], [matching, matching])
        with self.assertRaisesRegex(ValueError, 'duplicate training id'):
            audit([train, train], [matching])

    def test_audit_cli_emits_versioned_candidates_without_modifying_inputs(self):
        script = Path(__file__).resolve().parents[1] / 'scripts/audit_duplicates.py'
        train = [row('train', 'Patient received 1.0 mg. Which treatment should be chosen?')]
        evaluation = [row('evaluation', 'Patient received 10 mg. Which treatment should be chosen?')]
        # This is a deliberately unsafe semantic match. The tool may only flag it for review.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [root / 'train.jsonl', root / 'evaluation.jsonl']
            for path, values in zip(paths, (train, evaluation)):
                path.write_text(''.join(json.dumps(r) + '\n' for r in values), encoding='utf-8')
            before = [p.read_bytes() for p in paths]
            output = root / 'report.json'
            result = subprocess.run([sys.executable, str(script), '--train', str(paths[0]),
                                     '--evaluation', str(paths[1]), '--output', str(output)],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(output.read_text(encoding='utf-8'))
            self.assertTrue(report['algorithm']['version'])
            self.assertEqual(report['algorithm']['thresholds'], [0.9, 0.95])
            self.assertTrue(report['candidate_only'])
            candidate = report['candidates'][0]
            self.assertEqual(candidate['evaluation_id'], 'evaluation')
            self.assertEqual(candidate['train_id'], 'train')
            self.assertTrue(candidate['manual_review_required'])
            self.assertEqual(candidate['evaluation_question'], evaluation[0]['request']['state']['clinical_question'])
            self.assertEqual([p.read_bytes() for p in paths], before)
            blocked = subprocess.run([sys.executable, str(script), '--train', str(paths[0]),
                                      '--evaluation', str(paths[1]), '--output', str(paths[0])],
                                     capture_output=True, text=True)
            self.assertNotEqual(blocked.returncode, 0)
            self.assertIn('must not overwrite', blocked.stderr)
            self.assertEqual([p.read_bytes() for p in paths], before)


if __name__ == '__main__':
    unittest.main()
