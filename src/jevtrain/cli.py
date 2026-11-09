"""Medical choice model: split / sample / verify / bench / train / eval / calibrate / compare.

  python scripts/jev.py import-benchmark --benchmark <jev-healthcare-lab>/benchmarks/medical_decision_v1 \
                                --output work/benchmark/medical_decision_v1.jsonl
  python scripts/jev.py inventory --input pool.jsonl [dev.jsonl ...]        # per-task counts before training
  python scripts/jev.py split   --input pool.jsonl --per-task 60 --output work/splits [--exclude bench.jsonl]
  python scripts/jev.py sample  --config configs/jev/release/train.json --output work/train/sample.jsonl
  python scripts/jev.py verify  --config configs/jev/train.example.json --input train.jsonl --output work/verify
  python scripts/jev.py bench   --config configs/jev/train.example.json --input train.jsonl --output work/bench
  python scripts/jev.py train   --config configs/jev/train.example.json --output runs/train
  python scripts/jev.py eval    --config configs/jev/train.example.json --checkpoint runs/x/best.safetensors \\
                                --input dev.jsonl --views 4 --output runs/x/eval_dev
  python scripts/jev.py compare --base a/predictions.jsonl --candidate b/predictions.jsonl

See docs/TRAINING.md.
"""
import argparse
import json
from pathlib import Path
import sys


from jevtrain.data import exclusion_keys, read_rows, stratified_split, write_json, write_rows  # noqa: E402


from jevtrain.config import load_config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('import-benchmark')
    p.add_argument('--benchmark', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--all-scopes', action='store_true', help='keep non-clinical tasks too')
    p = sub.add_parser('inventory')
    p.add_argument('--input', nargs='+', required=True)
    p.add_argument('--target', type=int, default=1000, help='per-task row target used to list gaps')
    p.add_argument('--output')
    p = sub.add_parser('split')
    p.add_argument('--input', nargs='+', required=True)
    p.add_argument('--exclude', nargs='*', default=[])
    p.add_argument('--per-task', type=int, default=60)
    p.add_argument('--names', nargs='+', default=['development', 'test'])
    p.add_argument('--max-fraction', type=float, default=0.3)
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--output', required=True)
    p = sub.add_parser('sample', help='write the rows selected by data.train in a training config')
    p.add_argument('--config', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--supported-only', action='store_true', help='apply tokenizer length and option limits without loading weights')
    for name in ('verify', 'bench'):
        p = sub.add_parser(name)
        p.add_argument('--config', required=True)
        p.add_argument('--input', required=True)
        p.add_argument('--output', required=True)
        p.add_argument('--steps', type=int, default=100)
    p = sub.add_parser('train')
    p.add_argument('--config', required=True)
    p.add_argument('--output', required=True)
    p = sub.add_parser('eval')
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint')
    p.add_argument('--input', required=True)
    p.add_argument('--views', type=int, default=1)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--output', required=True)
    p.add_argument('--large-options', action='store_true', help='same as --tournament-above 26')
    p.add_argument('--tournament-above', type=int, help='block elimination above this option count; 52 = mixed policy')
    p.add_argument('--max-options', type=int, help='override model.max_options (release profile: 52)')
    p.add_argument('--max-tokens', type=int, help='override model.max_tokens (release profile: 8192)')
    p.add_argument('--sealed-ledger', help='reserve and record a one-time sealed evaluation attempt')
    p = sub.add_parser('calibrate', help='fit temperature on held-out validation predictions only')
    p.add_argument('--input', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    p = sub.add_parser('compare')
    p.add_argument('--base', required=True)
    p.add_argument('--candidate', required=True)
    args = parser.parse_args(argv)

    if args.command == 'import-benchmark':
        bench = Path(args.benchmark)
        scope = json.loads((bench / 'evaluation_scope.json').read_text())
        clinical = {t for t, v in scope['tasks'].items() if v.get('scope') == 'clinical'}
        answers = {r['id']: r for r in read_rows(bench / 'answers.jsonl')}
        rows, skipped = [], 0
        for r in read_rows(bench / 'requests.jsonl'):
            answer = answers.get(r['id'])
            if (not args.all_scopes and r['task'] not in clinical) or answer is None \
                    or answer['gold'] not in r['request']['questions']['decision']['criteria']:
                skipped += 1
                continue
            rows.append({'id': r['id'], 'group_id': answer.get('group_id', r['id']), 'task': r['task'],
                         'request': r['request'], 'gold': answer['gold'], 'split': 'benchmark'})
        write_rows(args.output, rows)
        print(json.dumps({'rows': len(rows), 'skipped': skipped}))
        return
    if args.command == 'inventory':
        from collections import Counter, defaultdict
        report = {}
        for path in args.input:
            rows = read_rows(path)
            tasks = defaultdict(lambda: {'rows': 0, 'groups': set(), 'options': Counter(), 'missing_gold': 0,
                                         'sources': Counter()})
            for r in rows:
                t = tasks[r.get('task', '<missing>')]
                t['rows'] += 1
                t['groups'].add(r.get('group_id'))
                t['options'][len(r['request']['questions']['decision']['criteria'])] += 1
                t['missing_gold'] += r.get('gold') is None
                t['sources'][r.get('source', 'unspecified')] += 1
            table = {name: {'rows': v['rows'], 'groups': len(v['groups']), 'missing_gold': v['missing_gold'],
                            'options': dict(sorted(v['options'].items())),
                            'top_sources': dict(v['sources'].most_common(3)),
                            'gap_to_target': max(0, args.target - v['rows'])}
                     for name, v in sorted(tasks.items(), key=lambda kv: -kv[1]['rows'])}
            report[path] = {'rows': len(rows), 'tasks': len(table), 'by_task': table}
            print(f'\n{path}: {len(rows)} rows, {len(table)} tasks')
            print(f"{'task':40s} {'rows':>8s} {'groups':>8s} {'>10opt':>7s} {'gap':>6s}")
            for name, v in table.items():
                wide = sum(c for n, c in v['options'].items() if n > 10)
                print(f"{name[:40]:40s} {v['rows']:8d} {v['groups']:8d} {wide:7d} {v['gap_to_target']:6d}")
        if args.output:
            write_json(args.output, report)
        return
    if args.command == 'split':
        rows = [r for path in args.input for r in read_rows(path)]
        groups, inputs = set(), set()
        for path in args.exclude:
            g, i = exclusion_keys(read_rows(path))
            groups |= g
            inputs |= i
        sets, pool, report = stratified_split(rows, args.per_task, tuple(args.names), args.seed,
                                              args.max_fraction, groups, inputs)
        out = Path(args.output)
        for name, values in sets.items():
            write_rows(out / f'{name}.jsonl', values)
        write_rows(out / 'pool.jsonl', pool)
        write_json(out / 'split_report.json', report)
        print(json.dumps(report['sizes'], ensure_ascii=False))
        return
    if args.command == 'calibrate':
        from jevtrain.calibrate import calibrate
        write_json(args.output, calibrate(args.input, args.checkpoint))
        return
    if args.command == 'compare':
        from jevtrain.evaluate import paired, task_table
        base, candidate = read_rows(args.base), read_rows(args.candidate)
        print(json.dumps({'paired': paired(base, candidate), 'tasks': task_table(base, candidate)},
                         ensure_ascii=False, indent=2))
        return
    config = load_config(args.config)
    if args.command == 'sample':
        from collections import Counter
        from jevtrain.train import load_training_rows
        rows = load_training_rows(config['data']['train'])
        rejected = Counter()
        if args.supported_only:
            from transformers import AutoTokenizer
            from jevtrain.prompt import prompt_for, SYMBOLS
            model = config['model']
            tokenizer = AutoTokenizer.from_pretrained(model['model_path'], local_files_only=True)
            kept = []
            for row in rows:
                count = len(row['request']['questions']['decision']['criteria'])
                if not 2 <= count <= min(model.get('max_options', 26), len(SYMBOLS)):
                    rejected['options'] += 1
                    continue
                prompt, _ = prompt_for(row['request'], tokenizer)
                if len(tokenizer.encode(prompt, add_special_tokens=False)) > model.get('max_tokens', 2048):
                    rejected['length'] += 1
                    continue
                kept.append(row)
            rows = kept
        write_rows(args.output, rows)
        print(json.dumps({'rows': len(rows), 'rejected': dict(rejected),
                          'by_task': dict(Counter(r['task'] for r in rows).most_common())},
                         ensure_ascii=False, indent=2))
        return
    if args.command == 'verify':
        from jevtrain.checks import verify
        print(json.dumps(verify(config, read_rows(args.input), args.output), ensure_ascii=False, indent=2))
    elif args.command == 'bench':
        from jevtrain.checks import bench
        print(json.dumps(bench(config, read_rows(args.input), args.output, steps=args.steps), indent=2))
    elif args.command == 'train':
        from jevtrain.train import run_training
        report, _ = run_training(config, args.output)
        print(json.dumps({k: report[k] for k in ('steps', 'best', 'selected', 'elapsed_seconds')}, indent=2))
    elif args.command == 'eval':
        from jevtrain.evaluate import predict, summary
        from jevtrain.model import ChoiceModel
        from contextlib import nullcontext
        from jevtrain.sealed import sealed_attempt
        if args.sealed_ledger and not args.checkpoint:
            parser.error('--sealed-ledger requires a fixed checkpoint')
        context = sealed_attempt(args.sealed_ledger, args.input, args.checkpoint) if args.sealed_ledger else nullcontext(None)
        with context as sealed_status:
            spec = dict(config['model'])
            if args.max_options is not None:
                spec['max_options'] = args.max_options
            if args.max_tokens is not None:
                spec['max_tokens'] = args.max_tokens
            above = 26 if args.large_options and args.tournament_above is None else args.tournament_above
            model = ChoiceModel(spec, checkpoint=args.checkpoint)
            records, unsupported = predict(model, read_rows(args.input), view_count=args.views,
                                           batch_size=args.batch_size, tournament_above=above)
            out = Path(args.output)
            write_rows(out / 'predictions.jsonl', records)
            write_rows(out / 'unsupported.jsonl', unsupported)
            result = summary(records, unsupported)
            result.update(checkpoint=args.checkpoint or 'readout', views=args.views, input=args.input,
                          policy={'max_options': spec.get('max_options', 26), 'max_tokens': spec.get('max_tokens', 2048),
                                  'tournament_above': above})
            result['sealed_test'] = sealed_status or {'evaluated': False, 'run_count': 0}
            write_json(out / 'summary.json', result)
            print(json.dumps({k: result[k] for k in ('n', 'correct', 'accuracy', 'accuracy_95ci', 'macro_accuracy',
                                                     'flip_rate', 'unsupported')}, indent=2))


if __name__ == '__main__':
    main()
