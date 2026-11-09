"""Predict choices from JSONL records containing a request or a bare request object."""
import argparse
import json
from pathlib import Path
import sys

from jevtrain.inference import ChoicePredictor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', required=True, help='downloaded model release directory')
    parser.add_argument('--base-model', help='optional existing, verified Qwen3.5-4B directory')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--input', required=True, help='JSONL input, or - for stdin')
    parser.add_argument('--output', default='-', help='JSONL output, defaults to stdout')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--large-options', action='store_true', help='experimental multi-pass elimination above 26 options')
    parser.add_argument('--tournament-above', type=int, help='eliminate in blocks only above this option count (26-52)')
    parser.add_argument('--absent-option', action=argparse.BooleanOptionalAction, default=None,
                        help='use the block-training subset format; default follows the bundle')
    args = parser.parse_args()
    stream = sys.stdin if args.input == '-' else Path(args.input).open(encoding='utf-8')
    try:
        records = [json.loads(line) for line in stream if line.strip()]
    finally:
        if stream is not sys.stdin:
            stream.close()
    if not records:
        parser.error('input is empty')
    predictor = ChoicePredictor(args.bundle, args.base_model, args.device)
    results = predictor.predict([row.get('request', row) for row in records], args.batch_size, large_options=args.large_options,
                                tournament_above=args.tournament_above, absent_option=args.absent_option)
    output = ''.join(json.dumps({'id': row.get('id', i), **result}, ensure_ascii=False, allow_nan=False) + '\n'
                     for i, (row, result) in enumerate(zip(records, results)))
    if args.output == '-':
        print(output, end='')
    else:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(output, encoding='utf-8')


if __name__ == '__main__':
    main()
