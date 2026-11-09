"""Render the model card from the source template."""
from pathlib import Path


def render(template, model_id=None):
    fields = {'MODEL_ID': model_id or 'JuneYao/medical-jev-4b', 'NAME': 'medical-jev-4b',
              'LICENSE': 'mit', 'METHOD': 'LoRA', 'ACCURACY': '67.90%', 'SUPPORTED': '72.70%',
              'VERSION': 'v1.1.0', 'WEIGHTS': '约 130 MB 适配器。',
              'TRAINING': '全层 rank 16 / alpha 32 LoRA，选择第 5,500 步。',
              'LICENSING': '代码与适配器采用 MIT，底座采用 Apache-2.0；数据遵循各来源许可。'}
    for key, value in fields.items():
        template = template.replace('{{' + key + '}}', value)
    return template


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--docs', default='docs')
    args = parser.parse_args()
    root = Path(args.docs)
    template = (root / 'MODEL_CARD.template.md').read_text()
    # Keep exporter substitution compatible with custom model repository IDs.
    (root / 'MODEL_CARD_V1.md').write_text(render(template, '{{MODEL_ID}}'))


if __name__ == '__main__':
    main()
