"""Choice scoring with the backbone's own answer-letter logits.

Scores are the LM logits of the positional answer letters at the last prompt
position. With a zero LoRA update this is exactly the frozen readout, so
training starts from the readout instead of a random head, and any option
count up to 52 is supported.
"""
import hashlib
import json
import math
from pathlib import Path
import re

from .prompt import SYMBOLS, option_keys, prompt_for, require

FORMAT = 'jev-lora-v1'


def fast_identity(model_path):
    """Config hash plus weight file names and sizes; cheap enough to run every load."""
    root = Path(model_path)
    config = (root / 'config.json').read_bytes()
    files = sorted((p.name, p.stat().st_size) for p in root.glob('*.safetensors'))
    files += sorted((p.name, p.stat().st_size) for p in root.glob('*.bin'))
    require(files, 'local model weights are required')
    return hashlib.sha256(json.dumps({'config': hashlib.sha256(config).hexdigest(), 'files': files}).encode()).hexdigest()


def decoder_layers(model):
    for path in ('model.layers', 'model.language_model.layers', 'language_model.model.layers',
                 'model.model.layers', 'transformer.h'):
        current = model
        try:
            for part in path.split('.'):
                current = getattr(current, part)
        except AttributeError:
            continue
        if len(current):
            return current
    raise ValueError('cannot locate decoder layers in this model class')


def selected_layers(total, spec):
    """'all', an int N (the last N layers), or an explicit list of indices."""
    if spec in (None, 'all'):
        return list(range(total))
    if type(spec) is int:
        require(1 <= spec <= total, 'invalid LoRA layer count')
        return list(range(total - spec, total))
    require(isinstance(spec, list) and spec and all(type(i) is int and -total <= i < total for i in spec),
            'invalid LoRA layer list')
    return sorted({i % total for i in spec})


def lora_spec(spec):
    lora = {'layers': 'all', 'rank': 16, 'alpha': 32, 'dropout': 0.0, 'targets': None, **spec.get('lora', {})}
    require(type(lora['rank']) is int and lora['rank'] > 0, 'invalid LoRA rank')
    require(isinstance(lora['alpha'], (int, float)) and lora['alpha'] > 0, 'invalid LoRA alpha')
    require(0 <= lora['dropout'] < 1, 'invalid LoRA dropout')
    require(lora['targets'] is None or (isinstance(lora['targets'], list) and lora['targets']),
            'targets must be null (every Linear) or a list of name regexes')
    return lora


def build_lora_class(torch):
    class LoRALinear(torch.nn.Module):
        def __init__(self, base, rank, alpha, dropout, generator):
            super().__init__()
            self.base = base
            self.scale = alpha / rank
            a = torch.empty(rank, base.in_features, dtype=torch.float32)
            bound = 1 / math.sqrt(base.in_features)
            a.uniform_(-bound, bound, generator=generator)
            self.lora_a = torch.nn.Parameter(a.to(base.weight.device))
            self.lora_b = torch.nn.Parameter(torch.zeros(base.out_features, rank, dtype=torch.float32,
                                                         device=base.weight.device))
            self.dropout = torch.nn.Dropout(dropout) if dropout else torch.nn.Identity()
            self.enabled = True

        def forward(self, x):
            out = self.base(x)
            if not self.enabled:
                return out
            update = self.dropout(x) @ self.lora_a.to(x.dtype).T @ self.lora_b.to(x.dtype).T
            return out + (update * self.scale).to(out.dtype)
    return LoRALinear


class ChoiceModel:
    """LoRA training and inference on the answer-letter scores.

    spec keys: model_path, device (cuda/cpu/mps), dtype (bf16/fp16/fp32),
    attn_implementation (sdpa/eager/flash_attention_2), max_tokens,
    max_options, gradient_checkpointing, pass_position_ids, lora{...}.
    """

    def __init__(self, spec, checkpoint=None, seed=17):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        metadata = read_metadata(checkpoint) if checkpoint else None
        if metadata:
            spec = {**spec, 'lora': metadata.get('lora') or {}}
        require(spec.get('training_mode', 'lora') == 'lora', 'only LoRA training is supported')
        self.spec = spec
        self.training_mode = 'lora'
        self.lora_config = lora_spec(spec)
        self.device = spec.get('device', 'cuda')
        dtypes = {'bf16': torch.bfloat16, 'fp16': torch.float16, 'fp32': torch.float32}
        self.dtype = dtypes[spec.get('dtype', 'bf16')]
        self.identity = fast_identity(spec['model_path'])
        if metadata:
            require(metadata['base_model'] == self.identity, 'adapter was trained on a different base model')
        self.tokenizer = AutoTokenizer.from_pretrained(spec['model_path'], local_files_only=True)
        options = {'local_files_only': True, 'attn_implementation': spec.get('attn_implementation', 'sdpa')}
        try:  # transformers >= 4.56 names it dtype; older releases use torch_dtype.
            self.model = AutoModelForCausalLM.from_pretrained(spec['model_path'], dtype=self.dtype, **options)
        except TypeError:
            self.model = AutoModelForCausalLM.from_pretrained(spec['model_path'], torch_dtype=self.dtype, **options)
        self.model = self.model.to(self.device)
        self.model.config.use_cache = False
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.max_options = spec.get('max_options', 26)
        self.letter_ids = []
        for letter in SYMBOLS[:self.max_options]:
            ids = self.tokenizer.encode(letter, add_special_tokens=False)
            require(len(ids) == 1, 'answer letter is not a single token: ' + letter)
            self.letter_ids.append(ids[0])
        require(len(set(self.letter_ids)) == len(self.letter_ids), 'answer letter token collision')
        self.pad_id = self.tokenizer.pad_token_id
        if self.pad_id is None:
            self.pad_id = self.tokenizer.eos_token_id
        require(self.pad_id is not None, 'tokenizer needs a pad or eos token')
        self._inject(seed)
        if checkpoint:
            self.load(checkpoint)
        if spec.get('gradient_checkpointing', False):
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        self.token_cache = {}

    def _inject(self, seed):
        torch = self.torch
        LoRALinear = build_lora_class(torch)
        layers = decoder_layers(self.model)
        self.layer_indices = selected_layers(len(layers), self.lora_config['layers'])
        patterns = [re.compile(p) for p in (self.lora_config['targets'] or [])]
        generator = torch.Generator().manual_seed(seed)
        self.adapters = {}
        for index in self.layer_indices:
            layer = layers[index]
            names = [n for n, m in layer.named_modules() if isinstance(m, torch.nn.Linear)
                     and (not patterns or any(p.search(n) for p in patterns))]
            for name in names:
                owner_name, _, attr = name.rpartition('.')
                owner = layer.get_submodule(owner_name) if owner_name else layer
                wrapped = LoRALinear(getattr(owner, attr), self.lora_config['rank'], self.lora_config['alpha'],
                                     self.lora_config['dropout'], generator)
                setattr(owner, attr, wrapped)
                self.adapters[f'layers.{index}.{name}'] = wrapped
        require(self.adapters, 'LoRA matched no Linear modules')
        self.parameters = {}
        for key, module in self.adapters.items():
            self.parameters[key + '.lora_a'] = module.lora_a
            self.parameters[key + '.lora_b'] = module.lora_b

    def trainable(self):
        return list(self.parameters.values())

    def trainable_count(self):
        return sum(p.numel() for p in self.parameters.values())

    def set_adapter_enabled(self, enabled):
        for module in self.adapters.values():
            module.enabled = enabled

    # ---- tokenization -------------------------------------------------
    def encode(self, row):
        """Prompt tokens for the row's displayed option order."""
        keys = option_keys(row)
        if len(keys) > self.max_options:
            raise UnsupportedRow(f'{len(keys)} options exceed max_options {self.max_options}')
        key = (row['id'], tuple(keys))
        cached = self.token_cache.get(key)
        if cached is None:
            prompt, _ = prompt_for(row['request'], self.tokenizer)
            cached = self.tokenizer.encode(prompt, add_special_tokens=False)
            if len(self.token_cache) < 500_000:
                self.token_cache[key] = cached
        if len(cached) > self.spec.get('max_tokens', 2048):
            raise UnsupportedRow(f'{len(cached)} tokens exceed max_tokens')
        return cached, keys

    def supported(self, row):
        try:
            self.encode(row)
            return True
        except UnsupportedRow:
            return False

    # ---- scoring ------------------------------------------------------
    def letter_logits(self, token_lists, n_options):
        """Left-padded batch -> float32 [B, max_n] option logits, padded with -1e4."""
        torch = self.torch
        width = max(map(len, token_lists))
        ids = torch.full((len(token_lists), width), self.pad_id, dtype=torch.long)
        mask = torch.zeros((len(token_lists), width), dtype=torch.long)
        for i, tokens in enumerate(token_lists):
            ids[i, width - len(tokens):] = torch.tensor(tokens)
            mask[i, width - len(tokens):] = 1
        ids, mask = ids.to(self.device), mask.to(self.device)
        kwargs = {'input_ids': ids, 'attention_mask': mask, 'use_cache': False}
        if self.spec.get('pass_position_ids', True):
            kwargs['position_ids'] = (mask.cumsum(-1) - 1).clamp_min(0)
        letters = torch.tensor(self.letter_ids[:max(n_options)], device=self.device)
        try:
            out = self.model(**kwargs, logits_to_keep=1)
            logits = out.logits[:, -1, :].float()[:, letters]
        except TypeError:
            out = self.model(**kwargs, output_hidden_states=True)
            hidden = out.hidden_states[-1][:, -1, :]
            weight = self.model.get_output_embeddings().weight[letters]
            logits = (hidden @ weight.T).float()
        columns = torch.arange(logits.shape[1], device=self.device)
        valid = columns[None, :] < torch.tensor(n_options, device=self.device)[:, None]
        return logits.masked_fill(~valid, -1e4)

    def probabilities(self, rows, batch_size=8):
        """Semantic-key probabilities for each row in its displayed order (no grad)."""
        torch = self.torch
        results = [None] * len(rows)
        encoded = [(i, *self.encode(r)) for i, r in enumerate(rows)]
        encoded.sort(key=lambda x: len(x[1]))
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad(), self.autocast():
            for start in range(0, len(encoded), batch_size):
                chunk = encoded[start:start + batch_size]
                logits = self.letter_logits([c[1] for c in chunk], [len(c[2]) for c in chunk])
                require(bool(torch.isfinite(logits).all()), 'nonfinite choice logits')
                for (index, _, keys), row_logits in zip(chunk, logits):
                    p = row_logits[:len(keys)].softmax(-1).cpu().tolist()
                    results[index] = dict(zip(keys, p))
        self.model.train(was_training)
        return results

    def autocast(self):
        torch = self.torch
        if self.device == 'cuda' and self.dtype != torch.float32:
            return torch.autocast('cuda', dtype=self.dtype)
        import contextlib
        return contextlib.nullcontext()

    # ---- persistence --------------------------------------------------
    def state(self):
        return {k: p.detach().cpu().contiguous() for k, p in self.parameters.items()}

    def save(self, path, extra=None):
        from safetensors.torch import save_file
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {'format': FORMAT,
                'training_mode': self.training_mode, 'base_model': self.identity, 'lora': self.lora_config,
                'layer_indices': self.layer_indices, 'letter_ids': self.letter_ids, **(extra or {})}
        temporary = path.with_name(path.name + '.partial')
        save_file(self.state(), str(temporary), metadata={'jev': json.dumps(meta, ensure_ascii=False)})
        temporary.replace(path)

    def load(self, path):
        from safetensors.torch import load_file
        state = load_file(str(path))
        meta = read_metadata(path)
        require(meta['base_model'] == self.identity, 'checkpoint was trained on a different base model')
        require(meta['letter_ids'][:len(self.letter_ids)] == self.letter_ids[:len(meta['letter_ids'])],
                'answer letter tokens differ from the checkpoint')
        require(set(state) == set(self.parameters), 'checkpoint parameter keys do not match the model')
        for key, p in self.parameters.items():
            require(state[key].shape == p.shape, 'adapter shape mismatch: ' + key)
            require(bool(self.torch.isfinite(state[key]).all()), 'nonfinite adapter tensor: ' + key)
        with self.torch.no_grad():
            for key, p in self.parameters.items():
                p.copy_(state[key].to(p.device))

    def load_state(self, state):
        with self.torch.no_grad():
            for key, p in self.parameters.items():
                p.copy_(state[key].to(p.device))


class UnsupportedRow(ValueError):
    pass


def read_metadata(path):
    from safetensors import safe_open
    with safe_open(str(path), framework='pt', device='cpu') as stream:
        meta = (stream.metadata() or {}).get('jev')
    require(meta is not None, 'not a JEV training checkpoint')
    meta = json.loads(meta)
    require(meta.get('format') == FORMAT, 'unsupported checkpoint format')
    return meta
