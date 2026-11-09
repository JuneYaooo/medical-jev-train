"""Frozen choice prompt used for training and standalone inference.

Serialization is shared by training and standalone inference.
"""
import json
import string

SYMBOLS = string.ascii_uppercase + string.ascii_lowercase
SYSTEM = '根据材料、题设规则和候选含义选择唯一最佳答案。只输出选项字母，不解释。'


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def option_keys(row):
    return list(row['request']['questions']['decision']['criteria'])


def prompt_for(request, tokenizer):
    question = request['questions']['decision']
    keys = list(question['criteria'])
    require(2 <= len(keys) <= len(SYMBOLS), 'supported candidate count is 2..52')
    choices = {SYMBOLS[i]: question['criteria'][key] for i, key in enumerate(keys)}
    payload = {'任务': question['instructions'], '材料': request['state'], '选项': choices}
    content = json.dumps(payload, ensure_ascii=False, separators=(',', ':'))
    prompt = tokenizer.apply_chat_template(
        [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': content}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return prompt, keys
