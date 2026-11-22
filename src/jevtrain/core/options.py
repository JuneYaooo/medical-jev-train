"""Guard permutations of choices with positional references."""
import re
import unicodedata


def positional_reasons(request):
    options = request['questions']['decision']['criteria']
    texts = [str(v) for v in options.values()]
    state = request.get('state', {})
    texts += [str(state[k]) for k in ('clinical_question', 'question') if k in state]
    reasons = set()
    for raw in texts:
        text = unicodedata.normalize('NFKC', raw)
        if re.search(r'(?:以上|上述|前述|下面|以下).{0,6}(?:均|都|全部|皆|各项|选项)|(?:all|none|both|any) of (?:the )?(?:above|below|these|those)|(?:all|none) (?:the )?(?:above|below)', text, re.I):
            reasons.add('relative_option_reference')
        # Implicit pairs do not name their referents. In a shuffled list,
        # "both" may no longer refer to the same two preceding choices.
        if re.search(r'(?:两者|二者|二项|两项)(?:均|都|皆|全)|^\s*(?:both|neither)(?:\s+(?:answers?|options?))?(?:\s+(?:are|is))?(?:\s+(?:correct|incorrect|true|false))?\s*[.!。]?\s*$', text, re.I):
            reasons.add('implicit_pair_reference')
        # Upper-case choice letters and explicit separators; do not match words like Core.
        if re.search(r'(?<![A-Za-z])[A-E](?:\s+(?:and|or)\s+|\s*[&、和及与或]\s*)[A-E](?![A-Za-z])', text):
            reasons.add('letter_combination_reference')
        if re.search(r'(?:选项|答案)\s*[A-E](?![A-Za-z])|(?:option|choice)\s+[A-E](?![A-Za-z])', text):
            reasons.add('explicit_option_reference')
    return sorted(reasons)
