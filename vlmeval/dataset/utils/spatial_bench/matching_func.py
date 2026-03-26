import re


def can_match_option(answer_text, choices=None, tail_lines=6, tail_window=800):
    if not isinstance(answer_text, str):
        return False

    text = answer_text.strip()
    letters = 'ABCDEFGHIJ'
    if choices:
        allowed = {str(choice).strip().upper()[:1] for choice in choices if str(choice).strip()}
        letters = ''.join(ch for ch in letters if ch in allowed) or 'ABCDEF'
    else:
        letters = 'ABCDEF'

    block_pattern = re.compile(
        rf'<\s*answer\b[^>]*>\s*([{letters}])(?:\s*[\.．:：\)\]】、])?.*?<\s*/\s*answer\s*>',
        flags=re.IGNORECASE | re.DOTALL,
    )
    match = block_pattern.search(text)
    if match:
        return match.group(1).upper()

    phrase_pattern = re.compile(
        rf'(?i)(?:final\s*answer|the\s*answer\s*is|answer(?:\s*is)?|correct\s*answer|'
        rf'答案|最终答案|结论|所以|因此|我选(?:择)?|选择|选)\s*[:：>＝=]?\s*'
        rf'[\(\[\{{（【]?\s*([{letters}])\s*[\)\]\}}）】]?(?:\b|[.)、。])'
    )
    match = phrase_pattern.search(text[-tail_window:])
    if match:
        return match.group(1).upper()

    line_patterns = [
        re.compile(rf'^\s*[*_`>（）\[\]【】\(\)]*\s*([{letters}])\s*[*_`（）\[\]【】\(\)]*\s*$'),
        re.compile(rf'^\s*([{letters}])\s*[\.．:：\)\]】、-]\s+', flags=re.IGNORECASE),
    ]
    tail_segment = text[-tail_window:].splitlines()[-tail_lines:]
    for line in reversed([line.strip() for line in tail_segment if line.strip()]):
        for pattern in line_patterns:
            match = pattern.match(line)
            if match:
                return match.group(1).upper()

    token_pattern = re.compile(rf'(?<![A-Za-z])([{letters}])(?![A-Za-z])')
    tokens = [token.upper() for token in token_pattern.findall(text)]
    unique_tokens = sorted(set(tokens))
    if len(unique_tokens) == 1:
        return unique_tokens[0]

    return False
