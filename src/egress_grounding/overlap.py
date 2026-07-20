from __future__ import annotations

import re


def english_tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+", text.lower())


def cjk_chars(text: str) -> list[str]:
    return re.findall(r"[\u3400-\u9fff]", text)


def longest_common_run(left: list[str], right: list[str]) -> int:
    if not left or not right:
        return 0
    prev = [0] * (len(right) + 1)
    best = 0
    for l_item in left:
        cur = [0] * (len(right) + 1)
        for j, r_item in enumerate(right, 1):
            if l_item == r_item:
                cur[j] = prev[j - 1] + 1
                best = max(best, cur[j])
        prev = cur
    return best


def suspicious_overlap(candidate: str, source: str, *, english_span: int = 8, cjk_span: int = 14) -> bool:
    if longest_common_run(english_tokens(candidate), english_tokens(source)) >= english_span:
        return True
    if longest_common_run(cjk_chars(candidate), cjk_chars(source)) >= cjk_span:
        return True
    return False


def has_suspicious_overlap(candidate: str, sources: list[str], *, english_span: int = 8, cjk_span: int = 14) -> bool:
    return any(suspicious_overlap(candidate, source, english_span=english_span, cjk_span=cjk_span) for source in sources if source)
