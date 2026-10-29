"""Bộ phân tích văn bản dùng chung cho đường build index và đường truy vấn.

Hai tokenizer:
- ``default``: baseline bắt buộc. Chữ thường, tách theo ký tự không phải chữ/số (``\\w+``).
  Không xử lý gì riêng cho Lean.
- ``lean``: giữ tên đầy đủ và tách tên theo ``.``, ``_``, camelCase; giữ ký hiệu toán
  (∀ → ↔ ≤ ∣ ⁻¹ ...) làm token; bỏ biến một chữ cái ASCII (n, m, a, x).

Không dùng NFKC vì NFKC đổi ℝ → R, ℕ → N và làm mất phân biệt kiểu.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Callable

TOKENIZER_VERSION = {"default": "default-v1", "lean": "lean-v1"}

_WORD = re.compile(r"\w+")
# Định danh Lean: bắt đầu bằng chữ/_ (gồm chữ Unicode như ℕ, α), có thể chứa số, ', !, ?,
# chỉ số dưới, và nhiều thành phần nối bằng dấu chấm.
_IDENT = re.compile(r"[^\W\d][\w'!?₀-ₜ]*(?:\.[^\W\d][\w'!?₀-ₜ]*)*")
_NUMBER = re.compile(r"\d+")
_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z0-9']+|[A-Z]+|[^\W_]+")
# Dấu câu/ngoặc không mang nghĩa toán học
_PUNCT = set("()[]{},:;.⟨⟩`\"@#$|«»⦃⦄") | {"‹", "›"}
_MULTI_SYMBOLS = ("⁻¹", ":=", "<|>", "<;>", "->", "<->", "≠", "^")


def default_tokens(text: str) -> list[str]:
    """Baseline: chữ thường + tách theo ``\\w+``."""
    return _WORD.findall(text.lower())


def _split_component(comp: str) -> list[str]:
    """``add_comm`` → [add, comm]; ``natAbs`` → [nat, abs]."""
    parts: list[str] = []
    for piece in comp.split("_"):
        if piece:
            parts.extend(m.group(0) for m in _CAMEL.finditer(piece))
    return parts


def _ident_tokens(ident: str) -> list[str]:
    comps = [c for c in ident.split(".") if c]
    out: list[str] = []
    if len(comps) > 1:
        out.append(ident.lower())  # tên đầy đủ, ví dụ nat.add_comm
    for c in comps:
        low = c.lower()
        if len(c) == 1 and c.isascii():
            continue  # biến một chữ cái: n, m, a, x
        out.append(low)
        sub = [s.lower() for s in _split_component(c)]
        if len(sub) > 1:
            out.extend(s for s in sub if not (len(s) == 1 and s.isascii()))
    return out


def lean_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch.isspace():
            i += 1
            continue
        m = _IDENT.match(text, i)
        if m:
            tokens.extend(_ident_tokens(m.group(0)))
            i = m.end()
            continue
        m = _NUMBER.match(text, i)
        if m:
            tokens.append(m.group(0))
            i = m.end()
            continue
        for sym in _MULTI_SYMBOLS:
            if text.startswith(sym, i):
                tokens.append(sym)
                i += len(sym)
                break
        else:
            if ch not in _PUNCT and ch != "_":
                tokens.append(ch)
            i += 1
    return tokens


TOKENIZERS: dict[str, Callable[[str], list[str]]] = {"default": default_tokens, "lean": lean_tokens}


def get_tokenizer(name: str) -> Callable[[str], list[str]]:
    try:
        return TOKENIZERS[name]
    except KeyError:
        raise ValueError(f"tokenizer không tồn tại: {name}; chọn một trong {sorted(TOKENIZERS)}") from None


def term_counts(text: str, tokenizer: str) -> Counter:
    return Counter(get_tokenizer(tokenizer)(text))
