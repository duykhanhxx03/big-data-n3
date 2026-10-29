from analysis.tokenizer import default_tokens, lean_tokens


def test_default_is_plain_word_split():
    assert default_tokens("theorem Nat.add_comm (n m : ℕ) : n + m = m + n") == [
        "theorem", "nat", "add_comm", "n", "m", "ℕ", "n", "m", "m", "n"]


def test_lean_splits_names_and_keeps_full_name():
    toks = lean_tokens("Nat.add_comm")
    assert toks[0] == "nat.add_comm"
    assert {"nat", "add_comm", "add", "comm"} <= set(toks)


def test_lean_keeps_math_symbols_and_drops_single_letter_vars():
    toks = lean_tokens("n m : ℕ\n⊢ n + m = m + n")
    assert "ℕ" in toks and "⊢" in toks and "+" in toks and "=" in toks
    assert "n" not in toks and "m" not in toks
    assert ":" not in toks


def test_lean_does_not_normalize_blackboard_letters():
    toks = lean_tokens("x : ℝ")
    assert "ℝ" in toks and "r" not in toks


def test_lean_camel_case_and_multi_char_symbols():
    toks = lean_tokens("Int.natAbs x⁻¹ ≠ 0")
    assert {"int.natabs", "natabs", "nat", "abs", "⁻¹", "≠", "0"} <= set(toks)


def test_lean_identifiers_with_primes_and_subscripts():
    toks = lean_tokens("h₁ : a' ≤ b")
    assert "h₁" in toks and "a'" in toks and "≤" in toks
