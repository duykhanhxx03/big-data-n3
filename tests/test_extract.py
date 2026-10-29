from extract.leandojo import decl_kind, library_of, statement_of


def test_theorem_without_proof_is_kept_whole():
    code = "protected theorem add_comm : ∀ (n m : Nat), n + m = m + n"
    assert statement_of(code) == code
    assert decl_kind(code) == "theorem"


def test_attributes_and_body_are_removed():
    code = "@[simp, norm_cast]\ndef of (C : Type u) [Quiver.{v + 1} C] : Quiv.{v, u} :=\n  Bundled.of C"
    assert statement_of(code) == "def of (C : Type u) [Quiver.{v + 1} C] : Quiv.{v, u}"
    assert decl_kind(code) == "def"


def test_where_and_nested_brackets():
    code = ("protected def mk' (s : Set R) (sm : Submonoid R) (hm : ↑sm = s) :\n"
            "    Subsemiring R where\n  carrier := s")
    assert statement_of(code).endswith("Subsemiring R")
    assert statement_of("def f (h : x := 1) : T := 2") == "def f (h : x := 1) : T"


def test_pattern_matching_arms():
    code = "def fib : Nat → Nat\n  | 0 => 0\n  | n + 1 => fib n"
    assert statement_of(code) == "def fib : Nat → Nat"


def test_set_builder_bar_is_not_an_arm():
    code = "theorem mem_setOf {p : α → Prop} : a ∈ { x | p x } ↔ p a"
    assert statement_of(code) == code


def test_library():
    assert library_of(".lake/packages/lean4/src/lean/Init/Data/Nat/Basic.lean") == "lean4"
    assert library_of("Mathlib/Order/Basic.lean") == "Mathlib"
