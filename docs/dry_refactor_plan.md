# DRY refactor plan: `cs336_basics/bpe_tokenizer`

Scope: duplication, dead code and general tidy-up in
[tokenizer.py](../cs336_basics/bpe_tokenizer/tokenizer.py),
[train_tokenizer.py](../cs336_basics/bpe_tokenizer/train_tokenizer.py) and
[utils.py](../cs336_basics/bpe_tokenizer/utils.py).
Goal: behavior-preserving. `tests/test_train_bpe.py` and `tests/test_tokenizer.py` are the safety net.

## Findings

### A. Duplicated logic
1. **Pre-tokenization regex** is defined in `utils.py` (the name is rebound from str to compiled) and again as a raw string in `tokenizer.py`.
2. **Splitting on special tokens**: `utils.split_on_special_tokens` drops the specials and whitespace-only pieces (used by training). `tokenizer._split_on_special_tokens` keeps the specials as their own pieces (used by encoding). The regex-building line is repeated, including a third time in `_pre_tokenize_text_in_chunks`. The behaviors differ on purpose, so they must become ONE function with an explicit "keep specials" choice.
3. **Pre-tokenizing text**: `utils.pre_tokenize_text_in_chunks` and `tokenizer._pre_tokenize_text_in_chunks` express the same idea. `utils.pre_tokenize_chunk` and `utils.get_pre_tokens_count` both re-inline the "read chunk, split, finditer" loop (the former only returns a length and looks unused).
4. **Pair/merge helpers**: `_merge_pair_in_token` exists 3 times (twice in `Tokenizer`, where the second shadows the first, and once in `BPETokenizer`). `_get_pair_from_token` exists in both classes. `_get_pair_freq` and `_get_pair_dict_and_freq` repeat the same pair-iteration loop.
5. **Counting**: `count_pre_tokens` and the manual cross-chunk merge in `BPETokenizer.train` hand-roll what `collections.Counter` provides.
6. **bytes <-> str conversion** has several variants: `Tokenizer.convert_bytes_to_str`, `Tokenizer.map_to_bytes`, `convert` inside train's `__main__`, `utils.render_token`, `convert_key_to_tuple_of_bytes`, `convert_merge_result`.
7. **Vocab/merges persistence** exists 3 times: `utils.save_vocab_and_merges` (used by scripts), an inline writer in `train_tokenizer.__main__`, and `Tokenizer.from_files` + `convert_merge_result` (while `utils.load_vocab_and_merges` is unused).
   - Risk: the writers use DIFFERENT encodings (utf-8 with `replace` vs chr-per-byte) and `from_files` expects the latter. Files from `save_vocab_and_merges` are lossy for partial-UTF-8 tokens and may not round-trip through `Tokenizer.from_files`. Needs a round-trip test.
8. **Special-token handling** (bytes encoding, longest-first sort, regex building) is recomputed ad hoc in the trainer, the tokenizer (per `encode` call) and the splitters. The trainer also hard-codes `b"<|endoftext|>"` instead of using its configured special tokens.

### B. Dead / suspicious code
- Unused (verify each with grep before deleting): `_merge_pair`, `_get_pair_freq`, `_load_corpus`, `utils.pre_tokenize_chunk`, `utils.load_vocab_and_merges`, `Tokenizer.convert_bytes_to_str`, unused imports in `utils.py` (`Process`, `Pool`, `defaultdict`).
- Large commented-out blocks and "temporary"/TODO markers in all three files.
- `train()` only defines `pre_tokens_count` inside `if enable_mp:`, so the single-process path fails.
- `logging.basicConfig(...)` runs at import time in `tokenizer.py` (library side effect).
- Two `if __name__ == "__main__":` demo blocks duplicate what `cs336_basics/scripts/train_bpe_*.py` already do.

## Target structure (names are suggestions)
- `pretokenization.py`: the single compiled pattern, special-token regex builder, ONE split function, ONE pre-tokenize function, chunk-boundary finder, per-chunk counting worker (Counter-based merge).
- `bpe_common.py` (shared by trainer and tokenizer): pair extraction, pair merge, string<->bytes conversions, special-token normalization (bytes + longest-first order).
- `serialization.py`: one save + one load for vocab/merges with a single documented on-disk scheme; `Tokenizer.from_files` delegates to it.
- `train_tokenizer.py`: only the trainer class (merge loop, frequency bookkeeping).
- `tokenizer.py`: only the encoder/decoder class.
- `utils.py`: removed (imports updated) or kept as a thin re-export for the scripts/tests that import it.
- Demo `__main__` blocks removed.

## Execution order (run the tests before and after each step)
1. Baseline: run both test files; save a small vocab/merges output for before/after comparison.
2. Delete confirmed-dead code and commented-out blocks.
3. Centralize the regex pattern and special-token regex/normalization.
4. Unify the two special-token splitters and two pre-tokenizers (explicit keep-specials option; preserve training's whitespace-only-piece behavior).
5. Move pair/merge helpers to the shared module; remove the duplicate copies.
6. Replace hand-rolled counting with Counter; fix the `enable_mp=False` path; stop hard-coding the split token.
7. Unify bytes<->str conversion and vocab/merges save/load; add a round-trip test (train -> save -> `from_files` -> encode/decode); make `from_files` delegate.
8. Remove `__main__` blocks and import-time `logging.basicConfig`; update imports in `tests/adapters.py` and `scripts/train_bpe_*.py`.
9. Final verification: tests, before/after output comparison, grep for leftover duplicates and unused imports.

## Notes
- Steps 4 and 7 are the risky ones: the duplicates are NOT behaviorally identical, so merge deliberately, not mechanically.
- `scripts/train_bpe_tinystories.py` and `train_bpe_owt.py` were not reviewed in depth; they are likely near-duplicates of each other (optional follow-up).
- Performance of `update_frequency_dict` (copying dicts every merge) is out of scope here.
