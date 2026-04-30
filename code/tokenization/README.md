# Tokenization

This directory defines the representation layer studied by the implementation.

| File | Purpose |
| --- | --- |
| `README.md` | This directory guide. |
| `__init__.py` | Exposes the tokenizer classes that should be importable from `code.tokenization`. |
| `base.py` | Abstract `TokenizationStrategy` interface shared by all tokenizer families. It defines fitting, state/goal transformation, save/load, and embedding-dimension contracts. |
| `factory.py` | Factory function that maps tokenizer names such as `wl` or `graphbpe` to concrete tokenizer instances. |
| `wl.py` | Weisfeiler-Leman tokenizer wrapper using the `wlplan` ILG feature generator with two refinement iterations in the reported configuration. |
| `simhash.py` | SimHash tokenizer that builds sparse symbolic count features and projects them to seeded binary hash vectors. |
| `shortest_path.py` | Shortest-path feature tokenizer over state and goal object graphs with predicate/count/path histogram features. |
| `graphbpe.py` | GraphBPE tokenizer that learns frequent adjacent label merges and embeds states as merged-label histograms. |
| `random.py` | Deterministic random baseline tokenizer that maps predicate/goal tokens to seeded vectors. |
| `multidomain.py` | All-domain tokenizer utilities, including pooled fitting for shared-feature tokenizers and the WL domain-block union representation. |

The study treats tokenizer choice as an experimental factor, so these files are the core representation implementations.
