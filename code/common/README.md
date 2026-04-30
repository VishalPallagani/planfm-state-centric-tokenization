# Common Utilities

| File | Purpose |
| --- | --- |
| `README.md` | This guide to shared utility files. |
| `__init__.py` | Makes `code.common` importable. |
| `utils.py` | Provides seed control, PyTorch data-worker seeding, and `validate_plan`, the VAL subprocess wrapper used during inference. |
| `fsf_wrapper.py` | Compatibility wrapper for fixed-size factored encodings because inference modules import `FSFEncoder` for optional fixed-size encoding support. It is not one of the tokenizer families in this study. |
| `wl_wrapper.py` | WL helper integration around `wlplan`, including trajectory parsing and state embedding support for WL-style graph features. |

The tokenizer comparison uses the implementations in `code/tokenization/`; these common files provide shared runtime services rather than defining the experimental factor levels.
