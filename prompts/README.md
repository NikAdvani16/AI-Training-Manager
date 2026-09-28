# Prompt Artifacts

The manager prompts are generated from runtime context, so the executable source
of truth is:

- `manager.py::build_analysis_prompt` for Stage 1;
- `manager.py::build_prompt` for Stage 2.

Regenerate the checked-in examples with:

```bash
python export_prompts.py --condition aggressive --seed 40
```

Every real manager decision also records the exact rendered Stage 1 prompt,
Stage 1 output, Stage 2 prompt(s), raw Stage 2 response(s), and validation result
in `controller_decisions.jsonl`.
