---
name: read-arxiv-paper
description: Read an arXiv paper from TeX source and write a WhiteHole-focused summary. Use when the user references an arXiv URL or paper ID.
---

# Read arXiv paper

Use this skill when the user provides an arXiv URL/ID and wants a technical summary.

## Core rules

* Always read the **source TeX** (`/src/`), not the PDF.
* Use the helper script: `~/bin/arxiv-src`.
* Write summaries to `docs/paper_summaries/summary_<arxiv-tag>_<short-name>/summary.md`. Use a short title or the technology presented in the paper as `<short-name>`. Put any relevant reference implementation copied from the paper's repository in the same folder.
* Focus the summary on WhiteHole: frozen JEPA/world-model adaptation, visual observation shifts, latent geometry, dynamics preservation, and planning evaluation.

## Workflow

1. Fetch + unpack source:

```bash
~/bin/arxiv-src "<arxiv-url-or-id>"
```

Use the printed fields (`EXTRACT_DIR`, `ENTRYPOINT`).

2. Read the entrypoint `.tex` and recursively follow relevant `\input{...}` / `\include{...}` files.

3. If needed, read related local docs/code for context before writing conclusions.

4. Create summary markdown in `docs/paper_summaries/` with this structure:
   - Problem and core idea
   - Method details (short)
   - Key results
   - What is relevant for WhiteHole
   - Concrete experiments to run next (3-6 bullets)
   - Risks / open questions

--- 

Conditional/Optional last step:
5. If the TeX or user provides a direct repository link, clone it under `/tmp/` and inspect its README and relevant source. Copy or translate only the implementation needed for the current WhiteHole question into the summary folder, then document what was added and how it relates to the paper.

For example, extract a paper's input-adapter mechanism rather than copying its entire training repository.

## Notes

* Do not overwrite an existing summary unless explicitly requested.
