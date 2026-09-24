# Publication boundary

This repository is intentionally curated for a public engineering profile. The source workspace remains the system of record for private implementation and evaluation material.

## Published

- architecture, design principles, and service boundaries;
- redacted JSON contracts and a non-secret configuration shape;
- a bounded public slice of the Temporal-V5 inference module;
- evidence framing, lifecycle semantics, and evaluation caveats;
- a static presentation page suitable for GitHub Pages.

## Withheld

- source for the private world-model producer, multimodal fusion, training losses, and production alarm head;
- model checkpoints, embeddings, datasets, captured video/audio, and test exports;
- API keys, device IDs, network addresses, cloud configuration, and personal documents;
- generated environments, local caches, experiment runners, and vendor forks.

## Review checklist before every push

1. `git status --short` contains only the showcase tree.
2. `git ls-files` contains no media, weights, archives, `.env`, or personal identifiers.
3. JSON examples contain no real hostnames, tokens, or device IDs.
4. Claims in README and evaluation notes are labeled as internal snapshots where appropriate.
5. The remote owner, repository visibility, and default branch are confirmed by the account holder.
