# CHANGELOG

<!-- version list -->

## v0.5.0 (2026-08-06)

### Bug Fixes

- **ci**: Point Environment A make targets at their post-reorg paths
  ([`c2294e0`](https://github.com/douglas-martins/methane-detection/commit/c2294e02e0bffd11f34afeda9358088a63e69108))

### Features

- **ci**: Add Environment B test job, coverage, and badges
  ([`b17bfdf`](https://github.com/douglas-martins/methane-detection/commit/b17bfdf4e8271450a617512c5793553738399ad3))

- **data**: Add DVC preprocessing pipeline for starcop_mini
  ([`03b6609`](https://github.com/douglas-martins/methane-detection/commit/03b6609a3f8fce580ad7bd72f6e1fae550b169d7))

- **models**: Track STARCOP baseline pretrained checkpoints with DVC
  ([`6162691`](https://github.com/douglas-martins/methane-detection/commit/6162691efaacb31ce3bc06c17989fd4f73c75dce))

### Refactoring

- **data**: Reorganize download_mini_dataset.py
  ([`4c10f59`](https://github.com/douglas-martins/methane-detection/commit/4c10f599d48eedfa465eb778d6c1d4561bbf8e85))


## v0.4.0 (2026-08-04)

### Bug Fixes

- **gitignore**: Stop blanket /data/ and /models/ rule from hiding DVC pointer files
  ([`5286d6b`](https://github.com/douglas-martins/methane-detection/commit/5286d6b0ef4949062b682a5f3ff1fb9e58d7db06))

### Features

- **data**: Initialize DVC with Google Drive remote and track STARCOP datasets
  ([`9752715`](https://github.com/douglas-martins/methane-detection/commit/9752715ca180b1c7dc811e64efd7bb45b9c272ea))


## v0.3.0 (2026-08-04)

### Bug Fixes

- **training**: Patch STARCOP train.py for Lightning 2.x and GCS-upload skip
  ([`95aea88`](https://github.com/douglas-martins/methane-detection/commit/95aea8804a480e43e21affa2df199c8b49337d97))

### Features

- **baseline**: Add local STARCOP baseline validation script and record pretrained metrics
  ([`e41c476`](https://github.com/douglas-martins/methane-detection/commit/e41c4761c1372dbb2290fc42d7c7a319c69f4b0a))

- **data**: Add script to download STARCOP mini demo dataset
  ([`b594a1a`](https://github.com/douglas-martins/methane-detection/commit/b594a1a1d58393e11b237d67589f913ed92265e7))


## v0.2.0 (2026-07-06)

### Features

- **env**: Add pyproject.toml with MLOps environment dependencies
  ([`17702f8`](https://github.com/douglas-martins/methane-detection/commit/17702f809d71bdd3e90e770056d69341bb516f3c))

- **release**: Configure python-semantic-release
  ([`0b47d11`](https://github.com/douglas-martins/methane-detection/commit/0b47d115655d83eb3b07593a3a9c2054340054b5))


## v0.1.0 (2026-07-06)

- Initial Release
