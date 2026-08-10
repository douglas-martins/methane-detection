# CHANGELOG

<!-- version list -->

## v0.6.0 (2026-08-10)

### Bug Fixes

- **ci**: Fall back to PR's own commitlint config when base commit lacks one
  ([`2ed71b9`](https://github.com/douglas-martins/methane-detection/commit/2ed71b99eefefef13d74f5493e482a5baff9ff19))

- **ci**: Harden commitlint checkout, push linting, and config file
  ([`c20b2d4`](https://github.com/douglas-martins/methane-detection/commit/c20b2d4b663c5a9e32c4e52cfb2ce5719d70d929))

- **ci**: Install commitlint dependencies from the trusted base commit's manifests
  ([`f4ecf69`](https://github.com/douglas-martins/methane-detection/commit/f4ecf69a80ba1896c2f433e6f8c4f304b079d92c))

- **ci**: Load commitlint config from the PR's base commit, not the merge ref and pin commitlint
  ([`444cbb6`](https://github.com/douglas-martins/methane-detection/commit/444cbb6af35ee1e48398c725aa15065c4b702dbe))

- **ci**: Pin commitlint dependencies via a committed lockfile
  ([`02ed9ea`](https://github.com/douglas-martins/methane-detection/commit/02ed9ead2dcb66ff9a5033a5ff5d51b8ab3a6464))

- **ci**: Skip commitlint validation on branch-deletion push events
  ([`1620b68`](https://github.com/douglas-martins/methane-detection/commit/1620b68e82a8f09c82948423ee7467b67f56a3fb))

- **ci**: Use an unpredictable temp file and EXIT trap for the base commitlint config
  ([`14336ad`](https://github.com/douglas-martins/methane-detection/commit/14336addaf33d0b5f58df80a06d5dd2b115c6348))

- **data**: Catch scene ambiguity between flat and nested layouts; clear stale missing-scenes report
  ([`a1fb841`](https://github.com/douglas-martins/methane-detection/commit/a1fb841f7330537f8a9144061c2edab3e941d402))

- **data**: Exclude non-directory matches from nested scene-folder glob
  ([`7fd94c8`](https://github.com/douglas-martins/methane-detection/commit/7fd94c8fb630687d6f69083827407e541cc6dc05))

- **data**: Reject path-traversal and absolute scene ids
  ([`7d7501e`](https://github.com/douglas-martins/methane-detection/commit/7d7501e00e58fdd269df18b38ce1e236385f1893))

### Features

- **ci**: Add commitlint enforcement for conventional commit messages
  ([`7f23097`](https://github.com/douglas-martins/methane-detection/commit/7f23097a29dc3cefd91a81e1d1145ab6ece52410))

- **data**: Add coordinates stage for per-scene geographic metadata
  ([`275989e`](https://github.com/douglas-martins/methane-detection/commit/275989e938ff68583b6a0a408c72b360764591b3))

- **data**: Compute pixel-level class distribution in stats stage
  ([`e4b518a`](https://github.com/douglas-martins/methane-detection/commit/e4b518ac6b41b39121ceaa098e4f99a5ad29ce25))

- **data**: Discover starcop_raw scenes from CSV manifest instead of directory listing
  ([`4902461`](https://github.com/douglas-martins/methane-detection/commit/49024619cf8f846327376b9c389120255bf9e723))

- **data**: Generate per-dataset DVC stages via foreach template
  ([`1cfb230`](https://github.com/douglas-martins/methane-detection/commit/1cfb230a96a0ee6786a75d9d432d1b0a590585ea))

- **data**: Parameterize preprocessing config for multi-dataset support
  ([`3962708`](https://github.com/douglas-martins/methane-detection/commit/396270892ee2a3ad53e62794f965873aba521b13))

- **data**: Wire configurable num_workers into patch_extract stage
  ([`683c514`](https://github.com/douglas-martins/methane-detection/commit/683c514d89be638653ca60d7c2504f4858d035fc))

### Performance Improvements

- **data**: Compute band stats incrementally to avoid OOM at starcop_raw scale
  ([`176ae83`](https://github.com/douglas-martins/methane-detection/commit/176ae83bf6ba03f0542e154a51e34e8360948679))

### Refactoring

- **test**: Avoid shadowing builtin id in fake_download
  ([`03d703d`](https://github.com/douglas-martins/methane-detection/commit/03d703d70c8c0c2ddf2ff16fb366882b375030fc))


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
