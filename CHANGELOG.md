# CHANGELOG

<!-- version list -->

## v0.15.0 (2026-08-15)

### Bug Fixes

- **cd**: Bound curl timeouts and track elapsed time in the smoke-test loop
  ([`00ce88f`](https://github.com/douglas-martins/methane-detection/commit/00ce88ffc2724befe877b63b952adfce40073a9b))

- **cd**: Poll Coolify's own deployment status instead of trusting the webhook call alone
  ([`f455f3b`](https://github.com/douglas-martins/methane-detection/commit/f455f3b6ee1e605169e5d8f76d75395991a50d4a))

- **cd**: Set persist-credentials: false on checkout
  ([`caccbaa`](https://github.com/douglas-martins/methane-detection/commit/caccbaaa3c8478a4c3148cc2941d80b8e69c8977))

- **cd**: Skip deploy when a release tag has no serving/deploy changes
  ([`0cec0ef`](https://github.com/douglas-martins/methane-detection/commit/0cec0ef5b2f7c6812bd3bf26d12ccc9431d593bf))

- **cd**: Use GITHUB_REF_NAME instead of interpolating github.ref_name
  ([`dcfc8cf`](https://github.com/douglas-martins/methane-detection/commit/dcfc8cf0d08a67b7fcfba6ae64d50332d0080a94))

### Features

- **cd**: Add CD workflow to build, push, and redeploy the inference API
  ([`6a370a6`](https://github.com/douglas-martins/methane-detection/commit/6a370a6e4e4cd42f51bc91b2be64e4efaf1605ad))


## v0.14.2 (2026-08-15)

### Bug Fixes

- **deploy**: Use python3 instead of curl for the bentoml healthcheck
  ([`82569f4`](https://github.com/douglas-martins/methane-detection/commit/82569f428841992bd66ac4c30f4ab9b7060fc6e1))


## v0.14.1 (2026-08-15)

### Bug Fixes

- **deploy**: Force a fresh pull on every bentoml redeploy
  ([`fcb4f63`](https://github.com/douglas-martins/methane-detection/commit/fcb4f63c7258356c2d9508bc8070fae59cae307b))

- **serving**: Add wandb and rasterio to serving requirements
  ([`73647a5`](https://github.com/douglas-martins/methane-detection/commit/73647a5078df13154272793daded6a5c65934c5b))


## v0.14.0 (2026-08-15)

### Bug Fixes

- **registry**: Pin mlflow.pytorch.log_model serialization_format to pickle
  ([`de0713b`](https://github.com/douglas-martins/methane-detection/commit/de0713bade099a834d00ce5619c38640ad76dd8a))

### Features

- **deploy**: Add BentoML Coolify docker-compose stack
  ([`3c93359`](https://github.com/douglas-martins/methane-detection/commit/3c93359c3a076939675945ab72a4ba1232831f74))


## v0.13.0 (2026-08-15)

### Bug Fixes

- **serving**: Load registered models via their recorded artifact path
  ([`dfe5fc8`](https://github.com/douglas-martins/methane-detection/commit/dfe5fc8c45f22f4a3d340197b02718ca96dfac45))

- **serving**: Reject ambiguous or unsafe-dtype arrays in assemble_input_tensor
  ([`e63fb94`](https://github.com/douglas-martins/methane-detection/commit/e63fb9495e6505b0a05bd7c060688324e2db38d9))

- **serving**: Reject complex-dtype arrays in assemble_input_tensor
  ([`62fdb60`](https://github.com/douglas-martins/methane-detection/commit/62fdb606a745d0f7315d69a973ace5a66f792dd5))

- **serving**: Reject non-array .npy uploads (e.g. .npz archives) with a 400
  ([`9418bda`](https://github.com/douglas-martins/methane-detection/commit/9418bdab9d2403223a5b54c5abe651fa5befe61f))

### Features

- **registry**: Add resolve_stage_version to look up the model version at a given stage
  ([`e80aa73`](https://github.com/douglas-martins/methane-detection/commit/e80aa73bd13aa1a25edb3b5ce4c86418eb277c0f))

- **serving**: Add BentoML inference API for the STARCOP segmentation model
  ([`45e3292`](https://github.com/douglas-martins/methane-detection/commit/45e3292a07bbe5d226818198c00cb40ef8dce3e2))


## v0.12.0 (2026-08-13)

### Bug Fixes

- **registry**: Pin checkpoint digest and download revision in HF baseline import
  ([`0405907`](https://github.com/douglas-martins/methane-detection/commit/0405907bf98c265f5e205f5ee0ede6fc2b9f84a4))

- **registry**: Pin HuggingFace revision to a reviewed commit instead of a live lookup
  ([`5bfd9f9`](https://github.com/douglas-martins/methane-detection/commit/5bfd9f9bfc8b5adfff7d5b08da80a53660dc4363))

### Features

- **registry**: Add HuggingFace STARCOP baseline import into MLflow
  ([`8ffb764`](https://github.com/douglas-martins/methane-detection/commit/8ffb764ce30130211a08e1d55ec28d91fcbaf6c5))


## v0.11.0 (2026-08-13)

### Features

- **training**: Add launch_profiles module for per-machine training CLI args
  ([`e27b8b3`](https://github.com/douglas-martins/methane-detection/commit/e27b8b336e04979b0027d263e58c6dd5317095d9))

- **training**: Add train_mac.sh launch script for M4 Pro MPS training
  ([`81da3a6`](https://github.com/douglas-martins/methane-detection/commit/81da3a6a822cd9eeea6c2193a2eacb4a267bfde3))


## v0.10.0 (2026-08-12)

### Bug Fixes

- **env-a**: Pin mlflow<3.7 to avoid torch.export ModuleNotFoundError under torch 1.13.1
  ([`e37daba`](https://github.com/douglas-martins/methane-detection/commit/e37daba5aed1ad060c8458a5f31fa814f13acf2c))

### Features

- **training**: Validate and enable MPS acceleration on Apple Silicon (M4 Pro)
  ([`75b672a`](https://github.com/douglas-martins/methane-detection/commit/75b672a9f745c8fd90af1ecbd7b033318b07f5c0))


## v0.9.1 (2026-08-11)

### Bug Fixes

- **ci**: Reset .npmrc to the trusted base commit before installing commitlint deps
  ([`8533a9c`](https://github.com/douglas-martins/methane-detection/commit/8533a9c4885d385a3e083294f97c576da3ed8202))

- **ci**: Reset .npmrc to the trusted base commit before installing commitlint deps
  ([`ecabe50`](https://github.com/douglas-martins/methane-detection/commit/ecabe50e719850545801d9ca25e198b3477612f5))

- **ci**: Retry badge-commit push after rebase to survive job races
  ([`80c7e73`](https://github.com/douglas-martins/methane-detection/commit/80c7e7319e468e7dc30c63058207fae76d872592))

- **ci**: Retry badge-commit push after rebase to survive job races
  ([`b54275d`](https://github.com/douglas-martins/methane-detection/commit/b54275db3e2c2029f8276e8f1a10a1c21b6302b6))

- **ci**: Serialize Release and Tests push-to-main runs to stop the git-push race
  ([`831681f`](https://github.com/douglas-martins/methane-detection/commit/831681f7fbad82ff4a6d7a2645a3e2ecf902e2fb))

- **release**: Skip CI on the semantic-release version-bump commit
  ([`adeb30c`](https://github.com/douglas-martins/methane-detection/commit/adeb30c793ffb504b97ee6a712f96db0d8755cb6))


## v0.9.0 (2026-08-11)

### Bug Fixes

- **registry**: Reject non-finite metrics and under-length loss histories
  ([`98a0184`](https://github.com/douglas-martins/methane-detection/commit/98a01844bccaa68bbc27d1f65a3eb6b2476fc991))

- **registry**: Reuse existing model version on repeat registration
  ([`b42e471`](https://github.com/douglas-martins/methane-detection/commit/b42e471d15abab8737b44562d2c4fda6804ef2f5))

### Features

- **registry**: Add MLflow model registry promotion workflow
  ([`7303ac0`](https://github.com/douglas-martins/methane-detection/commit/7303ac054532d1a2552dfdfe347b8ba43f20784c))

- **training**: Log held-out test-set metrics and pytorch model artifact to MLflow
  ([`de40934`](https://github.com/douglas-martins/methane-detection/commit/de409341fb9b3053679ce48bc03f0cbe585b8a52))


## v0.8.0 (2026-08-11)

### Bug Fixes

- **env-a**: Pin setuptools<81 and numpy<2 for pytorch_lightning/wandb imports
  ([`6e3c012`](https://github.com/douglas-martins/methane-detection/commit/6e3c01298dbd95225b331b6114dd806481c2e5f9))

- **training**: Resume training from last.ckpt instead of the checkpoint directory
  ([`300268c`](https://github.com/douglas-martins/methane-detection/commit/300268ce2065dedd1084d2d36b4c07d51234a63a))

- **training**: Validate MLflow tracking env vars before the first MLflow call
  ([`39f487d`](https://github.com/douglas-martins/methane-detection/commit/39f487d05e124eba79f986646a93080bf6373b8c))

### Build System

- **deps**: Add interrogate for docstring coverage checks
  ([`2495f7c`](https://github.com/douglas-martins/methane-detection/commit/2495f7c23ca138a7b21dd194a8e8cdf6ae26e0c7))

- **deps**: Add MLflow + boto3 client dependencies for Environment A
  ([`2f38254`](https://github.com/douglas-martins/methane-detection/commit/2f382548ce8a54c2706c188c6894769c37dd96b4))

### Features

- **training**: Add MLflow-tracked STARCOP training entrypoint
  ([`4b9ab57`](https://github.com/douglas-martins/methane-detection/commit/4b9ab57977d86adef50e4360d1daa49bafd9cb02))


## v0.7.0 (2026-08-10)

### Bug Fixes

- **deploy**: Pin mlflow image to 3.14.0 instead of :latest
  ([`45344c9`](https://github.com/douglas-martins/methane-detection/commit/45344c934793bf5525050c744bae7367dfac3dba))

- **deploy**: Pin mlflow image to 3.14.0 instead of :latest
  ([`e92835a`](https://github.com/douglas-martins/methane-detection/commit/e92835a6963cbf1f119019b91f7ad8b6a46ae827))

- **deploy**: Pin mlflow image to 3.14.0 instead of :latest
  ([`302d256`](https://github.com/douglas-martins/methane-detection/commit/302d256c48e2521219e5b7dc6036ee4c8f16339d))

### Features

- **deploy**: Add MLflow Coolify docker-compose stack
  ([`513d12e`](https://github.com/douglas-martins/methane-detection/commit/513d12eed0ab18de510a556e35b3759d0ae600d2))

- **deploy**: Fail fast on placeholder or non-hex secrets in mlflow startup
  ([`dd84de1`](https://github.com/douglas-martins/methane-detection/commit/dd84de10c212adf1c265c061e4c6cc3ad9ecd7b3))


## v0.6.1 (2026-08-10)

### Bug Fixes

- **ci**: Fail build_command if uv lock upgrade fails
  ([`de30aa2`](https://github.com/douglas-martins/methane-detection/commit/de30aa2afce6f1c36665603404c2880ff926e88a))

- **ci**: Keep RELEASE_TOKEN scoped to the git push only
  ([`ca954fc`](https://github.com/douglas-martins/methane-detection/commit/ca954fc291e79fed155077a0730bf0df4305a4b2))

- **ci**: Let badge-commit steps push to protected main
  ([`5e96347`](https://github.com/douglas-martins/methane-detection/commit/5e96347e6c8f55ea588875af3dae037a4712bd15))

- **ci**: Let semantic-release push version bumps to protected main
  ([`0b4ac95`](https://github.com/douglas-martins/methane-detection/commit/0b4ac95cf0b82d7d003179cefe6d004b95c4b4f8))

- **ci**: Make badge-commit push resilient and keep uv.lock in sync
  ([`a586cb1`](https://github.com/douglas-martins/methane-detection/commit/a586cb1ffc836bc73ab2f674bb8a3779751b4eab))

- **ci**: Scope RELEASE_TOKEN to only the badge-commit push
  ([`f6b69ac`](https://github.com/douglas-martins/methane-detection/commit/f6b69ac40319685cd0e193af13766617b0ff42f7))


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
