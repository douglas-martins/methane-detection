# DVC Remote Storage for STARCOP: Google Drive vs Backblaze B2

> Decision D-01 · methane-detection project · evaluated 2026-07-22. Converted from the original `docs/dvc-remote-comparison.html` (styled visual comparison) — table/content preserved, styling dropped. Supports [decisions.md](../decisions.md)'s D-01 entry.

## Recommendation

**Google Drive, with a dedicated Google Cloud OAuth client.**

You already pay for 5TB of Google storage, so the marginal cost of using it as the DVC remote is $0. The rate-limit risk that motivated this comparison is real but is a **shared quota** problem, not a Drive problem — it is solved by registering your own Google Cloud OAuth client, which moves you off the default DVC app's shared pool onto your own dedicated quota. Backblaze B2 stays documented below as the fallback if that mitigation proves insufficient.

## 1. Trade-off comparison

| Criterion | Google Drive | Backblaze B2 |
|---|---|---|
| Marginal cost | ✅ $0 — existing 5TB quota | ⚠️ New paid service |
| Rate limits (default) | 🔴 Shared DVC OAuth pool | ✅ Dedicated per-account |
| Rate limits (with own OAuth client) | ✅ ~20,000 req/100s, dedicated | ✅ Dedicated per-account |
| Throughput / protocol | ⚠️ Per-file API calls, no bulk PUT | ✅ S3-compatible, built for bulk transfer |
| Many-small-files behavior | ⚠️ Known DVC slowdown (hrs for GBs of small files) | ⚠️ Same DVC-side bottleneck, faster backend |
| Setup effort | ⚠️ GCP project + OAuth client + consent screen | ⚠️ B2 account + application key + bucket |
| Native DVC support | ✅ First-class (`dvc[gdrive]`) | ✅ First-class (S3-compatible remote) |
| Egress cost | ✅ Free | ✅ Free up to 3× avg. storage/month |

## 2. Monthly marginal cost as the dataset grows

Google Drive (existing 5TB plan) vs. Backblaze B2 ($0.006/GB/mo, 10GB free):

| Dataset size | Google Drive | Backblaze B2 |
|---|---|---|
| 50GB (mini + patch cache) | $0 | $0.24 |
| 300GB (full raw STARCOP) | $0 | $1.74 |
| 1TB (raw + processed patches + splits) | $0 | $6.14 |

Google Drive cost stays $0 at every size because it draws down storage you already pay for, up to the 5TB cap. Backblaze B2 is inexpensive in absolute terms (~$6/mo at 1TB) but is additive spend on top of the Drive plan you already carry — the constraint this comparison was asked to respect.

## 3. Why the rate limit isn't the blocker it first looked like

The rate-limit worry from D-01 was about the **default** DVC/pydrive2 OAuth app, whose quota is shared across every DVC user on the internet. Registering your own Google Cloud OAuth client (a one-time, free setup: create a GCP project → enable the Drive API → create a Desktop OAuth client ID) puts this project on its own quota — historically ~20,000 requests/100s per project/user, dedicated to you. That removes the main argument for paying for a second storage service.

The residual risk is DVC's own behavior with datasets containing many small files (patch-level hyperspectral tiles): it uploads file-by-file rather than in bulk, which is slow on *any* backend, Drive or B2. B2's S3-compatible API handles this somewhat better since bulk/multipart transfer is native to S3 semantics, but it does not eliminate DVC's per-file overhead.

## 4. When to revisit and move to B2

- **Sustained 403/429 errors** from the Drive API persist even after switching to a dedicated OAuth client.
- **Push/pull of the full raw dataset exceeds ~2–3 hours** on a stable connection, making iteration on preprocessing impractical.
- **5TB quota pressure** — if Drive usage across other tools pushes you close to the cap and DVC's storage need would tip you over.

If any of these trigger, B2 is a drop-in swap: DVC's S3-compatible remote type needs only a new `dvc remote add` pointing at the B2 bucket endpoint plus an application key — no pipeline or code changes.

---

*Sources: Backblaze B2 pricing (backblaze.com/cloud-storage/pricing) · Google Drive API usage limits (developers.google.com/workspace/drive/api/guides/limits) · DVC Google Drive remote docs (doc.dvc.org) · DVC large-dataset performance discussion (GitHub treeverse/dvc #7607, #7681).*
