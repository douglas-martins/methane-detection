# DVC Setup — Google Drive Remote

> Requirements for a contributor to pull/push this project's DVC-tracked data.
> This is a checklist of what you need and where it goes — not a walkthrough
> of this project's specific Drive folder or credentials. No real IDs or
> secrets are embedded here; see D-01 in `mlops-methane-detection-plan.md`
> for why Google Drive was chosen over Backblaze B2, and
> `docs/dvc-remote-comparison.html` for the full trade-off analysis.

## Why a personal OAuth client

The DVC `gdrive` remote's default OAuth app is shared across every DVC user
on the internet and shares its rate-limit quota accordingly. Each
contributor authenticates with their **own** Google Cloud OAuth client
instead, so pushes/pulls run against a dedicated per-user quota rather than
that shared pool.

## What you need

1. A **Google Cloud project** with the **Google Drive API** enabled
   (Google Cloud Console → APIs & Services → Library → "Google Drive API" →
   Enable).
2. An **OAuth 2.0 Client ID**, type **Desktop app**, created under that
   project's credentials (APIs & Services → Credentials → Create Credentials
   → OAuth client ID → Desktop app). Download the resulting
   `client_secret*.json` — you'll need the `client_id` and `client_secret`
   values from inside it, not the file itself.
3. Access to the **target Drive folder ID** this project's remote points at
   (ask the maintainer, or point at your own folder if standing up a
   separate remote for local experimentation — see below).
4. The OAuth consent screen for your Cloud project will be in **Testing**
   publishing status by default. The `drive` scope DVC requests is
   Google-classified as sensitive, so **you must add your own Google
   account as a Test user** under the consent screen's Test users section —
   otherwise the first auth attempt fails with an "app has not completed
   verification" error, even though it's your own client.

## Where credentials go

Never commit `client_secret*.json`, and never put the client ID/secret in
`.dvc/config` (that file is tracked by Git). They go into
`.dvc/config.local` instead, which is git-ignored by DVC's own
`.dvc/.gitignore`:

```bash
dvc remote modify --local gdrive gdrive_client_id <your-client-id>
dvc remote modify --local gdrive gdrive_client_secret <your-client-secret>
```

Before committing anything, double-check `.dvc/config.local` doesn't show up
in `git status` — it shouldn't, but it's worth a quick look given it holds
a secret.

## First pull

```bash
dvc pull
```

The first run opens a browser window for Google's OAuth consent flow
against your own client. Approve it (you'll likely see an "unverified app"
warning, expected for a personal, unpublished client) and DVC caches the
resulting token locally so subsequent `dvc pull`/`dvc push` calls don't
re-prompt. 

> Note: while the consent screen stays in Testing status, Google
> expires test-user refresh tokens after about 7 days, so re-authenticating
> periodically is expected — this isn't a bug.

## Everyday usage

```bash
dvc pull    # fetch the latest tracked data into your workspace
dvc push    # upload changes you've tracked with `dvc add`
dvc status  # check what's out of sync
```
