# Bakes prometheus.yml into the image at build time instead of relying on a
# Coolify bind mount. Confirmed against Coolify's own docs (Docker Compose
# Build Packs / Persistent Storage): Coolify's `volumes:` bind mounts do NOT
# sync files from the git checkout -- a `source: ./path` bind mount is
# either an empty `is_directory: true` placeholder or a literal inline
# `content:` block written into the compose YAML itself, never a copy of
# the referenced repo file. Confirmed live: prometheus.yml never appeared
# on the host, Prometheus crash-looped with "no such file or directory".
# A build-time COPY has no such gap -- one file, one source of truth,
# baked into the image every deploy.
FROM prom/prometheus:v3.5.0
COPY prometheus.yml /etc/prometheus/prometheus.yml
