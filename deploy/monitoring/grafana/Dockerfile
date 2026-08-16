# Bakes the provisioning tree into the image at build time instead of
# relying on a Coolify bind mount -- see prometheus/Dockerfile's comment
# for the full explanation (Coolify doesn't sync git-repo files into
# bind-mount sources; a build-time COPY has no such gap). Confirmed safe:
# grafana/grafana:11.6.0's own /etc/grafana/provisioning/ subdirectories
# (alerting/, datasources/, dashboards/, plugins/, notifiers/,
# access-control/) are all empty placeholders, nothing this COPY clobbers.
FROM grafana/grafana:11.6.0
COPY provisioning /etc/grafana/provisioning
