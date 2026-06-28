# Soccer Observability MVP

The soccer engine now has an opt-in telemetry envelope for long-running binaries:

- JSON structured logs to stdout for Kubernetes log collection.
- OpenTelemetry trace export over OTLP/HTTP.
- Collector-side Kubernetes metadata enrichment, attribute redaction, trace sampling, batching, retry, and disk-backed queues.
- CockroachDB TTL tables for recent high-value log slices and run/error events.

The design intentionally keeps raw logs out of CockroachDB by default. Cockroach stores recent indexed operational slices; a log backend or object storage should hold the raw stream.

## Rust Runtime

Call `soccer_engine::telemetry::init_soccer_telemetry(service_name)` once at process start. The following binaries already do this:

- `main_soccer_learning_run`
- `main_soccer_learning_queue`
- `main_soccer_learning_server`
- `main_soccer_tournament_run`
- `main_soccer_live`
- `main_soccer_live_5056`

Runtime env:

| Variable | Default | Purpose |
| --- | --- | --- |
| `SOCCER_TELEMETRY_ENABLED` | auto | Enables the tracing subscriber when true, or when JSON/OTLP/filter env vars imply telemetry. |
| `SOCCER_LOG_JSON` | `false` | Emits tracing events as newline-delimited JSON to stdout. |
| `SOCCER_RUST_LOG` | `info,soccer_engine=info` | Tracing filter, preferred over `RUST_LOG` for soccer jobs. |
| `SOCCER_SERVICE_NAME` | binary-specific | Overrides the service name sent to logs and OTel. |
| `SOCCER_CLUSTER_NAME` | unset/local | Adds the cluster name to logs and OTel resource attributes. |
| `SOCCER_OTEL_TRACES` | endpoint-driven | Enables OTLP trace export. |
| `SOCCER_OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` | `http://127.0.0.1:4318/v1/traces` | OTLP/HTTP trace endpoint. |

The existing stdout progress lines stay in place so scripts and old runbooks keep working.

## Kubernetes Collector

Apply `k8s/soccer-otel-collector.yaml` before enabling app telemetry in cluster jobs.

The collector:

- reads pod logs from `/var/log/pods/default_*/*/*.log`;
- receives app OTLP on `4317` and `4318`;
- enriches telemetry with Kubernetes pod/container/node metadata;
- deletes common sensitive attributes such as authorization headers and DB statements;
- samples traces with `SOCCER_OTEL_TRACE_SAMPLE_PERCENT`;
- batches export traffic; and
- uses `/var/cache/dd-soccer-otel-collector` as a hostPath-backed persistent queue.

By default it exports to:

```text
http://dd-soccer-otel-gateway.default.svc.cluster.local:4318
```

Point that service at the real upstream collector, log backend, or gateway. If the gateway is temporarily unavailable, the collector queues and retries instead of failing the soccer workload.

## CockroachDB TTL Slice

Apply `docs/sql/soccer_observability_cockroach.sql` to create:

- `soccer_observability.otel_log_slices`, 30-day TTL
- `soccer_observability.otel_run_events`, 90-day TTL
- `soccer_observability.otel_error_events`, 180-day TTL

Use a small sink/gateway to write selected warning/error/run events into these tables from OTLP or from the raw log backend. Keep table indexes lean; Cockroach TTL jobs delete rows in the background and should not carry the entire debug stream.
