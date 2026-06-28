CREATE DATABASE IF NOT EXISTS soccer_observability;

CREATE TABLE IF NOT EXISTS soccer_observability.otel_log_slices (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  observed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  expires_at TIMESTAMPTZ NOT NULL DEFAULT now() + INTERVAL '30 days',
  cluster_name STRING NOT NULL,
  namespace_name STRING,
  pod_name STRING,
  container_name STRING,
  service_name STRING NOT NULL,
  severity STRING NOT NULL,
  event_name STRING,
  trace_id STRING,
  span_id STRING,
  run_id STRING,
  tournament_id UUID,
  match_id UUID,
  source_commit STRING,
  body STRING,
  attributes JSONB NOT NULL DEFAULT '{}'::JSONB,
  resource JSONB NOT NULL DEFAULT '{}'::JSONB,
  INDEX otel_log_slices_recent_idx (observed_at DESC, cluster_name, service_name, severity),
  INDEX otel_log_slices_trace_idx (trace_id, observed_at DESC) WHERE trace_id IS NOT NULL,
  INDEX otel_log_slices_run_idx (run_id, observed_at DESC) WHERE run_id IS NOT NULL
) WITH (
  ttl_expiration_expression = 'expires_at',
  ttl_job_cron = '0 * * * *'
);

CREATE TABLE IF NOT EXISTS soccer_observability.otel_run_events (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  observed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  expires_at TIMESTAMPTZ NOT NULL DEFAULT now() + INTERVAL '90 days',
  cluster_name STRING NOT NULL,
  service_name STRING NOT NULL,
  run_id STRING NOT NULL,
  source_commit STRING,
  event_name STRING NOT NULL,
  status STRING,
  duration_ms INT8,
  metrics JSONB NOT NULL DEFAULT '{}'::JSONB,
  attributes JSONB NOT NULL DEFAULT '{}'::JSONB,
  INDEX otel_run_events_recent_idx (observed_at DESC, cluster_name, service_name),
  INDEX otel_run_events_run_idx (run_id, observed_at DESC)
) WITH (
  ttl_expiration_expression = 'expires_at',
  ttl_job_cron = '17 * * * *'
);

CREATE TABLE IF NOT EXISTS soccer_observability.otel_error_events (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  observed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  expires_at TIMESTAMPTZ NOT NULL DEFAULT now() + INTERVAL '180 days',
  cluster_name STRING NOT NULL,
  namespace_name STRING,
  pod_name STRING,
  service_name STRING NOT NULL,
  event_name STRING,
  trace_id STRING,
  span_id STRING,
  run_id STRING,
  source_commit STRING,
  error_class STRING,
  error_message STRING NOT NULL,
  attributes JSONB NOT NULL DEFAULT '{}'::JSONB,
  INDEX otel_error_events_recent_idx (observed_at DESC, cluster_name, service_name),
  INDEX otel_error_events_trace_idx (trace_id, observed_at DESC) WHERE trace_id IS NOT NULL
) WITH (
  ttl_expiration_expression = 'expires_at',
  ttl_job_cron = '31 * * * *'
);
