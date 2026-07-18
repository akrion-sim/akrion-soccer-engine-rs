use std::{
    collections::HashMap,
    env,
    sync::{Mutex, OnceLock},
    time::{Duration, Instant},
};

use opentelemetry::{
    global,
    metrics::{Counter, Histogram},
    trace::TracerProvider as _,
    KeyValue,
};
use opentelemetry_otlp::{Protocol, WithExportConfig};
use opentelemetry_sdk::{metrics::SdkMeterProvider, resource::Resource, trace::SdkTracerProvider};
use tracing_subscriber::prelude::*;
use tracing_subscriber::EnvFilter;

const DEFAULT_FILTER: &str = "info,soccer_engine=info";
const DEFAULT_OTLP_HTTP_BASE: &str = "http://127.0.0.1:4318";

pub struct SoccerTelemetryGuard {
    tracer_provider: Option<SdkTracerProvider>,
    meter_provider: Option<SdkMeterProvider>,
}

impl Drop for SoccerTelemetryGuard {
    fn drop(&mut self) {
        if let Some(provider) = self.meter_provider.take() {
            let _ = provider.shutdown();
        }
        if let Some(provider) = self.tracer_provider.take() {
            let _ = provider.shutdown();
        }
    }
}

pub fn init_soccer_telemetry(default_service_name: &'static str) -> SoccerTelemetryGuard {
    let config = SoccerTelemetryConfig::from_env(default_service_name);
    if !config.enabled {
        return SoccerTelemetryGuard {
            tracer_provider: None,
            meter_provider: None,
        };
    }

    let filter = telemetry_filter();
    let resource = telemetry_resource(&config.service_name);
    let tracer_provider = config.traces_endpoint.as_deref().and_then(|endpoint| {
        build_tracer_provider(endpoint, resource.clone())
            .map_err(|err| {
                eprintln!(
                    "soccer_telemetry_trace_init_failed service={} error={err}",
                    config.service_name
                );
            })
            .ok()
    });
    let meter_provider = config.metrics_endpoint.as_deref().and_then(|endpoint| {
        build_meter_provider(endpoint, resource)
            .map_err(|err| {
                eprintln!(
                    "soccer_telemetry_metric_init_failed service={} error={err}",
                    config.service_name
                );
            })
            .ok()
    });
    if let Some(provider) = meter_provider.as_ref() {
        global::set_meter_provider(provider.clone());
    }

    let init_result = match (&tracer_provider, config.json_logs) {
        (Some(provider), true) => {
            let tracer = provider.tracer(config.service_name.clone());
            tracing_subscriber::registry()
                .with(filter)
                .with(json_log_layer())
                .with(tracing_opentelemetry::layer().with_tracer(tracer))
                .try_init()
        }
        (Some(provider), false) => {
            let tracer = provider.tracer(config.service_name.clone());
            tracing_subscriber::registry()
                .with(filter)
                .with(compact_log_layer())
                .with(tracing_opentelemetry::layer().with_tracer(tracer))
                .try_init()
        }
        (None, true) => tracing_subscriber::registry()
            .with(filter)
            .with(json_log_layer())
            .try_init(),
        (None, false) => tracing_subscriber::registry()
            .with(filter)
            .with(compact_log_layer())
            .try_init(),
    };

    if let Err(err) = init_result {
        eprintln!(
            "soccer_telemetry_subscriber_init_failed service={} error={err}",
            config.service_name
        );
    }

    tracing::info!(
        event = "soccer_telemetry_initialized",
        service.name = %config.service_name,
        log.format = if config.json_logs { "json" } else { "compact" },
        otel.traces = tracer_provider.is_some(),
        otel.metrics = meter_provider.is_some(),
        log.destination = "stdout",
    );

    SoccerTelemetryGuard {
        tracer_provider,
        meter_provider,
    }
}

pub fn emit_process_start(service_name: &str) {
    process_metrics().start(service_name);
    tracing::info!(
        event = "soccer_process_start",
        service_name = service_name,
        cluster = telemetry_env("SOCCER_CLUSTER_NAME")
            .or_else(|| telemetry_env("OTEL_RESOURCE_ATTRIBUTES_CLUSTER"))
            .unwrap_or_else(|| "local".to_string()),
        pod = telemetry_env("HOSTNAME").unwrap_or_else(|| "local".to_string()),
        run_id = telemetry_env("SOCCER_RUN_ID").unwrap_or_else(|| "unset".to_string()),
        source_commit =
            telemetry_env("SOCCER_SOURCE_COMMIT").unwrap_or_else(|| "unknown".to_string())
    );
}

pub fn emit_process_error(service_name: &str, error: &str) {
    process_metrics().finish(service_name, "error");
    tracing::error!(
        event = "soccer_process_error",
        service_name = service_name,
        error = error
    );
}

pub fn emit_process_complete(service_name: &str) {
    process_metrics().finish(service_name, "ok");
    tracing::info!(
        event = "soccer_process_complete",
        service_name = service_name
    );
}

struct ProcessMetrics {
    starts: Counter<u64>,
    completions: Counter<u64>,
    duration: Histogram<f64>,
    started_at: Mutex<HashMap<String, Instant>>,
}

impl ProcessMetrics {
    fn start(&self, service_name: &str) {
        let attributes = [KeyValue::new("service.name", service_name.to_string())];
        self.starts.add(1, &attributes);
        self.started_at
            .lock()
            .expect("soccer process metric lock")
            .insert(service_name.to_string(), Instant::now());
    }

    fn finish(&self, service_name: &str, outcome: &'static str) {
        let attributes = [
            KeyValue::new("service.name", service_name.to_string()),
            KeyValue::new("process.outcome", outcome),
        ];
        self.completions.add(1, &attributes);
        if let Some(started) = self
            .started_at
            .lock()
            .expect("soccer process metric lock")
            .remove(service_name)
        {
            self.duration
                .record(started.elapsed().as_secs_f64(), &attributes);
        }
    }
}

fn process_metrics() -> &'static ProcessMetrics {
    static METRICS: OnceLock<ProcessMetrics> = OnceLock::new();
    METRICS.get_or_init(|| {
        let meter = global::meter("soccer_engine/process");
        ProcessMetrics {
            starts: meter
                .u64_counter("process.execution.started")
                .with_description("Started soccer service or job executions")
                .with_unit("{execution}")
                .build(),
            completions: meter
                .u64_counter("process.execution.completed")
                .with_description("Completed soccer service or job executions")
                .with_unit("{execution}")
                .build(),
            duration: meter
                .f64_histogram("process.execution.duration")
                .with_description("Soccer service or job execution duration")
                .with_unit("s")
                .build(),
            started_at: Mutex::new(HashMap::new()),
        }
    })
}

struct SoccerTelemetryConfig {
    enabled: bool,
    json_logs: bool,
    service_name: String,
    traces_endpoint: Option<String>,
    metrics_endpoint: Option<String>,
}

impl SoccerTelemetryConfig {
    fn from_env(default_service_name: &'static str) -> Self {
        let generic_endpoint = telemetry_env("SOCCER_OTEL_EXPORTER_OTLP_ENDPOINT")
            .or_else(|| telemetry_env("OTEL_EXPORTER_OTLP_ENDPOINT"));
        let endpoint_configured = generic_endpoint.is_some()
            || env::var_os("SOCCER_OTEL_EXPORTER_OTLP_TRACES_ENDPOINT").is_some()
            || env::var_os("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT").is_some()
            || env::var_os("SOCCER_OTEL_EXPORTER_OTLP_METRICS_ENDPOINT").is_some()
            || env::var_os("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT").is_some();
        let base = generic_endpoint
            .as_deref()
            .unwrap_or(DEFAULT_OTLP_HTTP_BASE);
        let json_logs = env_bool("SOCCER_LOG_JSON", false);
        let traces_enabled = env_bool("SOCCER_OTEL_TRACES", endpoint_configured);
        let metrics_enabled = env_bool("SOCCER_OTEL_METRICS", endpoint_configured);
        let enabled = env_bool(
            "SOCCER_TELEMETRY_ENABLED",
            json_logs
                || traces_enabled
                || metrics_enabled
                || env::var_os("SOCCER_RUST_LOG").is_some(),
        );
        let service_name = telemetry_env("SOCCER_SERVICE_NAME")
            .unwrap_or_else(|| default_service_name.to_string());

        Self {
            enabled,
            json_logs,
            service_name,
            traces_endpoint: traces_enabled.then(|| {
                telemetry_env("SOCCER_OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
                    .or_else(|| telemetry_env("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"))
                    .unwrap_or_else(|| signal_endpoint(base, "traces"))
            }),
            metrics_endpoint: metrics_enabled.then(|| {
                telemetry_env("SOCCER_OTEL_EXPORTER_OTLP_METRICS_ENDPOINT")
                    .or_else(|| telemetry_env("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT"))
                    .unwrap_or_else(|| signal_endpoint(base, "metrics"))
            }),
        }
    }
}

fn build_tracer_provider(
    endpoint: &str,
    resource: Resource,
) -> Result<SdkTracerProvider, Box<dyn std::error::Error + Send + Sync + 'static>> {
    let exporter = opentelemetry_otlp::SpanExporter::builder()
        .with_http()
        .with_protocol(Protocol::HttpBinary)
        .with_endpoint(endpoint)
        .with_timeout(Duration::from_secs(3))
        .build()?;
    Ok(SdkTracerProvider::builder()
        .with_resource(resource)
        .with_batch_exporter(exporter)
        .build())
}

fn build_meter_provider(
    endpoint: &str,
    resource: Resource,
) -> Result<SdkMeterProvider, Box<dyn std::error::Error + Send + Sync + 'static>> {
    let exporter = opentelemetry_otlp::MetricExporter::builder()
        .with_http()
        .with_protocol(Protocol::HttpBinary)
        .with_endpoint(endpoint)
        .with_timeout(Duration::from_secs(3))
        .build()?;
    Ok(SdkMeterProvider::builder()
        .with_resource(resource)
        .with_periodic_exporter(exporter)
        .build())
}

fn telemetry_filter() -> EnvFilter {
    EnvFilter::try_from_env("SOCCER_RUST_LOG")
        .or_else(|_| EnvFilter::try_from_default_env())
        .unwrap_or_else(|_| EnvFilter::new(DEFAULT_FILTER))
}

fn telemetry_resource(service_name: &str) -> Resource {
    let mut attributes = vec![
        KeyValue::new("service.namespace", "akrion"),
        KeyValue::new("service.version", env!("CARGO_PKG_VERSION")),
    ];
    push_env_attribute(&mut attributes, "SOCCER_CLUSTER_NAME", "k8s.cluster.name");
    push_env_attribute(&mut attributes, "POD_NAMESPACE", "k8s.namespace.name");
    push_env_attribute(&mut attributes, "HOSTNAME", "k8s.pod.name");
    push_env_attribute(&mut attributes, "NODE_NAME", "k8s.node.name");
    push_env_attribute(&mut attributes, "SOCCER_RUN_ID", "soccer.run_id");
    push_env_attribute(
        &mut attributes,
        "SOCCER_SOURCE_COMMIT",
        "service.instance.revision",
    );

    Resource::builder()
        .with_service_name(service_name.to_string())
        .with_attributes(attributes)
        .build()
}

fn push_env_attribute(attributes: &mut Vec<KeyValue>, env_name: &str, key: &'static str) {
    if let Some(value) = telemetry_env(env_name) {
        attributes.push(KeyValue::new(key, value));
    }
}

fn signal_endpoint(base: &str, signal: &str) -> String {
    format!("{}/v1/{signal}", base.trim_end_matches('/'))
}

fn json_log_layer<S>() -> tracing_subscriber::fmt::Layer<
    S,
    tracing_subscriber::fmt::format::JsonFields,
    tracing_subscriber::fmt::format::Format<tracing_subscriber::fmt::format::Json>,
>
where
    S: tracing::Subscriber + for<'a> tracing_subscriber::registry::LookupSpan<'a>,
{
    tracing_subscriber::fmt::layer()
        .json()
        .flatten_event(true)
        .with_ansi(false)
        .with_current_span(true)
        .with_span_list(true)
        .with_target(true)
        .with_thread_ids(false)
}

fn compact_log_layer<S>() -> tracing_subscriber::fmt::Layer<
    S,
    tracing_subscriber::fmt::format::DefaultFields,
    tracing_subscriber::fmt::format::Format<tracing_subscriber::fmt::format::Compact>,
>
where
    S: tracing::Subscriber + for<'a> tracing_subscriber::registry::LookupSpan<'a>,
{
    tracing_subscriber::fmt::layer()
        .compact()
        .with_ansi(false)
        .with_target(true)
        .with_thread_ids(false)
}

fn telemetry_env(name: &str) -> Option<String> {
    env::var(name)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn env_bool(name: &str, default: bool) -> bool {
    match telemetry_env(name) {
        Some(value) => matches!(
            value.to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        ),
        None => default,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generic_otlp_endpoint_gets_signal_path() {
        assert_eq!(
            signal_endpoint("http://collector:4318/", "metrics"),
            "http://collector:4318/v1/metrics"
        );
    }
}
