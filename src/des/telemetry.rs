use std::env;
use std::time::Duration;

use opentelemetry::trace::TracerProvider as _;
use opentelemetry::KeyValue;
use opentelemetry_otlp::{Protocol, WithExportConfig};
use opentelemetry_sdk::resource::Resource;
use opentelemetry_sdk::trace::SdkTracerProvider;
use tracing_subscriber::prelude::*;
use tracing_subscriber::EnvFilter;

const DEFAULT_FILTER: &str = "info,soccer_engine=info";
const DEFAULT_OTLP_HTTP_ENDPOINT: &str = "http://127.0.0.1:4318/v1/traces";

pub struct SoccerTelemetryGuard {
    provider: Option<SdkTracerProvider>,
}

impl Drop for SoccerTelemetryGuard {
    fn drop(&mut self) {
        if let Some(provider) = self.provider.take() {
            let _ = provider.shutdown();
        }
    }
}

pub fn init_soccer_telemetry(default_service_name: &'static str) -> SoccerTelemetryGuard {
    let config = SoccerTelemetryConfig::from_env(default_service_name);
    if !config.enabled {
        return SoccerTelemetryGuard { provider: None };
    }

    let filter = telemetry_filter();
    let provider = if config.otlp_traces {
        match build_tracer_provider(&config) {
            Ok(provider) => Some(provider),
            Err(err) => {
                eprintln!("soccer_telemetry_otlp_init_failed service={} error={err}", config.service_name);
                None
            }
        }
    } else {
        None
    };

    let init_result = match (&provider, config.json_logs) {
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
        eprintln!("soccer_telemetry_subscriber_init_failed service={} error={err}", config.service_name);
    }

    SoccerTelemetryGuard { provider }
}

pub fn emit_process_start(service_name: &str) {
    tracing::info!(
        event = "soccer_process_start",
        service_name = service_name,
        cluster = telemetry_env("SOCCER_CLUSTER_NAME")
            .or_else(|| telemetry_env("OTEL_RESOURCE_ATTRIBUTES_CLUSTER"))
            .unwrap_or_else(|| "local".to_string()),
        pod = telemetry_env("HOSTNAME").unwrap_or_else(|| "local".to_string()),
        run_id = telemetry_env("SOCCER_RUN_ID").unwrap_or_else(|| "unset".to_string()),
        source_commit = telemetry_env("SOCCER_SOURCE_COMMIT").unwrap_or_else(|| "unknown".to_string())
    );
}

pub fn emit_process_error(service_name: &str, error: &str) {
    tracing::error!(
        event = "soccer_process_error",
        service_name = service_name,
        error = error
    );
}

pub fn emit_process_complete(service_name: &str) {
    tracing::info!(
        event = "soccer_process_complete",
        service_name = service_name
    );
}

struct SoccerTelemetryConfig {
    enabled: bool,
    json_logs: bool,
    otlp_traces: bool,
    service_name: String,
    endpoint: String,
}

impl SoccerTelemetryConfig {
    fn from_env(default_service_name: &'static str) -> Self {
        let endpoint = telemetry_env("SOCCER_OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
            .or_else(|| telemetry_env("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"))
            .or_else(|| telemetry_env("SOCCER_OTEL_EXPORTER_OTLP_ENDPOINT"))
            .or_else(|| telemetry_env("OTEL_EXPORTER_OTLP_ENDPOINT"))
            .unwrap_or_else(|| DEFAULT_OTLP_HTTP_ENDPOINT.to_string());
        let explicit_endpoint = endpoint != DEFAULT_OTLP_HTTP_ENDPOINT
            || env::var_os("SOCCER_OTEL_EXPORTER_OTLP_TRACES_ENDPOINT").is_some()
            || env::var_os("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT").is_some()
            || env::var_os("SOCCER_OTEL_EXPORTER_OTLP_ENDPOINT").is_some()
            || env::var_os("OTEL_EXPORTER_OTLP_ENDPOINT").is_some();
        let json_logs = env_bool("SOCCER_LOG_JSON", false);
        let otlp_traces = env_bool("SOCCER_OTEL_TRACES", explicit_endpoint);
        let enabled = env_bool(
            "SOCCER_TELEMETRY_ENABLED",
            json_logs || otlp_traces || env::var_os("SOCCER_RUST_LOG").is_some(),
        );
        let service_name =
            telemetry_env("SOCCER_SERVICE_NAME").unwrap_or_else(|| default_service_name.to_string());

        Self {
            enabled,
            json_logs,
            otlp_traces,
            service_name,
            endpoint,
        }
    }
}

fn build_tracer_provider(
    config: &SoccerTelemetryConfig,
) -> Result<SdkTracerProvider, Box<dyn std::error::Error + Send + Sync + 'static>> {
    let exporter = opentelemetry_otlp::SpanExporter::builder()
        .with_http()
        .with_protocol(Protocol::HttpBinary)
        .with_endpoint(config.endpoint.clone())
        .with_timeout(Duration::from_secs(3))
        .build()?;
    let resource = Resource::builder()
        .with_service_name(config.service_name.clone())
        .with_attributes(telemetry_resource_attributes())
        .build();
    Ok(SdkTracerProvider::builder()
        .with_resource(resource)
        .with_batch_exporter(exporter)
        .build())
}

fn telemetry_filter() -> EnvFilter {
    EnvFilter::try_from_env("SOCCER_RUST_LOG")
        .or_else(|_| EnvFilter::try_from_default_env())
        .unwrap_or_else(|_| EnvFilter::new(DEFAULT_FILTER))
}

fn telemetry_resource_attributes() -> Vec<KeyValue> {
    let mut attrs = Vec::new();
    if let Some(cluster) = telemetry_env("SOCCER_CLUSTER_NAME") {
        attrs.push(KeyValue::new("k8s.cluster.name", cluster));
    }
    if let Some(run_id) = telemetry_env("SOCCER_RUN_ID") {
        attrs.push(KeyValue::new("soccer.run_id", run_id));
    }
    if let Some(commit) = telemetry_env("SOCCER_SOURCE_COMMIT") {
        attrs.push(KeyValue::new("soccer.source_commit", commit));
    }
    attrs
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
        .with_current_span(true)
        .with_span_list(false)
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
