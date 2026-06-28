//! Binary entrypoint for the revived live 2D soccer simulation server on port 5056.

fn main() {
    let service_name = "main_soccer_live_5056";
    let _telemetry = soccer_engine::telemetry::init_soccer_telemetry(service_name);
    soccer_engine::telemetry::emit_process_start(service_name);
    soccer_engine::des::main_soccer_live::run_5056();
    soccer_engine::telemetry::emit_process_complete(service_name);
}
