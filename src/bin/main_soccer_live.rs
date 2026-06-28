//! Binary entrypoint for the live 2D soccer simulation server.

fn main() {
    let service_name = "main_soccer_live";
    let _telemetry = soccer_engine::telemetry::init_soccer_telemetry(service_name);
    soccer_engine::telemetry::emit_process_start(service_name);
    soccer_engine::des::main_soccer_live::run();
    soccer_engine::telemetry::emit_process_complete(service_name);
}
