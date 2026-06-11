//! Binary entrypoint for the 2D live-match soccer prototype.

fn main() {
    if let Err(err) = soccer_engine::des::main_soccer::try_run() {
        eprintln!("main_soccer: {err}");
        std::process::exit(1);
    }
}
