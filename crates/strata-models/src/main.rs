use std::env;
use std::path::PathBuf;
use std::process::ExitCode;

use strata_models::gguf_header;

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.len() < 2 || args[0] != "gguf-header" {
        eprintln!("usage: strata-models gguf-header <path>");
        return ExitCode::from(2);
    }
    let path = PathBuf::from(&args[1]);
    match gguf_header(&path) {
        Ok(h) => {
            println!("{{\"bytes\":{},\"gguf_version\":{}}}", h.bytes, h.gguf_version);
            ExitCode::SUCCESS
        }
        Err(msg) => {
            eprintln!("{msg}");
            ExitCode::from(1)
        }
    }
}
