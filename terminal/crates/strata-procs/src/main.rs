//! `strata-procs --strata-protocol 1 <command> <pid>`: native-helper contract of `strata_control/native_helpers.py`.
//! Any other command line exits 2 with nothing on stdout.

use std::env;
use std::process;

use strata_procs::{command_line, exe_path, json_escape, parent_of, start_token};

const PROTOCOL: &str = "1";

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.len() != 4 || args[0] != "--strata-protocol" || args[1] != PROTOCOL {
        eprintln!("usage: strata-procs --strata-protocol 1 <start-token|command-line|parent-of|exe-path> <pid>");
        process::exit(2);
    }
    let cmd = args[2].as_str();
    let pid: u32 = match args[3].parse() {
        Ok(p) if p > 0 => p,
        _ => {
            eprintln!("invalid pid");
            process::exit(2);
        }
    };

    match cmd {
        "start-token" => {
            let val = match start_token(pid) {
                Some(tok) => json_escape(&tok),
                None => "null".to_string(),
            };
            println!("{{\"protocol\":1,\"ok\":true,\"result\":{{\"token\":{val}}}}}");
        }
        "command-line" => {
            let val = match command_line(pid) {
                Some(parts) => {
                    let items: Vec<String> = parts.iter().map(|p| json_escape(p)).collect();
                    format!("[{}]", items.join(","))
                }
                None => "null".to_string(),
            };
            println!("{{\"protocol\":1,\"ok\":true,\"result\":{{\"cmd\":{val}}}}}");
        }
        "parent-of" => {
            let val = match parent_of(pid) {
                Some(ppid) => ppid.to_string(),
                None => "null".to_string(),
            };
            println!("{{\"protocol\":1,\"ok\":true,\"result\":{{\"ppid\":{val}}}}}");
        }
        "exe-path" => {
            let val = match exe_path(pid) {
                Some(p) => json_escape(&p),
                None => "null".to_string(),
            };
            println!("{{\"protocol\":1,\"ok\":true,\"result\":{{\"exe\":{val}}}}}");
        }
        _ => {
            eprintln!("unknown command: {cmd}");
            process::exit(2);
        }
    }
}
