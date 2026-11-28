//! `strata-feed --strata-protocol 1 validate <path>`: native-helper contract of `strata_control/native_helpers.py`.
//! Any other command line exits 2 with nothing on stdout.

use std::env;
use std::fs;
use std::process;

const PROTOCOL: &str = "1";

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.len() != 4 || args[0] != "--strata-protocol" || args[1] != PROTOCOL || args[2] != "validate" {
        eprintln!("usage: strata-feed --strata-protocol 1 validate <feed.json>");
        process::exit(2);
    }
    let buf = match fs::read(&args[3]) {
        Ok(b) => b,
        Err(e) => {
            println!("{{\"protocol\":1,\"ok\":false,\"error\":{{\"code\":5,\"message\":{}}}}}", json_escape(&e.to_string()));
            return;
        }
    };
    match strata_feed::validate_feed_bytes(&buf) {
        Ok(()) => println!("{{\"protocol\":1,\"ok\":true,\"result\":{{\"valid\":true}}}}"),
        Err(msg) => println!("{{\"protocol\":1,\"ok\":false,\"error\":{{\"code\":5,\"message\":{}}}}}", json_escape(&msg)),
    }
}
