//! `strata-private-files --strata-protocol 1 <command> <args>`: native-helper contract of `strata_control/native_helpers.py`.
//! Any other command line exits 2 with nothing on stdout.

use std::env;
use std::path::PathBuf;
use std::process;

use strata_private_files::{current_user_sid, dacl_sddl, is_unshared, restrict_to_current_user};

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

fn err_out(msg: &str) {
    println!(
        "{{\"protocol\":1,\"ok\":false,\"error\":{{\"code\":5,\"message\":{}}}}}",
        json_escape(msg)
    );
}

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.len() < 3 || args[0] != "--strata-protocol" || args[1] != PROTOCOL {
        eprintln!("usage: strata-private-files --strata-protocol 1 <current-user-sid|dacl-sddl|restrict|is-unshared> ...");
        process::exit(2);
    }

    match args[2].as_str() {
        "current-user-sid" if args.len() == 3 => match current_user_sid() {
            Ok(sid) => println!("{{\"protocol\":1,\"ok\":true,\"result\":{{\"sid\":{}}}}}", json_escape(&sid)),
            Err(err) => err_out(&err.to_string()),
        },
        "dacl-sddl" if args.len() == 4 => match dacl_sddl(&PathBuf::from(&args[3])) {
            Ok(sddl) => println!("{{\"protocol\":1,\"ok\":true,\"result\":{{\"sddl\":{}}}}}", json_escape(&sddl)),
            Err(err) => err_out(&err.to_string()),
        },
        "restrict" if args.len() == 4 || args.len() == 5 => {
            let directory = match args.get(4).map(String::as_str) {
                Some("--dir") => Some(true),
                Some("--file") => Some(false),
                Some(other) => {
                    eprintln!("unknown flag: {other}");
                    process::exit(2);
                }
                None => None,
            };
            match restrict_to_current_user(&PathBuf::from(&args[3]), directory) {
                Ok(()) => println!("{{\"protocol\":1,\"ok\":true,\"result\":{{\"restricted\":true}}}}"),
                Err(err) => err_out(&err.to_string()),
            }
        }
        "is-unshared" if args.len() == 4 => match is_unshared(&PathBuf::from(&args[3])) {
            Ok(val) => println!("{{\"protocol\":1,\"ok\":true,\"result\":{{\"unshared\":{val}}}}}"),
            Err(err) => err_out(&err.to_string()),
        },
        _ => {
            eprintln!("usage: strata-private-files --strata-protocol 1 <current-user-sid|dacl-sddl|restrict|is-unshared> ...");
            process::exit(2);
        }
    }
}
