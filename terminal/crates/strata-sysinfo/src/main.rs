//! `strata-sysinfo --strata-protocol 1 gather`: native-helper contract of `strata_control/native_helpers.py`.
//! Any other command line exits 2 with nothing on stdout.

use std::env;
use std::process;

const PROTOCOL: &str = "1";

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.len() != 3 || args[0] != "--strata-protocol" || args[1] != PROTOCOL || args[2] != "gather" {
        eprintln!("usage: strata-sysinfo --strata-protocol 1 gather");
        process::exit(2);
    }
    let info = strata_sysinfo::gather();
    println!("{{\"protocol\":1,\"ok\":true,\"result\":{}}}", strata_sysinfo::to_json(&info));
}
