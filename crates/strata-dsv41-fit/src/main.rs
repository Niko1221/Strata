use std::env;
use std::process::ExitCode;

use strata_dsv41_fit::{place_ista_experts, split_layers};

fn parse_f64_list(s: &str) -> Result<Vec<f64>, String> {
    if s.trim().is_empty() {
        return Ok(Vec::new());
    }
    s.split(',')
        .map(|part| {
            part.trim()
                .parse::<f64>()
                .map_err(|e| format!("invalid float '{part}': {e}"))
        })
        .collect()
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: strata-dsv41-fit <split-layers|place-ista> ...");
        return ExitCode::from(2);
    }
    match args[0].as_str() {
        "split-layers" => {
            let Some(weights_str) = args.get(1) else {
                eprintln!("missing weights");
                return ExitCode::from(2);
            };
            let layers: usize = match args.get(2).map(|s| s.parse()) {
                Some(Ok(v)) => v,
                _ => {
                    eprintln!("invalid layers");
                    return ExitCode::from(2);
                }
            };
            let weights = match parse_f64_list(weights_str) {
                Ok(w) => w,
                Err(e) => {
                    eprintln!("{e}");
                    return ExitCode::from(2);
                }
            };
            let ranges = split_layers(&weights, layers);
            let pairs: Vec<String> = ranges.iter().map(|(a, b)| format!("[{a},{b}]")).collect();
            println!("[{}]", pairs.join(","));
            ExitCode::SUCCESS
        }
        "place-ista" => {
            if args.len() < 11 {
                eprintln!("missing args for place-ista");
                return ExitCode::from(2);
            }
            let vram_gbs = match parse_f64_list(&args[1]) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("{e}");
                    return ExitCode::from(2);
                }
            };
            let Ok(layers) = args[2].parse::<usize>() else { return ExitCode::from(2); };
            let Ok(epl) = args[3].parse::<usize>() else { return ExitCode::from(2); };
            let Ok(experts_gb) = args[4].parse::<f64>() else { return ExitCode::from(2); };
            let Ok(per_dense) = args[5].parse::<f64>() else { return ExitCode::from(2); };
            let Ok(engine_free) = args[6].parse::<f64>() else { return ExitCode::from(2); };
            let Ok(other) = args[7].parse::<f64>() else { return ExitCode::from(2); };
            let Ok(res_factor) = args[8].parse::<f64>() else { return ExitCode::from(2); };
            let amd = args[9] == "1" || args[9].eq_ignore_ascii_case("true");
            let Ok(amd_low) = args[10].parse::<f64>() else { return ExitCode::from(2); };
            let counts = place_ista_experts(
                &vram_gbs,
                layers,
                epl,
                experts_gb,
                per_dense,
                engine_free,
                other,
                res_factor,
                amd,
                amd_low,
            );
            let strs: Vec<String> = counts.iter().map(|c| c.to_string()).collect();
            println!("[{}]", strs.join(","));
            ExitCode::SUCCESS
        }
        other => {
            eprintln!("unknown command: {other}");
            ExitCode::from(2)
        }
    }
}
