//! `strata-fit --strata-protocol 1 <vram-plan|gpu-fit|gpu-options> <json>`:
//! native-helper contract of `strata_control/native_helpers.py`.
//! Any other command line exits 2 with nothing on stdout.

use std::collections::BTreeMap;
use std::env;
use std::process;

use strata_fit::gpu_fit;
use strata_fit::json::{self, Json};
use strata_fit::vram_fit;

const PROTOCOL: &str = "1";

fn err_out(code: i64, msg: &str) {
    let mut err_obj = BTreeMap::new();
    err_obj.insert("code".to_string(), Json::Int(code));
    err_obj.insert("message".to_string(), Json::Str(msg.to_string()));
    let mut env_obj = BTreeMap::new();
    env_obj.insert("protocol".to_string(), Json::Int(1));
    env_obj.insert("ok".to_string(), Json::Bool(false));
    env_obj.insert("error".to_string(), Json::Obj(err_obj));
    println!("{}", json::stringify(&Json::Obj(env_obj)));
}

fn ok_out(result: Json) {
    let mut env_obj = BTreeMap::new();
    env_obj.insert("protocol".to_string(), Json::Int(1));
    env_obj.insert("ok".to_string(), Json::Bool(true));
    env_obj.insert("result".to_string(), result);
    println!("{}", json::stringify(&Json::Obj(env_obj)));
}

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.len() != 4 || args[0] != "--strata-protocol" || args[1] != PROTOCOL {
        eprintln!("usage: strata-fit --strata-protocol 1 <vram-plan|gpu-fit|gpu-options> <json>");
        process::exit(2);
    }
    let cmd = args[2].as_str();
    if !matches!(cmd, "vram-plan" | "gpu-fit" | "gpu-options") {
        eprintln!("usage: strata-fit --strata-protocol 1 <vram-plan|gpu-fit|gpu-options> <json>");
        process::exit(2);
    }
    let req = match json::parse(&args[3]) {
        Ok(v) if matches!(v, Json::Obj(_)) => v,
        Ok(_) => {
            err_out(5, "payload must be a JSON object");
            return;
        }
        Err(e) => {
            err_out(5, &format!("bad json: {e}"));
            return;
        }
    };

    match cmd {
        "vram-plan" => {
            let s = |k: &str, d: &str| {
                req.get(k)
                    .and_then(|v| v.as_str())
                    .unwrap_or(d)
                    .to_string()
            };
            let f = |k: &str, d: f64| req.get(k).and_then(|v| v.as_f64()).unwrap_or(d);
            let i = |k: &str, d: i64| req.get(k).and_then(|v| v.as_i64()).unwrap_or(d);
            let b = |k: &str, d: bool| req.get(k).and_then(|v| v.as_bool()).unwrap_or(d);
            let res = vram_fit::plan(
                &s("model", ""),
                f("vram_gb", 0.0),
                i("ctx", 8192),
                &s("kv", "int8"),
                i("reserve_mib", 300),
                &s("prefill", "auto"),
                &s("vendor", "intel"),
                &s("title", ""),
                b("reserve_fixed", false),
                b("prefill_fixed", false),
                i("slots", 0),
                i("kv_resident", 0),
                &s("drafter", "mtp"),
                b("kv_fixed", false),
            );
            let mut out = BTreeMap::new();
            out.insert("ctx".to_string(), Json::Int(res.ctx));
            out.insert("reserve_mib".to_string(), Json::Int(res.reserve_mib));
            out.insert(
                "prefill".to_string(),
                res.prefill.map(Json::Str).unwrap_or(Json::Null),
            );
            out.insert("changed".to_string(), Json::Bool(res.changed));
            out.insert("fits".to_string(), Json::Bool(res.fits));
            out.insert("slack_mib".to_string(), Json::Num(res.slack_mib));
            out.insert("line".to_string(), Json::Str(res.line));
            out.insert(
                "changes".to_string(),
                Json::Arr(res.changes.into_iter().map(Json::Str).collect()),
            );
            out.insert(
                "kv_resident".to_string(),
                res.kv_resident.map(Json::Int).unwrap_or(Json::Null),
            );
            out.insert("slots".to_string(), Json::Int(res.slots));
            ok_out(Json::Obj(out));
        }
        "gpu-fit" => {
            let memory_gb = req.get("memory_gb").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let ram_gb = req.get("ram_gb").and_then(|v| v.as_f64());
            let vram_gb = req.get("vram_gb").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let out = gpu_fit::fit(memory_gb, ram_gb, vram_gb);
            ok_out(Json::Obj(out));
        }
        "gpu-options" => {
            let memory_gb = req.get("memory_gb").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let ram_gb = req.get("ram_gb").and_then(|v| v.as_f64());
            let cards = req
                .get("cards")
                .and_then(|v| v.as_arr())
                .unwrap_or(&[]);
            let vendor = req.get("vendor").and_then(|v| v.as_str());
            let out = gpu_fit::options(memory_gb, ram_gb, cards, vendor);
            ok_out(out);
        }
        _ => unreachable!(),
    }
}
