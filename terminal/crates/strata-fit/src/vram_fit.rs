pub const CHUNK_STEPS: [i64; 3] = [1024, 512, 256];
pub const RESERVE_FLOOR_MIB: i64 = 100;
pub const AUTO_CHUNK: i64 = 2048;
pub const MIN_SLOTS: f64 = 128.0;
pub const SLOT_MIB: f64 = 1.243;
pub const MARGIN_MIB: f64 = 64.0;
pub const HEAD_MIB: f64 = 110.0;
pub const SESSION_FIXED_MIB: f64 = 121.0;
pub const KV_MEASURED_FACTOR: f64 = 1.054;
pub const PROMPT_BASE_MIB: f64 = 170.0;
pub const PROMPT_PER_TOKEN_MIB: f64 = 0.66;
pub const GAP_OTHER_MIB: f64 = 250.0;
pub const MIN_CARD_GB: f64 = 5.0;
pub const BASE_GENERIC_MIB: f64 = 2448.0;
pub const SLOT_SLOTTED_MIB: f64 = 2.22;
pub const WORKING_CACHE_CUDA_MIB: f64 = 263.0 * 2.54;
pub const KV_RESIDENT_MIN: i64 = 4096;
pub const KV_RESIDENT_STEPS: [i64; 3] = [20480, 8192, 4096];
pub const STREAM_EXTRA_BYTES: f64 = 1400.0;
pub const SLOTS_MAX: i64 = 8;
pub const NOTE_SLOW: &str = "short of GPU memory: more is read from RAM and the SSD, expect it to be slower";

fn kv_bytes(kv: &str) -> f64 {
    match kv {
        "fp16" => 2112.0,
        "int8" => 1056.0,
        "q4_0" => 576.0,
        "k8v4" => 816.0,
        _ => 1056.0,
    }
}

fn gap_mib(vendor: &str) -> f64 {
    if vendor.eq_ignore_ascii_case("intel") {
        471.0
    } else {
        GAP_OTHER_MIB
    }
}

pub fn base_mib(model: &str) -> f64 {
    let low = model.to_ascii_lowercase().replace('_', "-");
    if low.contains("gyro-m") {
        5945.0
    } else if low.contains("gyro-s") {
        4720.0
    } else {
        BASE_GENERIC_MIB
    }
}

fn drafter_delta(drafter: &str) -> f64 {
    match drafter {
        "mtp" => 0.0,
        "dflash2" => -119.0,
        "dflash2+mtp" => 502.0,
        "none" => -835.0,
        _ => 0.0,
    }
}

pub fn slotted_base_mib(model: &str, vendor: &str, drafter: &str) -> Option<f64> {
    let low = model.to_ascii_lowercase().replace('_', "-");
    let v = vendor.to_ascii_lowercase();
    if !low.contains("flash-next") {
        return None;
    }
    let base = match v.as_str() {
        "intel" => 4500.0,
        "cuda" => 4990.0,
        "hip" => 3900.0,
        _ => return None,
    };
    Some(base + drafter_delta(drafter))
}

pub fn session_mib(ctx: i64, kv: &str, kv_resident: i64) -> f64 {
    let kb = kv_bytes(kv);
    if kv_resident > 0 && kv_resident < ctx {
        let cells = kv_resident.max(KV_RESIDENT_MIN);
        if cells < ctx {
            return SESSION_FIXED_MIB
                + (cells as f64) * 13.0 * kb * KV_MEASURED_FACTOR / 1_048_576.0
                + (ctx as f64) * STREAM_EXTRA_BYTES / 1_048_576.0;
        }
    }
    SESSION_FIXED_MIB + (ctx as f64) * 13.0 * kb * KV_MEASURED_FACTOR / 1_048_576.0
}

pub fn chunk_of(prefill: &str) -> i64 {
    match prefill.trim().parse::<i64>() {
        Ok(n) => n.max(1),
        Err(_) => AUTO_CHUNK,
    }
}

fn is_single_measured_vendor(vendor: &str) -> bool {
    let v = vendor.to_ascii_lowercase();
    v == "cuda" || v == "hip"
}

pub fn measured(model: &str, slots: i64, vendor: &str, drafter: &str) -> bool {
    if slotted_base_mib(model, vendor, drafter).is_none() {
        return false;
    }
    slots >= 2 || is_single_measured_vendor(vendor)
}

#[allow(clippy::too_many_arguments)]
pub fn slack_mib(
    model: &str,
    vram_gb: f64,
    ctx: i64,
    kv: &str,
    reserve_mib: i64,
    prefill: &str,
    vendor: &str,
    slots: i64,
    kv_resident: i64,
    drafter: &str,
) -> f64 {
    let gap = gap_mib(vendor);
    let prompt = PROMPT_BASE_MIB + PROMPT_PER_TOKEN_MIB * (chunk_of(prefill) as f64);
    let mut base = base_mib(model);
    let mut extra = 0.0;
    let mut main = session_mib(ctx, kv, 0);
    if measured(model, slots, vendor, drafter) {
        base = slotted_base_mib(model, vendor, drafter).unwrap_or(base);
        main = session_mib(ctx, kv, kv_resident);
        if slots >= 2 {
            extra = (slots.min(SLOTS_MAX) as f64) * main;
        }
    }
    vram_gb * 1024.0
        - gap
        - base
        - main
        - extra
        - HEAD_MIB
        - (reserve_mib.max(0) as f64)
        - prompt
}

pub fn needed_mib(model: &str, slots: i64, vendor: &str, drafter: &str) -> f64 {
    if measured(model, slots, vendor, drafter) {
        let v = vendor.to_ascii_lowercase();
        if slots >= 2 && (v == "cuda" || v == "hip") {
            return WORKING_CACHE_CUDA_MIB + MARGIN_MIB;
        }
        return SLOT_SLOTTED_MIB + MARGIN_MIB;
    }
    MIN_SLOTS * SLOT_MIB + MARGIN_MIB
}

fn knobs_for(vendor: &str) -> &'static [&'static str] {
    match vendor.to_ascii_lowercase().as_str() {
        "cuda" | "hip" => &["kv_resident"],
        "intel" => &["chunk", "reserve", "kv_resident"],
        _ => &["chunk", "reserve"],
    }
}

pub fn round_half_even_0(x: f64) -> i64 {
    let f = x.floor();
    let frac = x - f;
    if (frac - 0.5).abs() < 1e-12 {
        let fi = f as i64;
        if fi % 2 == 0 { fi } else { fi + 1 }
    } else {
        x.round() as i64
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct FitResult {
    pub ctx: i64,
    pub reserve_mib: i64,
    pub prefill: Option<String>,
    pub changed: bool,
    pub fits: bool,
    pub slack_mib: f64,
    pub line: String,
    pub changes: Vec<String>,
    pub kv_resident: Option<i64>,
    pub slots: i64,
}

#[allow(clippy::too_many_arguments)]
pub fn plan(
    model: &str,
    vram_gb: f64,
    ctx: i64,
    kv: &str,
    reserve_mib: i64,
    prefill: &str,
    vendor: &str,
    title: &str,
    reserve_fixed: bool,
    prefill_fixed: bool,
    slots_in: i64,
    kv_resident: i64,
    drafter: &str,
    kv_fixed: bool,
) -> FitResult {
    let slots = if slots_in >= 2 { slots_in } else { 0 };
    let mut keep = FitResult {
        ctx,
        reserve_mib,
        prefill: None,
        changed: false,
        fits: true,
        slack_mib: 0.0,
        line: String::new(),
        changes: Vec::new(),
        kv_resident: None,
        slots,
    };
    if vram_gb <= 0.0 || vram_gb < MIN_CARD_GB {
        return keep;
    }
    if (slots != 0 || is_single_measured_vendor(vendor)) && !measured(model, slots, vendor, drafter)
    {
        return keep;
    }
    let need = needed_mib(model, slots, vendor, drafter);
    keep.slack_mib = slack_mib(
        model,
        vram_gb,
        ctx,
        kv,
        reserve_mib,
        prefill,
        vendor,
        slots,
        kv_resident,
        drafter,
    );
    if keep.slack_mib >= need {
        return keep;
    }

    let mut kvr = kv_resident;
    let mut chunk = prefill.to_string();
    let mut res = reserve_mib;

    for &knob in knobs_for(vendor) {
        enum Step {
            Kv(i64),
            Chunk(String),
            Res(i64),
        }
        let steps: Vec<Step> = match knob {
            "kv_resident" => {
                let cur = if kvr != 0 { kvr } else { ctx };
                if kv_fixed || !measured(model, slots, vendor, drafter) {
                    Vec::new()
                } else {
                    KV_RESIDENT_STEPS
                        .iter()
                        .copied()
                        .filter(|&w| w < cur && w < ctx)
                        .map(Step::Kv)
                        .collect()
                }
            }
            "chunk" => {
                if prefill_fixed {
                    Vec::new()
                } else {
                    let cur_c = chunk_of(&chunk);
                    CHUNK_STEPS
                        .iter()
                        .copied()
                        .filter(|&c| c < cur_c)
                        .map(|c| Step::Chunk(c.to_string()))
                        .collect()
                }
            }
            _ => {
                if reserve_fixed || res <= RESERVE_FLOOR_MIB {
                    Vec::new()
                } else {
                    vec![Step::Res(RESERVE_FLOOR_MIB)]
                }
            }
        };

        for step in steps {
            match step {
                Step::Kv(v) => kvr = v,
                Step::Chunk(c) => chunk = c,
                Step::Res(r) => res = r,
            }
            if slack_mib(
                model, vram_gb, ctx, kv, res, &chunk, vendor, slots, kvr, drafter,
            ) >= need
            {
                break;
            }
        }
        if slack_mib(
            model, vram_gb, ctx, kv, res, &chunk, vendor, slots, kvr, drafter,
        ) >= need
        {
            break;
        }
    }

    let s = slack_mib(
        model, vram_gb, ctx, kv, res, &chunk, vendor, slots, kvr, drafter,
    );
    let ok = s >= need;
    let mut changes = Vec::new();
    if kvr != kv_resident {
        changes.push(format!(
            "KV window {kvr} tokens in VRAM, the rest read from RAM"
        ));
    }
    if chunk != prefill {
        changes.push(format!("prompts read in chunks of {chunk}"));
    }
    if res != reserve_mib {
        changes.push(format!("memory kept free for other programs {res} MiB"));
    }
    let name = if title.is_empty() {
        "This model"
    } else {
        title
    };
    let line = if !changes.is_empty() || !ok {
        let v_str = round_half_even_0(vram_gb);
        if changes.is_empty() {
            format!("{name} on this {v_str} GB card: {NOTE_SLOW}")
        } else {
            format!(
                "{name} on this {v_str} GB card: {NOTE_SLOW} ({})",
                changes.join(", ")
            )
        }
    } else {
        String::new()
    };

    FitResult {
        ctx,
        reserve_mib: res,
        prefill: if chunk != prefill { Some(chunk) } else { None },
        changed: !changes.is_empty(),
        fits: ok,
        slack_mib: s,
        line,
        changes,
        kv_resident: if kvr != kv_resident { Some(kvr) } else { None },
        slots,
    }
}
