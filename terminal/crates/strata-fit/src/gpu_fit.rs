use std::collections::{BTreeMap, BTreeSet};

use crate::json::Json;
use crate::vram_fit::round_half_even_0;

pub const DENSE_GB: f64 = 5.0;
pub const SYSTEM_ROOM_GB: f64 = 10.0;
pub const DENSE_EXTRA_GB: f64 = 1.0;
pub const SPLIT_CARD_OVERHEAD_GB: f64 = 2.0;
pub const SPLIT_MIN_VRAM_GB: f64 = 8.0;
pub const SPLIT_PROMPT_VRAM_GB: f64 = 12.0;
pub const MIN_CARD_GB: f64 = 4.0;
pub const FITS_SPARE_RAM_GB: f64 = 4.0;
pub const SPLIT_PROMPT_GAIN: &str = "about 20% faster";

pub fn round1(x: f64) -> f64 {
    let scaled = x * 10.0;
    let f = scaled.floor();
    let frac = scaled - f;
    let r = if (frac - 0.5).abs() < 1e-12 {
        let fi = f as i64;
        if fi % 2 == 0 { fi as f64 } else { (fi + 1) as f64 }
    } else {
        scaled.round()
    };
    r / 10.0
}

fn vendor_name(v: &str) -> String {
    match v {
        "nvidia" => "NVIDIA".to_string(),
        "amd" => "AMD".to_string(),
        "intel" => "Intel".to_string(),
        "apple" => "Apple".to_string(),
        other => {
            let mut c = other.chars();
            match c.next() {
                None => String::new(),
                Some(first) => first.to_uppercase().collect::<String>() + c.as_str(),
            }
        }
    }
}

fn card_vendor(c: &Json) -> Option<String> {
    c.get("vendor")
        .and_then(|v| match v {
            Json::Str(s) if !s.is_empty() => Some(s.to_ascii_lowercase()),
            _ => None,
        })
}

fn card_f64(c: &Json, key: &str) -> f64 {
    c.get(key).and_then(|v| v.as_f64()).unwrap_or(0.0)
}

fn shares_memory(c: &Json) -> bool {
    let truthy = |k: &str| match c.get(k) {
        Some(Json::Bool(b)) => *b,
        Some(Json::Int(n)) => *n != 0,
        Some(Json::Num(n)) => *n != 0.0,
        Some(Json::Str(s)) => !s.is_empty(),
        _ => false,
    };
    truthy("integrated")
        || truthy("uma")
        || truthy("unified_memory")
        || truthy("unified_gb")
        || card_vendor(c).as_deref() == Some("apple")
}

pub fn usable_cards(cards: &[Json]) -> (Vec<Json>, Vec<Json>) {
    let flagged: Vec<&Json> = cards
        .iter()
        .filter(|c| c.get("usable").and_then(|v| v.as_bool()) != Some(false))
        .collect();
    let has_real = flagged
        .iter()
        .any(|c| !shares_memory(c) && card_f64(c, "vram_gb") >= MIN_CARD_GB);

    let mut keep = Vec::new();
    let mut ignored = Vec::new();

    for c in cards {
        let vram = card_f64(c, "vram_gb");
        let why = if c.get("usable").and_then(|v| v.as_bool()) == Some(false) {
            Some(
                c.get("problem")
                    .and_then(|v| v.as_str())
                    .filter(|s| !s.is_empty())
                    .unwrap_or("Strata cannot use this card")
                    .to_string(),
            )
        } else if shares_memory(c) {
            if has_real {
                Some("it shares the system memory (an integrated graphics chip), so it adds nothing".to_string())
            } else {
                None
            }
        } else if vram < MIN_CARD_GB {
            Some(format!("only {} GB of graphics memory", round_half_even_0(vram)))
        } else {
            None
        };

        if let Some(w) = why {
            if let Json::Obj(mut map) = c.clone() {
                map.insert("why_ignored".to_string(), Json::Str(w));
                ignored.push(Json::Obj(map));
            }
        } else {
            keep.push(c.clone());
        }
    }

    keep.sort_by(|a, b| {
        let va = round_half_even_0(
            a.get("vram_gb")
                .or_else(|| a.get("unified_gb"))
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0),
        );
        let vb = round_half_even_0(
            b.get("vram_gb")
                .or_else(|| b.get("unified_gb"))
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0),
        );
        let ia = a.get("index").and_then(|v| v.as_i64()).unwrap_or(0);
        let ib = b.get("index").and_then(|v| v.as_i64()).unwrap_or(0);
        (-va, ia).cmp(&(-vb, ib))
    });

    (keep, ignored)
}

pub fn pick_vendor(cards: &[Json]) -> Option<String> {
    let (keep, _) = usable_cards(cards);
    let makes: BTreeSet<String> = keep.iter().filter_map(card_vendor).collect();
    if makes.len() <= 1 {
        return makes.into_iter().next();
    }
    if makes.contains("nvidia") {
        return Some("nvidia".to_string());
    }
    keep.first().and_then(card_vendor)
}

pub fn for_vendor(cards: &[Json], vendor: Option<&str>) -> (Vec<Json>, Vec<Json>) {
    let Some(v) = vendor else {
        return (cards.to_vec(), Vec::new());
    };
    let mut mine = Vec::new();
    let mut others = Vec::new();
    for c in cards {
        match card_vendor(c) {
            None => mine.push(c.clone()),
            Some(cv) if cv == v => mine.push(c.clone()),
            Some(_) => others.push(c.clone()),
        }
    }
    (mine, others)
}

fn short_name(c: &Json, vendor: &str) -> String {
    let mut n = c
        .get("name")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .unwrap_or("graphics card")
        .to_string();
    for junk in ["NVIDIA GeForce ", "NVIDIA ", "AMD Radeon ", "AMD ", "Intel "] {
        if let Some(rest) = n.strip_prefix(junk) {
            n = rest.to_string();
            break;
        }
    }
    format!("{} {}", vendor_name(vendor), n).trim().to_string()
}

pub fn other_vendor_note(cards: &[Json], vendor: Option<&str>) -> Option<String> {
    let v = vendor?;
    let (usable, _) = usable_cards(cards);
    let (mine, raw_others) = for_vendor(&usable, Some(v));
    let others: Vec<Json> = raw_others
        .into_iter()
        .filter(|c| card_vendor(c).is_some() && !shares_memory(c))
        .collect();
    if others.is_empty() {
        return None;
    }
    let intel: Vec<&Json> = others
        .iter()
        .filter(|c| card_vendor(c).as_deref() == Some("intel"))
        .collect();
    if !intel.is_empty() {
        let total: f64 = intel.iter().map(|c| card_f64(c, "vram_gb")).sum();
        let cards_str = if intel.len() > 1 {
            format!(
                "{} Intel Arc cards ({} GB together)",
                intel.len(),
                round_half_even_0(total)
            )
        } else {
            format!(
                "an {} ({} GB)",
                short_name(intel[0], "intel"),
                round_half_even_0(total)
            )
        };
        let s = if intel.len() > 1 { "s" } else { "" };
        return Some(format!(
            "This computer also has {cards_str}. Strata uses the {} card; to use the Intel card{s} instead, add --backend intel (it is remembered; --backend auto goes back).",
            vendor_name(v)
        ));
    }
    if v == "intel" {
        let mut names: BTreeSet<String> = BTreeSet::new();
        for c in &others {
            if let Some(cv) = card_vendor(c) {
                names.insert(short_name(c, &cv));
            }
        }
        let joined = names.into_iter().collect::<Vec<_>>().join(", ");
        return Some(format!(
            "This computer also has {joined}. Strata uses the Intel cards; --backend auto goes back to the PC's own choice (NVIDIA first)."
        ));
    }
    let best = others
        .iter()
        .max_by(|a, b| card_f64(a, "vram_gb").total_cmp(&card_f64(b, "vram_gb")))?;
    let mine_best = mine
        .iter()
        .map(|c| card_f64(c, "vram_gb"))
        .fold(0.0_f64, f64::max);
    let best_vram = card_f64(best, "vram_gb");
    if best_vram <= mine_best {
        return None;
    }
    let other = card_vendor(best)?;
    Some(format!(
        "This computer also has an {} ({} GB). The {} build of Strata can use it — it fits bigger models.",
        short_name(best, &other),
        round_half_even_0(best_vram),
        vendor_name(&other)
    ))
}

pub fn fit(memory_gb: f64, ram_gb: Option<f64>, vram_gb: f64) -> BTreeMap<String, Json> {
    let mut m = BTreeMap::new();
    let Some(ram) = ram_gb.filter(|&r| r > 0.0) else {
        m.insert("vram_use_gb".to_string(), Json::Num(0.0));
        m.insert("ram_use_gb".to_string(), Json::Num(0.0));
        m.insert("spill_gb".to_string(), Json::Num(0.0));
        m.insert("spare_ram_gb".to_string(), Json::Num(0.0));
        m.insert("known".to_string(), Json::Bool(false));
        return m;
    };
    let experts = (memory_gb - DENSE_EXTRA_GB).max(0.0);
    let on_gpu = (vram_gb - DENSE_GB).min(experts).max(0.0);
    let ram_room = (ram - SYSTEM_ROOM_GB).max(0.0);
    let spill = (experts - on_gpu - ram_room).max(0.0);
    let in_ram = (experts - on_gpu - spill).max(0.0);

    let vram_use = if vram_gb > 0.0 {
        round1(vram_gb.min(DENSE_GB + on_gpu))
    } else {
        0.0
    };
    let ram_use = round1(in_ram + if vram_gb <= 0.0 { DENSE_EXTRA_GB } else { 0.0 });
    m.insert("vram_use_gb".to_string(), Json::Num(vram_use));
    m.insert("ram_use_gb".to_string(), Json::Num(ram_use));
    m.insert("spill_gb".to_string(), Json::Num(round1(spill)));
    m.insert(
        "spare_ram_gb".to_string(),
        Json::Num(round1((ram_room - in_ram).max(0.0))),
    );
    m.insert("known".to_string(), Json::Bool(true));
    m
}

pub fn verdict(f: &BTreeMap<String, Json>) -> &'static str {
    if f.get("known").and_then(|v| v.as_bool()) != Some(true) {
        return "unknown";
    }
    let spill = f.get("spill_gb").and_then(|v| v.as_f64()).unwrap_or(0.0);
    if spill >= 1.0 {
        return "spills";
    }
    let spare = f.get("spare_ram_gb").and_then(|v| v.as_f64()).unwrap_or(0.0);
    let ram_use = f.get("ram_use_gb").and_then(|v| v.as_f64()).unwrap_or(0.0);
    if spare >= FITS_SPARE_RAM_GB || ram_use <= 0.0 {
        "fits"
    } else {
        "tight"
    }
}

fn own_memory_gb(c: &Json) -> f64 {
    let truthy = |k: &str| match c.get(k) {
        Some(Json::Bool(b)) => *b,
        Some(Json::Int(n)) => *n != 0,
        Some(Json::Num(n)) => *n != 0.0,
        Some(Json::Str(s)) => !s.is_empty(),
        _ => false,
    };
    if truthy("uma") || truthy("integrated") {
        card_f64(c, "dedicated_gb")
    } else {
        card_f64(c, "vram_gb")
    }
}

fn card_display_name(c: &Json, dup: bool) -> String {
    let idx = c.get("index").map(|v| match v {
        Json::Int(n) => format!("{n}"),
        Json::Num(n) => format!("{}", *n as i64),
        Json::Str(s) => s.clone(),
        _ => "None".to_string(),
    }).unwrap_or_else(|| "None".to_string());
    let mut n = c
        .get("name")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .unwrap_or_else(|| format!("card {idx}"));
    for junk in ["NVIDIA GeForce ", "NVIDIA ", "AMD Radeon ", "AMD "] {
        n = n.replace(junk, "");
    }
    if dup {
        format!("{n} (card {idx})")
    } else {
        n
    }
}

fn build_option(
    key: &str,
    cards: &[(&Json, bool)],
    memory_gb: f64,
    ram_gb: Option<f64>,
) -> BTreeMap<String, Json> {
    let n = cards.len();
    let total: f64 = cards.iter().map(|(c, _)| own_memory_gb(c)).sum();
    let eff = total - SPLIT_CARD_OVERHEAD_GB * ((n as f64) - 1.0);
    let f = fit(memory_gb, ram_gb, eff);
    let v = verdict(&f);
    let names = cards
        .iter()
        .map(|(c, dup)| card_display_name(c, *dup))
        .collect::<Vec<_>>()
        .join(" + ");
    let label = if n == 1 {
        format!("{names} + RAM")
    } else {
        format!("{names} + RAM (layer split)")
    };
    let have = eff + (ram_gb.unwrap_or(0.0) - SYSTEM_ROOM_GB).max(0.0);
    let spill_gb = f.get("spill_gb").and_then(|x| x.as_f64()).unwrap_or(0.0);
    let words = match v {
        "unknown" => "memory of this computer unknown".to_string(),
        "spills" => format!(
            "does not fit in memory: about {} GB is read from the SSD while answering (slower)",
            round_half_even_0(spill_gb)
        ),
        "tight" => format!(
            "fits, but tight: needs {} GB of the {} GB available",
            round_half_even_0(memory_gb),
            round_half_even_0(have)
        ),
        _ => format!(
            "fits: needs {} GB of the {} GB available",
            round_half_even_0(memory_gb),
            round_half_even_0(have)
        ),
    };
    let mut notes = Vec::new();
    if n >= 2 {
        notes.push(Json::Str(format!(
            "prompts are read {SPLIT_PROMPT_GAIN}; the cards share the layers (docs/MULTI_GPU.md)"
        )));
        let small: Vec<(&Json, bool)> = cards[1..]
            .iter()
            .copied()
            .filter(|(c, _)| card_f64(c, "vram_gb") < SPLIT_PROMPT_VRAM_GB - 0.5)
            .collect();
        if !small.is_empty() && card_f64(cards[0].0, "vram_gb") >= SPLIT_PROMPT_VRAM_GB - 0.5 {
            notes.push(Json::Str(format!(
                "{} is small: it can make prompts slower than the first card alone",
                card_display_name(small[0].0, small[0].1)
            )));
        }
    }
    let card_ids: Vec<Json> = cards
        .iter()
        .map(|(c, _)| c.get("index").cloned().unwrap_or(Json::Null))
        .collect();
    let mut m = BTreeMap::new();
    m.insert("key".to_string(), Json::Str(key.to_string()));
    m.insert("cards".to_string(), Json::Arr(card_ids));
    m.insert("label".to_string(), Json::Str(label.clone()));
    m.insert("verdict".to_string(), Json::Str(v.to_string()));
    m.insert("summary".to_string(), Json::Str(format!("{label}: {words}")));
    m.insert("words".to_string(), Json::Str(words));
    m.insert("vram_gb".to_string(), Json::Num(round1(total)));
    m.insert(
        "spill_gb".to_string(),
        f.get("spill_gb").cloned().unwrap_or(Json::Num(0.0)),
    );
    m.insert(
        "ram_use_gb".to_string(),
        f.get("ram_use_gb").cloned().unwrap_or(Json::Num(0.0)),
    );
    m.insert(
        "spare_ram_gb".to_string(),
        f.get("spare_ram_gb").cloned().unwrap_or(Json::Num(0.0)),
    );
    m.insert("notes".to_string(), Json::Arr(notes));
    m.insert("recommended".to_string(), Json::Bool(false));
    m
}

fn rank_of(v: &str) -> i32 {
    match v {
        "fits" => 0,
        "tight" => 1,
        "unknown" => 2,
        _ => 3,
    }
}

pub fn options(
    memory_gb: f64,
    ram_gb: Option<f64>,
    cards: &[Json],
    vendor_in: Option<&str>,
) -> Json {
    let vendor = vendor_in
        .map(|s| s.to_string())
        .or_else(|| pick_vendor(cards));
    let (mine, others) = for_vendor(cards, vendor.as_deref());
    let (keep, mut ignored) = usable_cards(&mine);
    for c in others {
        if let Json::Obj(mut map) = c.clone() {
            let cv = card_vendor(&c).unwrap_or_default();
            let v_str = vendor.as_deref().unwrap_or("");
            let msg = format!(
                "it needs the {} build of Strata, and this engine drives {} cards",
                vendor_name(&cv),
                vendor_name(v_str)
            );
            map.insert("why_ignored".to_string(), Json::Str(msg));
            ignored.push(Json::Obj(map));
        }
    }
    let note = other_vendor_note(cards, vendor.as_deref());
    let mut out = BTreeMap::new();
    out.insert("options".to_string(), Json::Arr(Vec::new()));
    out.insert("recommended".to_string(), Json::Null);
    out.insert("ignored".to_string(), Json::Arr(ignored));
    out.insert("cards".to_string(), Json::Int(keep.len() as i64));
    out.insert(
        "vendor".to_string(),
        vendor.clone().map(Json::Str).unwrap_or(Json::Null),
    );
    out.insert(
        "other_vendor_note".to_string(),
        note.map(Json::Str).unwrap_or(Json::Null),
    );
    if keep.is_empty() {
        return Json::Obj(out);
    }

    let names: Vec<String> = keep
        .iter()
        .map(|c| {
            c.get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("None")
                .to_string()
        })
        .collect();
    let keep_with_dup: Vec<(&Json, bool)> = keep
        .iter()
        .enumerate()
        .map(|(i, c)| (c, names.iter().filter(|n| *n == &names[i]).count() > 1))
        .collect();

    let mut opts = vec![build_option("single", &[keep_with_dup[0]], memory_gb, ram_gb)];
    let share: Vec<(&Json, bool)> = keep_with_dup
        .iter()
        .copied()
        .filter(|(c, _)| card_f64(c, "vram_gb") >= SPLIT_MIN_VRAM_GB - 0.5)
        .collect();
    if share.len() >= 2 {
        opts.push(build_option("split", &share[..2], memory_gb, ram_gb));
        if share.len() > 2 {
            opts.push(build_option("split-all", &share, memory_gb, ram_gb));
        }
    }
    for item in &keep_with_dup[1..] {
        let idx = item
            .0
            .get("index")
            .map(|v| match v {
                Json::Int(n) => format!("{n}"),
                Json::Num(n) => format!("{}", *n as i64),
                Json::Str(s) => s.clone(),
                _ => "None".to_string(),
            })
            .unwrap_or_else(|| "None".to_string());
        opts.push(build_option(
            &format!("single-{idx}"),
            &[*item],
            memory_gb,
            ram_gb,
        ));
    }

    let mut best_idx = 0usize;
    let single_verdict = opts[0]
        .get("verdict")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");
    if single_verdict != "fits" {
        for (i, o) in opts.iter().enumerate().skip(1) {
            let n_cards = o.get("cards").and_then(|v| v.as_arr()).map_or(0, |a| a.len());
            if n_cards > 1 {
                let ov = rank_of(o.get("verdict").and_then(|v| v.as_str()).unwrap_or("unknown"));
                let os = o.get("spill_gb").and_then(|v| v.as_f64()).unwrap_or(0.0);
                let bv = rank_of(opts[best_idx].get("verdict").and_then(|v| v.as_str()).unwrap_or("unknown"));
                let bs = opts[best_idx].get("spill_gb").and_then(|v| v.as_f64()).unwrap_or(0.0);
                if (ov, os) < (bv, bs) {
                    best_idx = i;
                }
            }
        }
    }

    let rec_key = opts[best_idx]
        .get("key")
        .and_then(|v| v.as_str())
        .unwrap_or("single")
        .to_string();
    opts[best_idx].insert("recommended".to_string(), Json::Bool(true));

    out.insert(
        "options".to_string(),
        Json::Arr(opts.into_iter().map(Json::Obj).collect()),
    );
    out.insert("recommended".to_string(), Json::Str(rec_key));
    Json::Obj(out)
}
