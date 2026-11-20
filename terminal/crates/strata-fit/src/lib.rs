pub mod gpu_fit;
pub mod json;
pub mod vram_fit;

#[cfg(test)]
mod tests {
    use super::{gpu_fit, json, vram_fit};

    #[test]
    fn vram_plan_fits_when_roomy() {
        let res = vram_fit::plan(
            "Qwen3.8-Flash-Next",
            24.0,
            131072,
            "int8",
            700,
            "auto",
            "cuda",
            "Flash-Next",
            true,
            true,
            0,
            32768,
            "dflash2",
            false,
        );
        assert!(res.fits);
        assert!(!res.changed);
    }

    #[test]
    fn vram_plan_steps_down_kv_resident_on_tight_cuda() {
        let res = vram_fit::plan(
            "Qwen3.8-Flash-Next",
            8.0,
            131072,
            "int8",
            700,
            "auto",
            "cuda",
            "Flash-Next",
            true,
            true,
            2,
            0,
            "mtp",
            false,
        );
        assert!(res.kv_resident.is_some());
        assert!(res.changed);
    }

    #[test]
    fn gpu_fit_and_options_work_for_dual_3090() {
        let cards = json::parse(
            r#"[
              {"index":0,"name":"NVIDIA GeForce RTX 3090","vendor":"nvidia","vram_gb":24,"usable":true},
              {"index":1,"name":"NVIDIA GeForce RTX 3090","vendor":"nvidia","vram_gb":24,"usable":true}
            ]"#,
        )
        .unwrap();
        let res = gpu_fit::options(83.6, Some(128.0), cards.as_arr().unwrap(), None);
        assert_eq!(res.get("recommended").and_then(|v| v.as_str()), Some("single"));
        assert_eq!(res.get("cards").and_then(|v| v.as_i64()), Some(2));
    }

    #[test]
    fn json_preserves_large_ints_and_surrogate_pairs_and_rejects_bad_json() {
        let parsed = json::parse(r#"{"big":9007199254740993,"emoji":"\uD83D\uDE00"}"#).unwrap();
        assert_eq!(parsed.get("big").and_then(|v| v.as_i64()), Some(9007199254740993));
        assert_eq!(parsed.get("emoji").and_then(|v| v.as_str()), Some("😀"));
        let out = json::stringify(&parsed);
        assert!(out.contains("9007199254740993"), "got {out}");
        assert!(json::parse(r#"{"a":1,"a":2}"#).is_err());
        assert!(json::parse(r#"{"a":NaN}"#).is_err());
        assert!(json::parse(r#"{"a":1} {}"#).is_err());
        assert!(json::parse(r#"{"a":"\uD83D"}"#).is_err());
    }
}
