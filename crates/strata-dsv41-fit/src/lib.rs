pub const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
pub const GB: f64 = 1_000_000_000.0;

pub fn cap_gb(vram_gb: f64) -> f64 {
    vram_gb * GIB / GB
}

pub fn uma_total_gb(vram_gb: f64, dedicated_gb: f64, shared_gb: f64, ram_gb: f64) -> f64 {
    if dedicated_gb > 0.0 {
        return ram_gb + dedicated_gb;
    }
    if vram_gb > 0.0 && shared_gb > 0.0 {
        return ram_gb + (vram_gb - shared_gb).max(0.0);
    }
    ram_gb.max(vram_gb)
}

pub fn split_layers(weights: &[f64], layers: usize) -> Vec<(usize, usize)> {
    let n = weights.len();
    if n <= 1 {
        return vec![(0, layers.saturating_sub(1))];
    }
    let sum_w: f64 = weights.iter().sum();
    let total = if sum_w == 0.0 { 1.0 } else { sum_w };
    let raw: Vec<f64> = weights.iter().map(|w| w / total * (layers as f64)).collect();
    let mut counts: Vec<usize> = raw.iter().map(|&r| (r as usize).max(1)).collect();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        let fa = raw[a] - (raw[a] as usize as f64);
        let fb = raw[b] - (raw[b] as usize as f64);
        fb.total_cmp(&fa)
    });
    let mut k = 0usize;
    while counts.iter().sum::<usize>() < layers {
        counts[order[k % n]] += 1;
        k += 1;
    }
    while counts.iter().sum::<usize>() > layers {
        let (max_idx, _) = counts
            .iter()
            .enumerate()
            .max_by_key(|&(_, &c)| c)
            .unwrap();
        counts[max_idx] -= 1;
    }
    let mut out = Vec::with_capacity(n);
    let mut first = 0usize;
    for c in counts {
        out.push((first, first + c - 1));
        first += c;
    }
    out
}

#[allow(clippy::too_many_arguments)]
pub fn place_ista_experts(
    vram_gbs: &[f64],
    layers: usize,
    experts_per_layer: usize,
    experts_gb: f64,
    per_dense: f64,
    engine_free_gb: f64,
    other_gb: f64,
    resident_expert_factor: f64,
    amd: bool,
    amd_low_share_gb: f64,
) -> Vec<usize> {
    let n_experts = layers * experts_per_layer;
    let expert_gb = experts_gb / (n_experts as f64);
    let res_gb = expert_gb * resident_expert_factor;
    let ranges = split_layers(vram_gbs, layers);

    let place = |other: f64| -> Vec<usize> {
        let cap: Vec<f64> = vram_gbs
            .iter()
            .map(|&v| (cap_gb(v) - per_dense - engine_free_gb - other).max(0.0))
            .collect();
        let cap_sum: f64 = cap.iter().sum();
        let vram_exp = (experts_gb * resident_expert_factor).min(cap_sum);
        cap.iter()
            .zip(ranges.iter())
            .map(|(&room, &(a, b))| {
                let gb = if cap_sum != 0.0 {
                    room.min(vram_exp * (room / cap_sum))
                } else {
                    0.0
                };
                let count = (gb / res_gb) as usize;
                count.min((b - a + 1) * experts_per_layer)
            })
            .collect()
    };

    let mut counts = place(other_gb);
    if amd && (counts.iter().sum::<usize>() as f64) < (n_experts as f64) / 2.0 {
        counts = place(amd_low_share_gb);
    }
    counts
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_layers_four_equal_cards() {
        let ranges = split_layers(&[32.0, 32.0, 32.0, 32.0], 40);
        assert_eq!(ranges, vec![(0, 9), (10, 19), (20, 29), (30, 39)]);
    }

    #[test]
    fn split_layers_single_card() {
        assert_eq!(split_layers(&[24.0], 40), vec![(0, 39)]);
    }

    #[test]
    fn uma_total_counts_carveout_once() {
        assert!((uma_total_gb(64.0, 64.0, 0.0, 64.0) - 128.0).abs() < 1e-9);
        assert!((uma_total_gb(80.0, 0.0, 32.0, 64.0) - 112.0).abs() < 1e-9);
    }

    #[test]
    fn place_ista_four_r9700() {
        let counts = place_ista_experts(
            &[32.0, 32.0, 32.0, 32.0],
            40,
            192,
            128.0,
            2.5,
            0.62,
            1.5,
            1.025,
            true,
            3.0,
        );
        assert_eq!(counts.len(), 4);
        assert!(counts.iter().all(|&c| c > 0));
    }
}
