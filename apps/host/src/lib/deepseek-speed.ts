import type { RunCard } from './native-fit';

/**
 * DeepSeek's Speed default, said the way the engine decides it (`--experts auto`, src/dsv41/ds4_experts_rule.hpp): Faster (5 experts a word) when
 * a card in use is an integrated / unified-memory GPU (Strix Halo, gfx1151: the model shares the PC's memory), when the one card in use is an
 * RDNA4 Radeon (gfx1200 / gfx1201: the CPU lane does most of the experts), when several RDNA4 cards hold half of the model's experts (the catalog's
 * `rdna4_faster`), or when the cards in use total 16 GiB or less; otherwise Full quality (6). The app never decides: the engine does; this only names the same default beside the choice.
 * The total is compared in whole MiB (a 16 GB card reports 16376 MiB = 15.99 GiB and counts as 16 GiB or less).
 */
export const FASTER_MIB = 16 * 1024;

export const isIntegrated = (card: RunCard) => !!(card.uma || card.integrated || (card.unified_gb ?? 0) > 0);

/** One RDNA4 Radeon (gfx1200 / gfx1201: Radeon AI PRO R9700, RX 9070 / 9060). */
export const isRdna4 = (card: RunCard) => (card.vendor ?? '').toLowerCase() === 'amd' && /^gfx120[01]/i.test(card.arch ?? '');

export function fasterByDefault(cards: readonly RunCard[] | null | undefined, rdna4Faster?: boolean | null): boolean {
  const used = (cards ?? []).filter(card => card.usable !== false);
  if (!used.length) return false;
  if (used.some(isIntegrated)) return true;
  if (used.length === 1 && isRdna4(used[0])) return true;
  // several RDNA4 cards: the engine gives 5 when together they hold half of the experts of this model (the catalog says, strata_control/dsv41_fit.py rdna4_faster)
  if (used.length > 1 && used.every(isRdna4) && rdna4Faster === true) return true;
  const mib = used.reduce((sum, card) => sum + Math.round((card.vram_gb ?? 0) * 1024), 0);
  return mib > 0 && mib <= FASTER_MIB;
}

/** The three entries of the Speed select: "Faster" or "Full quality" says "(default on this PC)" when it is what the engine picks by itself. */
export function speedOptions(cards: readonly RunCard[] | null | undefined, rdna4Faster?: boolean | null): { value: string; label: string }[] {
  const faster = fasterByDefault(cards, rdna4Faster);
  const mark = ' (default on this PC)';
  return [
    { value: '', label: faster ? 'Default (Faster on this PC)' : 'Default (Full quality on this PC)' },
    { value: '5', label: 'Faster' + (faster ? mark : '') },
    { value: '6', label: 'Full quality' + (faster ? '' : mark) },
  ];
}
