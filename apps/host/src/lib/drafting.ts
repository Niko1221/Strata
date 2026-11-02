/**
 * The Drafting choice the Models screen offers for a model. It exists only where there is something to choose: DFlash2 is for Qwen3.8-Flash-Next
 * only (the qwen and gyro families), and only while this computer's engine can run it: strata_control/drafting.py hardware_support (Control reports it as
 * `capabilities.dflash2`; Metal and Intel engines and PCs without a supported card have none). Every other model has no dropdown at all: DeepSeek has no
 * MTP (its own DSpark drafter is chosen by the engine), and the other families have the MTP and nothing to pick against it (Settings > Draft layer turns it off).
 */
export const DRAFTING_FAMILIES = ['qwen', 'gyro'];

/** The drafter a choice means for a family (one per run; strata_control/drafting.py choice_of): Auto is DFlash2 where the engine has it (`served`) and the family is one it was made for, else the MTP. */
export function resolveDrafter(choice: string | null | undefined, family: string | null | undefined, served: boolean | null | undefined): 'dflash2' | 'mtp' | 'off' {
  const c = String(choice ?? 'auto').toLowerCase();
  if (c === 'off' || c === 'mtp') return c;
  return served === true && DRAFTING_FAMILIES.includes(String(family)) ? 'dflash2' : 'mtp';
}

/** What was measured on this kind of card (Control's words): tested = faster on some text and slower on others, slower = slower on every prompt. */
export interface Dflash2Support { level: 'tested' | 'slower' | 'untested'; line: string }

/** The choices for a model, or null when there is nothing to choose (no dropdown). */
export function draftingChoices(family: string | null | undefined, dflash2Served: boolean | null | undefined): string[] | null {
  return DRAFTING_FAMILIES.includes(String(family)) && dflash2Served === true ? ['auto', 'mtp', 'dflash2', 'off'] : null;
}
