import type { ConfigDocument } from './config';

/** Hikari / Prism PQ2_0 and TurboQuant PTQ1_0 run on bundled PrismML, not native Q2_0 packs. */
export type StrataEngineBackend = 'PrismML' | 'TurboQuant';

export function resolveStrataEngineBackend(
  model: { local?: unknown; quant?: string } | null | undefined,
  backendHint?: string | null,
): StrataEngineBackend | null {
  if (backendHint === 'PrismML' || backendHint === 'TurboQuant') return backendHint;
  if (!model?.local) return null;
  if (model.quant === 'PQ2_0') return 'PrismML';
  if (model.quant === 'PTQ1_0') return 'TurboQuant';
  return null;
}

/** Pin runtime.backend so Load / agent path uses Strata’s bundled Prism/TurboQuant engine. */
export function pinStrataEngineBackend<T extends Pick<ConfigDocument, 'expert'>>(
  doc: T,
  model: { local?: unknown; quant?: string } | null | undefined,
  backendHint?: string | null,
): T {
  const backend = resolveStrataEngineBackend(model, backendHint);
  if (!backend) return doc;
  if (doc.expert['runtime.backend'] === backend) return doc;
  return { ...doc, expert: { ...doc.expert, 'runtime.backend': backend } };
}
