/** Adapt only host-injected settings; explicit API fields still reach backend validation. */
export function samplingForPolicy(sampling, policy) {
  if (policy !== 'q38dense' || !sampling) return sampling;
  const out = { ...sampling };
  // Dense native has no projection vector and rejects the host's unset seed sentinel.
  delete out.experimental_speed_projection;
  if (out.seed === -1) delete out.seed;
  return out;
}
