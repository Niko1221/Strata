/** Strata Dev: the UI extensions in dev/extensions/ui, by name (vite.config.ts, plugin "strata-dev-extensions"). Empty outside Dev. */
declare module 'virtual:strata-dev-extensions' {
  const modules: Record<string, () => Promise<unknown>>;
  export default modules;
}
