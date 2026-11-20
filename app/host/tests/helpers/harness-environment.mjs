import { join } from 'node:path';
import { isolatedTestEnvironment } from '../../scripts/test-environment.mjs';

/** Each real-Harness fixture owns its home and skill roots, including when run outside npm test. */
export function harnessFixtureEnvironment(directory, inherited = process.env) {
  return isolatedTestEnvironment(inherited, {
    HOME: directory, USERPROFILE: directory,
    APPDATA: join(directory, 'Roaming'), LOCALAPPDATA: join(directory, 'Local'),
    XDG_DATA_HOME: join(directory, 'xdg-data'), XDG_STATE_HOME: join(directory, 'xdg-state'),
    XDG_CONFIG_HOME: join(directory, 'xdg-config'),
  });
}
