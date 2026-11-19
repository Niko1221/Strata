import { existsSync } from 'node:fs';
import { chromium } from 'playwright-core';

// The CI browser's first cold process can take more than ten seconds to start.
// Keep a bounded launch deadline separate from each component assertion's timeout.
export const TEST_BROWSER_LAUNCH_TIMEOUT_MS = 30_000;

/** Use an installed or explicitly provisioned browser on each CI OS; never a user's browser profile. */
export function testBrowserExecutable() {
  const explicit = process.env.STRATA_TEST_BROWSER_EXE;
  if (explicit) {
    if (!existsSync(explicit)) throw new Error('STRATA_TEST_BROWSER_EXE does not name an installed browser');
    return explicit;
  }
  const paths = process.platform === 'win32' ? [
    'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
    'C:/Program Files/Microsoft/Edge/Application/msedge.exe',
    'C:/Program Files/Google/Chrome/Application/chrome.exe',
  ] : process.platform === 'darwin' ? [
    '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    '/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge',
  ] : [process.env.CHROME_BIN, '/usr/bin/google-chrome', '/usr/bin/google-chrome-stable', '/usr/bin/chromium', '/usr/bin/chromium-browser'];
  const executable = [...paths, chromium.executablePath()].find(path => path && existsSync(path));
  if (!executable) throw new Error('Mounted browser tests need Chromium, Chrome or Edge; provision one or set STRATA_TEST_BROWSER_EXE.');
  return executable;
}
