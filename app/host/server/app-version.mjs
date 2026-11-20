/**
 * Report the desktop shell's product version. Control and engine build numbers
 * have separate meanings and must never override it.
 */
import {readFileSync} from 'node:fs';
import {join} from 'node:path';

const VERSION = /^\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.+-]+)?$/;

export function appVersionFrom({env=process.env, here=''}) {
  const given=String(env.STRATA_APP_VERSION??'').trim();
  if (VERSION.test(given)) return given;
  try {
    const toml=readFileSync(join(here,'..','..','desktop','src-tauri','Cargo.toml'),'utf8');
    const found=/^version\s*=\s*"([^"]+)"/m.exec(toml);
    if (found && VERSION.test(found[1])) return found[1];
  } catch { /* An installed app receives its version from the shell. */ }
  return 'dev';
}

/** Prefer the product version; only a development host may fall back to Control. */
export function appVersionToShow(app, control) {
  const real=value=>typeof value==='string' && value.trim() && value.trim().toLowerCase()!=='dev' ? value.trim() : null;
  return real(app)??real(control)??(typeof app==='string' && app.trim() ? app.trim() : null);
}
