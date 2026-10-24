import {spawnSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';
const path=fileURLToPath(new URL('./tests.php',import.meta.url));
const win=process.platform==='win32';
const linuxPath=win?'/mnt/'+path[0].toLowerCase()+path.slice(2).replaceAll('\\','/'):path;
const r=spawnSync(win?'wsl.exe':'php',win?['-d','Ubuntu-24.04','--','php',linuxPath]:[path],{stdio:'inherit'});
process.exit(r.status??1);
