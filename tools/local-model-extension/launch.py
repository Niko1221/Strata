"""Open the existing Windows App with an optional, external local-model UI package."""
import argparse
import json
import os
from pathlib import Path
import subprocess


def plan(path):
    path=Path(path).resolve()
    if path.stat().st_size>65536:raise ValueError('Configuration Is Too Large')
    config=json.loads(path.read_text(encoding='utf-8-sig'))
    app=Path(config['app_executable'])
    if not app.is_absolute() or not app.is_file() or app.suffix.lower()!='.exe':
        raise ValueError('Choose The Existing Windows Strata App Executable')
    root=Path(__file__).resolve().parent
    modules=['host-preload.mjs','extension_bridge.mjs','owner_api.mjs']
    available=all((root/name).is_file() for name in modules)
    env=os.environ.copy()
    if available:
        env['STRATA_LOCAL_MODELS_CONFIG']=str(path)
        preload='--import='+ (root/'host-preload.mjs').as_uri()
        env['NODE_OPTIONS']=(env.get('NODE_OPTIONS','')+' '+preload).strip()
    return app,env,available


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    app,env,available=plan(args.config)
    if args.check:
        print(json.dumps({'app_ready':True,'preload_available':available}))
    else:
        child=subprocess.Popen([str(app)],cwd=app.parent,env=env,stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        print(json.dumps({'launched':True,'pid':child.pid,'preload_available':available}))


if __name__=='__main__':main()
