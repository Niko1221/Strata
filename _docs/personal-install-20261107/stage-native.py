"""Sign the prepared unsigned r4 only after the parent's reviewed source gate."""
from __future__ import annotations
import argparse
import json
import subprocess
from common import *

def verify(folder: Path, *, signed: bool) -> dict:
    env={k:v for k,v in os.environ.items() if not k.upper().startswith('STRATA_')}
    run=subprocess.run([str(NODE),str(HERE/'verify-native.mjs'),str(folder)],cwd=ROOT,env=env,capture_output=True,text=True,timeout=180)
    require(run.returncode==0,'Current-root native integrity verification failed')
    result=json.loads(run.stdout)
    if signed:
        sys.path.insert(0,str(ROOT))
        from terminal.strata_control import signing, engine_component
        manifest=load(folder/'engine-component.json')
        signing.verify_bytes(signing.load_trusted_keys(environ={}),engine_component.signing_payload(manifest),manifest.get('signature'))
        result['signatureVerified']=True
    return result

def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--reviewed-commit')
    args=parser.parse_args()
    plan={'preparationOnly':True,'signingKeyExists':KEY.is_file(),'wouldSignNewCopy':True,'wouldActivateInstalledEngine':False,
          'parentReviewRequired':True,'unsignedR4Prepared':COMPOSITE_RECEIPT.is_file(),
          'executionBlockers':blockers(resources())+['parent-reviewed-source'],'releaseQualified':False}
    if not args.execute:
        print(json.dumps(plan));return
    approved_source(args.reviewed_commit);require_resources();require(OUT.is_dir(),'Run input audit first')
    require(not (OUT/'staged-native.json').exists() and not SIGNED.exists(),'Fresh native staging required')
    require(KEY.is_file(),'Signing key unavailable')
    require(sha(UNSIGNED/'engine-component.json')==UNSIGNED_MANIFEST,'Original unsigned manifest drift')
    r4=load(COMPOSITE_RECEIPT)
    require(r4['ok'] and r4['unsigned'] and r4['releaseQualified'] is False and r4['prismManifestSHA256']==PRISM_MANIFEST_SHA,'Unsigned Prism r4 receipt missing')
    require(sha(COMPOSITE/'engine-component.json')==r4['manifestSHA256'],'Unsigned r4 manifest drift')
    verify(COMPOSITE,signed=False);quality=quality_record()
    copied=preserved_copy(COMPOSITE,SIGNED)
    result=subprocess.run([str(PYTHON),'-B','packaging/feed_tool.py','sign-package',str(SIGNED),'--key',str(KEY)],
        cwd=ROOT,env=build_env(),capture_output=True,text=True,timeout=60)
    # Never log key material or a raw signing failure. The tool alone reads the key.
    require(result.returncode==0,'Private component signing failed')
    bound=verify(SIGNED,signed=True)
    signed_tree=inventory(SIGNED)
    require({k:v for k,v in signed_tree.items() if k!='engine-component.json'}==
            {k:v for k,v in copied['inventory'].items() if k!='engine-component.json'},'Signing changed payload bytes')
    require(sha(UNSIGNED/'engine-component.json')==UNSIGNED_MANIFEST,'Original unsigned receipt changed')
    engine_bundle=RESOURCES/'engine-bundle'
    existing=[p.parent for p in engine_bundle.rglob('engine-component.json')]
    require(len(existing)==1 and sha(existing[0]/'engine-component.json')==r4['manifestSHA256'],'Expected exact unsigned r4 staging')
    backup=OUT/'resource-backups/engine-unsigned-r4'
    before=preserved_copy(existing[0],backup)
    require(inventory(existing[0])==before['inventory'],'Old staged engine changed during backup')
    # The copy is retained before replacing only this managed staging directory.
    shutil.rmtree(existing[0])
    staged=preserved_copy(SIGNED,engine_bundle/SIGNED.name)
    require(staged['inventory']==signed_tree,'Signed staging differs')
    verify(engine_bundle/SIGNED.name,signed=True)
    package_out=OUT/'engine-archive-r4';require(not package_out.exists(),'Fresh package output required')
    subprocess.run([str(PYTHON),'-B','packaging/feed_tool.py','pack-engine',str(SIGNED),'--out',str(package_out),'--format','zip','--url-prefix='],
        cwd=ROOT,env=build_env(),check=True,timeout=300)
    archive,=package_out.glob('*.zip')
    write_new(OUT/'native-quality.json',dict(quality,signed=True,installed=False,publicRelease=False,nativeSourceCommit=NATIVE_COMMIT))
    receipt={'ok':True,'releaseQualified':False,'unsignedManifestSHA256':r4['manifestSHA256'],'originalR3ManifestSHA256':UNSIGNED_MANIFEST,'signedManifestSHA256':sha(SIGNED/'engine-component.json'),
        'prismVersion':'0.4.1','prismManifestSHA256':PRISM_MANIFEST_SHA,'prismExeSHA256':PRISM_EXE_SHA,
        'engineSHA256':ENGINE_SHA,'q36SHA256':Q36_SHA,'nativeBuildSHA256':sha(SIGNED/'native-build.json'),
        'archive':str(archive),'archiveSHA256':sha(archive),'archiveBytes':archive.stat().st_size,
        'signatureVerified':True,'currentRootVerified':bound['currentRootVerified'],'stagedFiles':len(signed_tree),
        'payloadInventorySHA256':digest(signed_tree),'nativeSourceCommit':NATIVE_COMMIT,'installed':False,'quality':quality}
    write_new(OUT/'staged-native.json',receipt);print(json.dumps(receipt))

if __name__=='__main__':main()
