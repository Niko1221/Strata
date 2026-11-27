import React, { useState } from 'react';
import { createRoot } from 'react-dom/client';
import { ComposerControls } from '../../src/components/ComposerControls';
function Probe() {
  const [error, setError] = useState('');
  const [attached, setAttached] = useState('');
  return <><div id="notice">{error}</div><div id="attached">{attached}</div><ComposerControls session={{id:'test'} as any} draft="" onDraft={()=>{}} onPreview={()=>{}} onAttach={(_name, text)=>setAttached(text)} onError={setError} onSlash={()=>{}} streaming={false}/></>;
}
createRoot(document.getElementById('root')!).render(<Probe/>);
