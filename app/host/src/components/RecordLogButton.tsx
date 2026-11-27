// strata-mi50 (2026-10-27): the Record log button, one for the whole app, always in the same place (top right, 2026-10-27 (2); top centre before) on every screen and
// chat, so a recording in progress is always visible (it was in the Monitor's header, 2026-10-26).  The recorder is
// lib/diagnostics-recorder.ts.
// 2026-10-27 (4): a fixed offset (right: 76px, then 114px) covered the header's own buttons on some screens (Models, Monitor),
// whose right-hand controls differ in width; the button now sits just left of the right-hand controls of the header on show.
import { useLayoutEffect, useRef } from 'react';
import { tr } from '../lib/i18n';
import { startRecording, stopRecording, useDiagnosticsRecording } from '../lib/diagnostics-recorder';
import { pushToast } from '../lib/store';

const GAP = 12;

function visibleHeader(): HTMLElement | undefined {
    return [...document.querySelectorAll<HTMLElement>('.main__head, .chat-toolbar')].find(h => h.offsetParent !== null && h.getBoundingClientRect().height > 0);
}

/** Left edge of the right-hand controls in the visible screen header, or null when there is none. */
function headerControlsLeft(header: HTMLElement | undefined): number | null {
    if (!header) return null;
    // a screen header keeps its right-hand controls in one block (it can start left of the middle, as on Tasks)
    const actions = header.querySelector<HTMLElement>(':scope > .main__actions');
    if (actions && actions.getBoundingClientRect().width > 0) return actions.getBoundingClientRect().left;
    const box = header.getBoundingClientRect();
    const middle = box.left + box.width / 2;
    let left: number | null = null;
    for (const el of header.querySelectorAll<HTMLElement>('button, a, input, select, [role="button"], [role="tab"]')) {
        const r = el.getBoundingClientRect();
        if (r.width === 0 || r.left < middle) continue;
        left = left === null ? r.left : Math.min(left, r.left);
    }
    return left;
}

export function RecordLogButton() {
    const recorder = useDiagnosticsRecording();
    const ref = useRef<HTMLButtonElement>(null);
    useLayoutEffect(() => {
        let frame = 0;
        const place = () => {
            frame = 0;
            const button = ref.current;
            if (!button) return;
            const header = visibleHeader();
            const left = headerControlsLeft(header);
            const right = left === null ? '' : `${Math.max(GAP, window.innerWidth - left + GAP)}px`;
            if (button.style.right !== right) button.style.right = right;
            // centred on the header, which moves down when a banner (connection, focus) sits above it
            const box = header?.getBoundingClientRect();
            const top = box ? `${Math.max(0, Math.round(box.top + (box.height - button.offsetHeight) / 2))}px` : '';
            if (button.style.top !== top) button.style.top = top;
        };
        // a timer, not requestAnimationFrame: frames stop while the window is hidden and a pending one would hold every later update
        const schedule = () => { if (!frame) frame = window.setTimeout(place, 30); };
        place();
        const observer = new MutationObserver(schedule);
        observer.observe(document.body, { childList: true, subtree: true, attributes: true, attributeFilter: ['class', 'style', 'hidden'] });
        window.addEventListener('resize', schedule);
        void document.fonts?.ready.then(schedule);
        return () => { observer.disconnect(); window.removeEventListener('resize', schedule); if (frame) clearTimeout(frame); };
    }, []);
    function toggle() {
        if (!recorder.recording) {
            startRecording();
            pushToast('success', tr('Recording the log'), tr('The Monitor readings are saved once a second, on every screen, until you press Stop.'));
            return;
        }
        try {
            const name = stopRecording();
            pushToast('success', tr('Log saved'), `${name} · ${tr('It is in your Downloads folder.')}`);
        }
        catch {
            pushToast('error', tr('The log was not saved'), tr('Try again.'));
        }
    }
    return <button ref={ref} type="button" className="record-log" data-on={recorder.recording} onClick={toggle}
        title={tr(recorder.recording ? 'Stop and save the log' : 'Record a log of every Monitor reading, once a second')}>
      <span className="record-log__dot" aria-hidden="true"/>
      <span>{tr(recorder.recording ? `Recording · ${recorder.count} s · Stop` : 'Record log')}</span>
    </button>;
}
