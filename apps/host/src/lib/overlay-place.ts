/**
 * Where an anchored panel (a menu, a popover) goes, in a page that may be scaled (#4653).
 *
 * "Interface size" in Settings scales the whole page with the CSS `zoom` property on the root. In Chromium (WebView2) and the engines
 * that follow the standard, `getBoundingClientRect()` and `window.innerWidth` then report what is on the screen (visual pixels), but the
 * `left`, `top`, `width` and `max-height` written on an element are in the element's own zoomed pixels: 100 written is 125 on the screen
 * at 125%. A menu placed from the anchor's rectangle and written back unchanged therefore lands (scale - 1) of its distance too far
 * right and down, which at the bottom-left of the window (the sidebar's "More" item) is off the screen: the click "does nothing".
 *
 * Everything here works in visual pixels and divides by the scale only at the end, when the numbers become CSS values.
 * `measureScale()` reads the scale from the page itself (a probe element), so it is right whichever way an engine counts.
 */

export interface Box { left: number; top: number; right: number; bottom: number }
export interface Size { width: number; height: number }

export interface Placement {
    /** CSS values for `position: fixed` (the element's own pixels). */
    left: number;
    top: number;
    maxWidth: number;
    maxHeight: number;
    /** True when the panel sits above its anchor. */
    above: boolean;
    /** Where the panel is on the screen (visual pixels): for checks. */
    visual: Box;
}

/** The least room (screen pixels) on one side of the anchor that is worth shortening the panel to. */
const MIN_ROOM = 120;

const finite = (n: number, fallback: number) => (Number.isFinite(n) ? n : fallback);

/** A scale that makes sense: engines and tests may hand over 0, NaN or nonsense. */
export function sane(scale: number): number {
    return Number.isFinite(scale) && scale >= 0.25 && scale <= 4 ? scale : 1;
}

export interface PlaceOptions {
    anchor: Box;
    /** The panel's size on the screen when it has all the room it wants. */
    size: Size;
    viewport: Size;
    /** Visual pixels per CSS pixel of the panel (1.25 at "Interface size" 125%). */
    scale: number;
    /** Air between the screen edge and the panel. */
    margin?: number;
    /** Air between the anchor and the panel when it opens below / above. */
    gapBelow?: number;
    gapAbove?: number;
    /** Open to the right of the anchor, top aligned (a tooltip beside a sidebar row), instead of below / above. */
    side?: boolean;
}

/**
 * Below the anchor when it fits, otherwise above it, otherwise on the side with more room, shortened to fit (the panel scrolls).
 * Left aligned with the anchor and kept inside the screen. Works for any viewport and scale.
 */
export function placeAnchored(o: PlaceOptions): Placement {
    const scale = sane(o.scale);
    const m = o.margin ?? 12;
    const vw = Math.max(0, finite(o.viewport.width, 0)), vh = Math.max(0, finite(o.viewport.height, 0));
    const maxW = Math.max(0, vw - 2 * m);
    const width = Math.min(Math.max(0, finite(o.size.width, 0)), maxW);
    let left: number, top: number, above = false, maxH: number;
    if (o.side) {
        left = Math.min(o.anchor.right + (o.gapBelow ?? 12), Math.max(m, vw - width - m));
        maxH = Math.max(0, vh - 2 * m);
        const height = Math.min(Math.max(0, finite(o.size.height, 0)), maxH);
        top = Math.max(m, Math.min(o.anchor.top, vh - height - m));
    }
    else {
        left = Math.max(m, Math.min(o.anchor.left, vw - width - m));
        const gapBelow = o.gapBelow ?? 8, gapAbove = o.gapAbove ?? 6;
        const height = Math.max(0, finite(o.size.height, 0));
        const roomBelow = vh - (o.anchor.bottom + gapBelow) - m;
        const roomAbove = o.anchor.top - gapAbove - m;
        above = height > roomBelow && roomAbove > roomBelow;
        const room = Math.max(0, above ? roomAbove : roomBelow);
        if (room < Math.min(height, MIN_ROOM)) {
            // a short window: neither side has room worth using, so the panel covers part of its anchor rather than shrinking to nothing
            above = false;
            maxH = Math.max(0, vh - 2 * m);
            const shown = Math.min(height, maxH);
            top = Math.max(m, Math.min(o.anchor.top, vh - shown - m));
        }
        else {
            maxH = room;
            const shown = Math.min(height, room);
            top = above ? o.anchor.top - gapAbove - shown : o.anchor.bottom + gapBelow;
        }
    }
    const heightShown = Math.min(Math.max(0, finite(o.size.height, 0)), maxH);
    return {
        left: left / scale,
        top: top / scale,
        maxWidth: maxW / scale,
        maxHeight: maxH / scale,
        above,
        visual: { left, top, right: left + width, bottom: top + heightShown },
    };
}

/**
 * Visual pixels per CSS pixel where `host` lives, found by measuring: a probe 100 CSS pixels wide is put in, read and taken out.
 * 1 when there is no zoom, 1.25 at 125%, and whatever the engine really does if it counts differently.
 */
export function measureScale(host: HTMLElement = document.body): number {
    try {
        const probe = document.createElement('div');
        probe.style.cssText = 'position:fixed;left:0;top:0;width:100px;height:1px;visibility:hidden;pointer-events:none;';
        host.appendChild(probe);
        const scale = probe.getBoundingClientRect().width / 100;
        probe.remove();
        return sane(scale);
    }
    catch {
        return 1;
    }
}
