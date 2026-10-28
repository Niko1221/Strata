import { FOLLOW_SLACK, PILL_DISTANCE, distanceFromBottom, followAfterScroll, wheelLeavesBottom, KEYS_UP } from './follow-scroll';

/**
 * The helper panel follows the same reader-driven rule as the main chat (#4014):
 * a wheel or key moving up lets go at once, and being back at the bottom follows again.
 * The extra rule here is the "new content" count: it only grows while the reader is away.
 */
export { FOLLOW_SLACK, PILL_DISTANCE, distanceFromBottom, followAfterScroll, wheelLeavesBottom, KEYS_UP };

export function newContentCount(previous: number, current: number, following: boolean): number {
    if (following) return 0;
    return Math.max(0, current - previous);
}
