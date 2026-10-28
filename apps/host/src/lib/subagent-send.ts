/** Keep a failed direct message in place. The text is the only copy the person may have. */
export function markSendFailed<T extends object>(entry: T, error: string): T & { state: 'failed'; error: string } {
    return { ...entry, state: 'failed', error };
}

/** A retry keeps the same client id, so a request that was accepted just before a timeout is not sent twice. */
export function markSendRetrying<T extends object>(entry: T): T & { state: 'sending'; error: null } {
    return { ...entry, state: 'sending', error: null };
}
