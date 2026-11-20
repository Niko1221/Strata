import { useEffect, useRef } from 'react';

/** A response belongs to one visit to an editor, including A → B → A navigation. */
export function useRequestOwner(identity: string) {
    const owner = useRef({ identity });
    if (owner.current.identity !== identity)
        owner.current = { identity };
    useEffect(() => {
        const visit = owner.current;
        return () => {
            if (owner.current === visit)
                owner.current = { identity };
        };
    }, [identity]);
    return owner;
}
