/** Minimal typed event emitter (no dependencies — SDK size budget). */

export type Listener<T> = (e: T) => void;

export class Emitter<Events extends Record<string, unknown>> {
  private listeners = new Map<keyof Events, Set<Listener<never>>>();

  /** Returns an unsubscribe function. */
  on<K extends keyof Events>(event: K, cb: Listener<Events[K]>): () => void {
    let set = this.listeners.get(event);
    if (!set) {
      set = new Set();
      this.listeners.set(event, set);
    }
    set.add(cb as Listener<never>);
    return () => {
      set.delete(cb as Listener<never>);
    };
  }

  emit<K extends keyof Events>(event: K, payload: Events[K]): void {
    const set = this.listeners.get(event);
    if (!set) return;
    for (const cb of set) (cb as Listener<Events[K]>)(payload);
  }

  clear(): void {
    this.listeners.clear();
  }
}
