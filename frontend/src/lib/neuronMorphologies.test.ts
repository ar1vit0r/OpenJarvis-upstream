import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import {
  MORPHOLOGY_BUILDERS,
  getConfiguredMorphology,
  setConfiguredMorphology,
  type NeuronMorphology,
} from './neuronMorphologies';

// Minimal in-memory localStorage stub so these run under node (no jsdom
// dependency) — same pattern as api.auth.test.ts.
class MemoryStorage {
  private store = new Map<string, string>();
  getItem(k: string): string | null {
    return this.store.has(k) ? (this.store.get(k) as string) : null;
  }
  setItem(k: string, v: string): void {
    this.store.set(k, String(v));
  }
  removeItem(k: string): void {
    this.store.delete(k);
  }
  clear(): void {
    this.store.clear();
  }
}

beforeEach(() => {
  (globalThis as unknown as { localStorage: MemoryStorage }).localStorage = new MemoryStorage();
});

afterEach(() => {
  (globalThis as unknown as { localStorage?: MemoryStorage }).localStorage = undefined;
});

const MORPHOLOGIES = Object.keys(MORPHOLOGY_BUILDERS) as NeuronMorphology[];

describe('MORPHOLOGY_BUILDERS', () => {
  it.each(MORPHOLOGIES)('%s returns at least 1 soma position', (key) => {
    const spec = MORPHOLOGY_BUILDERS[key]();
    expect(spec.somaPositions.length).toBeGreaterThanOrEqual(1);
  });

  it('network returns exactly 5 soma positions', () => {
    const spec = MORPHOLOGY_BUILDERS.network();
    expect(spec.somaPositions.length).toBe(5);
  });

  it.each(MORPHOLOGIES)('%s has a sane positive node count', (key) => {
    const spec = MORPHOLOGY_BUILDERS[key]();
    const totalNodes = spec.branches.length + spec.twigs.length;
    expect(totalNodes).toBeGreaterThan(0);
  });

  it.each(MORPHOLOGIES)('%s has no zero-length/NaN branch or twig directions', (key) => {
    const spec = MORPHOLOGY_BUILDERS[key]();
    for (const b of spec.branches) {
      expect(Number.isFinite(b.dir.length())).toBe(true);
      expect(b.dir.length()).toBeGreaterThan(0);
    }
    for (const tw of spec.twigs) {
      expect(Number.isFinite(tw.dir.length())).toBe(true);
      expect(tw.dir.length()).toBeGreaterThan(0);
    }
  });

  it.each(MORPHOLOGIES)('%s has every branch somaIndex valid', (key) => {
    const spec = MORPHOLOGY_BUILDERS[key]();
    for (const b of spec.branches) {
      expect(b.somaIndex).toBeGreaterThanOrEqual(0);
      expect(b.somaIndex).toBeLessThan(spec.somaPositions.length);
    }
  });

  it('network has no duplicate/reciprocal synapse pairs', () => {
    const spec = MORPHOLOGY_BUILDERS.network();
    const seen = new Set<string>();
    for (const [a, b] of spec.synapses) {
      const key = a < b ? `${a}-${b}` : `${b}-${a}`;
      expect(seen.has(key)).toBe(false);
      seen.add(key);
    }
  });

  it('non-network morphologies have no synapses', () => {
    for (const key of MORPHOLOGIES) {
      if (key === 'network') continue;
      expect(MORPHOLOGY_BUILDERS[key]().synapses).toEqual([]);
    }
  });
});

describe('getConfiguredMorphology', () => {
  it('defaults to stellate when unset', () => {
    expect(getConfiguredMorphology()).toBe('stellate');
  });

  it('falls back to stellate for an invalid stored value', () => {
    localStorage.setItem('openjarvis-neuron-morphology', 'not-a-real-morphology');
    expect(getConfiguredMorphology()).toBe('stellate');
  });

  it('falls back to stellate for an Object.prototype key', () => {
    localStorage.setItem('openjarvis-neuron-morphology', 'constructor');
    expect(getConfiguredMorphology()).toBe('stellate');
  });

  it('round-trips a valid stored value via setConfiguredMorphology', () => {
    setConfiguredMorphology('network');
    expect(getConfiguredMorphology()).toBe('network');
  });
});
