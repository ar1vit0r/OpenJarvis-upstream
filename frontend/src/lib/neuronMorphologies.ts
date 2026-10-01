import * as THREE from 'three';

/**
 * Neuron morphology builders. Pure geometry
 * data (no THREE rendering/DOM), consumed by NeuronCoreMini, which owns the
 * combined-buffer-geometry rendering + per-frame animation.
 */

export type NeuronMorphology = 'pyramidal' | 'stellate' | 'purkinje' | 'bipolar' | 'network';

export interface NeuronBranch {
  somaIndex: number;
  dir: THREE.Vector3;
  length: number;
  curveSign: number;
  phase: number;
  perp: THREE.Vector3;
}

export interface NeuronTwig {
  parentIndex: number;
  tOnParent: number;
  dir: THREE.Vector3;
  length: number;
  curveSign: number;
  phase: number;
  perp: THREE.Vector3;
}

export interface NeuronSpec {
  somaPositions: THREE.Vector3[];
  branches: NeuronBranch[];
  twigs: NeuronTwig[];
  synapses: [number, number][];
}

export function perpOf(dir: THREE.Vector3): THREE.Vector3 {
  const ref = Math.abs(dir.y) < 0.9 ? new THREE.Vector3(0, 1, 0) : new THREE.Vector3(1, 0, 0);
  return new THREE.Vector3().crossVectors(dir, ref).normalize();
}

export function fibonacciSphereDirs(n: number): THREE.Vector3[] {
  const dirs: THREE.Vector3[] = [];
  const golden = Math.PI * (3 - Math.sqrt(5));
  for (let i = 0; i < n; i++) {
    const y = n === 1 ? 0 : 1 - (i / (n - 1)) * 2;
    const r = Math.sqrt(Math.max(0, 1 - y * y));
    const theta = golden * i;
    dirs.push(new THREE.Vector3(Math.cos(theta) * r, y, Math.sin(theta) * r));
  }
  return dirs;
}

export function coneSpreadDirs(
  n: number,
  axis: THREE.Vector3,
  angleDeg: number,
  planar = false,
): THREE.Vector3[] {
  const up = Math.abs(axis.y) < 0.9 ? new THREE.Vector3(0, 1, 0) : new THREE.Vector3(1, 0, 0);
  const basisX = new THREE.Vector3().crossVectors(axis, up).normalize();
  const basisZ = new THREE.Vector3().crossVectors(axis, basisX).normalize();
  const dirs: THREE.Vector3[] = [];
  for (let i = 0; i < n; i++) {
    const phi = (i / n) * Math.PI * 2 + (i % 2) * 0.3;
    const theta = THREE.MathUtils.degToRad(angleDeg) * (0.4 + 0.6 * ((i % 4) / 3));
    const planarPhi = planar ? (i / n) * Math.PI - Math.PI / 2 : phi;
    const dir = axis.clone().multiplyScalar(Math.cos(theta));
    const radial = basisX
      .clone()
      .multiplyScalar(Math.cos(planarPhi))
      .add(basisZ.clone().multiplyScalar(planar ? 0.15 * Math.sin(planarPhi) : Math.sin(planarPhi)))
      .multiplyScalar(Math.sin(theta));
    dir.add(radial);
    dirs.push(dir.normalize());
  }
  return dirs;
}

function createBuilder() {
  const somaPositions: THREE.Vector3[] = [];
  const branches: NeuronBranch[] = [];
  const twigs: NeuronTwig[] = [];
  const synapses: [number, number][] = [];

  function addBranch(somaIndex: number, dir: THREE.Vector3, length: number, curveSign: number): number {
    branches.push({
      somaIndex,
      dir: dir.clone(),
      length,
      curveSign,
      phase: Math.random() * Math.PI * 2,
      perp: perpOf(dir),
    });
    return branches.length - 1;
  }

  function addTwig(
    parentIndex: number,
    tOnParent: number,
    angleOffset: number,
    length: number,
    curveSign: number,
  ) {
    const parent = branches[parentIndex];
    const dir = parent.dir.clone().applyAxisAngle(parent.perp, angleOffset).normalize();
    twigs.push({
      parentIndex,
      tOnParent,
      dir,
      length,
      curveSign,
      phase: Math.random() * Math.PI * 2,
      perp: perpOf(dir),
    });
  }

  const build = (): NeuronSpec => ({ somaPositions, branches, twigs, synapses });

  return { somaPositions, branches, twigs, synapses, addBranch, addTwig, build };
}

// Evenly radial dendrites off one soma (50 branches x 9 twigs = 500 nodes).
// This is the baseline density every other morphology below is scaled to match.
const NUM_PRIMARY_BRANCHES = 50;
const TWIGS_PER_BRANCH = 9;

function buildStellate(): NeuronSpec {
  const b = createBuilder();
  b.somaPositions.push(new THREE.Vector3(0, 0, 0));

  fibonacciSphereDirs(NUM_PRIMARY_BRANCHES).forEach((d, i) => {
    const branch = b.addBranch(0, d, 1.0 + (i % 4) * 0.15, i % 2 === 0 ? 1 : -1);
    for (let j = 0; j < TWIGS_PER_BRANCH; j++) {
      const tOnParent = 0.22 + (j / TWIGS_PER_BRANCH) * 0.68;
      const angleOffset = (j % 2 === 0 ? 1 : -1) * (0.25 + (j / TWIGS_PER_BRANCH) * 0.55);
      const length = 0.22 + (j % 3) * 0.09;
      const curveSign = j % 2 === 0 ? 1 : -1;
      b.addTwig(branch, tOnParent, angleOffset, length, curveSign);
    }
  });

  return b.build();
}

// One long apical dendrite (with its own terminal tuft) + a spread of basal
// dendrites, at a node count comparable to stellate (~500).
const PYRAMIDAL_APICAL_TWIGS = 10;
const PYRAMIDAL_BASAL_BRANCHES = 44;
const PYRAMIDAL_BASAL_TWIGS_PER_BRANCH = 10;

function buildPyramidal(): NeuronSpec {
  const b = createBuilder();
  b.somaPositions.push(new THREE.Vector3(0, 0, 0));

  const apical = b.addBranch(0, new THREE.Vector3(0, 1, 0), 2.5, 1);
  for (let j = 0; j < PYRAMIDAL_APICAL_TWIGS; j++) {
    const tOnParent = 0.6 + (j / PYRAMIDAL_APICAL_TWIGS) * 0.35;
    const angleOffset = (j % 2 === 0 ? 1 : -1) * (0.35 + (j / PYRAMIDAL_APICAL_TWIGS) * 0.35);
    b.addTwig(apical, tOnParent, angleOffset, 0.3 + (j % 3) * 0.08, j % 2 === 0 ? 1 : -1);
  }

  coneSpreadDirs(PYRAMIDAL_BASAL_BRANCHES, new THREE.Vector3(0, -1, 0), 55).forEach((d, i) => {
    const branch = b.addBranch(0, d, 1.0 + (i % 4) * 0.1, i % 2 === 0 ? 1 : -1);
    for (let j = 0; j < PYRAMIDAL_BASAL_TWIGS_PER_BRANCH; j++) {
      const tOnParent = 0.25 + (j / PYRAMIDAL_BASAL_TWIGS_PER_BRANCH) * 0.65;
      const angleOffset = (j % 2 === 0 ? 1 : -1) * (0.3 + (j / PYRAMIDAL_BASAL_TWIGS_PER_BRANCH) * 0.4);
      b.addTwig(branch, tOnParent, angleOffset, 0.22 + (j % 3) * 0.07, j % 2 === 0 ? 1 : -1);
    }
  });

  return b.build();
}

// Dense fan of branching dendrites in one plane, at a node count comparable
// to stellate.
const PURKINJE_BRANCHES = 45;
const PURKINJE_TWIGS_PER_BRANCH = 10;

function buildPurkinje(): NeuronSpec {
  const b = createBuilder();
  b.somaPositions.push(new THREE.Vector3(0, 0, 0));

  coneSpreadDirs(PURKINJE_BRANCHES, new THREE.Vector3(0, 1, 0), 80, true).forEach((d, i) => {
    const branch = b.addBranch(0, d, 1.3 + (i % 3) * 0.15, i % 2 === 0 ? 1 : -1);
    for (let j = 0; j < PURKINJE_TWIGS_PER_BRANCH; j++) {
      const tOnParent = 0.35 + (j / PURKINJE_TWIGS_PER_BRANCH) * 0.55;
      const angleOffset = (j % 2 === 0 ? 1 : -1) * (0.3 + (j / PURKINJE_TWIGS_PER_BRANCH) * 0.4);
      b.addTwig(branch, tOnParent, angleOffset, 0.25 + (j % 3) * 0.08, j % 2 === 0 ? 1 : -1);
    }
  });

  return b.build();
}

// Two poles, each with a tuft of dendrites (a cone of branches around each
// pole axis, at a node count comparable to stellate).
const BIPOLAR_BRANCHES_PER_POLE = 25;
const BIPOLAR_TWIGS_PER_BRANCH = 9;

function buildBipolar(): NeuronSpec {
  const b = createBuilder();
  b.somaPositions.push(new THREE.Vector3(0, 0, 0));

  ([
    [new THREE.Vector3(0, 1, 0), 1],
    [new THREE.Vector3(0, -1, 0), -1],
  ] as const).forEach(([axis, poleSign]) => {
    coneSpreadDirs(BIPOLAR_BRANCHES_PER_POLE, axis, 30).forEach((d, i) => {
      const branch = b.addBranch(0, d, 1.7 + (i % 3) * 0.12, i % 2 === 0 ? poleSign : -poleSign);
      for (let j = 0; j < BIPOLAR_TWIGS_PER_BRANCH; j++) {
        const tOnParent = 0.5 + (j / BIPOLAR_TWIGS_PER_BRANCH) * 0.45;
        const angleOffset = (j % 2 === 0 ? 1 : -1) * (0.3 + (j / BIPOLAR_TWIGS_PER_BRANCH) * 0.4);
        b.addTwig(branch, tOnParent, angleOffset, 0.24 + (j % 3) * 0.08, j % 2 === 0 ? 1 : -1);
      }
    });
  });

  return b.build();
}

// A small cluster of neurons linked by synapses (5 somas x 20 branches x 4
// twigs = 500 nodes).
const NETWORK_SOMAS = 5;
const NETWORK_BRANCHES_PER_SOMA = 20;
const NETWORK_TWIGS_PER_BRANCH = 4;

function buildNetwork(): NeuronSpec {
  const b = createBuilder();
  fibonacciSphereDirs(NETWORK_SOMAS).forEach((d) => b.somaPositions.push(d.clone().multiplyScalar(1.6)));

  b.somaPositions.forEach((_pos, si) => {
    fibonacciSphereDirs(NETWORK_BRANCHES_PER_SOMA).forEach((d, i) => {
      const branch = b.addBranch(si, d, 0.5 + (i % 3) * 0.08, i % 2 === 0 ? 1 : -1);
      for (let j = 0; j < NETWORK_TWIGS_PER_BRANCH; j++) {
        const tOnParent = 0.35 + (j / NETWORK_TWIGS_PER_BRANCH) * 0.5;
        const angleOffset = (j % 2 === 0 ? 1 : -1) * (0.35 + (j / NETWORK_TWIGS_PER_BRANCH) * 0.35);
        b.addTwig(branch, tOnParent, angleOffset, 0.16 + (j % 2) * 0.05, j % 2 === 0 ? 1 : -1);
      }
    });
  });

  for (let i = 0; i < b.somaPositions.length; i++) {
    let best = -1;
    let bestDist = Infinity;
    for (let j = 0; j < b.somaPositions.length; j++) {
      if (i === j) continue;
      const d = b.somaPositions[i].distanceTo(b.somaPositions[j]);
      if (d < bestDist) {
        bestDist = d;
        best = j;
      }
    }
    if (best >= 0 && !b.synapses.some(([a, c]) => (a === i && c === best) || (a === best && c === i))) {
      b.synapses.push([i, best]);
    }
  }

  return b.build();
}

export const MORPHOLOGY_BUILDERS: Record<NeuronMorphology, () => NeuronSpec> = {
  pyramidal: buildPyramidal,
  stellate: buildStellate,
  purkinje: buildPurkinje,
  bipolar: buildBipolar,
  network: buildNetwork,
};

// Per-morphology camera framing. No group/camera rotation for any
// morphology (including network) — differing spatial extents are handled
// via distance/FOV only, so it never reads as a spinning logo. Starting
// placeholders; exact values are a manual visual tuning pass, not derived
// analytically.
export const MORPHOLOGY_CAMERA: Record<NeuronMorphology, { distance: number; fov: number }> = {
  stellate: { distance: 3.6, fov: 48 },
  pyramidal: { distance: 5.0, fov: 46 },
  purkinje: { distance: 4.0, fov: 46 },
  bipolar: { distance: 4.6, fov: 46 },
  network: { distance: 5.0, fov: 46 },
};

// [low-pitch, high-pitch] endpoints, lerped by the live pitch value.
export const PITCH_LENGTH_SCALE: [number, number] = [1.3, 0.75];
export const PITCH_CURVE_AMOUNT: [number, number] = [0.32, 0.08];
export const PITCH_SWAY_FREQ: [number, number] = [0.12, 0.35];

const MORPHOLOGY_STORAGE_KEY = 'openjarvis-neuron-morphology';

export function getConfiguredMorphology(): NeuronMorphology {
  try {
    const stored = localStorage.getItem(MORPHOLOGY_STORAGE_KEY);
    if (stored && Object.prototype.hasOwnProperty.call(MORPHOLOGY_BUILDERS, stored)) return stored as NeuronMorphology;
  } catch {
    // localStorage unavailable (e.g. privacy mode) — fall through to default.
  }
  return 'stellate';
}

export function setConfiguredMorphology(morphology: NeuronMorphology): void {
  try {
    localStorage.setItem(MORPHOLOGY_STORAGE_KEY, morphology);
  } catch {
    // Best-effort; nothing to recover here if storage is unavailable.
  }
}
