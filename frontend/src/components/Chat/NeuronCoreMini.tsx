import { useEffect, useRef } from 'react';
import * as THREE from 'three';
import {
  MORPHOLOGY_BUILDERS,
  MORPHOLOGY_CAMERA,
  PITCH_CURVE_AMOUNT,
  PITCH_LENGTH_SCALE,
  PITCH_SWAY_FREQ,
  getConfiguredMorphology,
  type NeuronMorphology,
} from '../../lib/neuronMorphologies';

/**
 * "Neuron Core" visual: a WebGL soma + dendrite tree driven by voice state,
 * mic amplitude, and pitch. Supports 5 morphologies (selectable via the
 * `morphology` prop, see neuronMorphologies.ts) and a pitch-driven shape
 * lerp fed by the `getPitch` prop.
 * No group/camera rotation for any morphology — only slow per-branch sway
 * (breathing, not spinning).
 */

export type VoiceAgentState = 'idle' | 'listening' | 'thinking' | 'speaking';

interface StateConfig {
  colorVar: string;
  speed: number;
  amp: number;
  label: string;
}

const STATE_CONFIG: Record<VoiceAgentState, StateConfig> = {
  idle: { colorVar: '--neuron-idle', speed: 0.3, amp: 0.15, label: 'Idle' },
  listening: { colorVar: '--neuron-listening', speed: 1.3, amp: 0.5, label: 'Listening' },
  thinking: { colorVar: '--neuron-thinking', speed: 2.4, amp: 0.25, label: 'Thinking' },
  speaking: { colorVar: '--neuron-speaking', speed: 1.7, amp: 0.55, label: 'Speaking' },
};

export function getStateLabel(state: VoiceAgentState): string {
  return STATE_CONFIG[state].label;
}

function resolveColor(el: HTMLElement, cssVar: string): string {
  const value = getComputedStyle(el).getPropertyValue(cssVar).trim();
  return value || '#888888';
}

export function withAlpha(hexOrColor: string, alphaHex: string): string {
  // Only hex colors carry an alpha channel this way; rgba()/named colors
  // pass through unchanged (still visible, just without the fade steps).
  if (/^#[0-9a-fA-F]{6}$/.test(hexOrColor)) {
    return `${hexOrColor}${alphaHex}`;
  }
  return hexOrColor;
}

const POINTS_PER_BRANCH = 10;
const SEGMENTS_PER_NODE = POINTS_PER_BRANCH - 1;

function writeQuadPoints(p0: THREE.Vector3, p1: THREE.Vector3, p2: THREE.Vector3, arr: Float32Array) {
  for (let k = 0; k < POINTS_PER_BRANCH; k++) {
    const tt = k / (POINTS_PER_BRANCH - 1);
    const mt = 1 - tt;
    arr[k * 3] = mt * mt * p0.x + 2 * mt * tt * p1.x + tt * tt * p2.x;
    arr[k * 3 + 1] = mt * mt * p0.y + 2 * mt * tt * p1.y + tt * tt * p2.y;
    arr[k * 3 + 2] = mt * mt * p0.z + 2 * mt * tt * p1.z + tt * tt * p2.z;
  }
}

const MIC_AMPLITUDE_GAIN = 2.5;
const MIC_ATTACK_TAU = 0.08;
const MIC_RELEASE_TAU = 0.25;
// Slower than the amplitude envelope above: a shape change reading jittery
// is more visually jarring than amplitude jitter, so pitch trades a little
// responsiveness for smoothness.
const PITCH_ATTACK_TAU = 0.15;
const PITCH_RELEASE_TAU = 0.4;
const PULSE_DECAY_SECONDS = 0.6;
const PULSE_AMPLITUDE_BOOST = 0.6;
const PULSE_REDUCED_MOTION_FLASH_MS = 220;

export interface NeuronPulse {
  /** 0-1ish; how strong the pulse reads (added to amplitude, see PULSE_AMPLITUDE_BOOST). */
  magnitude: number;
  /** CSS color the core briefly lerps toward; omit for an amplitude-only pulse (no hue shift). */
  tint?: string;
  /** Any value that changes per event — re-triggers the pulse even if magnitude/tint repeat. */
  nonce: number;
}

interface PulseTrigger {
  value: number;
  tint?: string;
  triggeredAt: number;
}

export function NeuronCoreMini({
  state,
  size = 40,
  color,
  morphology,
  getAmplitude,
  getPitch,
  pulse,
}: {
  state: VoiceAgentState;
  size?: number;
  /** Fixed CSS color; overrides the per-state CSS-variable color when set. */
  color?: string;
  /**
   * Which neuron morphology to render. Defaults to the user's configured
   * choice (see getConfiguredMorphology).
   * Read once at mount — changing this prop on an already-mounted instance
   * has no effect.
   */
  morphology?: NeuronMorphology;
  /**
   * Live mic-amplitude reader, 0-1. Polled every frame
   * while state === 'listening' and replaces the sine-breathing envelope
   * for that state only; ignored for idle/thinking/speaking.
   */
  getAmplitude?: () => number;
  /**
   * Live pitch reader, 0 (low) - 1 (high).
   * Polled every frame while state === 'listening', driving dendrite
   * length/curve/sway; ignored otherwise (shape stays at the neutral 0.5
   * midpoint, same as before pitch-reactivity existed).
   */
  getPitch?: () => number;
  /**
   * Discrete event pulse (e.g. a tool call starting) layered additively on
   * top of whatever continuous amplitude motion is running, independent of
   * `state`. Pass a new object (new `nonce`) to re-trigger.
   */
  pulse?: NeuronPulse | null;
}) {
  const mountRef = useRef<HTMLDivElement | null>(null);
  const stateRef = useRef(state);
  stateRef.current = state;
  const colorRef = useRef(color);
  colorRef.current = color;
  const getAmplitudeRef = useRef(getAmplitude);
  getAmplitudeRef.current = getAmplitude;
  const getPitchRef = useRef(getPitch);
  getPitchRef.current = getPitch;
  const pulseStateRef = useRef<PulseTrigger>({ value: 0, tint: undefined, triggeredAt: 0 });
  const reducedRef = useRef(
    typeof window !== 'undefined' ? window.matchMedia('(prefers-reduced-motion: reduce)').matches : false,
  );
  const requestFrameRef = useRef<(() => void) | null>(null);
  const morphologyRef = useRef<NeuronMorphology>(morphology ?? getConfiguredMorphology());

  useEffect(() => {
    if (!pulse) return;
    const prev = pulseStateRef.current;
    const now = performance.now();
    const elapsed = (now - prev.triggeredAt) / 1000;
    const prevDecayed = Math.max(0, prev.value * (1 - elapsed / PULSE_DECAY_SECONDS));
    pulseStateRef.current = {
      value: Math.max(prevDecayed, pulse.magnitude),
      tint: pulse.tint,
      triggeredAt: now,
    };

    if (!reducedRef.current) return;
    // No RAF loop is running to pick up the decay — render an instant
    // flash-then-clear instead of a smooth decay (a discrete state change,
    // not looping motion, so still shown under reduced motion).
    requestFrameRef.current?.();
    const timer = setTimeout(() => {
      pulseStateRef.current = { value: 0, tint: undefined, triggeredAt: performance.now() };
      requestFrameRef.current?.();
    }, PULSE_REDUCED_MOTION_FLASH_MS);
    return () => clearTimeout(timer);
  }, [pulse?.nonce]);

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;

    const cameraCfg = MORPHOLOGY_CAMERA[morphologyRef.current];
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(cameraCfg.fov, 1, 0.1, 100);
    camera.position.set(0, 0.1, cameraCfg.distance);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(size, size);
    mount.appendChild(renderer.domElement);

    const group = new THREE.Group();
    scene.add(group);

    const spec = MORPHOLOGY_BUILDERS[morphologyRef.current]();
    const totalNodes = spec.branches.length + spec.twigs.length;
    // Static per soma-owning-branch lookup, computed once outside the RAF
    // loop (soma positions never move).
    const branchOrigin = spec.branches.map((b) => spec.somaPositions[b.somaIndex]);

    const somaMeshes = spec.somaPositions.map((pos) => {
      const geo = new THREE.IcosahedronGeometry(0.32, 2);
      const mat = new THREE.MeshBasicMaterial({ color: 0x888888, transparent: true, opacity: 0.5 });
      const mesh = new THREE.Mesh(geo, mat);
      mesh.position.copy(pos);
      group.add(mesh);
      return { mesh, geo, mat };
    });

    // ~500 nodes as individual Line/Mesh objects would mean ~1000 draw
    // calls per frame, so lines and tips are each a single combined
    // geometry instead (2 draw calls total for the whole dendrite tree,
    // +1 more when synapses are present), with position buffers rewritten
    // in place every frame. Soma count is small and fixed per morphology
    // (<=5, for `network`) so per-soma mesh objects stay cheap.
    const lineGeo = new THREE.BufferGeometry();
    const linePositions = new Float32Array(totalNodes * SEGMENTS_PER_NODE * 2 * 3);
    lineGeo.setAttribute('position', new THREE.BufferAttribute(linePositions, 3));
    const lineMat = new THREE.LineBasicMaterial({ color: 0x888888, transparent: true, opacity: 0.6 });
    const lines = new THREE.LineSegments(lineGeo, lineMat);
    group.add(lines);

    const tipGeo = new THREE.BufferGeometry();
    const tipPositions = new Float32Array(totalNodes * 3);
    tipGeo.setAttribute('position', new THREE.BufferAttribute(tipPositions, 3));
    const tipMat = new THREE.PointsMaterial({
      color: 0x888888,
      size: 0.08,
      transparent: true,
      opacity: 0.8,
      sizeAttenuation: true,
    });
    const tips = new THREE.Points(tipGeo, tipMat);
    group.add(tips);

    let synapseLine: THREE.LineSegments | null = null;
    let synapseGeo: THREE.BufferGeometry | null = null;
    let synapseMat: THREE.LineBasicMaterial | null = null;
    if (spec.synapses.length > 0) {
      const arr = new Float32Array(spec.synapses.length * 2 * 3);
      spec.synapses.forEach(([a, c], i) => {
        const pa = spec.somaPositions[a];
        const pc = spec.somaPositions[c];
        arr[i * 6] = pa.x;
        arr[i * 6 + 1] = pa.y;
        arr[i * 6 + 2] = pa.z;
        arr[i * 6 + 3] = pc.x;
        arr[i * 6 + 4] = pc.y;
        arr[i * 6 + 5] = pc.z;
      });
      synapseGeo = new THREE.BufferGeometry();
      synapseGeo.setAttribute('position', new THREE.BufferAttribute(arr, 3));
      synapseMat = new THREE.LineBasicMaterial({ color: 0x888888, transparent: true, opacity: 0.35 });
      synapseLine = new THREE.LineSegments(synapseGeo, synapseMat);
      group.add(synapseLine);
    }

    const scratch = new Float32Array(POINTS_PER_BRANCH * 3);
    function writeNode(nodeIndex: number, p0: THREE.Vector3, p1: THREE.Vector3, p2: THREE.Vector3) {
      writeQuadPoints(p0, p1, p2, scratch);
      const segBase = nodeIndex * SEGMENTS_PER_NODE * 2 * 3;
      for (let k = 0; k < SEGMENTS_PER_NODE; k++) {
        const dst = segBase + k * 6;
        linePositions[dst] = scratch[k * 3];
        linePositions[dst + 1] = scratch[k * 3 + 1];
        linePositions[dst + 2] = scratch[k * 3 + 2];
        linePositions[dst + 3] = scratch[(k + 1) * 3];
        linePositions[dst + 4] = scratch[(k + 1) * 3 + 1];
        linePositions[dst + 5] = scratch[(k + 1) * 3 + 2];
      }
      const tipBase = nodeIndex * 3;
      const lastBase = (POINTS_PER_BRANCH - 1) * 3;
      tipPositions[tipBase] = scratch[lastBase];
      tipPositions[tipBase + 1] = scratch[lastBase + 1];
      tipPositions[tipBase + 2] = scratch[lastBase + 2];
    }

    const branchTip = spec.branches.map(() => new THREE.Vector3());

    let raf: number;
    let t = 0;
    let micSmoothed = 0;
    let pitchSmoothed = 0.5;
    const clock = new THREE.Clock();
    const branchColor = new THREE.Color();
    const pulseTintColor = new THREE.Color();
    const p1Scratch = new THREE.Vector3();
    const p2Scratch = new THREE.Vector3();
    const originScratch = new THREE.Vector3();
    const effDirScratch = new THREE.Vector3();

    // First JS-side prefers-reduced-motion check in this codebase (existing
    // instances are CSS-only @media blocks) — a canvas/WebGL RAF loop isn't
    // reachable via CSS media queries, so it needs its own gate here.
    const motionQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
    reducedRef.current = motionQuery.matches;

    function frame() {
      const dt = Math.min(clock.getDelta(), 0.05);
      t += dt * 0.8;

      const cfg = STATE_CONFIG[stateRef.current];
      let amplitude: number;
      if (stateRef.current === 'listening' && getAmplitudeRef.current && !reducedRef.current) {
        // Real mic input replaces the sine envelope for this state only —
        // fast attack / slower release so it reads as a VU meter rather
        // than jittering on every frame.
        const target = THREE.MathUtils.clamp(getAmplitudeRef.current() * MIC_AMPLITUDE_GAIN, 0, 1);
        const tau = target > micSmoothed ? MIC_ATTACK_TAU : MIC_RELEASE_TAU;
        micSmoothed += (target - micSmoothed) * (1 - Math.exp(-dt / tau));
        amplitude = cfg.amp * (0.4 + 1.2 * micSmoothed);
      } else {
        amplitude = cfg.amp * (0.6 + 0.4 * Math.sin(t * 1.5));
      }

      // Pitch drives dendrite shape (length/curve/sway), not color — kept a
      // separate channel from state/pulse color so the two signals don't
      // fight each other visually. Same listening-only gate as amplitude;
      // resets to the neutral 0.5 midpoint elsewhere so shape matches
      // pre-pitch-reactivity behavior in every other state.
      let pitchValue: number;
      if (stateRef.current === 'listening' && getPitchRef.current && !reducedRef.current) {
        const target = THREE.MathUtils.clamp(getPitchRef.current(), 0, 1);
        const tau = target > pitchSmoothed ? PITCH_ATTACK_TAU : PITCH_RELEASE_TAU;
        pitchSmoothed += (target - pitchSmoothed) * (1 - Math.exp(-dt / tau));
        pitchValue = pitchSmoothed;
      } else {
        pitchSmoothed = 0.5;
        pitchValue = 0.5;
      }
      const lengthScale = THREE.MathUtils.lerp(PITCH_LENGTH_SCALE[0], PITCH_LENGTH_SCALE[1], pitchValue);
      const curveAmount = THREE.MathUtils.lerp(PITCH_CURVE_AMOUNT[0], PITCH_CURVE_AMOUNT[1], pitchValue);
      const swayFreq = THREE.MathUtils.lerp(PITCH_SWAY_FREQ[0], PITCH_SWAY_FREQ[1], pitchValue);

      branchColor.set(colorRef.current || resolveColor(mount!, cfg.colorVar));

      // Event-pulse layer — additive on top of whichever amplitude branch
      // ran above, independent of state. Decays linearly from whatever
      // value/tint the pulse effect last set.
      const pulseState = pulseStateRef.current;
      const pulseElapsed = (performance.now() - pulseState.triggeredAt) / 1000;
      const pulseDecayed = Math.max(0, pulseState.value * (1 - pulseElapsed / PULSE_DECAY_SECONDS));
      if (pulseDecayed > 0) {
        amplitude = Math.min(1.4, amplitude + pulseDecayed * PULSE_AMPLITUDE_BOOST);
        if (pulseState.tint) {
          pulseTintColor.set(pulseState.tint);
          branchColor.lerp(pulseTintColor, pulseDecayed * 0.3);
        }
      }

      somaMeshes.forEach(({ mesh, mat }) => {
        mat.color.copy(branchColor);
        mat.opacity = 0.35 + amplitude * 0.4;
        mesh.scale.setScalar(1 + amplitude * 0.35 + 0.05 * Math.sin(t));
      });

      spec.branches.forEach((b, i) => {
        const origin = branchOrigin[i];
        const swayAngle = Math.sin(t * swayFreq + b.phase) * 0.1;
        const effDir = effDirScratch.copy(b.dir).applyAxisAngle(b.perp, swayAngle);
        const len = b.length * lengthScale * (0.92 + 0.08 * amplitude);
        p2Scratch.copy(origin).addScaledVector(effDir, len);
        p1Scratch
          .copy(origin)
          .addScaledVector(effDir, len * 0.5)
          .addScaledVector(b.perp, curveAmount * b.curveSign * len * 0.6);

        writeNode(i, origin, p1Scratch, p2Scratch);
        branchTip[i].copy(p2Scratch);
      });

      spec.twigs.forEach((tw, i) => {
        originScratch.copy(branchOrigin[tw.parentIndex]).lerp(branchTip[tw.parentIndex], tw.tOnParent);
        const swayAngle = Math.sin(t * swayFreq * 1.3 + tw.phase) * 0.12;
        const effDir = effDirScratch.copy(tw.dir).applyAxisAngle(tw.perp, swayAngle);
        const len = tw.length * lengthScale;
        p2Scratch.copy(originScratch).addScaledVector(effDir, len);
        p1Scratch
          .copy(originScratch)
          .addScaledVector(effDir, len * 0.5)
          .addScaledVector(tw.perp, curveAmount * tw.curveSign * len * 0.5);

        writeNode(spec.branches.length + i, originScratch, p1Scratch, p2Scratch);
      });

      lineGeo.attributes.position.needsUpdate = true;
      tipGeo.attributes.position.needsUpdate = true;
      lineMat.color.copy(branchColor);
      lineMat.opacity = 0.6 + amplitude * 0.35;
      tipMat.color.copy(branchColor);
      tipMat.opacity = 0.65 + amplitude * 0.4;
      if (synapseMat) {
        synapseMat.color.copy(branchColor);
        synapseMat.opacity = 0.2 + amplitude * 0.3;
      }

      renderer.render(scene, camera);
    }

    function loop() {
      frame();
      raf = requestAnimationFrame(loop);
    }

    function handleMotionChange(e: MediaQueryListEvent) {
      reducedRef.current = e.matches;
      if (reducedRef.current) {
        cancelAnimationFrame(raf);
        frame(); // one static frame, no sine breathing, no RAF scheduled
      } else {
        clock.getDelta(); // discard the delta accumulated while paused
        loop();
      }
    }
    motionQuery.addEventListener('change', handleMotionChange);
    requestFrameRef.current = frame;

    if (reducedRef.current) {
      frame();
    } else {
      loop();
    }

    return () => {
      cancelAnimationFrame(raf);
      motionQuery.removeEventListener('change', handleMotionChange);
      requestFrameRef.current = null;
      mount.removeChild(renderer.domElement);
      somaMeshes.forEach(({ geo, mat }) => {
        geo.dispose();
        mat.dispose();
      });
      lineGeo.dispose();
      lineMat.dispose();
      tipGeo.dispose();
      tipMat.dispose();
      synapseGeo?.dispose();
      synapseMat?.dispose();
      renderer.dispose();
    };
  }, [size]);

  return <div ref={mountRef} style={{ width: size, height: size }} />;
}
