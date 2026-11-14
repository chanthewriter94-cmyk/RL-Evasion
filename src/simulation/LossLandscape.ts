import { EvasionEngine, SIM_HEIGHT, SIM_WIDTH, SimParams } from './EvasionEngine';
import { NeuralNetwork } from './NeuralNetwork';

export type LandscapeMode = 'spatial' | 'weight';
export type LandscapeView = '1d' | '2d' | '3d' | '2d+3d';
export type LandscapeMetric = 'loss' | 'fitness';

export interface LandscapeSlicePoint {
  t: number;
  fitness: number;
  hit: boolean;
}

export interface LandscapeMarker {
  u: number;
  v: number;
  isBest?: boolean;
  isEnemy?: boolean;
}

export interface LandscapeData {
  mode: LandscapeMode;
  gridW: number;
  gridH: number;
  span: number;
  values: number[][];
  hitMask: boolean[][];
  minFitness: number;
  maxFitness: number;
  centerFitness: number;
  slice1D: LandscapeSlicePoint[];
  slice1DLabel: string;
  markerNorm: { u: number; v: number };
  markers: LandscapeMarker[];
  progress: number;
  evalSteps: number;
  generation: number;
}

interface WeightTensor {
  weights: number[][][];
  biases: number[][];
}

function createMulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function boxMuller(rng: () => number): number {
  const u1 = Math.max(1e-7, rng());
  const u2 = rng();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

function buildFilterNormalizedDirections(
  net: NeuralNetwork,
  seed: number
): { d1: WeightTensor; d2: WeightTensor } {
  const rng = createMulberry32(seed);
  const d1: WeightTensor = { weights: [], biases: [] };
  const d2: WeightTensor = { weights: [], biases: [] };

  for (let l = 0; l < net.weights.length; l++) {
    const wMat1: number[][] = [];
    const wMat2: number[][] = [];
    const bVec1: number[] = [];
    const bVec2: number[] = [];

    for (let r = 0; r < net.weights[l].length; r++) {
      const row = net.weights[l][r];
      const fRow = net.frozenWeights[l]?.[r];
      const isBiasFrozen = Boolean(net.frozenBiases[l]?.[r]);

      let wNormSq = 0;
      const raw1: number[] = [];
      const raw2: number[] = [];
      let raw1NormSq = 0;
      let raw2NormSq = 0;

      for (let c = 0; c < row.length; c++) {
        wNormSq += row[c] * row[c];
        if (fRow?.[c]) {
          raw1.push(0);
          raw2.push(0);
        } else {
          const g1 = boxMuller(rng);
          const g2 = boxMuller(rng);
          raw1.push(g1);
          raw2.push(g2);
          raw1NormSq += g1 * g1;
          raw2NormSq += g2 * g2;
        }
      }

      // Orthogonalize raw2 against raw1 per filter row
      let dot12 = 0;
      if (raw1NormSq > 1e-8) {
        for (let c = 0; c < raw1.length; c++) {
          dot12 += raw1[c] * raw2[c];
        }
        const proj = dot12 / raw1NormSq;
        raw2NormSq = 0;
        for (let c = 0; c < raw2.length; c++) {
          raw2[c] -= proj * raw1[c];
          raw2NormSq += raw2[c] * raw2[c];
        }
      }

      const targetNorm = Math.max(0.25, Math.sqrt(wNormSq));
      const scale1 = raw1NormSq > 1e-8 ? targetNorm / Math.sqrt(raw1NormSq) : 0;
      const scale2 = raw2NormSq > 1e-8 ? targetNorm / Math.sqrt(raw2NormSq) : 0;

      wMat1.push(raw1.map((v) => v * scale1));
      wMat2.push(raw2.map((v) => v * scale2));

      if (isBiasFrozen) {
        bVec1.push(0);
        bVec2.push(0);
      } else {
        const bMag = Math.max(0.1, Math.abs(net.biases[l][r]));
        bVec1.push(boxMuller(rng) * bMag * 0.35);
        bVec2.push(boxMuller(rng) * bMag * 0.35);
      }
    }

    d1.weights.push(wMat1);
    d2.weights.push(wMat2);
    d1.biases.push(bVec1);
    d2.biases.push(bVec2);
  }

  return { d1, d2 };
}

function projectNetworkOntoBasis(
  target: NeuralNetwork,
  base: NeuralNetwork,
  d1: WeightTensor,
  d2: WeightTensor
): { alpha: number; beta: number } {
  let dot1 = 0;
  let dot2 = 0;
  let norm1 = 0;
  let norm2 = 0;

  for (let l = 0; l < base.weights.length; l++) {
    for (let r = 0; r < base.weights[l].length; r++) {
      for (let c = 0; c < base.weights[l][r].length; c++) {
        const dw = (target.weights[l]?.[r]?.[c] ?? 0) - base.weights[l][r][c];
        const v1 = d1.weights[l][r][c];
        const v2 = d2.weights[l][r][c];
        dot1 += dw * v1;
        dot2 += dw * v2;
        norm1 += v1 * v1;
        norm2 += v2 * v2;
      }
    }
  }

  return {
    alpha: norm1 > 1e-8 ? dot1 / norm1 : 0,
    beta: norm2 > 1e-8 ? dot2 / norm2 : 0,
  };
}

function createPerturbedNetwork(
  base: NeuralNetwork,
  d1: WeightTensor,
  d2: WeightTensor,
  alpha: number,
  beta: number
): NeuralNetwork {
  const net = base.clone();
  for (let l = 0; l < net.weights.length; l++) {
    for (let r = 0; r < net.weights[l].length; r++) {
      net.biases[l][r] += alpha * d1.biases[l][r] + beta * d2.biases[l][r];
      for (let c = 0; c < net.weights[l][r].length; c++) {
        net.weights[l][r][c] += alpha * d1.weights[l][r][c] + beta * d2.weights[l][r][c];
      }
    }
  }
  return net;
}

function createInterpolatedNetwork(
  fromNet: NeuralNetwork,
  toNet: NeuralNetwork,
  t: number
): NeuralNetwork {
  const net = fromNet.clone();
  for (let l = 0; l < net.weights.length; l++) {
    for (let r = 0; r < net.weights[l].length; r++) {
      const tb = toNet.biases[l]?.[r] ?? net.biases[l][r];
      net.biases[l][r] = net.biases[l][r] + (tb - net.biases[l][r]) * t;
      for (let c = 0; c < net.weights[l][r].length; c++) {
        const tw = toNet.weights[l]?.[r]?.[c] ?? net.weights[l][r][c];
        net.weights[l][r][c] = net.weights[l][r][c] + (tw - net.weights[l][r][c]) * t;
      }
    }
  }
  return net;
}

export function scanSpatialLandscape(
  engine: EvasionEngine,
  gridW = 25,
  gridH = 17
): LandscapeData {
  const values: number[][] = [];
  const hitMask: boolean[][] = [];
  let minFitness = Infinity;
  let maxFitness = -Infinity;

  const pad = 10;
  for (let row = 0; row < gridH; row++) {
    const py = pad + (row / Math.max(1, gridH - 1)) * (SIM_HEIGHT - pad * 2);
    const valRow: number[] = [];
    const hitRow: boolean[] = [];
    for (let col = 0; col < gridW; col++) {
      const px = pad + (col / Math.max(1, gridW - 1)) * (SIM_WIDTH - pad * 2);
      const res = engine.evaluateSpatialSample(px, py);
      valRow.push(res.fitness);
      hitRow.push(res.hit);
      if (res.fitness < minFitness) minFitness = res.fitness;
      if (res.fitness > maxFitness) maxFitness = res.fitness;
    }
    values.push(valRow);
    hitMask.push(hitRow);
  }

  const slice1D: LandscapeSlicePoint[] = [];
  const sliceSamples = 35;
  for (let i = 0; i < sliceSamples; i++) {
    const t = i / (sliceSamples - 1);
    const px = pad + t * (SIM_WIDTH - pad * 2);
    const res = engine.evaluateSpatialSample(px, engine.player.y);
    slice1D.push({ t: px, fitness: res.fitness, hit: res.hit });
  }

  const centerRes = engine.evaluateSpatialSample(engine.player.x, engine.player.y);
  const markerU = Math.max(0, Math.min(1, (engine.player.x - pad) / (SIM_WIDTH - pad * 2)));
  const markerV = Math.max(0, Math.min(1, (engine.player.y - pad) / (SIM_HEIGHT - pad * 2)));

  const markers: LandscapeMarker[] = engine.enemies
    .filter((e) => !e.spawnDelay || e.spawnDelay <= 0)
    .map((e) => ({
      u: Math.max(0, Math.min(1, (e.x - pad) / (SIM_WIDTH - pad * 2))),
      v: Math.max(0, Math.min(1, (e.y - pad) / (SIM_HEIGHT - pad * 2))),
      isEnemy: true,
    }));

  if (!Number.isFinite(minFitness)) minFitness = 0;
  if (!Number.isFinite(maxFitness)) maxFitness = 1;

  return {
    mode: 'spatial',
    gridW,
    gridH,
    span: 1,
    values,
    hitMask,
    minFitness,
    maxFitness,
    centerFitness: centerRes.fitness,
    slice1D,
    slice1DLabel: `Arena X @ y=${Math.round(engine.player.y)}`,
    markerNorm: { u: markerU, v: markerV },
    markers,
    progress: 1,
    evalSteps: 1,
    generation: engine.generation,
  };
}

export class WeightSpaceScanner {
  private baseNet: NeuralNetwork;
  private bestNet: NeuralNetwork | null;
  private params: SimParams;
  private d1: WeightTensor;
  private d2: WeightTensor;
  private seedPlayer: { x: number; y: number; vx: number; vy: number };
  private seedEnemies: { x: number; y: number; spawnDelay: number; entered: boolean; speedPct?: number }[];
  private gridSize: number;
  private rolloutSteps: number;
  private span: number;
  private generation: number;
  private nextRow = 0;
  private sliceDone = false;

  data: LandscapeData;

  constructor(engine: EvasionEngine, gridSize = 13, rolloutSteps = 110, span = 1.2) {
    this.gridSize = gridSize;
    this.rolloutSteps = rolloutSteps;
    this.span = span;
    this.generation = engine.generation;
    this.params = { ...engine.params };
    this.baseNet = (engine.agent ?? engine.bestAgent ?? new NeuralNetwork(104, 1, 16, 2, 'tanh')).clone();
    this.bestNet =
      engine.bestAgent && engine.agent && engine.bestAgent !== engine.agent
        ? engine.bestAgent.clone()
        : null;

    // Seed deterministic enemy ring / snapshot from current episode or balanced ring
    this.seedPlayer = { x: SIM_WIDTH / 2, y: SIM_HEIGHT / 2, vx: 0, vy: 0 };
    const active = engine.enemies.filter((e) => !e.spawnDelay || e.spawnDelay <= 0);
    if (active.length > 0) {
      this.seedEnemies = active.map((e) => ({
        x: e.x,
        y: e.y,
        spawnDelay: 0,
        entered: true,
        speedPct: e.speedPct,
      }));
      this.seedPlayer = { ...engine.player };
    } else {
      const count = engine.params.numEnemies;
      this.seedEnemies = [];
      for (let i = 0; i < count; i++) {
        const ang = (i / Math.max(1, count)) * Math.PI * 2 + 0.35;
        this.seedEnemies.push({
          x: Math.max(16, Math.min(SIM_WIDTH - 16, SIM_WIDTH * 0.5 + Math.cos(ang) * 150)),
          y: Math.max(16, Math.min(SIM_HEIGHT - 16, SIM_HEIGHT * 0.5 + Math.sin(ang) * 120)),
          spawnDelay: 0,
          entered: true,
          speedPct: engine.params.enemySpeedPct,
        });
      }
    }

    const basisSeed = 1337 + engine.generation * 97 + this.baseNet.weights.length * 17;
    const { d1, d2 } = buildFilterNormalizedDirections(this.baseNet, basisSeed);
    this.d1 = d1;
    this.d2 = d2;

    const values: number[][] = [];
    const hitMask: boolean[][] = [];
    for (let r = 0; r < gridSize; r++) {
      values.push(new Array(gridSize).fill(0));
      hitMask.push(new Array(gridSize).fill(false));
    }

    const popMarkers: LandscapeMarker[] = [];
    for (const member of engine.population) {
      const proj = projectNetworkOntoBasis(member, this.baseNet, d1, d2);
      const u = (proj.alpha + span) / (span * 2);
      const v = (proj.beta + span) / (span * 2);
      if (u >= 0 && u <= 1 && v >= 0 && v <= 1) {
        popMarkers.push({ u, v, isBest: false });
      }
    }
    if (this.bestNet) {
      const bestProj = projectNetworkOntoBasis(this.bestNet, this.baseNet, d1, d2);
      const bu = (bestProj.alpha + span) / (span * 2);
      const bv = (bestProj.beta + span) / (span * 2);
      if (bu >= 0 && bu <= 1 && bv >= 0 && bv <= 1) {
        popMarkers.push({ u: bu, v: bv, isBest: true });
      }
    }

    this.data = {
      mode: 'weight',
      gridW: gridSize,
      gridH: gridSize,
      span,
      values,
      hitMask,
      minFitness: 0,
      maxFitness: 0,
      centerFitness: 0,
      slice1D: [],
      slice1DLabel: this.bestNet ? 'Current (0) -> Best (1)' : `α Slice [-${span}, +${span}]`,
      markerNorm: { u: 0.5, v: 0.5 },
      markers: popMarkers,
      progress: 0,
      evalSteps: rolloutSteps,
      generation: this.generation,
    };
  }

  get isComplete(): boolean {
    return this.nextRow >= this.gridSize && this.sliceDone;
  }

  private evaluateNet(net: NeuralNetwork): { fitness: number; hit: boolean } {
    const probe = new EvasionEngine(this.params);
    probe.population = [net];
    probe.popIndex = 0;
    probe.agent = net;
    probe.bestAgent = null;
    probe.player = { ...this.seedPlayer };
    probe.enemies = this.seedEnemies.map((e) => ({ ...e }));
    probe.stepCount = 0;
    probe.currentFitness = 0;
    probe.prevVx = 0;
    probe.prevVy = 0;
    probe.emaVx = 0;
    probe.emaVy = 0;
    probe.emaPx = probe.player.x;
    probe.emaPy = probe.player.y;
    probe.collectedSplitGaps = [];
    probe.splitGapHistory = [];

    let hit = false;
    for (let s = 0; s < this.rolloutSteps; s++) {
      probe.update();
      if (probe.stepCount === 0) {
        hit = true;
        return { fitness: net.fitness, hit };
      }
    }
    return { fitness: probe.currentFitness, hit };
  }

  stepRows(maxRows = 2): LandscapeData {
    if (!this.sliceDone) {
      const samples = 25;
      const slice: LandscapeSlicePoint[] = [];
      for (let i = 0; i < samples; i++) {
        if (this.bestNet) {
          const t = -0.2 + (i / (samples - 1)) * 1.4;
          const interp = createInterpolatedNetwork(this.baseNet, this.bestNet, t);
          const res = this.evaluateNet(interp);
          slice.push({ t, fitness: res.fitness, hit: res.hit });
        } else {
          const alpha = -this.span + (i / (samples - 1)) * (this.span * 2);
          const pert = createPerturbedNetwork(this.baseNet, this.d1, this.d2, alpha, 0);
          const res = this.evaluateNet(pert);
          slice.push({ t: alpha, fitness: res.fitness, hit: res.hit });
        }
      }
      this.data.slice1D = slice;
      this.sliceDone = true;
    }

    const endRow = Math.min(this.gridSize, this.nextRow + maxRows);
    for (let r = this.nextRow; r < endRow; r++) {
      const beta = -this.span + (r / Math.max(1, this.gridSize - 1)) * (this.span * 2);
      for (let c = 0; c < this.gridSize; c++) {
        const alpha = -this.span + (c / Math.max(1, this.gridSize - 1)) * (this.span * 2);
        const net =
          Math.abs(alpha) < 1e-6 && Math.abs(beta) < 1e-6
            ? this.baseNet
            : createPerturbedNetwork(this.baseNet, this.d1, this.d2, alpha, beta);
        const res = this.evaluateNet(net);
        this.data.values[r][c] = res.fitness;
        this.data.hitMask[r][c] = res.hit;
      }
    }
    this.nextRow = endRow;

    // Recompute min/max across scanned rows so far
    let minF = Infinity;
    let maxF = -Infinity;
    for (let r = 0; r < this.nextRow; r++) {
      for (let c = 0; c < this.gridSize; c++) {
        const val = this.data.values[r][c];
        if (val < minF) minF = val;
        if (val > maxF) maxF = val;
      }
    }
    const mid = Math.floor(this.gridSize / 2);
    this.data.minFitness = Number.isFinite(minF) ? minF : 0;
    this.data.maxFitness = Number.isFinite(maxF) ? maxF : 1;
    this.data.centerFitness = this.nextRow > mid ? this.data.values[mid][mid] : 0;
    this.data.progress = this.nextRow / this.gridSize;
    return this.data;
  }

  finishAll(): LandscapeData {
    while (!this.isComplete) {
      this.stepRows(this.gridSize);
    }
    return this.data;
  }
}

function sampleElevationColor(t: number, metric: LandscapeMetric): [number, number, number] {
  // t in [0, 1] where 0 = min value (low loss or low fitness), 1 = max value (high loss or high fitness)
  const clamped = Math.max(0, Math.min(1, t));
  if (metric === 'loss') {
    // Low loss (good) = cyan/teal -> mid = dark indigo/slate -> high loss (bad) = bright orange/red
    if (clamped < 0.5) {
      const k = clamped * 2;
      return [
        Math.round(10 + k * 40),
        Math.round(220 - k * 155),
        Math.round(235 - k * 135),
      ];
    }
    const k = (clamped - 0.5) * 2;
    return [
      Math.round(50 + k * 200),
      Math.round(65 + (k < 0.5 ? k * 80 : (1 - k) * 80)),
      Math.round(100 - k * 75),
    ];
  }
  // Fitness: low fitness = deep red -> mid = slate -> high fitness (good) = bright cyan
  if (clamped < 0.5) {
    const k = clamped * 2;
    return [
      Math.round(230 - k * 180),
      Math.round(35 + k * 35),
      Math.round(35 + k * 75),
    ];
  }
  const k = (clamped - 0.5) * 2;
  return [
    Math.round(50 - k * 40),
    Math.round(70 + k * 175),
    Math.round(110 + k * 145),
  ];
}

function toMetricValue(fitness: number, metric: LandscapeMetric): number {
  return metric === 'loss' ? -fitness : fitness;
}

function getMetricBounds(data: LandscapeData, metric: LandscapeMetric): { minV: number; maxV: number } {
  if (metric === 'loss') {
    return { minV: -data.maxFitness, maxV: -data.minFitness };
  }
  return { minV: data.minFitness, maxV: data.maxFitness };
}

function formatCompactNum(v: number): string {
  const abs = Math.abs(v);
  if (abs >= 10000) return `${(v / 1000).toFixed(1)}k`;
  if (abs >= 100) return `${Math.round(v)}`;
  return v.toFixed(1);
}

export function drawLandscape1D(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  data: LandscapeData,
  metric: LandscapeMetric
): void {
  ctx.save();
  ctx.translate(x, y);

  ctx.fillStyle = '#070707';
  ctx.fillRect(0, 0, w, h);
  ctx.strokeStyle = '#262626';
  ctx.lineWidth = 1;
  ctx.strokeRect(0.5, 0.5, w - 1, h - 1);

  const pts = data.slice1D;
  if (pts.length < 2) {
    ctx.fillStyle = '#777';
    ctx.font = '10px monospace';
    ctx.fillText('Scanning 1D slice...', 10, h * 0.5);
    ctx.restore();
    return;
  }

  const vals = pts.map((p) => toMetricValue(p.fitness, metric));
  let minV = Math.min(...vals);
  let maxV = Math.max(...vals);
  if (Math.abs(maxV - minV) < 1e-4) {
    minV -= 1;
    maxV += 1;
  }

  const padL = 38;
  const padR = 10;
  const padT = 20;
  const padB = 18;
  const plotW = Math.max(10, w - padL - padR);
  const plotH = Math.max(10, h - padT - padB);

  // Header
  ctx.fillStyle = '#aaa';
  ctx.font = '10px monospace';
  ctx.textBaseline = 'top';
  ctx.fillText(`1D ${metric === 'loss' ? 'Loss' : 'Fitness'} | ${data.slice1DLabel}`, 6, 4);

  // Horizontal grid lines
  ctx.strokeStyle = 'rgba(255,255,255,0.07)';
  ctx.fillStyle = '#666';
  ctx.font = '9px monospace';
  ctx.textBaseline = 'middle';
  for (let g = 0; g <= 2; g++) {
    const gy = padT + (g / 2) * plotH;
    ctx.beginPath();
    ctx.moveTo(padL, gy);
    ctx.lineTo(padL + plotW, gy);
    ctx.stroke();
    const gv = maxV - (g / 2) * (maxV - minV);
    ctx.fillText(formatCompactNum(gv), 4, gy);
  }

  // Filled curve
  const coords = pts.map((p, i) => {
    const px = padL + (i / (pts.length - 1)) * plotW;
    const norm = (vals[i] - minV) / (maxV - minV);
    const py = padT + (1 - norm) * plotH;
    return { px, py, hit: p.hit, t: p.t, v: vals[i] };
  });

  const grad = ctx.createLinearGradient(0, padT, 0, padT + plotH);
  if (metric === 'loss') {
    grad.addColorStop(0, 'rgba(255, 85, 65, 0.34)');
    grad.addColorStop(1, 'rgba(0, 220, 235, 0.12)');
  } else {
    grad.addColorStop(0, 'rgba(0, 235, 245, 0.35)');
    grad.addColorStop(1, 'rgba(255, 70, 70, 0.1)');
  }

  ctx.beginPath();
  ctx.moveTo(coords[0].px, padT + plotH);
  for (const c of coords) ctx.lineTo(c.px, c.py);
  ctx.lineTo(coords[coords.length - 1].px, padT + plotH);
  ctx.closePath();
  ctx.fillStyle = grad;
  ctx.fill();

  ctx.strokeStyle = metric === 'loss' ? '#f86' : '#0ee';
  ctx.lineWidth = 1.8;
  ctx.beginPath();
  coords.forEach((c, idx) => {
    if (idx === 0) ctx.moveTo(c.px, c.py);
    else ctx.lineTo(c.px, c.py);
  });
  ctx.stroke();

  // Hit cliff markers
  for (const c of coords) {
    if (c.hit) {
      ctx.fillStyle = '#f33';
      ctx.beginPath();
      ctx.arc(c.px, c.py, 2.5, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  // Current position marker
  const markerU = data.mode === 'spatial' ? data.markerNorm.u : data.slice1DLabel.startsWith('Current') ? (0 - -0.2) / 1.4 : 0.5;
  const markerIdx = Math.max(0, Math.min(coords.length - 1, Math.round(markerU * (coords.length - 1))));
  const mc = coords[markerIdx];
  ctx.strokeStyle = 'rgba(255, 255, 0, 0.5)';
  ctx.setLineDash([3, 3]);
  ctx.beginPath();
  ctx.moveTo(mc.px, padT);
  ctx.lineTo(mc.px, padT + plotH);
  ctx.stroke();
  ctx.setLineDash([]);

  ctx.fillStyle = '#ff0';
  ctx.beginPath();
  ctx.arc(mc.px, mc.py, 3.5, 0, Math.PI * 2);
  ctx.fill();

  ctx.restore();
}

export function drawLandscape2D(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  data: LandscapeData,
  metric: LandscapeMetric,
  hoverCell?: { row: number; col: number } | null
): void {
  ctx.save();
  ctx.translate(x, y);

  ctx.fillStyle = '#070707';
  ctx.fillRect(0, 0, w, h);
  ctx.strokeStyle = '#262626';
  ctx.lineWidth = 1;
  ctx.strokeRect(0.5, 0.5, w - 1, h - 1);

  const { minV, maxV } = getMetricBounds(data, metric);
  const spanV = Math.max(1e-4, maxV - minV);

  const padL = 8;
  const padR = 8;
  const padT = 18;
  const padB = 14;
  const plotW = Math.max(10, w - padL - padR);
  const plotH = Math.max(10, h - padT - padB);

  const gw = data.gridW;
  const gh = data.gridH;
  const cellW = plotW / gw;
  const cellH = plotH / gh;

  for (let r = 0; r < gh; r++) {
    for (let c = 0; c < gw; c++) {
      const val = toMetricValue(data.values[r][c], metric);
      const t = (val - minV) / spanV;
      const [cr, cg, cb] = sampleElevationColor(t, metric);
      ctx.fillStyle = `rgb(${cr},${cg},${cb})`;
      ctx.fillRect(padL + c * cellW, padT + r * cellH, Math.ceil(cellW), Math.ceil(cellH));

      if (data.hitMask[r]?.[c]) {
        ctx.fillStyle = 'rgba(255, 35, 35, 0.35)';
        ctx.fillRect(padL + c * cellW, padT + r * cellH, Math.ceil(cellW), Math.ceil(cellH));
      }
    }
  }

  // Iso-contour lines via cell neighbor level crossings
  const levels = 6;
  ctx.strokeStyle = 'rgba(0, 0, 0, 0.36)';
  ctx.lineWidth = 1;
  for (let lv = 1; lv < levels; lv++) {
    const thr = minV + (lv / levels) * spanV;
    ctx.beginPath();
    for (let r = 0; r < gh - 1; r++) {
      for (let c = 0; c < gw - 1; c++) {
        const v00 = toMetricValue(data.values[r][c], metric);
        const v10 = toMetricValue(data.values[r][c + 1], metric);
        const v01 = toMetricValue(data.values[r + 1][c], metric);
        const b00 = v00 >= thr;
        const b10 = v10 >= thr;
        const b01 = v01 >= thr;
        const cx = padL + (c + 0.5) * cellW;
        const cy = padT + (r + 0.5) * cellH;
        if (b00 !== b10) {
          ctx.moveTo(cx + cellW * 0.5, cy - cellH * 0.4);
          ctx.lineTo(cx + cellW * 0.5, cy + cellH * 0.4);
        }
        if (b00 !== b01) {
          ctx.moveTo(cx - cellW * 0.4, cy + cellH * 0.5);
          ctx.lineTo(cx + cellW * 0.4, cy + cellH * 0.5);
        }
      }
    }
    ctx.stroke();
  }

  // Secondary markers (population or enemies)
  for (const m of data.markers) {
    const mx = padL + m.u * plotW;
    const my = padT + m.v * plotH;
    if (m.isEnemy) {
      ctx.fillStyle = '#f33';
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 1;
      ctx.fillRect(mx - 3, my - 3, 6, 6);
      ctx.strokeRect(mx - 3, my - 3, 6, 6);
    } else if (m.isBest) {
      ctx.fillStyle = '#0ff';
      ctx.strokeStyle = '#000';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.arc(mx, my, 3.5, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    } else {
      ctx.fillStyle = 'rgba(255, 255, 255, 0.65)';
      ctx.beginPath();
      ctx.arc(mx, my, 2, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  // Primary current marker crosshair
  const cmx = padL + data.markerNorm.u * plotW;
  const cmy = padT + data.markerNorm.v * plotH;
  ctx.strokeStyle = '#ff0';
  ctx.lineWidth = 1.3;
  ctx.beginPath();
  ctx.arc(cmx, cmy, 4.5, 0, Math.PI * 2);
  ctx.moveTo(cmx - 7, cmy);
  ctx.lineTo(cmx + 7, cmy);
  ctx.moveTo(cmx, cmy - 7);
  ctx.lineTo(cmx, cmy + 7);
  ctx.stroke();

  if (hoverCell) {
    const hx = padL + hoverCell.col * cellW;
    const hy = padT + hoverCell.row * cellH;
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 1.5;
    ctx.strokeRect(hx, hy, cellW, cellH);
  }

  // Labels
  ctx.fillStyle = '#bbb';
  ctx.font = '10px monospace';
  ctx.textBaseline = 'top';
  ctx.fillText(
    `2D Contour (${data.mode === 'spatial' ? 'Arena X,Y' : 'α,β Weights'})`,
    6,
    4
  );

  ctx.fillStyle = '#888';
  ctx.font = '9px monospace';
  ctx.textBaseline = 'bottom';
  ctx.fillText(
    `${metric === 'loss' ? 'Loss' : 'Fit'}: ${formatCompactNum(minV)}..${formatCompactNum(maxV)}`,
    6,
    h - 2
  );

  ctx.restore();
}

export function drawLandscape3D(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  data: LandscapeData,
  metric: LandscapeMetric,
  yaw: number,
  pitch: number
): void {
  ctx.save();
  ctx.translate(x, y);

  ctx.fillStyle = '#060606';
  ctx.fillRect(0, 0, w, h);
  ctx.strokeStyle = '#262626';
  ctx.lineWidth = 1;
  ctx.strokeRect(0.5, 0.5, w - 1, h - 1);

  const gw = data.gridW;
  const gh = data.gridH;
  const { minV, maxV } = getMetricBounds(data, metric);
  const spanV = Math.max(1e-4, maxV - minV);

  const cosY = Math.cos(yaw);
  const sinY = Math.sin(yaw);
  const cosP = Math.cos(pitch);
  const sinP = Math.sin(pitch);

  const scaleXY = Math.min(w, h) * 0.37;
  const heightScale = 0.65;
  const camDist = 3.2;
  const cx = w * 0.5;
  const cy = h * 0.55;

  const project = (nx: number, ny: number, nz: number) => {
    // nx, ny in [-1, 1], nz in [0, 1]
    const rx = nx * cosY - ny * sinY;
    const ry = nx * sinY + ny * cosY;
    const rz = (nz - 0.42) * heightScale;

    const ty = ry * cosP - rz * sinP;
    const tz = ry * sinP + rz * cosP;

    const persp = camDist / (camDist + ty * 0.55 + 1.1);
    return {
      sx: cx + rx * scaleXY * persp,
      sy: cy - tz * scaleXY * persp,
      depth: ty,
    };
  };

  // Floor reference grid
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
  ctx.lineWidth = 1;
  for (let i = -2; i <= 2; i++) {
    const k = i / 2;
    const a1 = project(k, -1, 0);
    const b1 = project(k, 1, 0);
    ctx.beginPath();
    ctx.moveTo(a1.sx, a1.sy);
    ctx.lineTo(b1.sx, b1.sy);
    ctx.stroke();

    const a2 = project(-1, k, 0);
    const b2 = project(1, k, 0);
    ctx.beginPath();
    ctx.moveTo(a2.sx, a2.sy);
    ctx.lineTo(b2.sx, b2.sy);
    ctx.stroke();
  }

  // Project mesh vertices
  const verts: { sx: number; sy: number; depth: number; norm: number }[][] = [];
  for (let r = 0; r < gh; r++) {
    const ny = (r / Math.max(1, gh - 1)) * 2 - 1;
    const rowArr: { sx: number; sy: number; depth: number; norm: number }[] = [];
    for (let c = 0; c < gw; c++) {
      const nx = (c / Math.max(1, gw - 1)) * 2 - 1;
      const val = toMetricValue(data.values[r][c], metric);
      const norm = Math.max(0, Math.min(1, (val - minV) / spanV));
      const p = project(nx, ny, norm);
      rowArr.push({ ...p, norm });
    }
    verts.push(rowArr);
  }

  // Build painter-sorted quads
  const quads: {
    p0: { sx: number; sy: number };
    p1: { sx: number; sy: number };
    p2: { sx: number; sy: number };
    p3: { sx: number; sy: number };
    depth: number;
    avgNorm: number;
    hit: boolean;
  }[] = [];

  for (let r = 0; r < gh - 1; r++) {
    for (let c = 0; c < gw - 1; c++) {
      const v0 = verts[r][c];
      const v1 = verts[r][c + 1];
      const v2 = verts[r + 1][c + 1];
      const v3 = verts[r + 1][c];
      quads.push({
        p0: v0,
        p1: v1,
        p2: v2,
        p3: v3,
        depth: (v0.depth + v1.depth + v2.depth + v3.depth) * 0.25,
        avgNorm: (v0.norm + v1.norm + v2.norm + v3.norm) * 0.25,
        hit: Boolean(data.hitMask[r]?.[c]),
      });
    }
  }

  // Sort farthest first (largest depth first)
  quads.sort((a, b) => b.depth - a.depth);

  for (const q of quads) {
    const [cr, cg, cb] = sampleElevationColor(q.avgNorm, metric);
    ctx.fillStyle = q.hit
      ? `rgba(${Math.min(255, cr + 35)}, ${Math.max(10, cg - 10)}, ${Math.max(10, cb - 10)}, 0.88)`
      : `rgba(${cr}, ${cg}, ${cb}, 0.85)`;
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.16)';
    ctx.lineWidth = 0.6;
    ctx.beginPath();
    ctx.moveTo(q.p0.sx, q.p0.sy);
    ctx.lineTo(q.p1.sx, q.p1.sy);
    ctx.lineTo(q.p2.sx, q.p2.sy);
    ctx.lineTo(q.p3.sx, q.p3.sy);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
  }

  // Beacon pin at current agent / player position
  const mc = Math.max(0, Math.min(gw - 1, Math.round(data.markerNorm.u * (gw - 1))));
  const mr = Math.max(0, Math.min(gh - 1, Math.round(data.markerNorm.v * (gh - 1))));
  const mVal = toMetricValue(data.values[mr]?.[mc] ?? data.centerFitness, metric);
  const mNorm = Math.max(0, Math.min(1, (mVal - minV) / spanV));
  const mx = data.markerNorm.u * 2 - 1;
  const my = data.markerNorm.v * 2 - 1;
  const basePt = project(mx, my, mNorm);
  const topPt = project(mx, my, Math.min(1.22, mNorm + 0.28));

  ctx.strokeStyle = '#ff0';
  ctx.lineWidth = 1.8;
  ctx.beginPath();
  ctx.moveTo(basePt.sx, basePt.sy);
  ctx.lineTo(topPt.sx, topPt.sy);
  ctx.stroke();

  ctx.fillStyle = '#ff0';
  ctx.beginPath();
  ctx.arc(topPt.sx, topPt.sy, 3.5, 0, Math.PI * 2);
  ctx.fill();

  ctx.fillStyle = '#bbb';
  ctx.font = '10px monospace';
  ctx.textBaseline = 'top';
  ctx.fillText(
    `3D ${metric === 'loss' ? 'Loss' : 'Fitness'} Surface`,
    6,
    4
  );

  ctx.fillStyle = '#888';
  ctx.font = '9px monospace';
  ctx.textBaseline = 'bottom';
  ctx.fillText(
    `Now: ${formatCompactNum(toMetricValue(data.centerFitness, metric))}`,
    6,
    h - 2
  );

  ctx.restore();
}
