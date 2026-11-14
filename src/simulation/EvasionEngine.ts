import { ActivationType, NeuralNetwork } from './NeuralNetwork';

export const SIM_WIDTH = 600;
export const SIM_HEIGHT = 400;
export const NN_WIDTH = 570;
export const NN_HEIGHT = 475;

export type SpawnMode = 'outer' | 'central';
export type EnemySpeedMode = 'fixed' | 'rarity';

export interface SimParams {
  hiddenLayersCount: number;
  neuronsPerLayer: number;
  activationType: ActivationType;
  numEnemies: number;
  enemySpeedPct: number;
  enemySpeedMode: EnemySpeedMode;
  spawnMode: SpawnMode;
  simSpeed: number;
  turboMode: boolean;
  rewardEnemyDist: boolean;
  rewardWallDist: boolean;
  rewardGroupEnemies: boolean;
  rewardSpreadEnemies: boolean;
  rewardEdgeEnemies: boolean;
  rewardOppHeading: boolean;
  rewardSmoothVel: boolean;
  rewardSplitGap: boolean;
  rewardEdgeSense: boolean;
  rewardOrbitKite: boolean;
  rewardMatadorCut: boolean;
  rewardAnkleBreaker: boolean;
  rewardPincerEscape: boolean;
  rewardOpenCorridor: boolean;
  rewardWallSlingshot: boolean;
  rewardPackTangle: boolean;
  rewardSpideySense: boolean;
  rewardFarWallSense: boolean;
  rewardSplitGapCollect: boolean;
  rewardSplitGapPredict: boolean;
  rewardHit: boolean;
  rewardAvgEnemyCenter: boolean;
  rewardVectorStabbing: boolean;
  rewardTwitch: boolean;
  weightEnemyDist: number;
  weightWallDist: number;
  weightGroupEnemies: number;
  weightSpreadEnemies: number;
  weightEdgeEnemies: number;
  weightOppHeading: number;
  weightSmoothVel: number;
  weightSplitGap: number;
  weightEdgeSense: number;
  weightOrbitKite: number;
  weightMatadorCut: number;
  weightAnkleBreaker: number;
  weightPincerEscape: number;
  weightOpenCorridor: number;
  weightWallSlingshot: number;
  weightPackTangle: number;
  weightSpideySense: number;
  weightFarWallSense: number;
  weightSplitGapCollect: number;
  weightSplitGapPredict: number;
  weightHit: number;
  weightAvgEnemyCenter: number;
  weightVectorStabbing: number;
  weightTwitch: number;
}

export interface Enemy {
  x: number;
  y: number;
  spawnDelay?: number;
  entered?: boolean;
  speedPct?: number;
}

export interface Player {
  x: number;
  y: number;
  vx: number;
  vy: number;
}

export class EvasionEngine {
  readonly playerSpeed = 3.5;
  readonly maxSteps = 18000; // 5 minutes at 60fps
  readonly popSize = 15;

  params: SimParams;
  enemySpeed: number;
  player: Player = { x: SIM_WIDTH / 2, y: SIM_HEIGHT / 2, vx: 0, vy: 0 };
  enemies: Enemy[] = [];
  agent: NeuralNetwork | null = null;
  bestAgent: NeuralNetwork | null = null;
  population: NeuralNetwork[] = [];
  popIndex = 0;
  generation = 1;
  stepCount = 0;
  currentFitness = 0;
  bestFitness = -Infinity;
  isFrozen = false;
  prevVx = 0;
  prevVy = 0;
  emaVx = 0;
  emaVy = 0;
  emaPx = SIM_WIDTH / 2;
  emaPy = SIM_HEIGHT / 2;
  collectedSplitGaps: { pairKey: number; mx: number; my: number; step: number }[] = [];
  splitGapHistory: { angle: number; x: number; y: number }[] = [];
  onStatsChange?: (statsText: string) => void;

  constructor(initialParams: SimParams, onStatsChange?: (statsText: string) => void) {
    this.params = { ...initialParams };
    this.enemySpeed = this.playerSpeed * (this.params.enemySpeedPct / 100);
    this.onStatsChange = onStatsChange;
  }

  private sampleRaritySpeedPct(): number {
    const u1 = Math.max(1e-7, Math.random());
    const u2 = Math.random();
    const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    const raw = Math.random() < 0.92 ? 60 + z * 13 : 1 + Math.random() * 89;
    return Math.min(90, Math.max(1, Math.round(raw)));
  }

  private spawnEnemy(spawnDelay = 0): Enemy {
    const speedPct =
      this.params.enemySpeedMode === 'rarity'
        ? this.sampleRaritySpeedPct()
        : this.params.enemySpeedPct;

    if (this.params.spawnMode === 'central') {
      let ex = 0;
      let ey = 0;
      let dist = 0;
      let attempts = 0;
      do {
        ex = 10 + Math.random() * (SIM_WIDTH - 20);
        ey = 10 + Math.random() * (SIM_HEIGHT - 20);
        dist = Math.hypot(ex - this.player.x, ey - this.player.y);
        attempts++;
      } while (dist < 60 && attempts < 25);
      return { x: ex, y: ey, spawnDelay: 0, entered: true, speedPct };
    }

    const edge = Math.floor(Math.random() * 4);
    const offset = 14;
    let ex = 0;
    let ey = 0;
    if (edge === 0) {
      ex = Math.random() * SIM_WIDTH;
      ey = -offset;
    } else if (edge === 1) {
      ex = SIM_WIDTH + offset;
      ey = Math.random() * SIM_HEIGHT;
    } else if (edge === 2) {
      ex = Math.random() * SIM_WIDTH;
      ey = SIM_HEIGHT + offset;
    } else {
      ex = -offset;
      ey = Math.random() * SIM_HEIGHT;
    }
    return { x: ex, y: ey, spawnDelay, entered: false, speedPct };
  }

  setSpawnMode(mode: SpawnMode): void {
    this.params.spawnMode = mode;
  }

  setEnemySpeedMode(mode: EnemySpeedMode): void {
    this.params.enemySpeedMode = mode;
    if (mode === 'rarity') {
      for (const e of this.enemies) {
        e.speedPct = this.sampleRaritySpeedPct();
      }
    } else {
      for (const e of this.enemies) {
        e.speedPct = this.params.enemySpeedPct;
      }
    }
  }

  updateEnemyCount(targetCount: number): void {
    const clamped = Math.min(50, Math.max(1, targetCount));
    this.params.numEnemies = clamped;
    while (this.enemies.length < clamped) {
      this.enemies.push(this.spawnEnemy());
    }
    if (this.enemies.length > clamped) {
      this.enemies.length = clamped;
    }
  }

  updateEnemySpeedPct(pct: number): void {
    const clamped = Math.max(1, pct);
    this.params.enemySpeedPct = clamped;
    this.enemySpeed = this.playerSpeed * (clamped / 100);
  }

  resetSimulation(): void {
    const layers = Math.max(1, this.params.hiddenLayersCount || 1);
    const neurons = Math.max(2, this.params.neuronsPerLayer || 16);
    const activation = this.params.activationType || 'tanh';
    this.params.hiddenLayersCount = layers;
    this.params.neuronsPerLayer = neurons;
    this.params.numEnemies = Math.min(50, Math.max(1, this.params.numEnemies || 3));
    this.updateEnemySpeedPct(this.params.enemySpeedPct || 80);

    this.population = [];
    for (let i = 0; i < this.popSize; i++) {
      this.population.push(new NeuralNetwork(104, layers, neurons, 2, activation));
    }
    this.popIndex = 0;
    this.generation = 1;
    this.bestFitness = -Infinity;
    this.bestAgent = null;
    this.isFrozen = false;
    this.startEpisode();
  }

  startEpisode(): void {
    this.player.x = SIM_WIDTH / 2;
    this.player.y = SIM_HEIGHT / 2;
    this.player.vx = 0;
    this.player.vy = 0;

    this.enemies = [];
    if (this.params.spawnMode === 'central') {
      for (let i = 0; i < this.params.numEnemies; i++) {
        this.enemies.push(this.spawnEnemy(0));
      }
    } else {
      const maxStagger = Math.min(75, 16 + this.params.numEnemies * 5);
      for (let i = 0; i < this.params.numEnemies; i++) {
        const delay = i === 0 ? 0 : Math.floor(4 + Math.random() * maxStagger);
        this.enemies.push(this.spawnEnemy(delay));
      }
    }

    this.agent = this.population[this.popIndex];
    this.stepCount = 0;
    this.currentFitness = 0;
    this.prevVx = 0;
    this.prevVy = 0;
    this.emaVx = 0;
    this.emaVy = 0;
    this.emaPx = this.player.x;
    this.emaPy = this.player.y;
    this.collectedSplitGaps = [];
    this.splitGapHistory = [];
  }

  private computeSplitGapAngle(activeEnemies: Enemy[]): number | null {
    if (activeEnemies.length === 0) return null;
    const sortedNearby = activeEnemies
      .map((e) => {
        const dx = e.x - this.player.x;
        const dy = e.y - this.player.y;
        return { dist: Math.hypot(dx, dy), angle: Math.atan2(dy, dx) };
      })
      .sort((a, b) => a.dist - b.dist);

    const withinRadius = sortedNearby.filter((e) => e.dist <= 220);
    const nearby =
      withinRadius.length >= 2
        ? withinRadius.slice(0, 12)
        : sortedNearby.slice(0, Math.min(sortedNearby.length, 8));

    if (nearby.length === 1) {
      return nearby[0].angle + Math.PI;
    }

    const angles = nearby.map((e) => e.angle).sort((a, b) => a - b);
    let maxGap = -1;
    let bestMidAngle = 0;
    for (let i = 0; i < angles.length; i++) {
      const a1 = angles[i];
      const a2 = i === angles.length - 1 ? angles[0] + Math.PI * 2 : angles[i + 1];
      const gap = a2 - a1;
      if (gap > maxGap) {
        maxGap = gap;
        bestMidAngle = a1 + gap * 0.5;
      }
    }
    return bestMidAngle;
  }

  update(): void {
    if (!this.agent) return;

    const inputs: number[] = [
      (this.player.x / SIM_WIDTH) * 2 - 1,
      (this.player.y / SIM_HEIGHT) * 2 - 1,
      this.player.vx / this.playerSpeed,
      this.player.vy / this.playerSpeed,
    ];

    const activeEnemies = this.enemies.filter((e) => !e.spawnDelay || e.spawnDelay <= 0);

    for (let i = 0; i < 50; i++) {
      if (i < activeEnemies.length) {
        inputs.push((activeEnemies[i].x - this.player.x) / SIM_WIDTH);
        inputs.push((activeEnemies[i].y - this.player.y) / SIM_HEIGHT);
      } else {
        inputs.push(0);
        inputs.push(0);
      }
    }

    const outputs = this.agent.predict(inputs);
    let ax = outputs[0];
    let ay = outputs[1];

    const mag = Math.sqrt(ax * ax + ay * ay);
    if (mag > 1) {
      ax /= mag;
      ay /= mag;
    }

    const prevPx = this.player.x;
    const prevPy = this.player.y;

    this.player.vx = ax * this.playerSpeed;
    this.player.vy = ay * this.playerSpeed;

    this.player.x += this.player.vx;
    this.player.y += this.player.vy;

    this.player.x = Math.max(6, Math.min(SIM_WIDTH - 6, this.player.x));
    this.player.y = Math.max(6, Math.min(SIM_HEIGHT - 6, this.player.y));

    const activeEnemyIndices: number[] = [];
    for (let i = 0; i < this.enemies.length; i++) {
      const e = this.enemies[i];
      if (!e.spawnDelay || e.spawnDelay <= 0) {
        activeEnemyIndices.push(i);
      }
    }

    let minEnemyDist = Infinity;
    const activeEnemyVels: { vx: number; vy: number }[] = [];
    for (let i = 0; i < activeEnemies.length; i++) {
      const e = activeEnemies[i];
      const dx = this.player.x - e.x;
      const dy = this.player.y - e.y;
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (dist < minEnemyDist) minEnemyDist = dist;

      let vx = dist > 0.1 ? dx / dist : 0;
      let vy = dist > 0.1 ? dy / dist : 0;

      let sepX = 0;
      let sepY = 0;
      const desiredSep = 26;
      for (let j = 0; j < activeEnemies.length; j++) {
        if (i === j) continue;
        const other = activeEnemies[j];
        const sdx = e.x - other.x;
        const sdy = e.y - other.y;
        const sDist = Math.sqrt(sdx * sdx + sdy * sdy);
        if (sDist > 0.001 && sDist < desiredSep) {
          const weight = (desiredSep - sDist) / desiredSep;
          sepX += (sdx / sDist) * weight;
          sepY += (sdy / sDist) * weight;
        }
      }

      vx += sepX * 1.6;
      vy += sepY * 1.6;

      const vMag = Math.sqrt(vx * vx + vy * vy);
      if (vMag > 0.001) {
        const effSpeed =
          this.params.enemySpeedMode === 'rarity'
            ? this.playerSpeed * (Math.min(90, Math.max(1, e.speedPct ?? 60)) / 100)
            : this.enemySpeed;
        const stepVx = (vx / vMag) * effSpeed;
        const stepVy = (vy / vMag) * effSpeed;
        e.x += stepVx;
        e.y += stepVy;
        activeEnemyVels.push({ vx: stepVx, vy: stepVy });
      } else {
        activeEnemyVels.push({ vx: 0, vy: 0 });
      }

      if (dist < 11) {
        if (this.params.rewardHit) {
          this.currentFitness += this.params.weightHit;
        }
        this.endEpisode();
        return;
      }
    }

    const minEnemyGap = 11;
    for (let iter = 0; iter < 2; iter++) {
      for (let i = 0; i < activeEnemies.length; i++) {
        for (let j = i + 1; j < activeEnemies.length; j++) {
          const dx = activeEnemies[j].x - activeEnemies[i].x;
          const dy = activeEnemies[j].y - activeEnemies[i].y;
          const d = Math.sqrt(dx * dx + dy * dy);
          if (d < minEnemyGap) {
            const nx = d > 0.001 ? dx / d : Math.cos(i + j);
            const ny = d > 0.001 ? dy / d : Math.sin(i + j);
            const overlap = (minEnemyGap - d) * 0.5;
            activeEnemies[i].x -= nx * overlap;
            activeEnemies[i].y -= ny * overlap;
            activeEnemies[j].x += nx * overlap;
            activeEnemies[j].y += ny * overlap;
          }
        }
      }
      for (let i = 0; i < activeEnemies.length; i++) {
        const e = activeEnemies[i];
        if (!e.entered) {
          if (e.x >= 5 && e.x <= SIM_WIDTH - 5 && e.y >= 5 && e.y <= SIM_HEIGHT - 5) {
            e.entered = true;
          }
        }
        if (e.entered) {
          e.x = Math.max(5, Math.min(SIM_WIDTH - 5, e.x));
          e.y = Math.max(5, Math.min(SIM_HEIGHT - 5, e.y));
        }
      }
    }

    if (this.params.rewardEnemyDist && activeEnemies.length > 0) {
      this.currentFitness += minEnemyDist * this.params.weightEnemyDist;
    }

    if (this.params.rewardWallDist) {
      const wallDistX = Math.min(this.player.x - 6, SIM_WIDTH - 6 - this.player.x);
      const wallDistY = Math.min(this.player.y - 6, SIM_HEIGHT - 6 - this.player.y);
      const minWallDist = Math.min(wallDistX, wallDistY);
      this.currentFitness += minWallDist * this.params.weightWallDist;
    }

    if (this.params.rewardGroupEnemies && activeEnemies.length > 1) {
      let cx = 0;
      let cy = 0;
      for (let i = 0; i < activeEnemies.length; i++) {
        cx += activeEnemies[i].x;
        cy += activeEnemies[i].y;
      }
      cx /= activeEnemies.length;
      cy /= activeEnemies.length;
      let totalSpread = 0;
      for (let i = 0; i < activeEnemies.length; i++) {
        totalSpread += Math.hypot(activeEnemies[i].x - cx, activeEnemies[i].y - cy);
      }
      const avgSpread = totalSpread / activeEnemies.length;
      const maxArenaRadius = Math.hypot(SIM_WIDTH, SIM_HEIGHT) * 0.5;
      const tightness = Math.max(0, maxArenaRadius - avgSpread);
      this.currentFitness += tightness * this.params.weightGroupEnemies;
    }

    if (this.params.rewardSpreadEnemies && activeEnemies.length > 1) {
      let cx = 0;
      let cy = 0;
      for (let i = 0; i < activeEnemies.length; i++) {
        cx += activeEnemies[i].x;
        cy += activeEnemies[i].y;
      }
      cx /= activeEnemies.length;
      cy /= activeEnemies.length;
      let totalSpread = 0;
      for (let i = 0; i < activeEnemies.length; i++) {
        totalSpread += Math.hypot(activeEnemies[i].x - cx, activeEnemies[i].y - cy);
      }
      const avgSpread = totalSpread / activeEnemies.length;
      this.currentFitness += avgSpread * this.params.weightSpreadEnemies;
    }

    if (this.params.rewardEdgeEnemies && activeEnemies.length > 0) {
      let totalEdgeCloseness = 0;
      for (let i = 0; i < activeEnemies.length; i++) {
        const e = activeEnemies[i];
        const distToLeft = e.x;
        const distToRight = SIM_WIDTH - e.x;
        const distToTop = e.y;
        const distToBottom = SIM_HEIGHT - e.y;
        const minDistToEdge = Math.min(distToLeft, distToRight, distToTop, distToBottom);
        const maxPossibleEdgeDist = Math.min(SIM_WIDTH, SIM_HEIGHT) * 0.5;
        const edgeCloseness = Math.max(0, maxPossibleEdgeDist - minDistToEdge);
        totalEdgeCloseness += edgeCloseness;
      }
      this.currentFitness +=
        (totalEdgeCloseness / activeEnemies.length) * this.params.weightEdgeEnemies;
    }

    if (this.params.rewardAvgEnemyCenter && activeEnemies.length > 0) {
      let cx = 0;
      let cy = 0;
      for (let i = 0; i < activeEnemies.length; i++) {
        cx += activeEnemies[i].x;
        cy += activeEnemies[i].y;
      }
      cx /= activeEnemies.length;
      cy /= activeEnemies.length;
      const distFromCenter = Math.hypot(cx - SIM_WIDTH * 0.5, cy - SIM_HEIGHT * 0.5);
      const maxArenaRadius = Math.hypot(SIM_WIDTH, SIM_HEIGHT) * 0.5;
      const centerScore = Math.max(0, maxArenaRadius - distFromCenter);
      this.currentFitness += centerScore * this.params.weightAvgEnemyCenter;
    }

    if (this.params.rewardOppHeading && activeEnemies.length > 0) {
      let avgEx = 0;
      let avgEy = 0;
      for (let i = 0; i < activeEnemies.length; i++) {
        const dx = activeEnemies[i].x - this.player.x;
        const dy = activeEnemies[i].y - this.player.y;
        const d = Math.hypot(dx, dy);
        if (d > 0.001) {
          avgEx += dx / d;
          avgEy += dy / d;
        }
      }
      avgEx /= activeEnemies.length;
      avgEy /= activeEnemies.length;

      const avgMag = Math.hypot(avgEx, avgEy);
      const pMag = Math.hypot(this.player.vx, this.player.vy);
      if (avgMag > 0.001 && pMag > 0.001) {
        const awayX = -avgEx / avgMag;
        const awayY = -avgEy / avgMag;
        const headX = this.player.vx / pMag;
        const headY = this.player.vy / pMag;
        const alignment = headX * awayX + headY * awayY;
        this.currentFitness += alignment * 100 * this.params.weightOppHeading;
      }
    }

    const prevStepVx = this.prevVx;
    const prevStepVy = this.prevVy;
    const nvx = this.player.vx / this.playerSpeed;
    const nvy = this.player.vy / this.playerSpeed;
    if (this.stepCount === 0) {
      this.prevVx = nvx;
      this.prevVy = nvy;
      this.emaVx = nvx;
      this.emaVy = nvy;
    } else {
      if (this.params.rewardSmoothVel) {
        const speedNorm = Math.hypot(nvx, nvy);
        const jerk = Math.hypot(nvx - this.prevVx, nvy - this.prevVy);
        const noise = Math.hypot(nvx - this.emaVx, nvy - this.emaVy);
        const smoothScore = (speedNorm - 1.5 * jerk - 1.0 * noise) * 80;
        this.currentFitness += smoothScore * this.params.weightSmoothVel;
      }
      this.prevVx = nvx;
      this.prevVy = nvy;
      this.emaVx = 0.88 * this.emaVx + 0.12 * nvx;
      this.emaVy = 0.88 * this.emaVy + 0.12 * nvy;
    }

    if (this.params.rewardSplitGap && activeEnemies.length > 0) {
      const bestMidAngle = this.computeSplitGapAngle(activeEnemies);
      if (bestMidAngle !== null) {
        const gapX = Math.cos(bestMidAngle);
        const gapY = Math.sin(bestMidAngle);
        const gapAlignment = nvx * gapX + nvy * gapY;
        this.currentFitness += gapAlignment * 100 * this.params.weightSplitGap;
      }
    }

    if (this.params.rewardSplitGapPredict && activeEnemies.length > 0) {
      const currAngle = this.computeSplitGapAngle(activeEnemies);
      if (currAngle !== null) {
        const currX = Math.cos(currAngle);
        const currY = Math.sin(currAngle);

        const wrapAngle = (diff: number): number => {
          let a = diff;
          while (a > Math.PI) a -= Math.PI * 2;
          while (a < -Math.PI) a += Math.PI * 2;
          return a;
        };

        if (this.splitGapHistory.length > 0) {
          const prevSample = this.splitGapHistory[this.splitGapHistory.length - 1];
          const stepJump = Math.abs(wrapAngle(currAngle - prevSample.angle));
          if (stepJump > 0.85) {
            this.splitGapHistory = [];
          }
        }

        let predictedAngle = currAngle;
        let anticipationScore = 0;

        if (this.splitGapHistory.length > 0) {
          const pastSample = this.splitGapHistory[0];
          const dt = this.splitGapHistory.length;
          const rawDiff = wrapAngle(currAngle - pastSample.angle);
          const perStepDelta = rawDiff / dt;
          const leadAngle = Math.max(-0.65, Math.min(0.65, perStepDelta * 5));
          predictedAngle = currAngle + leadAngle;

          const prevMag = Math.hypot(this.prevVx, this.prevVy);
          const speedNorm = Math.hypot(nvx, nvy);
          if (this.stepCount > 0 && prevMag > 0.01 && speedNorm > 0.01 && Math.abs(rawDiff) > 0.01) {
            const prevUx = this.prevVx / prevMag;
            const prevUy = this.prevVy / prevMag;
            const alignCurr = prevUx * currX + prevUy * currY;
            const alignPast = prevUx * pastSample.x + prevUy * pastSample.y;
            anticipationScore =
              Math.max(-1, Math.min(1, (alignCurr - alignPast) * 6)) * 35 * speedNorm;
          }
        }

        const predX = Math.cos(predictedAngle);
        const predY = Math.sin(predictedAngle);
        const predAlignment = nvx * predX + nvy * predY;
        this.currentFitness +=
          (predAlignment * 100 + anticipationScore) * this.params.weightSplitGapPredict;

        this.splitGapHistory.push({ angle: currAngle, x: currX, y: currY });
        if (this.splitGapHistory.length > 5) {
          this.splitGapHistory.shift();
        }
      }
    }

    if (this.params.rewardEdgeSense && activeEnemies.length > 0) {
      const speedNorm = Math.hypot(nvx, nvy);
      if (speedNorm > 0.01) {
        const ux = nvx / speedNorm;
        const uy = nvy / speedNorm;
        const rx = -uy;
        const ry = ux;
        const senseRadius = 140;

        let leftThreat = 0;
        let rightThreat = 0;
        let frontThreat = 0;

        for (let i = 0; i < activeEnemies.length; i++) {
          const dx = activeEnemies[i].x - this.player.x;
          const dy = activeEnemies[i].y - this.player.y;
          const d = Math.hypot(dx, dy);
          if (d <= 0.001 || d >= senseRadius) continue;

          const closeness = 1 - d / senseRadius;
          const ndx = dx / d;
          const ndy = dy / d;
          const fwd = ndx * ux + ndy * uy;
          const lat = ndx * rx + ndy * ry;
          const periphThreat = closeness * Math.abs(lat);

          if (lat < 0) {
            if (periphThreat > leftThreat) leftThreat = periphThreat;
          } else if (lat > 0) {
            if (periphThreat > rightThreat) rightThreat = periphThreat;
          }

          if (fwd > 0.75) {
            const ft = closeness * ((fwd - 0.75) / 0.25);
            if (ft > frontThreat) frontThreat = ft;
          }
        }

        const edgeScore =
          (Math.abs(leftThreat - rightThreat) -
            Math.min(leftThreat, rightThreat) -
            0.5 * frontThreat) *
          speedNorm;
        this.currentFitness += edgeScore * 100 * this.params.weightEdgeSense;
      }
    }

    const velSpeedNorm = Math.hypot(nvx, nvy);
    if (velSpeedNorm > 0.01) {
      const ux = nvx / velSpeedNorm;
      const uy = nvy / velSpeedNorm;

      if (this.params.rewardOrbitKite && activeEnemies.length > 0) {
        let nearDx = 0;
        let nearDy = 0;
        let nearDist = Infinity;
        for (let i = 0; i < activeEnemies.length; i++) {
          const dx = activeEnemies[i].x - this.player.x;
          const dy = activeEnemies[i].y - this.player.y;
          const d = Math.hypot(dx, dy);
          if (d < nearDist) {
            nearDist = d;
            nearDx = dx;
            nearDy = dy;
          }
        }
        if (nearDist > 0.001) {
          const ex = nearDx / nearDist;
          const ey = nearDy / nearDist;
          const tangentAlign = 1 - Math.abs(ux * ex + uy * ey);
          const sweetSpot = Math.max(0, 1 - Math.abs(nearDist - 85) / 55);
          const tooClosePenalty = nearDist < 35 ? ((35 - nearDist) / 35) * 0.6 : 0;
          const orbitScore =
            (sweetSpot * (tangentAlign - 0.15) - tooClosePenalty) * velSpeedNorm;
          this.currentFitness += orbitScore * 100 * this.params.weightOrbitKite;
        }
      }

      if (this.params.rewardMatadorCut && activeEnemies.length > 0) {
        const cutRadius = 115;
        let bestPos = 0;
        let worstNeg = 0;
        for (let i = 0; i < activeEnemies.length; i++) {
          const dx = activeEnemies[i].x - this.player.x;
          const dy = activeEnemies[i].y - this.player.y;
          const d = Math.hypot(dx, dy);
          if (d <= 0.001 || d >= cutRadius) continue;

          const ev = activeEnemyVels[i];
          const evMag = Math.hypot(ev.vx, ev.vy);
          if (evMag <= 0.001) continue;

          const evx = ev.vx / evMag;
          const evy = ev.vy / evMag;
          const closeness = 1 - d / cutRadius;
          const towardEnemy = ux * (dx / d) + uy * (dy / d);

          if (towardEnemy > 0.6) {
            const pen = -closeness * ((towardEnemy - 0.6) / 0.4);
            if (pen < worstNeg) worstNeg = pen;
          } else {
            const againstEnemyHeading = -(ux * evx + uy * evy);
            const perpToEnemyHeading = Math.abs(-ux * evy + uy * evx);
            const safeMask = 1 - Math.max(0, towardEnemy) / 0.6;
            const score =
              closeness * (0.75 * againstEnemyHeading + 0.25 * perpToEnemyHeading) * safeMask;
            if (score > bestPos) bestPos = score;
            if (score < worstNeg) worstNeg = score;
          }
        }
        this.currentFitness += (bestPos + worstNeg) * velSpeedNorm * 100 * this.params.weightMatadorCut;
      }

      if (this.params.rewardAnkleBreaker && activeEnemies.length > 0) {
        const radius = 135;
        let bestPos = 0;
        let worstNeg = 0;
        for (let i = 0; i < activeEnemies.length; i++) {
          const dx = activeEnemies[i].x - this.player.x;
          const dy = activeEnemies[i].y - this.player.y;
          const d = Math.hypot(dx, dy);
          if (d <= 0.001 || d >= radius) continue;

          const closeness = 1 - d / radius;
          const towardEnemy = ux * (dx / d) + uy * (dy / d);
          const rvx = activeEnemyVels[i].vx - this.player.vx;
          const rvy = activeEnemyVels[i].vy - this.player.vy;
          const angRate = Math.abs(dx * rvy - dy * rvx) / (d * d);
          const normRate = Math.min(1.5, angRate * 14);

          if (towardEnemy > 0.65) {
            const pen = -closeness * ((towardEnemy - 0.65) / 0.35);
            if (pen < worstNeg) worstNeg = pen;
          } else {
            const pos = closeness * normRate * (1 - Math.max(0, towardEnemy) * 0.5);
            if (pos > bestPos) bestPos = pos;
          }
        }
        this.currentFitness +=
          (bestPos + worstNeg) * velSpeedNorm * 100 * this.params.weightAnkleBreaker;
      }

      if (this.params.rewardPincerEscape && activeEnemies.length >= 2) {
        const pincerRadius = 185;
        let bestSeverity = 0;
        let bestAx = 0;
        let bestAy = 0;

        for (let i = 0; i < activeEnemies.length; i++) {
          const dx1 = activeEnemies[i].x - this.player.x;
          const dy1 = activeEnemies[i].y - this.player.y;
          const d1 = Math.hypot(dx1, dy1);
          if (d1 <= 0.001 || d1 >= pincerRadius) continue;
          const u1x = dx1 / d1;
          const u1y = dy1 / d1;

          for (let j = i + 1; j < activeEnemies.length; j++) {
            const dx2 = activeEnemies[j].x - this.player.x;
            const dy2 = activeEnemies[j].y - this.player.y;
            const d2 = Math.hypot(dx2, dy2);
            if (d2 <= 0.001 || d2 >= pincerRadius) continue;

            const dot = u1x * (dx2 / d2) + u1y * (dy2 / d2);
            if (dot < -0.35) {
              const opp = (-dot - 0.35) / 0.65;
              const prox = 1 - Math.max(d1, d2) / pincerRadius;
              const severity = opp * prox;
              if (severity > bestSeverity) {
                bestSeverity = severity;
                bestAx = activeEnemies[j].x - activeEnemies[i].x;
                bestAy = activeEnemies[j].y - activeEnemies[i].y;
              }
            }
          }
        }

        if (bestSeverity > 0) {
          const aLen = Math.hypot(bestAx, bestAy);
          if (aLen > 0.001) {
            const axisX = bestAx / aLen;
            const axisY = bestAy / aLen;
            const escX = -axisY;
            const escY = axisX;
            const escAlign = Math.abs(ux * escX + uy * escY);
            const axisAlign = Math.abs(ux * axisX + uy * axisY);
            const pincerScore = (escAlign - 0.5 * axisAlign) * bestSeverity * velSpeedNorm;
            this.currentFitness += pincerScore * 100 * this.params.weightPincerEscape;
          }
        }
      }

      if (this.params.rewardOpenCorridor) {
        const baseAngle = Math.atan2(uy, ux);
        const offsets = [-0.42, 0, 0.42];
        const maxHorizon = 240;
        let minRay = 1;
        let sumRay = 0;

        for (let k = 0; k < offsets.length; k++) {
          const rx = Math.cos(baseAngle + offsets[k]);
          const ry = Math.sin(baseAngle + offsets[k]);
          const tx =
            rx > 1e-5
              ? (SIM_WIDTH - 6 - this.player.x) / rx
              : rx < -1e-5
              ? (6 - this.player.x) / rx
              : maxHorizon;
          const ty =
            ry > 1e-5
              ? (SIM_HEIGHT - 6 - this.player.y) / ry
              : ry < -1e-5
              ? (6 - this.player.y) / ry
              : maxHorizon;
          let clearance = Math.max(0, Math.min(maxHorizon, tx, ty));

          for (let i = 0; i < activeEnemies.length; i++) {
            const dx = activeEnemies[i].x - this.player.x;
            const dy = activeEnemies[i].y - this.player.y;
            const proj = dx * rx + dy * ry;
            if (proj > 0 && proj < clearance + 11) {
              const perp = Math.abs(dx * ry - dy * rx);
              if (perp < 22) {
                const hitDist = Math.max(0, proj - 11);
                if (hitDist < clearance) clearance = hitDist;
              }
            }
          }

          const normClear = clearance / maxHorizon;
          if (normClear < minRay) minRay = normClear;
          sumRay += normClear;
        }

        const avgRay = sumRay / offsets.length;
        const corridorScore = (0.5 * minRay + 0.5 * avgRay) * velSpeedNorm;
        this.currentFitness += corridorScore * 100 * this.params.weightOpenCorridor;
      }

      if (this.params.rewardWallSlingshot && activeEnemies.length > 0) {
        const wallThresh = 65;
        const dLeft = this.player.x - 6;
        const dRight = SIM_WIDTH - 6 - this.player.x;
        const dTop = this.player.y - 6;
        const dBot = SIM_HEIGHT - 6 - this.player.y;
        const minWall = Math.min(dLeft, dRight, dTop, dBot);

        if (minWall < wallThresh) {
          let nearDx = 0;
          let nearDy = 0;
          let nearDist = Infinity;
          for (let i = 0; i < activeEnemies.length; i++) {
            const dx = activeEnemies[i].x - this.player.x;
            const dy = activeEnemies[i].y - this.player.y;
            const d = Math.hypot(dx, dy);
            if (d < nearDist) {
              nearDist = d;
              nearDx = dx;
              nearDy = dy;
            }
          }

          if (nearDist > 0.001 && nearDist < 170) {
            const wnxRaw =
              (dLeft < wallThresh ? 1 - dLeft / wallThresh : 0) -
              (dRight < wallThresh ? 1 - dRight / wallThresh : 0);
            const wnyRaw =
              (dTop < wallThresh ? 1 - dTop / wallThresh : 0) -
              (dBot < wallThresh ? 1 - dBot / wallThresh : 0);
            const wnMag = Math.hypot(wnxRaw, wnyRaw);
            if (wnMag > 0.001) {
              const wnx = wnxRaw / wnMag;
              const wny = wnyRaw / wnMag;
              const tx = -wny;
              const ty = wnx;
              const ex = nearDx / nearDist;
              const ey = nearDy / nearDist;
              const sign = ex * tx + ey * ty > 0 ? -1 : 1;
              const escX = tx * sign;
              const escY = ty * sign;
              const wallProx = 1 - minWall / wallThresh;
              const enemyProx = 1 - nearDist / 170;
              const slingScore = (ux * escX + uy * escY) * wallProx * enemyProx * velSpeedNorm;
              this.currentFitness += slingScore * 100 * this.params.weightWallSlingshot;
            }
          }
        }
      }

      if (this.params.rewardPackTangle && activeEnemies.length >= 2) {
        const pairRadius = 65;
        let totalTightness = 0;
        let weightedCx = 0;
        let weightedCy = 0;

        for (let i = 0; i < activeEnemies.length; i++) {
          for (let j = i + 1; j < activeEnemies.length; j++) {
            const dij = Math.hypot(
              activeEnemies[j].x - activeEnemies[i].x,
              activeEnemies[j].y - activeEnemies[i].y
            );
            if (dij < pairRadius) {
              const t = 1 - dij / pairRadius;
              totalTightness += t;
              weightedCx += (activeEnemies[i].x + activeEnemies[j].x) * 0.5 * t;
              weightedCy += (activeEnemies[i].y + activeEnemies[j].y) * 0.5 * t;
            }
          }
        }

        if (totalTightness > 0.001) {
          const cx = weightedCx / totalTightness;
          const cy = weightedCy / totalTightness;
          const cdx = cx - this.player.x;
          const cdy = cy - this.player.y;
          const cd = Math.hypot(cdx, cdy);
          if (cd > 0.001) {
            const clusterDensity = Math.min(1, totalTightness / Math.max(1, activeEnemies.length * 0.75));
            const behindFactor = -(ux * (cdx / cd) + uy * (cdy / cd));
            const safeMargin = minEnemyDist > 26 ? Math.min(1, (minEnemyDist - 26) / 60) : -0.5;
            const tangleScore =
              clusterDensity *
              (behindFactor > 0 ? behindFactor * safeMargin : behindFactor * 0.5) *
              velSpeedNorm;
            this.currentFitness += tangleScore * 100 * this.params.weightPackTangle;
          }
        }
      }

      if (this.params.rewardSpideySense && activeEnemies.length > 0) {
        const senseRadius = 200;
        let omegaX = 0;
        let omegaY = 0;

        for (let i = 0; i < activeEnemies.length; i++) {
          const dx = activeEnemies[i].x - this.player.x;
          const dy = activeEnemies[i].y - this.player.y;
          const d = Math.hypot(dx, dy);
          if (d <= 0.001 || d >= senseRadius) continue;

          const gx = dx / d;
          const gy = dy / d;
          const prox = 1 - d / senseRadius;
          const proxWeight = prox * prox;
          const closingSpeed =
            ((this.player.vx - activeEnemyVels[i].vx) * gx +
              (this.player.vy - activeEnemyVels[i].vy) * gy) /
            this.playerSpeed;
          const closeWeight = 0.35 + Math.max(0, closingSpeed);
          const gradMag = proxWeight * closeWeight;

          omegaX += gx * gradMag;
          omegaY += gy * gradMag;
        }

        const uphillPierce = omegaX * nvx + omegaY * nvy;
        if (uphillPierce > 0) {
          this.currentFitness -= uphillPierce * 100 * this.params.weightSpideySense;
        }
      }

      if (this.params.rewardVectorStabbing) {
        const stabBuffer = 24;
        const rawNextX = prevPx + this.player.vx;
        const rawNextY = prevPy + this.player.vy;

        const dLeft = Math.max(0, prevPx - 6);
        const dRight = Math.max(0, SIM_WIDTH - 6 - prevPx);
        const dTop = Math.max(0, prevPy - 6);
        const dBot = Math.max(0, SIM_HEIGHT - 6 - prevPy);

        const outLeft = Math.max(0, -nvx);
        const outRight = Math.max(0, nvx);
        const outTop = Math.max(0, -nvy);
        const outBot = Math.max(0, nvy);

        const pierceLeft = Math.max(0, 6 - rawNextX) / this.playerSpeed;
        const pierceRight = Math.max(0, rawNextX - (SIM_WIDTH - 6)) / this.playerSpeed;
        const pierceTop = Math.max(0, 6 - rawNextY) / this.playerSpeed;
        const pierceBot = Math.max(0, rawNextY - (SIM_HEIGHT - 6)) / this.playerSpeed;

        const proxLeft = dLeft < stabBuffer ? (1 - dLeft / stabBuffer) * outLeft : 0;
        const proxRight = dRight < stabBuffer ? (1 - dRight / stabBuffer) * outRight : 0;
        const proxTop = dTop < stabBuffer ? (1 - dTop / stabBuffer) * outTop : 0;
        const proxBot = dBot < stabBuffer ? (1 - dBot / stabBuffer) * outBot : 0;

        const stabLeft = Math.max(pierceLeft, 0.5 * proxLeft);
        const stabRight = Math.max(pierceRight, 0.5 * proxRight);
        const stabTop = Math.max(pierceTop, 0.5 * proxTop);
        const stabBot = Math.max(pierceBot, 0.5 * proxBot);

        const totalStab = Math.min(1.5, stabLeft + stabRight + stabTop + stabBot);
        if (totalStab > 0) {
          this.currentFitness += totalStab * 100 * this.params.weightVectorStabbing;
        }
      }

      if (this.params.rewardTwitch && activeEnemies.length > 0 && this.stepCount > 0) {
        const prevStepMag = Math.hypot(prevStepVx, prevStepVy);
        if (prevStepMag > 0.05) {
          const fx = prevStepVx / prevStepMag;
          const fy = prevStepVy / prevStepMag;
          const twitchRadius = 115;
          const turnSnap = Math.min(1, Math.hypot(nvx - prevStepVx, nvy - prevStepVy) / 1.1);
          const latPrev = Math.abs(-ux * fy + uy * fx);

          let bestPos = 0;
          let worstNeg = 0;
          for (let i = 0; i < activeEnemies.length; i++) {
            const dx = activeEnemies[i].x - this.player.x;
            const dy = activeEnemies[i].y - this.player.y;
            const d = Math.hypot(dx, dy);
            if (d <= 0.001 || d >= twitchRadius) continue;

            const ex = dx / d;
            const ey = dy / d;
            const frontDot = fx * ex + fy * ey;
            if (frontDot <= 0.45) continue;

            const closeness = 1 - d / twitchRadius;
            const frontWeight = closeness * ((frontDot - 0.45) / 0.55);
            const towardEnemy = ux * ex + uy * ey;

            if (towardEnemy > 0.65) {
              const pen = -frontWeight * ((towardEnemy - 0.65) / 0.35);
              if (pen < worstNeg) worstNeg = pen;
            } else {
              const latEnemy = Math.abs(-ux * ey + uy * ex);
              const score =
                frontWeight *
                (0.55 * latPrev + 0.45 * latEnemy) *
                (0.35 + 0.65 * turnSnap);
              if (score > bestPos) bestPos = score;
            }
          }
          if (bestPos > 0 || worstNeg < 0) {
            this.currentFitness +=
              (bestPos + worstNeg) * velSpeedNorm * 100 * this.params.weightTwitch;
          }
        }
      }

      if (this.params.rewardFarWallSense) {
        const farX = Math.max(
          6,
          Math.min(SIM_WIDTH - 6, SIM_WIDTH * 0.5 + (SIM_WIDTH * 0.5 - this.emaPx) * 1.5)
        );
        const farY = Math.max(
          6,
          Math.min(SIM_HEIGHT - 6, SIM_HEIGHT * 0.5 + (SIM_HEIGHT * 0.5 - this.emaPy) * 1.5)
        );
        const toFarX = farX - this.player.x;
        const toFarY = farY - this.player.y;
        const distToFar = Math.hypot(toFarX, toFarY);
        if (distToFar > 0.001) {
          const maxSpan = Math.hypot(SIM_WIDTH, SIM_HEIGHT);
          const alignment = (ux * toFarX + uy * toFarY) / distToFar;
          const farWeight = Math.min(1, distToFar / (maxSpan * 0.45));
          const farScore = alignment * farWeight * velSpeedNorm;
          this.currentFitness += farScore * 100 * this.params.weightFarWallSense;
        }
      }

      if (this.params.rewardSplitGapCollect && activeEnemies.length >= 2) {
        const sortedEntries = activeEnemies
          .map((e, localIdx) => {
            const dx = e.x - this.player.x;
            const dy = e.y - this.player.y;
            return {
              x: e.x,
              y: e.y,
              idx: activeEnemyIndices[localIdx] ?? localIdx,
              dist: Math.hypot(dx, dy),
              angle: Math.atan2(dy, dx),
            };
          })
          .sort((a, b) => a.dist - b.dist);

        const withinRadius = sortedEntries.filter((e) => e.dist <= 220);
        const nearby = (
          withinRadius.length >= 2
            ? withinRadius.slice(0, 12)
            : sortedEntries.slice(0, Math.min(sortedEntries.length, 8))
        ).sort((a, b) => a.angle - b.angle);

        if (nearby.length >= 2) {
          let maxGap = -1;
          let widestIndex = 0;
          for (let i = 0; i < nearby.length; i++) {
            const a1 = nearby[i].angle;
            const a2 =
              i === nearby.length - 1 ? nearby[0].angle + Math.PI * 2 : nearby[i + 1].angle;
            const gap = a2 - a1;
            if (gap > maxGap) {
              maxGap = gap;
              widestIndex = i;
            }
          }

          const velAngle = Math.atan2(uy, ux);
          const isGapAlreadyIdentified = (pairKey: number, mx: number, my: number): boolean => {
            for (let k = this.collectedSplitGaps.length - 1; k >= 0; k--) {
              const rec = this.collectedSplitGaps[k];
              const age = this.stepCount - rec.step;
              const spatialDist = Math.hypot(mx - rec.mx, my - rec.my);
              if (rec.pairKey === pairKey && (age < 240 || spatialDist < 100)) {
                return true;
              }
              if (spatialDist < 48 && age < 360) {
                return true;
              }
            }
            return false;
          };

          let bestForwardShape = 0;
          let crossedStepDelta = 0;
          const rX = this.player.x - prevPx;
          const rY = this.player.y - prevPy;

          for (let i = 0; i < nearby.length; i++) {
            const e1 = nearby[i];
            const e2 = i === nearby.length - 1 ? nearby[0] : nearby[i + 1];
            const chordLen = Math.hypot(e2.x - e1.x, e2.y - e1.y);
            if (chordLen < 24 || chordLen > 280) continue;

            const pairKey = Math.min(e1.idx, e2.idx) * 100 + Math.max(e1.idx, e2.idx);
            const mx = (e1.x + e2.x) * 0.5;
            const my = (e1.y + e2.y) * 0.5;
            const alreadyCollected = isGapAlreadyIdentified(pairKey, mx, my);

            // Check segment crossing between player step (prevP -> currP) and gap gate (e1 -> e2)
            const sX = e2.x - e1.x;
            const sY = e2.y - e1.y;
            const denom = rX * sY - rY * sX;
            if (Math.abs(denom) > 1e-6) {
              const qpx = e1.x - prevPx;
              const qpy = e1.y - prevPy;
              const s = (qpx * sY - qpy * sX) / denom;
              const t = (qpx * rY - qpy * rX) / denom;
              if (s >= 0 && s <= 1 && t >= 0.1 && t <= 0.9) {
                const postClearance = Math.min(
                  Math.hypot(this.player.x - e1.x, this.player.y - e1.y),
                  Math.hypot(this.player.x - e2.x, this.player.y - e2.y)
                );
                if (postClearance >= 13) {
                  const centerQuality = Math.max(0.25, 4 * t * (1 - t));
                  const widestMult = i === widestIndex ? 1.25 : 1.0;
                  if (alreadyCollected) {
                    crossedStepDelta -= 400 * centerQuality;
                  } else {
                    crossedStepDelta += 600 * centerQuality * widestMult;
                    this.collectedSplitGaps.push({ pairKey, mx, my, step: this.stepCount });
                    if (this.collectedSplitGaps.length > 64) {
                      this.collectedSplitGaps.shift();
                    }
                  }
                }
              }
            }

            // Forward gap approach shaping: reward heading through uncollected forward gaps,
            // penalize re-heading into already-collected gaps
            const toMidX = mx - this.player.x;
            const toMidY = my - this.player.y;
            const md = Math.hypot(toMidX, toMidY);
            if (md > 0.001 && md < 190 && minEnemyDist > 16) {
              const forwardDot = (ux * toMidX + uy * toMidY) / md;
              if (forwardDot > 0.25) {
                const a1 = e1.angle;
                const a2 = i === nearby.length - 1 ? e2.angle + Math.PI * 2 : e2.angle;
                let va = velAngle;
                while (va < a1) va += Math.PI * 2;
                while (va > a1 + Math.PI * 2) va -= Math.PI * 2;
                if (va >= a1 && va <= a2) {
                  const span = a2 - a1;
                  const frac = span > 0.001 ? (va - a1) / span : 0.5;
                  const centerAim = Math.max(0, 1 - Math.abs(frac - 0.5) * 2);
                  const prox = 1 - md / 190;
                  const widestMult = i === widestIndex ? 1.35 : 1.0;
                  const shape = centerAim * prox * widestMult * velSpeedNorm;
                  if (alreadyCollected) {
                    if (-shape * 55 < bestForwardShape) bestForwardShape = -shape * 55;
                  } else {
                    if (shape * 60 > bestForwardShape) bestForwardShape = shape * 60;
                  }
                }
              }
            }
          }

          this.currentFitness +=
            (crossedStepDelta + bestForwardShape) * this.params.weightSplitGapCollect;
        }
      }
    }

    this.emaPx = 0.98 * this.emaPx + 0.02 * this.player.x;
    this.emaPy = 0.98 * this.emaPy + 0.02 * this.player.y;

    for (let i = 0; i < this.enemies.length; i++) {
      const e = this.enemies[i];
      if (e.spawnDelay && e.spawnDelay > 0) {
        e.spawnDelay--;
      }
    }

    this.stepCount++;
    if (this.stepCount >= this.maxSteps) {
      this.endEpisode();
    }
  }

  evaluateSpatialSample(px: number, py: number): { fitness: number; hit: boolean } {
    if (!this.agent) return { fitness: 0, hit: false };

    const activeEnemies = this.enemies.filter((e) => !e.spawnDelay || e.spawnDelay <= 0);
    const inputs: number[] = [
      (px / SIM_WIDTH) * 2 - 1,
      (py / SIM_HEIGHT) * 2 - 1,
      this.player.vx / this.playerSpeed,
      this.player.vy / this.playerSpeed,
    ];
    for (let i = 0; i < 50; i++) {
      if (i < activeEnemies.length) {
        inputs.push((activeEnemies[i].x - px) / SIM_WIDTH);
        inputs.push((activeEnemies[i].y - py) / SIM_HEIGHT);
      } else {
        inputs.push(0);
        inputs.push(0);
      }
    }

    const outputs = this.agent.forwardOnly(inputs);
    let ax = outputs[0] ?? 0;
    let ay = outputs[1] ?? 0;
    const mag = Math.hypot(ax, ay);
    if (mag > 1) {
      ax /= mag;
      ay /= mag;
    }

    const vx = ax * this.playerSpeed;
    const vy = ay * this.playerSpeed;
    const nx = Math.max(6, Math.min(SIM_WIDTH - 6, px + vx));
    const ny = Math.max(6, Math.min(SIM_HEIGHT - 6, py + vy));
    const nvx = vx / this.playerSpeed;
    const nvy = vy / this.playerSpeed;

    let fitness = 0;
    let minEnemyDist = Infinity;
    const enemyVels: { vx: number; vy: number }[] = [];

    for (let i = 0; i < activeEnemies.length; i++) {
      const e = activeEnemies[i];
      const dx = nx - e.x;
      const dy = ny - e.y;
      const d = Math.hypot(dx, dy);
      if (d < minEnemyDist) minEnemyDist = d;
      const effSpeed =
        this.params.enemySpeedMode === 'rarity'
          ? this.playerSpeed * (Math.min(90, Math.max(1, e.speedPct ?? 60)) / 100)
          : this.enemySpeed;
      enemyVels.push(d > 0.1 ? { vx: (dx / d) * effSpeed, vy: (dy / d) * effSpeed } : { vx: 0, vy: 0 });
    }

    const hit = activeEnemies.length > 0 && minEnemyDist < 11;
    if (hit && this.params.rewardHit) {
      fitness += this.params.weightHit;
    }

    if (this.params.rewardEnemyDist && activeEnemies.length > 0) {
      fitness += minEnemyDist * this.params.weightEnemyDist;
    }

    if (this.params.rewardWallDist) {
      const minWallDist = Math.min(nx - 6, SIM_WIDTH - 6 - nx, ny - 6, SIM_HEIGHT - 6 - ny);
      fitness += minWallDist * this.params.weightWallDist;
    }

    if (this.params.rewardGroupEnemies && activeEnemies.length > 1) {
      let cx = 0;
      let cy = 0;
      for (const e of activeEnemies) {
        cx += e.x;
        cy += e.y;
      }
      cx /= activeEnemies.length;
      cy /= activeEnemies.length;
      let spread = 0;
      for (const e of activeEnemies) spread += Math.hypot(e.x - cx, e.y - cy);
      const maxArenaRadius = Math.hypot(SIM_WIDTH, SIM_HEIGHT) * 0.5;
      fitness += Math.max(0, maxArenaRadius - spread / activeEnemies.length) * this.params.weightGroupEnemies;
    }

    if (this.params.rewardSpreadEnemies && activeEnemies.length > 1) {
      let cx = 0;
      let cy = 0;
      for (const e of activeEnemies) {
        cx += e.x;
        cy += e.y;
      }
      cx /= activeEnemies.length;
      cy /= activeEnemies.length;
      let spread = 0;
      for (const e of activeEnemies) spread += Math.hypot(e.x - cx, e.y - cy);
      fitness += (spread / activeEnemies.length) * this.params.weightSpreadEnemies;
    }

    if (this.params.rewardEdgeEnemies && activeEnemies.length > 0) {
      const maxPossibleEdgeDist = Math.min(SIM_WIDTH, SIM_HEIGHT) * 0.5;
      let total = 0;
      for (const e of activeEnemies) {
        const minEdge = Math.min(e.x, SIM_WIDTH - e.x, e.y, SIM_HEIGHT - e.y);
        total += Math.max(0, maxPossibleEdgeDist - minEdge);
      }
      fitness += (total / activeEnemies.length) * this.params.weightEdgeEnemies;
    }

    if (this.params.rewardAvgEnemyCenter && activeEnemies.length > 0) {
      let cx = 0;
      let cy = 0;
      for (const e of activeEnemies) {
        cx += e.x;
        cy += e.y;
      }
      cx /= activeEnemies.length;
      cy /= activeEnemies.length;
      const distFromCenter = Math.hypot(cx - SIM_WIDTH * 0.5, cy - SIM_HEIGHT * 0.5);
      const maxArenaRadius = Math.hypot(SIM_WIDTH, SIM_HEIGHT) * 0.5;
      fitness += Math.max(0, maxArenaRadius - distFromCenter) * this.params.weightAvgEnemyCenter;
    }

    if (this.params.rewardOppHeading && activeEnemies.length > 0) {
      let avgEx = 0;
      let avgEy = 0;
      for (const e of activeEnemies) {
        const dx = e.x - nx;
        const dy = e.y - ny;
        const d = Math.hypot(dx, dy);
        if (d > 0.001) {
          avgEx += dx / d;
          avgEy += dy / d;
        }
      }
      avgEx /= activeEnemies.length;
      avgEy /= activeEnemies.length;
      const avgMag = Math.hypot(avgEx, avgEy);
      const pMag = Math.hypot(vx, vy);
      if (avgMag > 0.001 && pMag > 0.001) {
        const alignment = (vx / pMag) * (-avgEx / avgMag) + (vy / pMag) * (-avgEy / avgMag);
        fitness += alignment * 100 * this.params.weightOppHeading;
      }
    }

    if (this.params.rewardSmoothVel) {
      const speedNorm = Math.hypot(nvx, nvy);
      const jerk = Math.hypot(nvx - this.prevVx, nvy - this.prevVy);
      const noise = Math.hypot(nvx - this.emaVx, nvy - this.emaVy);
      fitness += (speedNorm - 1.5 * jerk - 1.0 * noise) * 80 * this.params.weightSmoothVel;
    }

    if ((this.params.rewardSplitGap || this.params.rewardSplitGapPredict || this.params.rewardSplitGapCollect) && activeEnemies.length > 0) {
      const sortedNearby = activeEnemies
        .map((e) => ({ dist: Math.hypot(e.x - nx, e.y - ny), angle: Math.atan2(e.y - ny, e.x - nx) }))
        .sort((a, b) => a.dist - b.dist);
      const within = sortedNearby.filter((e) => e.dist <= 220);
      const nearby = within.length >= 2 ? within.slice(0, 12) : sortedNearby.slice(0, Math.min(sortedNearby.length, 8));
      let bestMidAngle = 0;
      if (nearby.length === 1) {
        bestMidAngle = nearby[0].angle + Math.PI;
      } else {
        const angles = nearby.map((e) => e.angle).sort((a, b) => a - b);
        let maxGap = -1;
        for (let i = 0; i < angles.length; i++) {
          const a1 = angles[i];
          const a2 = i === angles.length - 1 ? angles[0] + Math.PI * 2 : angles[i + 1];
          if (a2 - a1 > maxGap) {
            maxGap = a2 - a1;
            bestMidAngle = a1 + maxGap * 0.5;
          }
        }
      }
      const gapAlign = nvx * Math.cos(bestMidAngle) + nvy * Math.sin(bestMidAngle);
      if (this.params.rewardSplitGap) fitness += gapAlign * 100 * this.params.weightSplitGap;
      if (this.params.rewardSplitGapPredict) fitness += gapAlign * 100 * this.params.weightSplitGapPredict;
      if (this.params.rewardSplitGapCollect) fitness += Math.max(0, gapAlign) * 60 * this.params.weightSplitGapCollect;
    }

    const speedNorm = Math.hypot(nvx, nvy);
    if (speedNorm > 0.01) {
      const ux = nvx / speedNorm;
      const uy = nvy / speedNorm;

      if (this.params.rewardEdgeSense && activeEnemies.length > 0) {
        const rx = -uy;
        const ry = ux;
        let leftT = 0;
        let rightT = 0;
        let frontT = 0;
        for (const e of activeEnemies) {
          const dx = e.x - nx;
          const dy = e.y - ny;
          const d = Math.hypot(dx, dy);
          if (d <= 0.001 || d >= 140) continue;
          const closeness = 1 - d / 140;
          const ndx = dx / d;
          const ndy = dy / d;
          const fwd = ndx * ux + ndy * uy;
          const lat = ndx * rx + ndy * ry;
          const pt = closeness * Math.abs(lat);
          if (lat < 0 && pt > leftT) leftT = pt;
          if (lat > 0 && pt > rightT) rightT = pt;
          if (fwd > 0.75) {
            const ft = closeness * ((fwd - 0.75) / 0.25);
            if (ft > frontT) frontT = ft;
          }
        }
        fitness += (Math.abs(leftT - rightT) - Math.min(leftT, rightT) - 0.5 * frontT) * speedNorm * 100 * this.params.weightEdgeSense;
      }

      if (this.params.rewardOrbitKite && activeEnemies.length > 0 && minEnemyDist > 0.001) {
        let nearDx = 0;
        let nearDy = 0;
        let nd = Infinity;
        for (const e of activeEnemies) {
          const d = Math.hypot(e.x - nx, e.y - ny);
          if (d < nd) {
            nd = d;
            nearDx = e.x - nx;
            nearDy = e.y - ny;
          }
        }
        const tangent = 1 - Math.abs(ux * (nearDx / nd) + uy * (nearDy / nd));
        const sweet = Math.max(0, 1 - Math.abs(nd - 85) / 55);
        const closePen = nd < 35 ? ((35 - nd) / 35) * 0.6 : 0;
        fitness += (sweet * (tangent - 0.15) - closePen) * speedNorm * 100 * this.params.weightOrbitKite;
      }

      if (this.params.rewardOpenCorridor) {
        const baseAngle = Math.atan2(uy, ux);
        const offsets = [-0.42, 0, 0.42];
        const maxHorizon = 240;
        let minRay = 1;
        let sumRay = 0;
        for (const off of offsets) {
          const rx = Math.cos(baseAngle + off);
          const ry = Math.sin(baseAngle + off);
          const tx = rx > 1e-5 ? (SIM_WIDTH - 6 - nx) / rx : rx < -1e-5 ? (6 - nx) / rx : maxHorizon;
          const ty = ry > 1e-5 ? (SIM_HEIGHT - 6 - ny) / ry : ry < -1e-5 ? (6 - ny) / ry : maxHorizon;
          let clearance = Math.max(0, Math.min(maxHorizon, tx, ty));
          for (const e of activeEnemies) {
            const dx = e.x - nx;
            const dy = e.y - ny;
            const proj = dx * rx + dy * ry;
            if (proj > 0 && proj < clearance + 11 && Math.abs(dx * ry - dy * rx) < 22) {
              clearance = Math.min(clearance, Math.max(0, proj - 11));
            }
          }
          const nc = clearance / maxHorizon;
          if (nc < minRay) minRay = nc;
          sumRay += nc;
        }
        fitness += (0.5 * minRay + 0.5 * (sumRay / offsets.length)) * speedNorm * 100 * this.params.weightOpenCorridor;
      }

      if (this.params.rewardSpideySense && activeEnemies.length > 0) {
        let ox = 0;
        let oy = 0;
        for (let i = 0; i < activeEnemies.length; i++) {
          const dx = activeEnemies[i].x - nx;
          const dy = activeEnemies[i].y - ny;
          const d = Math.hypot(dx, dy);
          if (d <= 0.001 || d >= 200) continue;
          const gx = dx / d;
          const gy = dy / d;
          const prox = 1 - d / 200;
          const closing = ((vx - enemyVels[i].vx) * gx + (vy - enemyVels[i].vy) * gy) / this.playerSpeed;
          const gm = prox * prox * (0.35 + Math.max(0, closing));
          ox += gx * gm;
          oy += gy * gm;
        }
        const pierce = ox * nvx + oy * nvy;
        if (pierce > 0) fitness -= pierce * 100 * this.params.weightSpideySense;
      }

      if (this.params.rewardVectorStabbing) {
        const stabBuffer = 24;
        const rawNextX = px + vx;
        const rawNextY = py + vy;
        const dLeft = Math.max(0, px - 6);
        const dRight = Math.max(0, SIM_WIDTH - 6 - px);
        const dTop = Math.max(0, py - 6);
        const dBot = Math.max(0, SIM_HEIGHT - 6 - py);
        const stabLeft = Math.max(Math.max(0, 6 - rawNextX) / this.playerSpeed, 0.5 * (dLeft < stabBuffer ? (1 - dLeft / stabBuffer) * Math.max(0, -nvx) : 0));
        const stabRight = Math.max(Math.max(0, rawNextX - (SIM_WIDTH - 6)) / this.playerSpeed, 0.5 * (dRight < stabBuffer ? (1 - dRight / stabBuffer) * Math.max(0, nvx) : 0));
        const stabTop = Math.max(Math.max(0, 6 - rawNextY) / this.playerSpeed, 0.5 * (dTop < stabBuffer ? (1 - dTop / stabBuffer) * Math.max(0, -nvy) : 0));
        const stabBot = Math.max(Math.max(0, rawNextY - (SIM_HEIGHT - 6)) / this.playerSpeed, 0.5 * (dBot < stabBuffer ? (1 - dBot / stabBuffer) * Math.max(0, nvy) : 0));
        const totalStab = Math.min(1.5, stabLeft + stabRight + stabTop + stabBot);
        if (totalStab > 0) fitness += totalStab * 100 * this.params.weightVectorStabbing;
      }

      if (this.params.rewardTwitch && activeEnemies.length > 0) {
        const prevStepMag = Math.hypot(this.prevVx, this.prevVy);
        if (prevStepMag > 0.05) {
          const fx = this.prevVx / prevStepMag;
          const fy = this.prevVy / prevStepMag;
          const twitchRadius = 115;
          const turnSnap = Math.min(1, Math.hypot(nvx - this.prevVx, nvy - this.prevVy) / 1.1);
          const latPrev = Math.abs(-ux * fy + uy * fx);
          let bestPos = 0;
          let worstNeg = 0;
          for (let i = 0; i < activeEnemies.length; i++) {
            const dx = activeEnemies[i].x - nx;
            const dy = activeEnemies[i].y - ny;
            const d = Math.hypot(dx, dy);
            if (d <= 0.001 || d >= twitchRadius) continue;
            const ex = dx / d;
            const ey = dy / d;
            const frontDot = fx * ex + fy * ey;
            if (frontDot <= 0.45) continue;
            const frontWeight = (1 - d / twitchRadius) * ((frontDot - 0.45) / 0.55);
            const towardEnemy = ux * ex + uy * ey;
            if (towardEnemy > 0.65) {
              const pen = -frontWeight * ((towardEnemy - 0.65) / 0.35);
              if (pen < worstNeg) worstNeg = pen;
            } else {
              const latEnemy = Math.abs(-ux * ey + uy * ex);
              const score =
                frontWeight *
                (0.55 * latPrev + 0.45 * latEnemy) *
                (0.35 + 0.65 * turnSnap);
              if (score > bestPos) bestPos = score;
            }
          }
          if (bestPos > 0 || worstNeg < 0) {
            fitness += (bestPos + worstNeg) * speedNorm * 100 * this.params.weightTwitch;
          }
        }
      }

      if (this.params.rewardFarWallSense) {
        const farX = Math.max(6, Math.min(SIM_WIDTH - 6, SIM_WIDTH * 0.5 + (SIM_WIDTH * 0.5 - this.emaPx) * 1.5));
        const farY = Math.max(6, Math.min(SIM_HEIGHT - 6, SIM_HEIGHT * 0.5 + (SIM_HEIGHT * 0.5 - this.emaPy) * 1.5));
        const toFarX = farX - nx;
        const toFarY = farY - ny;
        const dFar = Math.hypot(toFarX, toFarY);
        if (dFar > 0.001) {
          const align = (ux * toFarX + uy * toFarY) / dFar;
          const fw = Math.min(1, dFar / (Math.hypot(SIM_WIDTH, SIM_HEIGHT) * 0.45));
          fitness += align * fw * speedNorm * 100 * this.params.weightFarWallSense;
        }
      }
    }

    return { fitness, hit };
  }

  endEpisode(): void {
    if (!this.agent) return;
    this.agent.fitness = this.currentFitness;
    if (this.currentFitness > this.bestFitness) {
      this.bestFitness = this.currentFitness;
      this.bestAgent = this.agent.clone();
      this.bestAgent.fitness = this.agent.fitness;
    }

    this.popIndex++;
    if (this.popIndex >= this.popSize) {
      this.population.sort((a, b) => b.fitness - a.fitness);
      const nextGen: NeuralNetwork[] = [];
      for (let i = 0; i < 3; i++) {
        nextGen.push(this.population[i].clone());
      }
      while (nextGen.length < this.popSize) {
        const parent = this.population[Math.floor(Math.random() * 3)];
        const child = parent.clone();
        child.mutate(0.2);
        nextGen.push(child);
      }
      this.population = nextGen;
      this.generation++;
      this.popIndex = 0;
    }

    const statsText = `${this.generation} | ${Math.round(this.currentFitness)} | ${Math.round(
      this.bestFitness === -Infinity ? 0 : this.bestFitness
    )}`;
    this.onStatsChange?.(statsText);
    this.startEpisode();
  }

  stepFrame(): void {
    if (this.params.turboMode) {
      const startTime = performance.now();
      while (performance.now() - startTime < 14) {
        this.update();
      }
    } else {
      for (let i = 0; i < this.params.simSpeed; i++) {
        this.update();
      }
    }
  }

  drawArena(ctx: CanvasRenderingContext2D): void {
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, SIM_WIDTH, SIM_HEIGHT);

    ctx.fillStyle = '#0ff';
    ctx.fillRect(this.player.x - 6, this.player.y - 6, 12, 12);

    ctx.fillStyle = '#f00';
    for (const e of this.enemies) {
      if (!e.spawnDelay || e.spawnDelay <= 0) {
        ctx.fillRect(e.x - 5, e.y - 5, 10, 10);
      }
    }
  }

  getNNNodeLayout(): { x: number; y: number }[][] {
    if (!this.agent) return [];
    const activeInputCount = 4 + this.params.numEnemies * 2;
    const hiddenSizes = this.agent.getHiddenLayerSizes();
    const layerCounts = [activeInputCount, ...hiddenSizes, 2];
    const numLayers = layerCounts.length;
    const padX = 30;
    const padY = 15;
    const layerSpacing = (NN_WIDTH - padX * 2) / Math.max(1, numLayers - 1);
    const layout: { x: number; y: number }[][] = [];

    for (let l = 0; l < numLayers; l++) {
      const count = layerCounts[l];
      const x = padX + l * layerSpacing;
      const stepY = (NN_HEIGHT - padY * 2) / Math.max(1, count - 1);
      const col: { x: number; y: number }[] = [];
      for (let n = 0; n < count; n++) {
        const y = count === 1 ? NN_HEIGHT / 2 : padY + n * stepY;
        col.push({ x, y });
      }
      layout.push(col);
    }
    return layout;
  }

  findHiddenNodeAt(
    x: number,
    y: number,
    radius = 9
  ): { hiddenLayer: number; nodeIndex: number } | null {
    if (!this.agent) return null;
    const layout = this.getNNNodeLayout();
    let bestDist = radius;
    let match: { hiddenLayer: number; nodeIndex: number } | null = null;

    for (let l = 1; l < layout.length - 1; l++) {
      const nodes = layout[l];
      for (let n = 0; n < nodes.length; n++) {
        const d = Math.hypot(nodes[n].x - x, nodes[n].y - y);
        if (d <= bestDist) {
          bestDist = d;
          match = { hiddenLayer: l - 1, nodeIndex: n };
        }
      }
    }
    return match;
  }

  toggleHiddenNodeFreeze(
    hiddenLayer: number,
    nodeIndex: number
  ): { isFrozen: boolean; canPrune: boolean } {
    if (!this.agent) {
      return { isFrozen: this.isFrozen, canPrune: false };
    }
    const currentlyFrozen = Boolean(this.agent.frozenBiases[hiddenLayer]?.[nodeIndex]);
    const nextFrozen = !currentlyFrozen;

    for (const net of this.population) {
      net.setHiddenNodeFrozen(hiddenLayer, nodeIndex, nextFrozen);
    }
    if (this.bestAgent) {
      this.bestAgent.setHiddenNodeFrozen(hiddenLayer, nodeIndex, nextFrozen);
    }

    this.isFrozen = this.agent.hasFrozenWeights();
    return {
      isFrozen: this.isFrozen,
      canPrune: this.canPrune(),
    };
  }

  toggleHiddenNodesInRect(
    x1: number,
    y1: number,
    x2: number,
    y2: number
  ): { changed: boolean; isFrozen: boolean; canPrune: boolean } {
    if (!this.agent) {
      return { changed: false, isFrozen: this.isFrozen, canPrune: false };
    }
    const minX = Math.min(x1, x2) - 5;
    const maxX = Math.max(x1, x2) + 5;
    const minY = Math.min(y1, y2) - 5;
    const maxY = Math.max(y1, y2) + 5;
    const layout = this.getNNNodeLayout();
    const selected: { hiddenLayer: number; nodeIndex: number; frozen: boolean }[] = [];

    for (let l = 1; l < layout.length - 1; l++) {
      const nodes = layout[l];
      for (let n = 0; n < nodes.length; n++) {
        const pt = nodes[n];
        if (pt.x >= minX && pt.x <= maxX && pt.y >= minY && pt.y <= maxY) {
          selected.push({
            hiddenLayer: l - 1,
            nodeIndex: n,
            frozen: Boolean(this.agent.frozenBiases[l - 1]?.[n]),
          });
        }
      }
    }

    if (selected.length === 0) {
      return { changed: false, isFrozen: this.isFrozen, canPrune: this.canPrune() };
    }

    const shouldFreeze = selected.some((s) => !s.frozen);
    for (const item of selected) {
      for (const net of this.population) {
        net.setHiddenNodeFrozen(item.hiddenLayer, item.nodeIndex, shouldFreeze);
      }
      if (this.bestAgent) {
        this.bestAgent.setHiddenNodeFrozen(item.hiddenLayer, item.nodeIndex, shouldFreeze);
      }
    }

    this.isFrozen = this.agent.hasFrozenWeights();
    return {
      changed: true,
      isFrozen: this.isFrozen,
      canPrune: this.canPrune(),
    };
  }

  drawNN(
    nnCtx: CanvasRenderingContext2D,
    selectionBox?: { x1: number; y1: number; x2: number; y2: number } | null
  ): void {
    nnCtx.fillStyle = '#000';
    nnCtx.fillRect(0, 0, NN_WIDTH, NN_HEIGHT);
    if (!this.agent || !this.agent.layerValues || this.agent.layerValues.length === 0) return;

    const activeInputCount = 4 + this.params.numEnemies * 2;
    const layers = this.agent.layerValues.map((vals, idx) =>
      idx === 0 ? vals.slice(0, activeInputCount) : vals
    );
    const numLayers = layers.length;
    const layout = this.getNNNodeLayout();
    const nodeCoords: { x: number; y: number; val: number }[][] = layout.map((col, l) =>
      col.map((pt, n) => ({ x: pt.x, y: pt.y, val: layers[l]?.[n] ?? 0 }))
    );

    const frozenBoxes: {
      layerEnd: number;
      left: number;
      right: number;
      top: number;
      bottom: number;
    }[] = [];

    for (let l = 1; l < numLayers - 1; l++) {
      const nodes = nodeCoords[l];
      let runStart = -1;
      for (let n = 0; n <= nodes.length; n++) {
        const isNodeFrozen =
          n < nodes.length && Boolean(this.agent.frozenBiases[l - 1]?.[n]);
        if (isNodeFrozen && runStart === -1) {
          runStart = n;
        } else if (!isNodeFrozen && runStart !== -1) {
          const top = nodes[runStart].y - 9;
          const bottom = nodes[n - 1].y + 9;
          const left = nodes[runStart].x - 11;
          const right = nodes[runStart].x + 11;
          const prev = frozenBoxes[frozenBoxes.length - 1];
          if (
            prev &&
            prev.layerEnd === l - 1 &&
            Math.abs(prev.top - top) < 1 &&
            Math.abs(prev.bottom - bottom) < 1
          ) {
            prev.layerEnd = l;
            prev.right = right;
          } else {
            frozenBoxes.push({ layerEnd: l, left, right, top, bottom });
          }
          runStart = -1;
        }
      }
    }

    for (const box of frozenBoxes) {
      const w = box.right - box.left;
      const h = box.bottom - box.top;
      nnCtx.fillStyle = 'rgba(38, 92, 175, 0.16)';
      nnCtx.strokeStyle = 'rgba(95, 165, 255, 0.52)';
      nnCtx.lineWidth = 1;
      nnCtx.beginPath();
      nnCtx.roundRect(box.left, box.top, w, h, 4);
      nnCtx.fill();
      nnCtx.stroke();
    }

    const firingEdges: {
      x1: number;
      y1: number;
      x2: number;
      y2: number;
      signal: number;
      mag: number;
      phaseOffset: number;
      frozen: boolean;
    }[] = [];

    for (let l = 0; l < this.agent.weights.length; l++) {
      const wMatrix = this.agent.weights[l];
      const fwMatrix = this.agent.frozenWeights[l];
      const fromNodes = nodeCoords[l];
      const toNodes = nodeCoords[l + 1];
      for (let r = 0; r < toNodes.length; r++) {
        for (let c = 0; c < fromNodes.length; c++) {
          const w = wMatrix[r][c];
          const isEdgeFrozen = Boolean(fwMatrix?.[r]?.[c]);
          const baseAlpha = Math.min(0.24, Math.abs(w) * 0.18);
          const signal = fromNodes[c].val * w;
          const signalMag = Math.abs(signal);

          if (baseAlpha >= 0.025) {
            nnCtx.strokeStyle = isEdgeFrozen
              ? w > 0
                ? `rgba(80, 150, 235, ${baseAlpha})`
                : `rgba(145, 115, 230, ${baseAlpha})`
              : w > 0
              ? `rgba(0, 200, 200, ${baseAlpha})`
              : `rgba(210, 60, 60, ${baseAlpha})`;
            nnCtx.lineWidth = Math.min(1.3, 0.45 + Math.abs(w) * 0.4);
            nnCtx.beginPath();
            nnCtx.moveTo(fromNodes[c].x, fromNodes[c].y);
            nnCtx.lineTo(toNodes[r].x, toNodes[r].y);
            nnCtx.stroke();
          }

          if (signalMag > 0.04) {
            firingEdges.push({
              x1: fromNodes[c].x,
              y1: fromNodes[c].y,
              x2: toNodes[r].x,
              y2: toNodes[r].y,
              signal,
              mag: signalMag,
              phaseOffset: l * 0.27 + (c * 0.13 + r * 0.19),
              frozen: isEdgeFrozen,
            });
          }
        }
      }
    }

    const pulsePhase = performance.now() * 0.0016;
    for (const edge of firingEdges) {
      const alpha = Math.min(0.88, edge.mag * 0.72);
      nnCtx.strokeStyle = edge.frozen
        ? edge.signal > 0
          ? `rgba(105, 185, 255, ${alpha})`
          : `rgba(165, 130, 255, ${alpha})`
        : edge.signal > 0
        ? `rgba(0, 255, 255, ${alpha})`
        : `rgba(255, 65, 65, ${alpha})`;
      nnCtx.lineWidth = Math.min(2.2, 0.7 + edge.mag * 1.05);
      nnCtx.beginPath();
      nnCtx.moveTo(edge.x1, edge.y1);
      nnCtx.lineTo(edge.x2, edge.y2);
      nnCtx.stroke();

      if (edge.mag > 0.2) {
        const t = (pulsePhase + edge.phaseOffset) % 1;
        const px = edge.x1 + (edge.x2 - edge.x1) * t;
        const py = edge.y1 + (edge.y2 - edge.y1) * t;
        const dotAlpha = Math.min(0.85, edge.mag * 0.7);
        nnCtx.fillStyle = edge.frozen
          ? edge.signal > 0
            ? `rgba(205, 235, 255, ${dotAlpha})`
            : `rgba(225, 210, 255, ${dotAlpha})`
          : edge.signal > 0
          ? `rgba(180, 255, 255, ${dotAlpha})`
          : `rgba(255, 165, 165, ${dotAlpha})`;
        nnCtx.beginPath();
        nnCtx.arc(px, py, 1.35, 0, Math.PI * 2);
        nnCtx.fill();
      }
    }

    for (let l = 0; l < numLayers; l++) {
      for (let n = 0; n < nodeCoords[l].length; n++) {
        const node = nodeCoords[l][n];
        const v = Math.max(-1, Math.min(1, node.val));
        const absV = Math.abs(v);
        const isHiddenFrozen =
          l > 0 &&
          l < numLayers - 1 &&
          Boolean(this.agent.frozenBiases[l - 1]?.[n]);

        nnCtx.beginPath();
        nnCtx.arc(node.x, node.y, 4, 0, Math.PI * 2);
        if (isHiddenFrozen) {
          if (v >= 0) {
            nnCtx.fillStyle = `rgb(${Math.floor(25 + v * 75)}, ${Math.floor(
              70 + v * 130
            )}, ${Math.floor(145 + v * 110)})`;
          } else {
            const neg = -v;
            nnCtx.fillStyle = `rgb(${Math.floor(65 + neg * 115)}, ${Math.floor(
              55 + neg * 60
            )}, ${Math.floor(135 + neg * 120)})`;
          }
          nnCtx.fill();
          nnCtx.strokeStyle =
            absV > 0.35 ? (v >= 0 ? '#93c5fd' : '#c4b5fd') : '#5b82b0';
          nnCtx.lineWidth = 1.35;
        } else {
          if (v >= 0) {
            const intensity = Math.floor(v * 255);
            nnCtx.fillStyle = `rgb(0, ${intensity}, ${intensity})`;
          } else {
            const intensity = Math.floor(-v * 255);
            nnCtx.fillStyle = `rgb(${intensity}, 40, 40)`;
          }
          nnCtx.fill();
          nnCtx.strokeStyle =
            absV > 0.35 ? (v >= 0 ? '#0aa' : '#a44') : '#555';
          nnCtx.lineWidth = 1;
        }
        nnCtx.stroke();
      }
    }

    const outputNodes = nodeCoords[numLayers - 1];
    if (outputNodes && outputNodes.length === 2) {
      nnCtx.fillStyle = '#666';
      nnCtx.font = '9px monospace';
      nnCtx.textBaseline = 'middle';
      nnCtx.fillText('vx', outputNodes[0].x + 8, outputNodes[0].y);
      nnCtx.fillText('vy', outputNodes[1].x + 8, outputNodes[1].y);
    }

    if (selectionBox) {
      const sx = Math.min(selectionBox.x1, selectionBox.x2);
      const sy = Math.min(selectionBox.y1, selectionBox.y2);
      const sw = Math.abs(selectionBox.x2 - selectionBox.x1);
      const sh = Math.abs(selectionBox.y2 - selectionBox.y1);
      if (sw >= 2 || sh >= 2) {
        nnCtx.save();
        nnCtx.fillStyle = 'rgba(95, 165, 255, 0.14)';
        nnCtx.strokeStyle = 'rgba(130, 200, 255, 0.85)';
        nnCtx.lineWidth = 1;
        nnCtx.setLineDash([4, 3]);
        nnCtx.fillRect(sx, sy, sw, sh);
        nnCtx.strokeRect(sx, sy, sw, sh);
        nnCtx.restore();
      }
    }
  }

  canPrune(): boolean {
    return this.agent ? this.agent.canPrune() : false;
  }

  pruneNetwork(mode: 'least' | 'strongest' = 'least'): {
    hiddenLayerSizes: number[];
    canPruneMore: boolean;
  } | null {
    if (!this.agent) return null;
    const pruned = this.agent.pruneNode(mode);
    if (!pruned) return null;

    for (const net of this.population) {
      if (net !== this.agent) {
        net.pruneNode(mode, pruned.layer);
      }
    }

    if (this.bestAgent) {
      this.bestAgent.pruneNode(mode, pruned.layer);
    }

    const sizes = this.agent.getHiddenLayerSizes();
    if (sizes.length === 1) {
      this.params.neuronsPerLayer = sizes[0];
    }

    return {
      hiddenLayerSizes: sizes,
      canPruneMore: this.agent.canPrune(),
    };
  }

  setFrozen(frozen: boolean): { isFrozen: boolean; canPrune: boolean } {
    this.isFrozen = frozen;
    for (const net of this.population) {
      if (frozen) net.freezeHiddenAll();
      else net.unfreezeAll();
    }
    if (this.bestAgent) {
      if (frozen) this.bestAgent.freezeHiddenAll();
      else this.bestAgent.unfreezeAll();
    }
    return {
      isFrozen: this.isFrozen,
      canPrune: this.canPrune(),
    };
  }

  scaleFrozenNetwork(
    targetLayers: number,
    targetNeurons: number,
    activation: ActivationType
  ): { layers: number; neurons: number; canPrune: boolean } {
    return this.resizeNetwork(targetLayers, targetNeurons, activation);
  }

  resizeNetwork(
    targetLayers: number,
    targetNeurons: number,
    activation: ActivationType
  ): { layers: number; neurons: number; canPrune: boolean } {
    if (!this.agent) {
      return {
        layers: this.params.hiddenLayersCount,
        neurons: this.params.neuronsPerLayer,
        canPrune: false,
      };
    }

    const frozenBounds = this.isFrozen
      ? this.agent.getMinimumFrozenDimensions()
      : { minLayers: 1, minNeurons: 2 };
    const clampedLayers = Math.min(5, Math.max(frozenBounds.minLayers, targetLayers));
    const clampedNeurons = Math.min(64, Math.max(frozenBounds.minNeurons, targetNeurons));

    for (let i = 0; i < this.population.length; i++) {
      this.population[i].resize(clampedLayers, clampedNeurons, activation, i < 3, this.isFrozen);
    }
    if (this.bestAgent) {
      this.bestAgent.resize(clampedLayers, clampedNeurons, activation, true, this.isFrozen);
    }

    if (this.agent.layerValues.length > 0) {
      this.agent.predict(this.agent.layerValues[0]);
    }

    this.params.hiddenLayersCount = clampedLayers;
    this.params.neuronsPerLayer = clampedNeurons;
    this.params.activationType = activation;

    return {
      layers: clampedLayers,
      neurons: clampedNeurons,
      canPrune: this.canPrune(),
    };
  }

  setActivation(activation: ActivationType): void {
    this.params.activationType = activation;
    for (const net of this.population) {
      net.activation = activation;
    }
    if (this.bestAgent) {
      this.bestAgent.activation = activation;
    }
    if (this.agent && this.agent.layerValues.length > 0) {
      this.agent.predict(this.agent.layerValues[0]);
    }
  }

  exportWeightsPayload() {
    const liveFitness = this.agent ? this.currentFitness : -Infinity;
    const savedBestFitness = this.bestAgent ? this.bestFitness : -Infinity;
    const source =
      this.bestAgent && savedBestFitness >= liveFitness
        ? this.bestAgent
        : this.agent ?? this.bestAgent;
    if (!source) return null;

    const hiddenLayerSizes = source.getHiddenLayerSizes();
    const effectiveBestFitness = Math.max(savedBestFitness, liveFitness);

    return {
      format: 'rl-evasion-baseline-weights-v2',
      exportedAt: new Date().toISOString(),
      generation: this.generation,
      fitness: effectiveBestFitness !== -Infinity ? effectiveBestFitness : 0,
      architecture: {
        inputSize: 104,
        hiddenLayers: hiddenLayerSizes.length,
        hiddenLayerSizes,
        outputSize: 2,
        activation: source.activation,
      },
      weights: JSON.parse(JSON.stringify(source.weights)) as number[][][],
      biases: JSON.parse(JSON.stringify(source.biases)) as number[][],
      frozenWeights: JSON.parse(JSON.stringify(source.frozenWeights)) as boolean[][][],
      frozenBiases: JSON.parse(JSON.stringify(source.frozenBiases)) as boolean[][],
      frozenHiddenNodes: source.getFrozenHiddenMask(),
      isFrozen: this.isFrozen,
      params: { ...this.params },
      simulationState: {
        popIndex: this.popIndex,
        stepCount: this.stepCount,
        currentFitness: this.currentFitness,
        bestFitness: this.bestFitness !== -Infinity ? this.bestFitness : null,
        prevVx: this.prevVx,
        prevVy: this.prevVy,
        emaVx: this.emaVx,
        emaVy: this.emaVy,
        emaPx: this.emaPx,
        emaPy: this.emaPy,
        collectedSplitGaps: this.collectedSplitGaps.map((g) => ({ ...g })),
        splitGapHistory: this.splitGapHistory.map((h) => ({ ...h })),
        player: { ...this.player },
        enemies: this.enemies.map((e) => ({
          x: e.x,
          y: e.y,
          spawnDelay: e.spawnDelay ?? 0,
          entered: e.entered ?? true,
          ...(typeof e.speedPct === 'number' ? { speedPct: e.speedPct } : {}),
        })),
        population: this.population.map((net) => ({
          activation: net.activation,
          fitness: Number.isFinite(net.fitness) ? net.fitness : null,
          weights: JSON.parse(JSON.stringify(net.weights)) as number[][][],
          biases: JSON.parse(JSON.stringify(net.biases)) as number[][],
          frozenWeights: JSON.parse(JSON.stringify(net.frozenWeights)) as boolean[][][],
          frozenBiases: JSON.parse(JSON.stringify(net.frozenBiases)) as boolean[][],
        })),
        bestAgent: this.bestAgent
          ? {
              activation: this.bestAgent.activation,
              fitness: Number.isFinite(this.bestAgent.fitness) ? this.bestAgent.fitness : null,
              weights: JSON.parse(JSON.stringify(this.bestAgent.weights)) as number[][][],
              biases: JSON.parse(JSON.stringify(this.bestAgent.biases)) as number[][],
              frozenWeights: JSON.parse(JSON.stringify(this.bestAgent.frozenWeights)) as boolean[][][],
              frozenBiases: JSON.parse(JSON.stringify(this.bestAgent.frozenBiases)) as boolean[][],
            }
          : null,
      },
    };
  }

  importWeightsPayload(data: unknown): {
    params: SimParams;
    generation: number;
    canPrune: boolean;
    isFrozen: boolean;
  } {
    const parsed = NeuralNetwork.fromJSON(data);
    const { network, generation, fitness, hiddenLayerSizes } = parsed;
    const record = data as Record<string, unknown>;

    if (record.params && typeof record.params === 'object') {
      const p = record.params as Partial<SimParams>;
      if (typeof p.numEnemies === 'number' && Number.isFinite(p.numEnemies)) {
        this.params.numEnemies = Math.min(50, Math.max(1, Math.round(p.numEnemies)));
      }
      if (typeof p.enemySpeedPct === 'number' && Number.isFinite(p.enemySpeedPct)) {
        this.updateEnemySpeedPct(p.enemySpeedPct);
      }
      if (p.enemySpeedMode === 'fixed' || p.enemySpeedMode === 'rarity') {
        this.params.enemySpeedMode = p.enemySpeedMode;
      }
      if (p.spawnMode === 'outer' || p.spawnMode === 'central') {
        this.params.spawnMode = p.spawnMode;
      }
      if (typeof p.simSpeed === 'number' && Number.isFinite(p.simSpeed)) {
        this.params.simSpeed = Math.min(100, Math.max(1, Math.round(p.simSpeed)));
      }
      if (typeof p.turboMode === 'boolean') {
        this.params.turboMode = p.turboMode;
      }
      if (typeof p.rewardEnemyDist === 'boolean') this.params.rewardEnemyDist = p.rewardEnemyDist;
      if (typeof p.rewardWallDist === 'boolean') this.params.rewardWallDist = p.rewardWallDist;
      if (typeof p.rewardGroupEnemies === 'boolean') {
        this.params.rewardGroupEnemies = p.rewardGroupEnemies;
      }
      if (typeof p.rewardSpreadEnemies === 'boolean') {
        this.params.rewardSpreadEnemies = p.rewardSpreadEnemies;
      }
      if (typeof p.rewardEdgeEnemies === 'boolean') {
        this.params.rewardEdgeEnemies = p.rewardEdgeEnemies;
      }
      if (typeof p.rewardOppHeading === 'boolean') this.params.rewardOppHeading = p.rewardOppHeading;
      if (typeof p.rewardSmoothVel === 'boolean') this.params.rewardSmoothVel = p.rewardSmoothVel;
      if (typeof p.rewardSplitGap === 'boolean') this.params.rewardSplitGap = p.rewardSplitGap;
      if (typeof p.rewardEdgeSense === 'boolean') this.params.rewardEdgeSense = p.rewardEdgeSense;
      if (typeof p.rewardOrbitKite === 'boolean') this.params.rewardOrbitKite = p.rewardOrbitKite;
      if (typeof p.rewardMatadorCut === 'boolean') this.params.rewardMatadorCut = p.rewardMatadorCut;
      if (typeof p.rewardAnkleBreaker === 'boolean') {
        this.params.rewardAnkleBreaker = p.rewardAnkleBreaker;
      }
      if (typeof p.rewardPincerEscape === 'boolean') {
        this.params.rewardPincerEscape = p.rewardPincerEscape;
      }
      if (typeof p.rewardOpenCorridor === 'boolean') {
        this.params.rewardOpenCorridor = p.rewardOpenCorridor;
      }
      if (typeof p.rewardWallSlingshot === 'boolean') {
        this.params.rewardWallSlingshot = p.rewardWallSlingshot;
      }
      if (typeof p.rewardPackTangle === 'boolean') this.params.rewardPackTangle = p.rewardPackTangle;
      if (typeof p.rewardSpideySense === 'boolean') this.params.rewardSpideySense = p.rewardSpideySense;
      if (typeof p.rewardFarWallSense === 'boolean') {
        this.params.rewardFarWallSense = p.rewardFarWallSense;
      }
      if (typeof p.rewardSplitGapCollect === 'boolean') {
        this.params.rewardSplitGapCollect = p.rewardSplitGapCollect;
      }
      if (typeof p.rewardSplitGapPredict === 'boolean') {
        this.params.rewardSplitGapPredict = p.rewardSplitGapPredict;
      }
      if (typeof p.rewardHit === 'boolean') {
        this.params.rewardHit = p.rewardHit;
      }
      if (typeof p.rewardAvgEnemyCenter === 'boolean') {
        this.params.rewardAvgEnemyCenter = p.rewardAvgEnemyCenter;
      }
      if (typeof p.rewardVectorStabbing === 'boolean') {
        this.params.rewardVectorStabbing = p.rewardVectorStabbing;
      }
      if (typeof p.rewardTwitch === 'boolean') {
        this.params.rewardTwitch = p.rewardTwitch;
      }

      if (typeof p.weightEnemyDist === 'number' && Number.isFinite(p.weightEnemyDist)) {
        this.params.weightEnemyDist = p.weightEnemyDist;
      }
      if (typeof p.weightWallDist === 'number' && Number.isFinite(p.weightWallDist)) {
        this.params.weightWallDist = p.weightWallDist;
      }
      if (typeof p.weightGroupEnemies === 'number' && Number.isFinite(p.weightGroupEnemies)) {
        this.params.weightGroupEnemies = p.weightGroupEnemies;
      }
      if (typeof p.weightSpreadEnemies === 'number' && Number.isFinite(p.weightSpreadEnemies)) {
        this.params.weightSpreadEnemies = p.weightSpreadEnemies;
      }
      if (typeof p.weightEdgeEnemies === 'number' && Number.isFinite(p.weightEdgeEnemies)) {
        this.params.weightEdgeEnemies = p.weightEdgeEnemies;
      }
      if (typeof p.weightOppHeading === 'number' && Number.isFinite(p.weightOppHeading)) {
        this.params.weightOppHeading = p.weightOppHeading;
      }
      if (typeof p.weightSmoothVel === 'number' && Number.isFinite(p.weightSmoothVel)) {
        this.params.weightSmoothVel = p.weightSmoothVel;
      }
      if (typeof p.weightSplitGap === 'number' && Number.isFinite(p.weightSplitGap)) {
        this.params.weightSplitGap = p.weightSplitGap;
      }
      if (typeof p.weightEdgeSense === 'number' && Number.isFinite(p.weightEdgeSense)) {
        this.params.weightEdgeSense = p.weightEdgeSense;
      }
      if (typeof p.weightOrbitKite === 'number' && Number.isFinite(p.weightOrbitKite)) {
        this.params.weightOrbitKite = p.weightOrbitKite;
      }
      if (typeof p.weightMatadorCut === 'number' && Number.isFinite(p.weightMatadorCut)) {
        this.params.weightMatadorCut = p.weightMatadorCut;
      }
      if (typeof p.weightAnkleBreaker === 'number' && Number.isFinite(p.weightAnkleBreaker)) {
        this.params.weightAnkleBreaker = p.weightAnkleBreaker;
      }
      if (typeof p.weightPincerEscape === 'number' && Number.isFinite(p.weightPincerEscape)) {
        this.params.weightPincerEscape = p.weightPincerEscape;
      }
      if (typeof p.weightOpenCorridor === 'number' && Number.isFinite(p.weightOpenCorridor)) {
        this.params.weightOpenCorridor = p.weightOpenCorridor;
      }
      if (typeof p.weightWallSlingshot === 'number' && Number.isFinite(p.weightWallSlingshot)) {
        this.params.weightWallSlingshot = p.weightWallSlingshot;
      }
      if (typeof p.weightPackTangle === 'number' && Number.isFinite(p.weightPackTangle)) {
        this.params.weightPackTangle = p.weightPackTangle;
      }
      if (typeof p.weightSpideySense === 'number' && Number.isFinite(p.weightSpideySense)) {
        this.params.weightSpideySense = p.weightSpideySense;
      }
      if (typeof p.weightFarWallSense === 'number' && Number.isFinite(p.weightFarWallSense)) {
        this.params.weightFarWallSense = p.weightFarWallSense;
      }
      if (typeof p.weightSplitGapCollect === 'number' && Number.isFinite(p.weightSplitGapCollect)) {
        this.params.weightSplitGapCollect = p.weightSplitGapCollect;
      }
      if (typeof p.weightSplitGapPredict === 'number' && Number.isFinite(p.weightSplitGapPredict)) {
        this.params.weightSplitGapPredict = p.weightSplitGapPredict;
      }
      if (typeof p.weightHit === 'number' && Number.isFinite(p.weightHit)) {
        this.params.weightHit = p.weightHit;
      }
      if (typeof p.weightAvgEnemyCenter === 'number' && Number.isFinite(p.weightAvgEnemyCenter)) {
        this.params.weightAvgEnemyCenter = p.weightAvgEnemyCenter;
      }
      if (typeof p.weightVectorStabbing === 'number' && Number.isFinite(p.weightVectorStabbing)) {
        this.params.weightVectorStabbing = p.weightVectorStabbing;
      }
      if (typeof p.weightTwitch === 'number' && Number.isFinite(p.weightTwitch)) {
        this.params.weightTwitch = p.weightTwitch;
      }
    }

    this.params.hiddenLayersCount = hiddenLayerSizes.length;
    if (
      hiddenLayerSizes.length === 1 ||
      !(record.params && typeof (record.params as Partial<SimParams>).neuronsPerLayer === 'number')
    ) {
      this.params.neuronsPerLayer = hiddenLayerSizes[0] ?? 16;
    } else {
      this.params.neuronsPerLayer = Math.min(
        64,
        Math.max(2, Math.round((record.params as Partial<SimParams>).neuronsPerLayer!))
      );
    }
    this.params.activationType = network.activation;

    let restoredExactState = false;
    if (record.simulationState && typeof record.simulationState === 'object') {
      const sim = record.simulationState as Record<string, unknown>;
      if (Array.isArray(sim.population) && sim.population.length > 0) {
        const restoredPop: NeuralNetwork[] = [];
        for (const item of sim.population) {
          if (item && typeof item === 'object') {
            const netItem = item as Record<string, unknown>;
            const parsedMember = NeuralNetwork.fromJSON({
              weights: netItem.weights,
              biases: netItem.biases,
              frozenWeights: netItem.frozenWeights,
              frozenBiases: netItem.frozenBiases,
              frozenHiddenNodes: netItem.frozenHiddenNodes ?? record.frozenHiddenNodes,
              isFrozen: record.isFrozen,
              fitness: netItem.fitness,
              architecture: { activation: netItem.activation ?? network.activation },
            });
            restoredPop.push(parsedMember.network);
          }
        }

        if (restoredPop.length > 0) {
          this.population = restoredPop;
          const rawPopIdx = typeof sim.popIndex === 'number' ? Math.floor(sim.popIndex) : 0;
          this.popIndex = Math.max(0, Math.min(this.population.length - 1, rawPopIdx));
          this.agent = this.population[this.popIndex];
          this.generation = generation;
          this.stepCount =
            typeof sim.stepCount === 'number' && Number.isFinite(sim.stepCount) && sim.stepCount >= 0
              ? Math.floor(sim.stepCount)
              : 0;
          this.currentFitness =
            typeof sim.currentFitness === 'number' && Number.isFinite(sim.currentFitness)
              ? sim.currentFitness
              : 0;
          this.bestFitness =
            'bestFitness' in sim
              ? typeof sim.bestFitness === 'number' && Number.isFinite(sim.bestFitness)
                ? sim.bestFitness
                : -Infinity
              : fitness;
          this.prevVx =
            typeof sim.prevVx === 'number' && Number.isFinite(sim.prevVx) ? sim.prevVx : 0;
          this.prevVy =
            typeof sim.prevVy === 'number' && Number.isFinite(sim.prevVy) ? sim.prevVy : 0;
          this.emaVx =
            typeof sim.emaVx === 'number' && Number.isFinite(sim.emaVx) ? sim.emaVx : 0;
          this.emaVy =
            typeof sim.emaVy === 'number' && Number.isFinite(sim.emaVy) ? sim.emaVy : 0;
          this.emaPx =
            typeof sim.emaPx === 'number' && Number.isFinite(sim.emaPx)
              ? sim.emaPx
              : SIM_WIDTH / 2;
          this.emaPy =
            typeof sim.emaPy === 'number' && Number.isFinite(sim.emaPy)
              ? sim.emaPy
              : SIM_HEIGHT / 2;
          if (Array.isArray(sim.collectedSplitGaps)) {
            this.collectedSplitGaps = sim.collectedSplitGaps
              .filter(
                (g): g is { pairKey: number; mx: number; my: number; step: number } =>
                  Boolean(
                    g &&
                      typeof g === 'object' &&
                      typeof (g as Record<string, unknown>).pairKey === 'number' &&
                      typeof (g as Record<string, unknown>).mx === 'number' &&
                      typeof (g as Record<string, unknown>).my === 'number' &&
                      typeof (g as Record<string, unknown>).step === 'number'
                  )
              )
              .slice(-64);
          } else {
            this.collectedSplitGaps = [];
          }
          if (Array.isArray(sim.splitGapHistory)) {
            this.splitGapHistory = sim.splitGapHistory
              .filter(
                (h): h is { angle: number; x: number; y: number } =>
                  Boolean(
                    h &&
                      typeof h === 'object' &&
                      typeof (h as Record<string, unknown>).angle === 'number' &&
                      typeof (h as Record<string, unknown>).x === 'number' &&
                      typeof (h as Record<string, unknown>).y === 'number'
                  )
              )
              .slice(-5);
          } else {
            this.splitGapHistory = [];
          }

          if (sim.bestAgent && typeof sim.bestAgent === 'object') {
            const b = sim.bestAgent as Record<string, unknown>;
            this.bestAgent = NeuralNetwork.fromJSON({
              weights: b.weights,
              biases: b.biases,
              frozenWeights: b.frozenWeights,
              frozenBiases: b.frozenBiases,
              frozenHiddenNodes: b.frozenHiddenNodes ?? record.frozenHiddenNodes,
              isFrozen: record.isFrozen,
              fitness: b.fitness,
              architecture: { activation: b.activation ?? network.activation },
            }).network;
          } else {
            this.bestAgent = network.clone();
          }

          if (sim.player && typeof sim.player === 'object') {
            const pl = sim.player as Record<string, unknown>;
            if (
              typeof pl.x === 'number' &&
              typeof pl.y === 'number' &&
              typeof pl.vx === 'number' &&
              typeof pl.vy === 'number'
            ) {
              this.player = { x: pl.x, y: pl.y, vx: pl.vx, vy: pl.vy };
            }
          }

          if (Array.isArray(sim.enemies)) {
            const restoredEnemies: Enemy[] = [];
            for (const e of sim.enemies) {
              if (
                e &&
                typeof e === 'object' &&
                typeof (e as Record<string, unknown>).x === 'number' &&
                typeof (e as Record<string, unknown>).y === 'number'
              ) {
                const rec = e as Record<string, unknown>;
                restoredEnemies.push({
                  x: rec.x as number,
                  y: rec.y as number,
                  spawnDelay:
                    typeof rec.spawnDelay === 'number' && Number.isFinite(rec.spawnDelay)
                      ? Math.max(0, Math.floor(rec.spawnDelay))
                      : 0,
                  entered: typeof rec.entered === 'boolean' ? rec.entered : true,
                  speedPct:
                    typeof rec.speedPct === 'number' && Number.isFinite(rec.speedPct)
                      ? this.params.enemySpeedMode === 'rarity'
                        ? Math.min(90, Math.max(1, rec.speedPct))
                        : Math.max(1, rec.speedPct)
                      : this.params.enemySpeedMode === 'rarity'
                      ? this.sampleRaritySpeedPct()
                      : this.params.enemySpeedPct,
                });
              }
            }
            if (restoredEnemies.length > 0) {
              this.enemies = restoredEnemies;
              this.params.numEnemies = restoredEnemies.length;
            }
          }

          restoredExactState = true;
        }
      }
    }

    if (!restoredExactState) {
      const nextPop: NeuralNetwork[] = [];
      for (let i = 0; i < this.popSize; i++) {
        const clone = network.clone();
        if (i >= 3) {
          clone.mutate(0.2);
        }
        nextPop.push(clone);
      }

      this.population = nextPop;
      this.popIndex = 0;
      this.generation = generation;
      this.bestFitness = fitness;
      this.bestAgent = network.clone();
      this.bestAgent.fitness = fitness;
      this.startEpisode();
    }

    this.isFrozen = Boolean(this.agent?.hasFrozenWeights());

    const statsText = `${this.generation} | ${Math.round(this.currentFitness)} | ${Math.round(
      this.bestFitness === -Infinity ? 0 : this.bestFitness
    )}`;
    this.onStatsChange?.(statsText);

    return {
      params: { ...this.params },
      generation: this.generation,
      canPrune: this.canPrune(),
      isFrozen: this.isFrozen,
    };
  }
}
