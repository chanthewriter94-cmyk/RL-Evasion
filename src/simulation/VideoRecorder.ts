import { EvasionEngine, NN_HEIGHT, NN_WIDTH, SIM_HEIGHT, SIM_WIDTH } from './EvasionEngine';
import {
  drawLandscape1D,
  drawLandscape2D,
  drawLandscape3D,
  LandscapeData,
  LandscapeMetric,
  LandscapeMode,
} from './LossLandscape';

export interface RecordingOptions {
  recordTraining: boolean;
  recordLossLandscape: boolean;
  recordNeuralNet: boolean;
  landscapeMode: LandscapeMode;
  landscapeMetric: LandscapeMetric;
  activeRewardLabels: string[];
  getLandscapeData?: () => LandscapeData | null;
}

const HUD_HEIGHT = 36;
const PAD = 10;
const LANDSCAPE_REC_WIDTH = 380;

function pickSupportedMimeType(): { mimeType: string; ext: 'mp4' | 'webm' } {
  if (typeof MediaRecorder === 'undefined') {
    return { mimeType: '', ext: 'webm' };
  }
  const candidates: { mime: string; ext: 'mp4' | 'webm' }[] = [
    { mime: 'video/mp4;codecs=avc1.42E01E', ext: 'mp4' },
    { mime: 'video/mp4', ext: 'mp4' },
    { mime: 'video/webm;codecs=vp9', ext: 'webm' },
    { mime: 'video/webm;codecs=vp8', ext: 'webm' },
    { mime: 'video/webm', ext: 'webm' },
  ];
  for (const c of candidates) {
    if (typeof MediaRecorder.isTypeSupported === 'function' && MediaRecorder.isTypeSupported(c.mime)) {
      return { mimeType: c.mime, ext: c.ext };
    }
  }
  return { mimeType: '', ext: 'webm' };
}

export class VideoRecorder {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private recorder: MediaRecorder | null = null;
  private stream: MediaStream | null = null;
  private chunks: BlobPart[] = [];
  private options: RecordingOptions = {
    recordTraining: true,
    recordLossLandscape: true,
    recordNeuralNet: true,
    landscapeMode: 'weight',
    landscapeMetric: 'loss',
    activeRewardLabels: [],
  };
  private trail: { x: number; y: number }[] = [];
  private lastStepCount = -1;
  private ext: 'mp4' | 'webm' = 'webm';
  private mimeType = 'video/webm';
  private isStopping = false;
  private startedAtMs = 0;
  private contentHeight = NN_HEIGHT;

  constructor() {
    this.canvas = document.createElement('canvas');
    const ctx = this.canvas.getContext('2d');
    if (!ctx) {
      throw new Error('Unable to initialize offscreen 2D context for recording.');
    }
    this.ctx = ctx;
  }

  get isRecording(): boolean {
    return this.recorder !== null && this.recorder.state === 'recording';
  }

  start(
    engine: EvasionEngine,
    options: RecordingOptions,
    onStopComplete: (fileName: string) => void,
    onError: (msg: string) => void
  ): boolean {
    if (!options.recordTraining && !options.recordLossLandscape && !options.recordNeuralNet) {
      onError('Select at least one view to record.');
      return false;
    }
    if (typeof MediaRecorder === 'undefined' || typeof this.canvas.captureStream !== 'function') {
      onError('Video recording is not supported in this browser.');
      return false;
    }

    this.options = {
      ...options,
      activeRewardLabels: [...options.activeRewardLabels],
    };
    this.trail = [];
    this.lastStepCount = -1;
    this.chunks = [];
    this.isStopping = false;
    this.startedAtMs = Date.now();

    let totalPanels = 0;
    let width = PAD;
    if (options.recordTraining) {
      width += SIM_WIDTH + PAD;
      totalPanels++;
    }
    if (options.recordLossLandscape) {
      width += LANDSCAPE_REC_WIDTH + PAD;
      totalPanels++;
    }
    if (options.recordNeuralNet) {
      width += NN_WIDTH + PAD;
      totalPanels++;
    }

    const contentHeight = options.recordNeuralNet
      ? NN_HEIGHT
      : options.recordTraining
      ? SIM_HEIGHT
      : 440;
    this.contentHeight = contentHeight;
    const height = HUD_HEIGHT + contentHeight + PAD;

    this.canvas.width = width;
    this.canvas.height = height;

    this.renderFrame(engine);

    const { mimeType, ext } = pickSupportedMimeType();
    this.ext = ext;
    this.mimeType = mimeType || 'video/webm';

    const bps = totalPanels >= 3 ? 4_500_000 : 3_500_000;

    try {
      this.stream = this.canvas.captureStream(30);
      this.recorder = mimeType
        ? new MediaRecorder(this.stream, { mimeType, videoBitsPerSecond: bps })
        : new MediaRecorder(this.stream, { videoBitsPerSecond: bps });
    } catch (err) {
      this.cleanupStream();
      onError(err instanceof Error ? err.message : 'Failed to start MediaRecorder.');
      return false;
    }

    const genAtStart = engine.generation;

    this.recorder.ondataavailable = (ev: BlobEvent) => {
      if (ev.data && ev.data.size > 0) {
        this.chunks.push(ev.data);
      }
    };

    this.recorder.onerror = () => {
      this.cleanupStream();
      this.recorder = null;
      this.isStopping = false;
      onError('Recording error occurred.');
    };

    this.recorder.onstop = () => {
      const recordedChunks = this.chunks;
      this.chunks = [];
      const outMime = this.mimeType;
      const outExt = this.ext;
      this.cleanupStream();
      this.recorder = null;
      this.isStopping = false;

      if (recordedChunks.length === 0) {
        onError('No video data captured.');
        return;
      }

      const blob = new Blob(recordedChunks, { type: outMime });
      const url = URL.createObjectURL(blob);
      const stamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
      const fileName = `rl-evasion-gen-${genAtStart}-${stamp}.${outExt}`;
      const link = document.createElement('a');
      link.href = url;
      link.download = fileName;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      setTimeout(() => URL.revokeObjectURL(url), 1500);
      onStopComplete(fileName);
    };

    this.recorder.start(250);
    return true;
  }

  renderFrame(engine: EvasionEngine): void {
    const ctx = this.ctx;
    const w = this.canvas.width;
    const h = this.canvas.height;
    const {
      recordTraining,
      recordLossLandscape,
      recordNeuralNet,
      landscapeMode,
      landscapeMetric,
      activeRewardLabels,
      getLandscapeData,
    } = this.options;
    const contentH = this.contentHeight;

    ctx.fillStyle = '#090909';
    ctx.fillRect(0, 0, w, h);

    // Enhanced HUD header strip
    ctx.fillStyle = '#121212';
    ctx.fillRect(0, 0, w, HUD_HEIGHT - 2);
    ctx.strokeStyle = '#2a2a2a';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, HUD_HEIGHT - 2);
    ctx.lineTo(w, HUD_HEIGHT - 2);
    ctx.stroke();

    // Live REC dot + telemetry
    ctx.fillStyle = '#f33';
    ctx.beginPath();
    ctx.arc(16, 17, 4, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = '#eee';
    ctx.font = 'bold 12px monospace';
    ctx.textBaseline = 'middle';
    const bestFit =
      engine.bestFitness === -Infinity ? 0 : Math.round(engine.bestFitness);
    const hudLeft = `GEN ${engine.generation}  |  FIT ${Math.round(
      engine.currentFitness
    )}  |  BEST ${bestFit}  |  STEP ${engine.stepCount}  |  E:${engine.params.numEnemies}`;
    ctx.fillText(hudLeft, 28, 17);

    if (activeRewardLabels.length > 0 && w >= 600) {
      const maxShown = w >= 1200 ? 6 : 4;
      const rewardSummary =
        activeRewardLabels.slice(0, maxShown).join(' ') +
        (activeRewardLabels.length > maxShown ? ` +${activeRewardLabels.length - maxShown}` : '');
      ctx.fillStyle = '#0cc';
      ctx.font = '11px monospace';
      ctx.textAlign = 'right';
      ctx.fillText(rewardSummary, w - 12, 17);
      ctx.textAlign = 'left';
    }

    let cursorX = PAD;

    // 1. Left: Main Training Arena Footage
    if (recordTraining) {
      const simX = cursorX;
      const simY = HUD_HEIGHT + Math.floor((contentH - SIM_HEIGHT) * 0.5);

      ctx.save();
      ctx.translate(simX, simY);
      engine.drawArena(ctx);

      // Subtle arena grid overlay for enhanced footage readability
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
      ctx.lineWidth = 1;
      for (let gx = 50; gx < SIM_WIDTH; gx += 50) {
        ctx.beginPath();
        ctx.moveTo(gx, 0);
        ctx.lineTo(gx, SIM_HEIGHT);
        ctx.stroke();
      }
      for (let gy = 50; gy < SIM_HEIGHT; gy += 50) {
        ctx.beginPath();
        ctx.moveTo(0, gy);
        ctx.lineTo(SIM_WIDTH, gy);
        ctx.stroke();
      }

      // Reset trail on episode rollover
      if (engine.stepCount < this.lastStepCount) {
        this.trail = [];
      }
      this.lastStepCount = engine.stepCount;
      const lastPt = this.trail[this.trail.length - 1];
      if (!lastPt || Math.hypot(engine.player.x - lastPt.x, engine.player.y - lastPt.y) >= 1) {
        this.trail.push({ x: engine.player.x, y: engine.player.y });
        if (this.trail.length > 28) {
          this.trail.shift();
        }
      }

      // Fading cyan motion trail
      for (let i = 1; i < this.trail.length; i++) {
        const alpha = (i / this.trail.length) * 0.55;
        ctx.strokeStyle = `rgba(0, 255, 255, ${alpha.toFixed(2)})`;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(this.trail[i - 1].x, this.trail[i - 1].y);
        ctx.lineTo(this.trail[i].x, this.trail[i].y);
        ctx.stroke();
      }

      // Steering velocity vector indicator from player
      const vMag = Math.hypot(engine.player.vx, engine.player.vy);
      if (vMag > 0.1) {
        const nx = engine.player.vx / vMag;
        const ny = engine.player.vy / vMag;
        const arrowLen = 14 + (vMag / engine.playerSpeed) * 14;
        const ax = engine.player.x + nx * arrowLen;
        const ay = engine.player.y + ny * arrowLen;
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.78)';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(engine.player.x, engine.player.y);
        ctx.lineTo(ax, ay);
        ctx.stroke();
      }

      ctx.strokeStyle = '#333';
      ctx.lineWidth = 1;
      ctx.strokeRect(0, 0, SIM_WIDTH, SIM_HEIGHT);
      ctx.restore();

      cursorX += SIM_WIDTH + PAD;
    }

    // 2. Center: Loss Landscape Bridge (3D Surface + 2D Contour + 1D Slice)
    if (recordLossLandscape) {
      const lsX = cursorX;
      const lsY = HUD_HEIGHT;
      const lsW = LANDSCAPE_REC_WIDTH;
      const lsH = contentH;

      ctx.save();
      ctx.translate(lsX, lsY);

      ctx.fillStyle = '#070707';
      ctx.fillRect(0, 0, lsW, lsH);

      const headerH = 22;
      ctx.fillStyle = '#121212';
      ctx.fillRect(0, 0, lsW, headerH);
      ctx.strokeStyle = '#242424';
      ctx.beginPath();
      ctx.moveTo(0, headerH);
      ctx.lineTo(lsW, headerH);
      ctx.stroke();

      const landscapeData = getLandscapeData ? getLandscapeData() : null;
      const modeTag = landscapeMode === 'weight' ? 'WEIGHT SPACE' : 'SPATIAL ARENA';
      const metricTag = landscapeMetric === 'loss' ? 'LOSS' : 'FITNESS';
      ctx.fillStyle = '#0dd';
      ctx.font = 'bold 10px monospace';
      ctx.textBaseline = 'middle';
      ctx.fillText(`LANDSCAPE • ${modeTag} (${metricTag})`, 8, headerH * 0.5);

      if (landscapeData) {
        const pct = Math.round(landscapeData.progress * 100);
        if (pct < 100) {
          ctx.fillStyle = '#fb0';
          ctx.font = '10px monospace';
          ctx.textAlign = 'right';
          ctx.fillText(`SCAN ${pct}%`, lsW - 8, headerH * 0.5);
          ctx.textAlign = 'left';
        }

        const innerPad = 6;
        const innerW = lsW - innerPad * 2;
        const availH = lsH - headerH - innerPad * 4;
        const h3d = Math.floor(availH * 0.44);
        const h2d = Math.floor(availH * 0.32);
        const h1d = availH - h3d - h2d;

        const y3d = headerH + innerPad;
        const y2d = y3d + h3d + innerPad;
        const y1d = y2d + h2d + innerPad;

        const elapsedSec = (Date.now() - this.startedAtMs) * 0.001;
        const autoYaw = 0.68 + elapsedSec * 0.25;

        drawLandscape3D(ctx, innerPad, y3d, innerW, h3d, landscapeData, landscapeMetric, autoYaw, 0.58);
        drawLandscape2D(ctx, innerPad, y2d, innerW, h2d, landscapeData, landscapeMetric);
        drawLandscape1D(ctx, innerPad, y1d, innerW, h1d, landscapeData, landscapeMetric);
      } else {
        ctx.fillStyle = '#777';
        ctx.font = '11px monospace';
        ctx.textBaseline = 'middle';
        ctx.fillText('Scanning loss landscape...', 14, lsH * 0.5);
      }

      ctx.strokeStyle = '#333';
      ctx.lineWidth = 1;
      ctx.strokeRect(0, 0, lsW, lsH);
      ctx.restore();

      cursorX += lsW + PAD;
    }

    // 3. Right: Neural Network Scene
    if (recordNeuralNet) {
      const nnX = cursorX;
      const nnY = HUD_HEIGHT + Math.floor((contentH - NN_HEIGHT) * 0.5);
      ctx.save();
      ctx.translate(nnX, nnY);
      engine.drawNN(ctx, null);
      ctx.strokeStyle = '#333';
      ctx.lineWidth = 1;
      ctx.strokeRect(0, 0, NN_WIDTH, NN_HEIGHT);
      ctx.restore();
    }
  }

  stop(): void {
    if (!this.recorder || this.isStopping) return;
    if (this.recorder.state === 'inactive') return;
    this.isStopping = true;
    this.recorder.stop();
  }

  private cleanupStream(): void {
    if (this.stream) {
      for (const track of this.stream.getTracks()) {
        track.stop();
      }
      this.stream = null;
    }
  }
}
