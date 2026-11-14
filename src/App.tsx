import { useCallback, useEffect, useRef, useState } from 'react';
import { ControlBar, RewardConfigItem } from './components/ControlBar';
import { LossLandscapePanel } from './components/LossLandscapePanel';
import { OfflineIndicator } from './components/OfflineIndicator';
import { WeightInspector } from './components/WeightInspector';
import {
  EnemySpeedMode,
  EvasionEngine,
  NN_HEIGHT,
  NN_WIDTH,
  SIM_HEIGHT,
  SIM_WIDTH,
  SimParams,
  SpawnMode,
} from './simulation/EvasionEngine';
import {
  LandscapeData,
  LandscapeMetric,
  LandscapeMode,
  LandscapeView,
  scanSpatialLandscape,
  WeightSpaceScanner,
} from './simulation/LossLandscape';
import { ActivationType } from './simulation/NeuralNetwork';
import { VideoRecorder } from './simulation/VideoRecorder';

const INITIAL_PARAMS: SimParams = {
  hiddenLayersCount: 1,
  neuronsPerLayer: 16,
  activationType: 'tanh',
  numEnemies: 3,
  enemySpeedPct: 80,
  enemySpeedMode: 'fixed',
  spawnMode: 'outer',
  simSpeed: 1,
  turboMode: false,
  rewardEnemyDist: true,
  rewardWallDist: false,
  rewardGroupEnemies: false,
  rewardSpreadEnemies: false,
  rewardEdgeEnemies: false,
  rewardOppHeading: false,
  rewardSmoothVel: false,
  rewardSplitGap: false,
  rewardEdgeSense: false,
  rewardOrbitKite: false,
  rewardMatadorCut: false,
  rewardAnkleBreaker: false,
  rewardPincerEscape: false,
  rewardOpenCorridor: false,
  rewardWallSlingshot: false,
  rewardPackTangle: false,
  rewardSpideySense: false,
  rewardFarWallSense: false,
  rewardSplitGapCollect: false,
  rewardSplitGapPredict: false,
  rewardHit: true,
  rewardAvgEnemyCenter: false,
  rewardVectorStabbing: false,
  rewardTwitch: false,
  weightEnemyDist: 1.0,
  weightWallDist: 1.5,
  weightGroupEnemies: 0.8,
  weightSpreadEnemies: 0.8,
  weightEdgeEnemies: 1.0,
  weightOppHeading: 1.0,
  weightSmoothVel: 1.0,
  weightSplitGap: 1.0,
  weightEdgeSense: 1.0,
  weightOrbitKite: 1.0,
  weightMatadorCut: 1.0,
  weightAnkleBreaker: 1.0,
  weightPincerEscape: 1.0,
  weightOpenCorridor: 1.0,
  weightWallSlingshot: 1.0,
  weightPackTangle: 1.0,
  weightSpideySense: 1.0,
  weightFarWallSense: 1.0,
  weightSplitGapCollect: 1.0,
  weightSplitGapPredict: 1.0,
  weightHit: -500,
  weightAvgEnemyCenter: 1.0,
  weightVectorStabbing: -1.0,
  weightTwitch: 1.0,
};

const INITIAL_REWARDS: RewardConfigItem[] = [
  {
    id: 'R21',
    label: 'Hit',
    category: 'spacing',
    active: true,
    isPositive: false,
    weightText: '500',
    title: 'Fitness change applied when an enemy collides with the player (default -500)',
  },
  {
    id: 'R1',
    label: '+EnemyDist',
    category: 'spacing',
    active: true,
    isPositive: true,
    weightText: '1',
    title: 'Distance from nearest enemy',
  },
  {
    id: 'R2',
    label: '+WallDist',
    category: 'spacing',
    active: false,
    isPositive: true,
    weightText: '1.5',
    title: 'Distance from nearest arena wall',
  },
  {
    id: 'R3',
    label: '+GroupEnemies',
    category: 'spacing',
    active: false,
    isPositive: true,
    weightText: '0.8',
    title: 'Weight for enemy grouping reward',
  },
  {
    id: 'R4',
    label: '+SpreadEnemies',
    category: 'spacing',
    active: false,
    isPositive: true,
    weightText: '0.8',
    title: 'Weight for enemy spreading reward',
  },
  {
    id: 'R5',
    label: '+EdgeEnemies',
    category: 'spacing',
    active: false,
    isPositive: true,
    weightText: '1.0',
    title: 'Weight for enemies near map edge reward',
  },
  {
    id: 'R22',
    label: '+AvgEnemyCenter',
    category: 'spacing',
    active: false,
    isPositive: true,
    weightText: '1.0',
    title: 'Reward when the average enemy position (centroid) is in the center of the arena',
  },
  {
    id: 'R6',
    label: '+OppHeading',
    category: 'flow',
    active: false,
    isPositive: true,
    weightText: '1.0',
    title: 'Weight for player heading away from average enemy vector',
  },
  {
    id: 'R7',
    label: '+SmoothVel',
    category: 'flow',
    active: false,
    isPositive: true,
    weightText: '1.0',
    title: 'Weight for smooth average velocity and penalizing jerky or noisy movement',
  },
  {
    id: 'R8',
    label: '+SplitGap',
    category: 'flow',
    active: false,
    isPositive: true,
    weightText: '1.0',
    title: 'Weight for velocity splitting the largest angular gap between nearby enemies',
  },
  {
    id: 'R19',
    label: '+SplitGapCollect',
    category: 'flow',
    active: false,
    isPositive: true,
    weightText: '1.0',
    title: 'Meta-reward for SplitGap: rewards collecting unique forward enemy gaps and penalizes re-crossing already-collected gaps',
  },
  {
    id: 'R20',
    label: '+SplitGapPredict',
    category: 'flow',
    active: false,
    isPositive: true,
    weightText: '1.0',
    title: 'Meta-reward for SplitGap: learns the shift from where the split gap was moments before to predict and steer toward where the gap is going to be',
  },
  {
    id: 'R14',
    label: '+OpenCorridor',
    category: 'flow',
    active: false,
    isPositive: true,
    weightText: '1.0',
    title: 'Forward cone clearance to both arena walls and enemy intercepts',
  },
  {
    id: 'R18',
    label: '+FarWallSense',
    category: 'flow',
    active: false,
    isPositive: true,
    weightText: '1.0',
    title: 'Prefer moving toward the opposite side of the arena from recent position',
  },
  {
    id: 'R9',
    label: '+EdgeSense',
    category: 'tactical',
    active: false,
    isPositive: true,
    weightText: '1.0',
    title: 'Favor velocities with a close enemy in only one peripheral while the opposite peripheral is clear',
  },
  {
    id: 'R10',
    label: '+OrbitKite',
    category: 'tactical',
    active: false,
    isPositive: true,
    weightText: '1.0',
    title: 'Orbit nearest enemy at 90 degrees inside a 60-110px sweet-spot band',
  },
  {
    id: 'R11',
    label: '+MatadorCut',
    category: 'tactical',
    active: false,
    isPositive: true,
    weightText: '1.0',
    title: 'Cut past a close enemy shoulder behind its pursuit velocity vector',
  },
  {
    id: 'R12',
    label: '+AnkleBreaker',
    category: 'tactical',
    active: false,
    isPositive: true,
    weightText: '1.0',
    title: 'Maximize angular bearing sweep rate of nearby enemies around the player',
  },
  {
    id: 'R13',
    label: '+PincerEscape',
    category: 'tactical',
    active: false,
    isPositive: true,
    weightText: '1.0',
    title: 'Break perpendicularly out of opposing two-enemy squeezes',
  },
  {
    id: 'R15',
    label: '+WallSlingshot',
    category: 'tactical',
    active: false,
    isPositive: true,
    weightText: '1.0',
    title: 'Run parallel along walls away from nearby threats when cornered near edges',
  },
  {
    id: 'R16',
    label: '+PackTangle',
    category: 'tactical',
    active: false,
    isPositive: true,
    weightText: '1.0',
    title: 'Lead trailing enemies into tightly crowded clusters behind the player heading',
  },
  {
    id: 'R17',
    label: '+SpideySense',
    category: 'tactical',
    active: false,
    isPositive: true,
    weightText: '1.0',
    title: 'Sum spatial danger gradients within 200px weighted by proximity and closing speed, penalizing velocity piercing uphill into danger',
  },
  {
    id: 'R23',
    label: 'VectorStabbing',
    category: 'tactical',
    active: false,
    isPositive: false,
    weightText: '1.0',
    title: 'Penalize velocity vectors that pierce into or drive straight toward the arena walls',
  },
  {
    id: 'R24',
    label: '+Twitch',
    category: 'tactical',
    active: false,
    isPositive: true,
    weightText: '1.0',
    title: 'Reward sharp perpendicular lateral movement when an enemy is right in front of the player',
  },
];

export default function App() {
  const [layers, setLayers] = useState<number>(INITIAL_PARAMS.hiddenLayersCount);
  const [neurons, setNeurons] = useState<number>(INITIAL_PARAMS.neuronsPerLayer);
  const [activation, setActivation] = useState<ActivationType>(INITIAL_PARAMS.activationType);
  const [enemyCount, setEnemyCount] = useState<number>(INITIAL_PARAMS.numEnemies);
  const [enemySpeedPct, setEnemySpeedPct] = useState<number>(INITIAL_PARAMS.enemySpeedPct);
  const [enemySpeedMode, setEnemySpeedMode] = useState<EnemySpeedMode>(
    INITIAL_PARAMS.enemySpeedMode
  );
  const [spawnMode, setSpawnMode] = useState<SpawnMode>(INITIAL_PARAMS.spawnMode);
  const [simSpeed, setSimSpeed] = useState<number>(INITIAL_PARAMS.simSpeed);
  const [turboMode, setTurboMode] = useState<boolean>(INITIAL_PARAMS.turboMode);
  const [rewards, setRewards] = useState<RewardConfigItem[]>(INITIAL_REWARDS);
  const [statsText, setStatsText] = useState<string>('0 | 0 | 0');
  const [canPrune, setCanPrune] = useState<boolean>(true);
  const [isFrozen, setIsFrozen] = useState<boolean>(false);
  const [nnCursor, setNnCursor] = useState<'default' | 'pointer' | 'crosshair'>('default');
  const [bottomTab, setBottomTab] = useState<'nn' | 'weights' | 'landscape' | null>('nn');
  const [inspectorTick, setInspectorTick] = useState<number>(0);
  const [showRecorderSuite, setShowRecorderSuite] = useState<boolean>(false);
  const [recNeuralNet, setRecNeuralNet] = useState<boolean>(true);
  const [recLossLandscape, setRecLossLandscape] = useState<boolean>(true);
  const [recTraining, setRecTraining] = useState<boolean>(true);
  const [isRecording, setIsRecording] = useState<boolean>(false);
  const [recElapsedSec, setRecElapsedSec] = useState<number>(0);
  const [recStatus, setRecStatus] = useState<{ text: string; isError: boolean } | null>(null);
  const [landscapeMode, setLandscapeMode] = useState<LandscapeMode>('weight');
  const [landscapeView, setLandscapeView] = useState<LandscapeView>('2d+3d');
  const [landscapeMetric, setLandscapeMetric] = useState<LandscapeMetric>('loss');
  const [landscapeData, setLandscapeData] = useState<LandscapeData | null>(null);

  const rewardsRef = useRef<RewardConfigItem[]>(INITIAL_REWARDS);
  const simCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const nnCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const engineRef = useRef<EvasionEngine | null>(null);
  const recorderRef = useRef<VideoRecorder | null>(null);
  const dragBoxRef = useRef<{ x1: number; y1: number; x2: number; y2: number } | null>(null);
  const landscapeDataRef = useRef<LandscapeData | null>(null);
  const weightScannerRef = useRef<WeightSpaceScanner | null>(null);
  const lastLandscapeGenRef = useRef<number>(-1);
  const lastScanStartedAtRef = useRef<number>(0);

  if (!engineRef.current) {
    engineRef.current = new EvasionEngine(INITIAL_PARAMS);
  }

  useEffect(() => {
    const engine = engineRef.current!;
    engine.onStatsChange = setStatsText;
    engine.resetSimulation();

    let animId = 0;
    const loop = () => {
      engine.stepFrame();

      const simCtx = simCanvasRef.current?.getContext('2d');
      if (simCtx) {
        engine.drawArena(simCtx);
      }

      const nnCtx = nnCanvasRef.current?.getContext('2d');
      if (nnCtx) {
        engine.drawNN(nnCtx, dragBoxRef.current);
      }

      if (recorderRef.current?.isRecording) {
        recorderRef.current.renderFrame(engine);
      }

      animId = requestAnimationFrame(loop);
    };

    animId = requestAnimationFrame(loop);

    const preventFileNav = (e: DragEvent) => {
      e.preventDefault();
    };
    window.addEventListener('dragover', preventFileNav);
    window.addEventListener('drop', preventFileNav);

    return () => {
      cancelAnimationFrame(animId);
      recorderRef.current?.stop();
      window.removeEventListener('dragover', preventFileNav);
      window.removeEventListener('drop', preventFileNav);
    };
  }, []);

  useEffect(() => {
    if (!isRecording) {
      setRecElapsedSec(0);
      return;
    }
    const startedAt = Date.now();
    const id = window.setInterval(() => {
      setRecElapsedSec(Math.floor((Date.now() - startedAt) / 1000));
    }, 500);
    return () => window.clearInterval(id);
  }, [isRecording]);

  useEffect(() => {
    if (bottomTab !== 'weights') return;
    setInspectorTick((t) => t + 1);
    const id = window.setInterval(() => {
      setInspectorTick((t) => t + 1);
    }, 150);
    return () => window.clearInterval(id);
  }, [bottomTab]);

  const startFreshLandscapeScan = useCallback(
    (modeOverride?: LandscapeMode) => {
      const engine = engineRef.current;
      if (!engine) return;
      const activeMode = modeOverride ?? landscapeMode;
      lastLandscapeGenRef.current = engine.generation;
      lastScanStartedAtRef.current = Date.now();

      if (activeMode === 'spatial') {
        weightScannerRef.current = null;
        const snap = scanSpatialLandscape(engine, 25, 17);
        landscapeDataRef.current = snap;
        setLandscapeData(snap);
      } else {
        const scanner = new WeightSpaceScanner(engine, 13, 110, 1.2);
        const initial = scanner.stepRows(2);
        weightScannerRef.current = scanner;
        const copy = { ...initial };
        landscapeDataRef.current = copy;
        setLandscapeData(copy);
      }
    },
    [landscapeMode]
  );

  const needsLandscapeUpdates =
    bottomTab === 'landscape' || (isRecording && recLossLandscape);

  useEffect(() => {
    if (!needsLandscapeUpdates) return;

    startFreshLandscapeScan(landscapeMode);

    if (landscapeMode === 'spatial') {
      const id = window.setInterval(() => {
        if (!engineRef.current) return;
        const snap = scanSpatialLandscape(engineRef.current, 25, 17);
        landscapeDataRef.current = snap;
        setLandscapeData(snap);
      }, 120);
      return () => window.clearInterval(id);
    }

    const id = window.setInterval(() => {
      const engine = engineRef.current;
      if (!engine) return;
      const scanner = weightScannerRef.current;

      if (!scanner) {
        startFreshLandscapeScan('weight');
        return;
      }

      if (!scanner.isComplete) {
        const next = scanner.stepRows(1);
        const copy = { ...next };
        landscapeDataRef.current = copy;
        setLandscapeData(copy);
      } else if (
        engine.generation !== lastLandscapeGenRef.current &&
        Date.now() - lastScanStartedAtRef.current >= 2500
      ) {
        startFreshLandscapeScan('weight');
      }
    }, 65);

    return () => window.clearInterval(id);
  }, [needsLandscapeUpdates, landscapeMode, startFreshLandscapeScan]);

  // Refresh landscape immediately when active rewards or architecture change while visible/recording
  useEffect(() => {
    if (!needsLandscapeUpdates) return;
    startFreshLandscapeScan(landscapeMode);
  }, [
    rewards,
    layers,
    neurons,
    activation,
    enemyCount,
    enemySpeedPct,
    enemySpeedMode,
    spawnMode,
    needsLandscapeUpdates,
    landscapeMode,
    startFreshLandscapeScan,
  ]);

  const handleReset = useCallback(() => {
    if (!engineRef.current) return;
    engineRef.current.resetSimulation();
    setIsFrozen(false);
    setCanPrune(engineRef.current.canPrune());
    setInspectorTick((t) => t + 1);
    weightScannerRef.current = null;
    if (bottomTab === 'landscape' || (isRecording && recLossLandscape)) {
      startFreshLandscapeScan();
    }
  }, [bottomTab, isRecording, recLossLandscape, startFreshLandscapeScan]);

  const handleToggleFreeze = useCallback(() => {
    if (!engineRef.current) return;
    const next = !engineRef.current.isFrozen;
    const res = engineRef.current.setFrozen(next);
    setIsFrozen(res.isFrozen);
    setCanPrune(res.canPrune);
    setInspectorTick((t) => t + 1);
  }, []);

  const handleToggleInspectorNodeFreeze = useCallback((hiddenLayer: number, nodeIndex: number) => {
    if (!engineRef.current) return;
    const res = engineRef.current.toggleHiddenNodeFreeze(hiddenLayer, nodeIndex);
    setIsFrozen(res.isFrozen);
    setCanPrune(res.canPrune);
    setInspectorTick((t) => t + 1);
  }, []);

  const handleStartRecording = useCallback(() => {
    if (!engineRef.current || isRecording) return;
    if (!recorderRef.current) {
      recorderRef.current = new VideoRecorder();
    }
    if (recLossLandscape && !landscapeDataRef.current) {
      startFreshLandscapeScan();
    }
    const activeLabels = rewardsRef.current
      .filter((r) => r.active)
      .map((r) => `${r.isPositive ? '+' : '-'}${r.label.replace(/^[+-]/, '')}`);
    setRecStatus(null);
    const ok = recorderRef.current.start(
      engineRef.current,
      {
        recordTraining: recTraining,
        recordLossLandscape: recLossLandscape,
        recordNeuralNet: recNeuralNet,
        landscapeMode,
        landscapeMetric,
        activeRewardLabels: activeLabels,
        getLandscapeData: () => landscapeDataRef.current,
      },
      (fileName) => {
        setIsRecording(false);
        setRecStatus({ text: `Saved ${fileName}`, isError: false });
      },
      (errMsg) => {
        setIsRecording(false);
        setRecStatus({ text: errMsg, isError: true });
      }
    );
    if (ok) {
      setIsRecording(true);
    }
  }, [
    isRecording,
    landscapeMetric,
    landscapeMode,
    recLossLandscape,
    recNeuralNet,
    recTraining,
    startFreshLandscapeScan,
  ]);

  const handleStopRecording = useCallback(() => {
    recorderRef.current?.stop();
  }, []);

  const getNNCanvasPoint = useCallback((clientX: number, clientY: number) => {
    const canvas = nnCanvasRef.current;
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return null;
    return {
      x: ((clientX - rect.left) / rect.width) * NN_WIDTH,
      y: ((clientY - rect.top) / rect.height) * NN_HEIGHT,
    };
  }, []);

  const finishNNDrag = useCallback(
    (clientX: number, clientY: number) => {
      const box = dragBoxRef.current;
      dragBoxRef.current = null;
      if (!box || !engineRef.current) {
        setNnCursor('default');
        return;
      }
      const pt = getNNCanvasPoint(clientX, clientY);
      const endX = pt ? pt.x : box.x2;
      const endY = pt ? pt.y : box.y2;
      const dragDist = Math.hypot(endX - box.x1, endY - box.y1);

      if (dragDist < 5) {
        const hit = engineRef.current.findHiddenNodeAt(endX, endY);
        if (hit) {
          const res = engineRef.current.toggleHiddenNodeFreeze(hit.hiddenLayer, hit.nodeIndex);
          setIsFrozen(res.isFrozen);
          setCanPrune(res.canPrune);
        }
      } else {
        const res = engineRef.current.toggleHiddenNodesInRect(box.x1, box.y1, endX, endY);
        if (res.changed) {
          setIsFrozen(res.isFrozen);
          setCanPrune(res.canPrune);
        }
      }
      setNnCursor(pt && engineRef.current.findHiddenNodeAt(pt.x, pt.y) ? 'pointer' : 'default');
    },
    [getNNCanvasPoint]
  );

  useEffect(() => {
    const handleWindowPointerUp = (e: PointerEvent) => {
      if (dragBoxRef.current) {
        finishNNDrag(e.clientX, e.clientY);
      }
    };
    window.addEventListener('pointerup', handleWindowPointerUp);
    window.addEventListener('pointercancel', handleWindowPointerUp);
    return () => {
      window.removeEventListener('pointerup', handleWindowPointerUp);
      window.removeEventListener('pointercancel', handleWindowPointerUp);
    };
  }, [finishNNDrag]);

  const handleLayersChange = useCallback((val: number) => {
    const clamped = Math.min(5, Math.max(1, val));
    if (!engineRef.current) return;
    const resized = engineRef.current.resizeNetwork(
      clamped,
      engineRef.current.params.neuronsPerLayer,
      engineRef.current.params.activationType
    );
    setLayers(resized.layers);
    setNeurons(resized.neurons);
    setCanPrune(resized.canPrune);
    setInspectorTick((t) => t + 1);
  }, []);

  const handleNeuronsChange = useCallback((val: number) => {
    const clamped = Math.min(64, Math.max(2, val));
    if (!engineRef.current) return;
    const resized = engineRef.current.resizeNetwork(
      engineRef.current.params.hiddenLayersCount,
      clamped,
      engineRef.current.params.activationType
    );
    setLayers(resized.layers);
    setNeurons(resized.neurons);
    setCanPrune(resized.canPrune);
    setInspectorTick((t) => t + 1);
  }, []);

  const handlePruneNetwork = useCallback(() => {
    if (!engineRef.current) return;
    const result = engineRef.current.pruneNetwork('least');
    if (!result) {
      setCanPrune(false);
      return;
    }
    if (result.hiddenLayerSizes.length === 1) {
      setNeurons(result.hiddenLayerSizes[0]);
    }
    setCanPrune(result.canPruneMore);
    setInspectorTick((t) => t + 1);
  }, []);

  const handleReversePrune = useCallback(() => {
    if (!engineRef.current) return;
    const result = engineRef.current.pruneNetwork('strongest');
    if (!result) {
      setCanPrune(false);
      return;
    }
    if (result.hiddenLayerSizes.length === 1) {
      setNeurons(result.hiddenLayerSizes[0]);
    }
    setCanPrune(result.canPruneMore);
    setInspectorTick((t) => t + 1);
  }, []);

  const handleDownloadWeights = useCallback(() => {
    const payload = engineRef.current?.exportWeightsPayload();
    if (!payload) return;
    const json = JSON.stringify(payload, null, 2);
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `rl-evasion-weights-gen-${payload.generation}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }, []);

  const handleGetWeightsJson = useCallback(() => {
    const payload = engineRef.current?.exportWeightsPayload();
    if (!payload) return '';
    return JSON.stringify(payload);
  }, []);

  const handleUploadWeights = useCallback((data: unknown) => {
    if (!engineRef.current) {
      throw new Error('Simulation engine not ready.');
    }
    const result = engineRef.current.importWeightsPayload(data);
    const p = result.params;
    setLayers(p.hiddenLayersCount);
    setNeurons(p.neuronsPerLayer);
    setActivation(p.activationType);
    setEnemyCount(p.numEnemies);
    setEnemySpeedPct(p.enemySpeedPct);
    setEnemySpeedMode(p.enemySpeedMode);
    setSpawnMode(p.spawnMode);
    setSimSpeed(p.simSpeed);
    setTurboMode(p.turboMode);
    setCanPrune(result.canPrune);
    setIsFrozen(result.isFrozen);

    const restoredRewards: RewardConfigItem[] = [
      {
        id: 'R21',
        label: 'Hit',
        category: 'spacing',
        active: typeof p.rewardHit === 'boolean' ? p.rewardHit : true,
        isPositive: (p.weightHit ?? -500) >= 0,
        weightText: String(Math.abs(p.weightHit ?? -500)),
        title: 'Fitness change applied when an enemy collides with the player (default -500)',
      },
      {
        id: 'R1',
        label: '+EnemyDist',
        category: 'spacing',
        active: p.rewardEnemyDist,
        isPositive: p.weightEnemyDist >= 0,
        weightText: String(Math.abs(p.weightEnemyDist)),
        title: 'Distance from nearest enemy',
      },
      {
        id: 'R2',
        label: '+WallDist',
        category: 'spacing',
        active: p.rewardWallDist,
        isPositive: p.weightWallDist >= 0,
        weightText: String(Math.abs(p.weightWallDist)),
        title: 'Distance from nearest arena wall',
      },
      {
        id: 'R3',
        label: '+GroupEnemies',
        category: 'spacing',
        active: p.rewardGroupEnemies,
        isPositive: p.weightGroupEnemies >= 0,
        weightText: String(Math.abs(p.weightGroupEnemies)),
        title: 'Weight for enemy grouping reward',
      },
      {
        id: 'R4',
        label: '+SpreadEnemies',
        category: 'spacing',
        active: p.rewardSpreadEnemies,
        isPositive: p.weightSpreadEnemies >= 0,
        weightText: String(Math.abs(p.weightSpreadEnemies)),
        title: 'Weight for enemy spreading reward',
      },
      {
        id: 'R5',
        label: '+EdgeEnemies',
        category: 'spacing',
        active: p.rewardEdgeEnemies,
        isPositive: p.weightEdgeEnemies >= 0,
        weightText: String(Math.abs(p.weightEdgeEnemies)),
        title: 'Weight for enemies near map edge reward',
      },
      {
        id: 'R22',
        label: '+AvgEnemyCenter',
        category: 'spacing',
        active: Boolean(p.rewardAvgEnemyCenter),
        isPositive: (p.weightAvgEnemyCenter ?? 1.0) >= 0,
        weightText: String(Math.abs(p.weightAvgEnemyCenter ?? 1.0)),
        title: 'Reward when the average enemy position (centroid) is in the center of the arena',
      },
      {
        id: 'R6',
        label: '+OppHeading',
        category: 'flow',
        active: p.rewardOppHeading,
        isPositive: p.weightOppHeading >= 0,
        weightText: String(Math.abs(p.weightOppHeading)),
        title: 'Weight for player heading away from average enemy vector',
      },
      {
        id: 'R7',
        label: '+SmoothVel',
        category: 'flow',
        active: p.rewardSmoothVel,
        isPositive: p.weightSmoothVel >= 0,
        weightText: String(Math.abs(p.weightSmoothVel)),
        title: 'Weight for smooth average velocity and penalizing jerky or noisy movement',
      },
      {
        id: 'R8',
        label: '+SplitGap',
        category: 'flow',
        active: p.rewardSplitGap,
        isPositive: p.weightSplitGap >= 0,
        weightText: String(Math.abs(p.weightSplitGap)),
        title: 'Weight for velocity splitting the largest angular gap between nearby enemies',
      },
      {
        id: 'R19',
        label: '+SplitGapCollect',
        category: 'flow',
        active: p.rewardSplitGapCollect,
        isPositive: p.weightSplitGapCollect >= 0,
        weightText: String(Math.abs(p.weightSplitGapCollect)),
        title: 'Meta-reward for SplitGap: rewards collecting unique forward enemy gaps and penalizes re-crossing already-collected gaps',
      },
      {
        id: 'R20',
        label: '+SplitGapPredict',
        category: 'flow',
        active: Boolean(p.rewardSplitGapPredict),
        isPositive: (p.weightSplitGapPredict ?? 1.0) >= 0,
        weightText: String(Math.abs(p.weightSplitGapPredict ?? 1.0)),
        title: 'Meta-reward for SplitGap: learns the shift from where the split gap was moments before to predict and steer toward where the gap is going to be',
      },
      {
        id: 'R14',
        label: '+OpenCorridor',
        category: 'flow',
        active: p.rewardOpenCorridor,
        isPositive: p.weightOpenCorridor >= 0,
        weightText: String(Math.abs(p.weightOpenCorridor)),
        title: 'Forward cone clearance to both arena walls and enemy intercepts',
      },
      {
        id: 'R18',
        label: '+FarWallSense',
        category: 'flow',
        active: p.rewardFarWallSense,
        isPositive: p.weightFarWallSense >= 0,
        weightText: String(Math.abs(p.weightFarWallSense)),
        title: 'Prefer moving toward the opposite side of the arena from recent position',
      },
      {
        id: 'R9',
        label: '+EdgeSense',
        category: 'tactical',
        active: p.rewardEdgeSense,
        isPositive: p.weightEdgeSense >= 0,
        weightText: String(Math.abs(p.weightEdgeSense)),
        title: 'Favor velocities with a close enemy in only one peripheral while the opposite peripheral is clear',
      },
      {
        id: 'R10',
        label: '+OrbitKite',
        category: 'tactical',
        active: p.rewardOrbitKite,
        isPositive: p.weightOrbitKite >= 0,
        weightText: String(Math.abs(p.weightOrbitKite)),
        title: 'Orbit nearest enemy at 90 degrees inside a 60-110px sweet-spot band',
      },
      {
        id: 'R11',
        label: '+MatadorCut',
        category: 'tactical',
        active: p.rewardMatadorCut,
        isPositive: p.weightMatadorCut >= 0,
        weightText: String(Math.abs(p.weightMatadorCut)),
        title: 'Cut past a close enemy shoulder behind its pursuit velocity vector',
      },
      {
        id: 'R12',
        label: '+AnkleBreaker',
        category: 'tactical',
        active: p.rewardAnkleBreaker,
        isPositive: p.weightAnkleBreaker >= 0,
        weightText: String(Math.abs(p.weightAnkleBreaker)),
        title: 'Maximize angular bearing sweep rate of nearby enemies around the player',
      },
      {
        id: 'R13',
        label: '+PincerEscape',
        category: 'tactical',
        active: p.rewardPincerEscape,
        isPositive: p.weightPincerEscape >= 0,
        weightText: String(Math.abs(p.weightPincerEscape)),
        title: 'Break perpendicularly out of opposing two-enemy squeezes',
      },
      {
        id: 'R15',
        label: '+WallSlingshot',
        category: 'tactical',
        active: p.rewardWallSlingshot,
        isPositive: p.weightWallSlingshot >= 0,
        weightText: String(Math.abs(p.weightWallSlingshot)),
        title: 'Run parallel along walls away from nearby threats when cornered near edges',
      },
      {
        id: 'R16',
        label: '+PackTangle',
        category: 'tactical',
        active: p.rewardPackTangle,
        isPositive: p.weightPackTangle >= 0,
        weightText: String(Math.abs(p.weightPackTangle)),
        title: 'Lead trailing enemies into tightly crowded clusters behind the player heading',
      },
      {
        id: 'R17',
        label: '+SpideySense',
        category: 'tactical',
        active: p.rewardSpideySense,
        isPositive: p.weightSpideySense >= 0,
        weightText: String(Math.abs(p.weightSpideySense)),
        title: 'Sum spatial danger gradients within 200px weighted by proximity and closing speed, penalizing velocity piercing uphill into danger',
      },
      {
        id: 'R23',
        label: 'VectorStabbing',
        category: 'tactical',
        active: Boolean(p.rewardVectorStabbing),
        isPositive: (p.weightVectorStabbing ?? -1.0) >= 0,
        weightText: String(Math.abs(p.weightVectorStabbing ?? -1.0)),
        title: 'Penalize velocity vectors that pierce into or drive straight toward the arena walls',
      },
      {
        id: 'R24',
        label: '+Twitch',
        category: 'tactical',
        active: Boolean(p.rewardTwitch),
        isPositive: (p.weightTwitch ?? 1.0) >= 0,
        weightText: String(Math.abs(p.weightTwitch ?? 1.0)),
        title: 'Reward sharp perpendicular lateral movement when an enemy is right in front of the player',
      },
    ];
    rewardsRef.current = restoredRewards;
    setRewards(restoredRewards);
    return { generation: result.generation };
  }, []);

  const handleActivationChange = useCallback((val: ActivationType) => {
    setActivation(val);
    engineRef.current?.setActivation(val);
  }, []);

  const handleEnemyCountChange = useCallback((val: number) => {
    const clamped = Math.min(50, Math.max(1, val));
    setEnemyCount(clamped);
    engineRef.current?.updateEnemyCount(clamped);
  }, []);

  const handleEnemySpeedPctChange = useCallback((val: number) => {
    const clamped = Math.max(1, val);
    setEnemySpeedPct(clamped);
    engineRef.current?.updateEnemySpeedPct(clamped);
  }, []);

  const handleEnemySpeedModeChange = useCallback((mode: EnemySpeedMode) => {
    setEnemySpeedMode(mode);
    engineRef.current?.setEnemySpeedMode(mode);
  }, []);

  const handleSpawnModeChange = useCallback((mode: SpawnMode) => {
    setSpawnMode(mode);
    engineRef.current?.setSpawnMode(mode);
  }, []);

  const handleSimSpeedChange = useCallback((val: number) => {
    setSimSpeed(val);
    setTurboMode(false);
    if (engineRef.current) {
      engineRef.current.params.simSpeed = val;
      engineRef.current.params.turboMode = false;
    }
  }, []);

  const handleToggleTurbo = useCallback(() => {
    if (!engineRef.current) return;
    const next = !engineRef.current.params.turboMode;
    engineRef.current.params.turboMode = next;
    setTurboMode(next);
  }, []);

  const syncRewardFlag = (id: RewardConfigItem['id'], active: boolean) => {
    if (!engineRef.current) return;
    switch (id) {
      case 'R1':
        engineRef.current.params.rewardEnemyDist = active;
        break;
      case 'R2':
        engineRef.current.params.rewardWallDist = active;
        break;
      case 'R3':
        engineRef.current.params.rewardGroupEnemies = active;
        break;
      case 'R4':
        engineRef.current.params.rewardSpreadEnemies = active;
        break;
      case 'R5':
        engineRef.current.params.rewardEdgeEnemies = active;
        break;
      case 'R6':
        engineRef.current.params.rewardOppHeading = active;
        break;
      case 'R7':
        engineRef.current.params.rewardSmoothVel = active;
        break;
      case 'R8':
        engineRef.current.params.rewardSplitGap = active;
        break;
      case 'R9':
        engineRef.current.params.rewardEdgeSense = active;
        break;
      case 'R10':
        engineRef.current.params.rewardOrbitKite = active;
        break;
      case 'R11':
        engineRef.current.params.rewardMatadorCut = active;
        break;
      case 'R12':
        engineRef.current.params.rewardAnkleBreaker = active;
        break;
      case 'R13':
        engineRef.current.params.rewardPincerEscape = active;
        break;
      case 'R14':
        engineRef.current.params.rewardOpenCorridor = active;
        break;
      case 'R15':
        engineRef.current.params.rewardWallSlingshot = active;
        break;
      case 'R16':
        engineRef.current.params.rewardPackTangle = active;
        break;
      case 'R17':
        engineRef.current.params.rewardSpideySense = active;
        break;
      case 'R18':
        engineRef.current.params.rewardFarWallSense = active;
        break;
      case 'R19':
        engineRef.current.params.rewardSplitGapCollect = active;
        break;
      case 'R20':
        engineRef.current.params.rewardSplitGapPredict = active;
        break;
      case 'R21':
        engineRef.current.params.rewardHit = active;
        break;
      case 'R22':
        engineRef.current.params.rewardAvgEnemyCenter = active;
        break;
      case 'R23':
        engineRef.current.params.rewardVectorStabbing = active;
        break;
      case 'R24':
        engineRef.current.params.rewardTwitch = active;
        break;
    }
  };

  const syncRewardWeight = (id: RewardConfigItem['id'], signedWeight: number) => {
    if (!engineRef.current) return;
    switch (id) {
      case 'R1':
        engineRef.current.params.weightEnemyDist = signedWeight;
        break;
      case 'R2':
        engineRef.current.params.weightWallDist = signedWeight;
        break;
      case 'R3':
        engineRef.current.params.weightGroupEnemies = signedWeight;
        break;
      case 'R4':
        engineRef.current.params.weightSpreadEnemies = signedWeight;
        break;
      case 'R5':
        engineRef.current.params.weightEdgeEnemies = signedWeight;
        break;
      case 'R6':
        engineRef.current.params.weightOppHeading = signedWeight;
        break;
      case 'R7':
        engineRef.current.params.weightSmoothVel = signedWeight;
        break;
      case 'R8':
        engineRef.current.params.weightSplitGap = signedWeight;
        break;
      case 'R9':
        engineRef.current.params.weightEdgeSense = signedWeight;
        break;
      case 'R10':
        engineRef.current.params.weightOrbitKite = signedWeight;
        break;
      case 'R11':
        engineRef.current.params.weightMatadorCut = signedWeight;
        break;
      case 'R12':
        engineRef.current.params.weightAnkleBreaker = signedWeight;
        break;
      case 'R13':
        engineRef.current.params.weightPincerEscape = signedWeight;
        break;
      case 'R14':
        engineRef.current.params.weightOpenCorridor = signedWeight;
        break;
      case 'R15':
        engineRef.current.params.weightWallSlingshot = signedWeight;
        break;
      case 'R16':
        engineRef.current.params.weightPackTangle = signedWeight;
        break;
      case 'R17':
        engineRef.current.params.weightSpideySense = signedWeight;
        break;
      case 'R18':
        engineRef.current.params.weightFarWallSense = signedWeight;
        break;
      case 'R19':
        engineRef.current.params.weightSplitGapCollect = signedWeight;
        break;
      case 'R20':
        engineRef.current.params.weightSplitGapPredict = signedWeight;
        break;
      case 'R21':
        engineRef.current.params.weightHit = signedWeight;
        break;
      case 'R22':
        engineRef.current.params.weightAvgEnemyCenter = signedWeight;
        break;
      case 'R23':
        engineRef.current.params.weightVectorStabbing = signedWeight;
        break;
      case 'R24':
        engineRef.current.params.weightTwitch = signedWeight;
        break;
    }
  };

  const handleToggleReward = useCallback((id: RewardConfigItem['id']) => {
    const nextRewards = rewardsRef.current.map((item) => {
      if (item.id !== id) return item;
      const nextActive = !item.active;
      syncRewardFlag(id, nextActive);
      return { ...item, active: nextActive };
    });
    rewardsRef.current = nextRewards;
    setRewards(nextRewards);
  }, []);

  const handleToggleRewardSign = useCallback((id: RewardConfigItem['id']) => {
    const nextRewards = rewardsRef.current.map((item) => {
      if (item.id !== id) return item;
      const nextPositive = !item.isPositive;
      const raw = parseFloat(item.weightText) || 0;
      const signed = nextPositive ? Math.abs(raw) : -Math.abs(raw);
      syncRewardWeight(id, signed);
      return { ...item, isPositive: nextPositive };
    });
    rewardsRef.current = nextRewards;
    setRewards(nextRewards);
  }, []);

  const handleRewardWeightChange = useCallback((id: RewardConfigItem['id'], text: string) => {
    const nextRewards = rewardsRef.current.map((item) => {
      if (item.id !== id) return item;
      const parsed = parseFloat(text);
      if (!isNaN(parsed)) {
        const signed = item.isPositive ? Math.abs(parsed) : -Math.abs(parsed);
        syncRewardWeight(id, signed);
      }
      return { ...item, weightText: text };
    });
    rewardsRef.current = nextRewards;
    setRewards(nextRewards);
  }, []);

  const handleClearAllRewards = useCallback(() => {
    const nextRewards = rewardsRef.current.map((item) => {
      if (item.active) {
        syncRewardFlag(item.id, false);
      }
      return { ...item, active: false };
    });
    rewardsRef.current = nextRewards;
    setRewards(nextRewards);
  }, []);

  return (
    <div className="flex flex-col items-center">
      <ControlBar
        activation={activation}
        enemyCount={enemyCount}
        enemySpeedPct={enemySpeedPct}
        enemySpeedMode={enemySpeedMode}
        spawnMode={spawnMode}
        simSpeed={simSpeed}
        turboMode={turboMode}
        rewards={rewards}
        onActivationChange={handleActivationChange}
        onEnemyCountChange={handleEnemyCountChange}
        onEnemySpeedPctChange={handleEnemySpeedPctChange}
        onEnemySpeedModeChange={handleEnemySpeedModeChange}
        onSpawnModeChange={handleSpawnModeChange}
        onSimSpeedChange={handleSimSpeedChange}
        onToggleReward={handleToggleReward}
        onToggleRewardSign={handleToggleRewardSign}
        onRewardWeightChange={handleRewardWeightChange}
        onClearAllRewards={handleClearAllRewards}
        onToggleTurbo={handleToggleTurbo}
        isFrozen={isFrozen}
        onToggleFreeze={handleToggleFreeze}
        onPruneNetwork={handlePruneNetwork}
        onReversePrune={handleReversePrune}
        onDownloadWeights={handleDownloadWeights}
        onGetWeightsJson={handleGetWeightsJson}
        onUploadWeights={handleUploadWeights}
        canPrune={canPrune}
        onReset={handleReset}
      />
      <canvas
        ref={simCanvasRef}
        id="simCanvas"
        width={SIM_WIDTH}
        height={SIM_HEIGHT}
        className="bg-[#000] border border-[#333] block max-w-full"
      />
      <div id="stats" className="text-[11px] my-[5px] text-[#888] tabular-nums">
        {statsText}
      </div>
      <div className="w-full max-w-[600px] flex flex-col items-center">
        <div className="w-full flex items-center justify-between gap-[6px] flex-wrap mb-[4px]">
          <div className="flex items-center gap-[4px] flex-wrap">
            <button
              id="tabNeuralNet"
              type="button"
              onClick={() => {
                dragBoxRef.current = null;
                setNnCursor('default');
                setBottomTab((prev) => (prev === 'nn' ? null : 'nn'));
              }}
              className={`${
                bottomTab === 'nn' ? 'bg-[#066]' : 'bg-[#222] hover:bg-[#333]'
              } text-[#eee] border border-[#444] px-[6px] py-[3px] font-mono text-[12px] leading-normal cursor-pointer whitespace-nowrap`}
            >
              Neural Net {bottomTab === 'nn' ? '▾' : '▸'}
            </button>
            <button
              id="tabWeightInspector"
              type="button"
              onClick={() => {
                dragBoxRef.current = null;
                setNnCursor('default');
                setBottomTab((prev) => (prev === 'weights' ? null : 'weights'));
              }}
              className={`${
                bottomTab === 'weights' ? 'bg-[#066]' : 'bg-[#222] hover:bg-[#333]'
              } text-[#eee] border border-[#444] px-[6px] py-[3px] font-mono text-[12px] leading-normal cursor-pointer whitespace-nowrap`}
            >
              Weight Inspector {bottomTab === 'weights' ? '▾' : '▸'}
            </button>
            <button
              id="tabLossLandscape"
              type="button"
              onClick={() => {
                dragBoxRef.current = null;
                setNnCursor('default');
                setBottomTab((prev) => (prev === 'landscape' ? null : 'landscape'));
              }}
              className={`${
                bottomTab === 'landscape' ? 'bg-[#066]' : 'bg-[#222] hover:bg-[#333]'
              } text-[#eee] border border-[#444] px-[6px] py-[3px] font-mono text-[12px] leading-normal cursor-pointer whitespace-nowrap`}
            >
              Loss Landscape {bottomTab === 'landscape' ? '▾' : '▸'}
            </button>
            <button
              id="tabRecSuite"
              type="button"
              onClick={() => {
                if (isRecording) {
                  handleStopRecording();
                } else {
                  setShowRecorderSuite((prev) => !prev);
                }
              }}
              title={isRecording ? 'Stop recording and download video' : 'Open video recording suite'}
              className={`${
                isRecording
                  ? 'bg-[#6b1414] border-[#c44] text-[#fff] hover:bg-[#821b1b]'
                  : showRecorderSuite
                  ? 'bg-[#066] border-[#444] text-[#eee]'
                  : 'bg-[#222] border-[#444] text-[#eee] hover:bg-[#333]'
              } border px-[6px] py-[3px] font-mono text-[12px] leading-normal cursor-pointer whitespace-nowrap`}
            >
              {isRecording
                ? `■ Stop ${Math.floor(recElapsedSec / 60)}:${String(recElapsedSec % 60).padStart(2, '0')}`
                : `● Rec ${showRecorderSuite ? '▾' : '▸'}`}
            </button>
          </div>

          <div className="flex items-center gap-[8px] font-mono text-[12px]">
            <span className="inline-flex items-center">
              <span className="mr-[4px] text-[#bbb]">L:</span>
              <span className="inline-flex items-center border border-[#444] bg-[#222]">
                <button
                  id="btnDecLayers"
                  type="button"
                  disabled={layers <= (isFrozen ? layers : 1)}
                  onClick={() => handleLayersChange(layers - 1)}
                  title="Decrease hidden layers"
                  className="bg-[#222] hover:bg-[#333] text-[#eee] border-none px-[7px] py-[2px] font-mono text-[12px] leading-normal cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed"
                >
                  -
                </button>
                <span
                  id="layersVal"
                  className="px-[7px] py-[2px] border-x border-[#444] text-[#eee] tabular-nums min-w-[22px] text-center"
                >
                  {layers}
                </span>
                <button
                  id="btnIncLayers"
                  type="button"
                  disabled={layers >= 5}
                  onClick={() => handleLayersChange(layers + 1)}
                  title="Increase hidden layers"
                  className="bg-[#222] hover:bg-[#333] text-[#eee] border-none px-[7px] py-[2px] font-mono text-[12px] leading-normal cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed"
                >
                  +
                </button>
              </span>
            </span>

            <span className="inline-flex items-center">
              <span className="mr-[4px] text-[#bbb]">N:</span>
              <span className="inline-flex items-center border border-[#444] bg-[#222]">
                <button
                  id="btnDecNeurons"
                  type="button"
                  disabled={neurons <= (isFrozen ? neurons : 2)}
                  onClick={() => handleNeuronsChange(neurons - 1)}
                  title="Decrease nodes per layer"
                  className="bg-[#222] hover:bg-[#333] text-[#eee] border-none px-[7px] py-[2px] font-mono text-[12px] leading-normal cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed"
                >
                  -
                </button>
                <span
                  id="neuronsVal"
                  className="px-[7px] py-[2px] border-x border-[#444] text-[#eee] tabular-nums min-w-[28px] text-center"
                >
                  {neurons}
                </span>
                <button
                  id="btnIncNeurons"
                  type="button"
                  disabled={neurons >= 64}
                  onClick={() => handleNeuronsChange(neurons + 1)}
                  title="Increase nodes per layer"
                  className="bg-[#222] hover:bg-[#333] text-[#eee] border-none px-[7px] py-[2px] font-mono text-[12px] leading-normal cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed"
                >
                  +
                </button>
              </span>
            </span>
          </div>
        </div>

        {showRecorderSuite && (
          <div
            id="recordingSuitePanel"
            className="w-full max-w-[570px] mb-[6px] p-[6px] bg-[#161616] border border-[#333] flex items-center justify-between gap-[8px] flex-wrap font-mono text-[11px]"
          >
            <div className="flex items-center gap-[10px] flex-wrap">
              <label className="inline-flex items-center gap-[4px] cursor-pointer text-[#ddd]">
                <input
                  id="chkRecNeuralNet"
                  type="checkbox"
                  checked={recNeuralNet}
                  disabled={isRecording}
                  onChange={(e) => setRecNeuralNet(e.target.checked)}
                  className="accent-[#0aa] cursor-pointer disabled:opacity-50"
                />
                Record Neural Net
              </label>
              <label className="inline-flex items-center gap-[4px] cursor-pointer text-[#ddd]">
                <input
                  id="chkRecLandscape"
                  type="checkbox"
                  checked={recLossLandscape}
                  disabled={isRecording}
                  onChange={(e) => setRecLossLandscape(e.target.checked)}
                  className="accent-[#0aa] cursor-pointer disabled:opacity-50"
                />
                Record Loss Landscape
              </label>
              <label className="inline-flex items-center gap-[4px] cursor-pointer text-[#ddd]">
                <input
                  id="chkRecTraining"
                  type="checkbox"
                  checked={recTraining}
                  disabled={isRecording}
                  onChange={(e) => setRecTraining(e.target.checked)}
                  className="accent-[#0aa] cursor-pointer disabled:opacity-50"
                />
                Record Training
              </label>
            </div>

            <div className="flex items-center gap-[8px]">
              {recStatus && (
                <span className={recStatus.isError ? 'text-[#f55]' : 'text-[#0cc]'}>
                  {recStatus.text}
                </span>
              )}
              {!isRecording ? (
                <button
                  id="btnStartRec"
                  type="button"
                  disabled={!recNeuralNet && !recLossLandscape && !recTraining}
                  onClick={handleStartRecording}
                  className="bg-[#1f4f4f] hover:bg-[#276868] text-[#fff] border border-[#0aa] px-[8px] py-[3px] font-mono text-[11px] cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed whitespace-nowrap"
                >
                  ● Record
                </button>
              ) : (
                <button
                  id="btnStopRec"
                  type="button"
                  onClick={handleStopRecording}
                  className="bg-[#7a1c1c] hover:bg-[#962222] text-[#fff] border border-[#d55] px-[8px] py-[3px] font-mono text-[11px] cursor-pointer whitespace-nowrap tabular-nums"
                >
                  ■ Stop ({Math.floor(recElapsedSec / 60)}:
                  {String(recElapsedSec % 60).padStart(2, '0')})
                </button>
              )}
            </div>
          </div>
        )}

        {bottomTab === 'nn' && (
          <canvas
            ref={nnCanvasRef}
            id="nnCanvas"
            width={NN_WIDTH}
            height={NN_HEIGHT}
            onPointerDown={(e) => {
              if (e.button !== 0 && e.pointerType === 'mouse') return;
              const pt = getNNCanvasPoint(e.clientX, e.clientY);
              if (!pt) return;
              e.currentTarget.setPointerCapture?.(e.pointerId);
              dragBoxRef.current = { x1: pt.x, y1: pt.y, x2: pt.x, y2: pt.y };
              setNnCursor('crosshair');
            }}
            onPointerMove={(e) => {
              const pt = getNNCanvasPoint(e.clientX, e.clientY);
              if (!pt) return;
              if (dragBoxRef.current) {
                dragBoxRef.current.x2 = pt.x;
                dragBoxRef.current.y2 = pt.y;
                return;
              }
              const hit = engineRef.current?.findHiddenNodeAt(pt.x, pt.y);
              setNnCursor(hit ? 'pointer' : 'default');
            }}
            onPointerUp={(e) => {
              if (dragBoxRef.current) {
                finishNNDrag(e.clientX, e.clientY);
              }
            }}
            onPointerCancel={() => {
              dragBoxRef.current = null;
              setNnCursor('default');
            }}
            onPointerLeave={() => {
              if (!dragBoxRef.current) {
                setNnCursor('default');
              }
            }}
            className={`bg-[#000] border border-[#333] block w-[95%] max-w-[570px] h-auto mx-auto select-none touch-none ${
              nnCursor === 'pointer'
                ? 'cursor-pointer'
                : nnCursor === 'crosshair'
                ? 'cursor-crosshair'
                : 'cursor-default'
            }`}
          />
        )}

        {bottomTab === 'weights' && engineRef.current && (
          <WeightInspector
            engine={engineRef.current}
            tick={inspectorTick}
            onToggleHiddenNodeFreeze={handleToggleInspectorNodeFreeze}
          />
        )}

        {bottomTab === 'landscape' && engineRef.current && (
          <LossLandscapePanel
            engine={engineRef.current}
            mode={landscapeMode}
            onModeChange={setLandscapeMode}
            view={landscapeView}
            onViewChange={setLandscapeView}
            metric={landscapeMetric}
            onMetricChange={setLandscapeMetric}
            data={landscapeData}
            activeRewardLabels={rewards
              .filter((r) => r.active)
              .map((r) => `${r.isPositive ? '+' : '-'}${r.weightText} ${r.label.replace(/^[+-]/, '')}`)}
            onRescan={() => startFreshLandscapeScan()}
          />
        )}
      </div>
      <OfflineIndicator />
    </div>
  );
}
