import React, { useEffect, useRef, useState } from 'react';
import { EvasionEngine, SIM_HEIGHT, SIM_WIDTH } from '../simulation/EvasionEngine';
import {
  drawLandscape1D,
  drawLandscape2D,
  drawLandscape3D,
  LandscapeData,
  LandscapeMetric,
  LandscapeMode,
  LandscapeView,
} from '../simulation/LossLandscape';

interface LossLandscapePanelProps {
  engine: EvasionEngine;
  mode: LandscapeMode;
  onModeChange: (mode: LandscapeMode) => void;
  view: LandscapeView;
  onViewChange: (view: LandscapeView) => void;
  metric: LandscapeMetric;
  onMetricChange: (metric: LandscapeMetric) => void;
  data: LandscapeData | null;
  activeRewardLabels: string[];
  onRescan: () => void;
}

const CANVAS_W = 554;
const SPLIT_H = 390;
const TOP_3D_H = 214;
const BOT_Y = 220;
const BOT_LEFT_W = 274;
const BOT_RIGHT_X = 280;
const BOT_RIGHT_W = 274;
const BOT_H = 170;

export const LossLandscapePanel: React.FC<LossLandscapePanelProps> = ({
  engine,
  mode,
  onModeChange,
  view,
  onViewChange,
  metric,
  onMetricChange,
  data,
  activeRewardLabels,
  onRescan,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [yaw, setYaw] = useState<number>(0.68);
  const [pitch, setPitch] = useState<number>(0.58);
  const [hoverCell, setHoverCell] = useState<{ row: number; col: number } | null>(null);
  const orbitDragRef = useRef<{ lastX: number; lastY: number } | null>(null);

  const canvasHeight = view === '2d+3d' ? SPLIT_H : view === '1d' ? 236 : 310;

  useEffect(() => {
    setHoverCell(null);
  }, [mode, view]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.fillStyle = '#050505';
    ctx.fillRect(0, 0, CANVAS_W, canvasHeight);

    if (!data) {
      ctx.fillStyle = '#888';
      ctx.font = '11px monospace';
      ctx.textBaseline = 'middle';
      ctx.fillText('Preparing loss landscape scan...', 16, canvasHeight * 0.5);
      return;
    }

    if (view === '2d+3d') {
      drawLandscape3D(ctx, 0, 0, CANVAS_W, TOP_3D_H, data, metric, yaw, pitch);
      drawLandscape2D(ctx, 0, BOT_Y, BOT_LEFT_W, BOT_H, data, metric, hoverCell);
      drawLandscape1D(ctx, BOT_RIGHT_X, BOT_Y, BOT_RIGHT_W, BOT_H, data, metric);
    } else if (view === '3d') {
      drawLandscape3D(ctx, 0, 0, CANVAS_W, canvasHeight, data, metric, yaw, pitch);
    } else if (view === '2d') {
      drawLandscape2D(ctx, 0, 0, CANVAS_W, canvasHeight, data, metric, hoverCell);
    } else {
      drawLandscape1D(ctx, 0, 0, CANVAS_W, canvasHeight, data, metric);
    }
  }, [data, metric, view, yaw, pitch, hoverCell, canvasHeight]);

  const toCanvasCoords = (clientX: number, clientY: number) => {
    const canvas = canvasRef.current;
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return null;
    return {
      x: ((clientX - rect.left) / rect.width) * CANVAS_W,
      y: ((clientY - rect.top) / rect.height) * canvasHeight,
    };
  };

  const inspect2DAt = (px: number, py: number) => {
    if (!data) return false;
    let boxX = 0;
    let boxY = 0;
    let boxW = CANVAS_W;
    let boxH = canvasHeight;

    if (view === '2d+3d') {
      if (py < BOT_Y || px > BOT_LEFT_W) return false;
      boxX = 0;
      boxY = BOT_Y;
      boxW = BOT_LEFT_W;
      boxH = BOT_H;
    } else if (view !== '2d') {
      return false;
    }

    const padL = 8;
    const padR = 8;
    const padT = 18;
    const padB = 14;
    const plotW = Math.max(10, boxW - padL - padR);
    const plotH = Math.max(10, boxH - padT - padB);
    const relX = px - (boxX + padL);
    const relY = py - (boxY + padT);
    if (relX < 0 || relX > plotW || relY < 0 || relY > plotH) {
      return false;
    }
    const col = Math.max(0, Math.min(data.gridW - 1, Math.floor((relX / plotW) * data.gridW)));
    const row = Math.max(0, Math.min(data.gridH - 1, Math.floor((relY / plotH) * data.gridH)));
    setHoverCell({ row, col });
    return true;
  };

  const isIn3DRegion = (px: number, py: number) => {
    if (view === '3d') return true;
    if (view === '2d+3d') return py <= TOP_3D_H && px >= 0 && px <= CANVAS_W;
    return false;
  };

  let hoverReadout: string | null = null;
  if (data && hoverCell) {
    const rawFit = data.values[hoverCell.row]?.[hoverCell.col] ?? 0;
    const shownVal = metric === 'loss' ? -rawFit : rawFit;
    const isHit = Boolean(data.hitMask[hoverCell.row]?.[hoverCell.col]);
    const metricLabel = metric === 'loss' ? 'Loss' : 'Fitness';

    if (data.mode === 'weight') {
      const alpha =
        -data.span + (hoverCell.col / Math.max(1, data.gridW - 1)) * (data.span * 2);
      const beta =
        -data.span + (hoverCell.row / Math.max(1, data.gridH - 1)) * (data.span * 2);
      hoverReadout = `α=${alpha >= 0 ? '+' : ''}${alpha.toFixed(2)}, β=${beta >= 0 ? '+' : ''}${beta.toFixed(2)} | ${metricLabel}: ${shownVal.toFixed(1)}${isHit ? ' | HIT' : ''}`;
    } else {
      const pad = 10;
      const sx = Math.round(pad + (hoverCell.col / Math.max(1, data.gridW - 1)) * (SIM_WIDTH - pad * 2));
      const sy = Math.round(pad + (hoverCell.row / Math.max(1, data.gridH - 1)) * (SIM_HEIGHT - pad * 2));
      hoverReadout = `Arena (${sx}, ${sy}) | ${metricLabel}: ${shownVal.toFixed(1)}${isHit ? ' | HIT' : ''}`;
    }
  }

  const progressPct = data ? Math.round(data.progress * 100) : 0;
  const rewardSummary =
    activeRewardLabels.length > 0
      ? activeRewardLabels.slice(0, 6).join(', ') +
        (activeRewardLabels.length > 6 ? ` +${activeRewardLabels.length - 6}` : '')
      : 'None (Survival steps only)';

  return (
    <div
      id="lossLandscapePanel"
      className="w-[95%] max-w-[570px] mx-auto bg-[#0a0a0a] border border-[#333] p-[8px] font-mono text-[11px] text-[#ddd]"
    >
      <div className="flex items-center justify-between gap-[6px] flex-wrap mb-[6px]">
        <div className="flex items-center gap-[4px] flex-wrap">
          <button
            id="btnLandscapeWeightMode"
            type="button"
            onClick={() => onModeChange('weight')}
            title="Filter-normalized 2D weight perturbations (α, β) + Current→Best 1D slice"
            className={`${
              mode === 'weight' ? 'bg-[#066] border-[#0aa] text-[#fff]' : 'bg-[#1a1a1a] border-[#444] text-[#ccc] hover:bg-[#2a2a2a]'
            } border px-[6px] py-[2px] cursor-pointer`}
          >
            Weight Space
          </button>
          <button
            id="btnLandscapeSpatialMode"
            type="button"
            onClick={() => onModeChange('spatial')}
            title="Probe active reward landscape across arena (X, Y) coordinates"
            className={`${
              mode === 'spatial' ? 'bg-[#066] border-[#0aa] text-[#fff]' : 'bg-[#1a1a1a] border-[#444] text-[#ccc] hover:bg-[#2a2a2a]'
            } border px-[6px] py-[2px] cursor-pointer`}
          >
            Spatial Arena
          </button>

          <span className="text-[#444] mx-[2px]">|</span>

          <button
            id="btnLandscapeMetricLoss"
            type="button"
            onClick={() => onMetricChange('loss')}
            className={`${
              metric === 'loss' ? 'bg-[#5c1f1f] border-[#c55] text-[#fff]' : 'bg-[#1a1a1a] border-[#444] text-[#ccc] hover:bg-[#2a2a2a]'
            } border px-[6px] py-[2px] cursor-pointer`}
          >
            Loss
          </button>
          <button
            id="btnLandscapeMetricFit"
            type="button"
            onClick={() => onMetricChange('fitness')}
            className={`${
              metric === 'fitness' ? 'bg-[#066] border-[#0aa] text-[#fff]' : 'bg-[#1a1a1a] border-[#444] text-[#ccc] hover:bg-[#2a2a2a]'
            } border px-[6px] py-[2px] cursor-pointer`}
          >
            Fitness
          </button>
        </div>

        <div className="flex items-center gap-[4px] flex-wrap">
          {(['2d+3d', '3d', '2d', '1d'] as LandscapeView[]).map((v) => (
            <button
              key={v}
              type="button"
              onClick={() => onViewChange(v)}
              className={`${
                view === v ? 'bg-[#066] border-[#0aa] text-[#fff]' : 'bg-[#1a1a1a] border-[#444] text-[#ccc] hover:bg-[#2a2a2a]'
              } border px-[6px] py-[2px] uppercase cursor-pointer`}
            >
              {v}
            </button>
          ))}
          <button
            id="btnLandscapeRescan"
            type="button"
            onClick={onRescan}
            title="Re-scan landscape with current network weights and active rewards"
            className="bg-[#222] hover:bg-[#333] border border-[#555] text-[#eee] px-[6px] py-[2px] cursor-pointer tabular-nums"
          >
            {mode === 'weight' && progressPct < 100 ? `Scan ${progressPct}%` : 'Rescan'}
          </button>
        </div>
      </div>

      <div className="mb-[6px] px-[6px] py-[3px] bg-[#141414] border border-[#2a2a2a] flex items-center justify-between gap-[6px] flex-wrap text-[10px] text-[#999]">
        <span>
          Rewards ({activeRewardLabels.length}): <span className="text-[#0cc]">{rewardSummary}</span>
        </span>
        <span className="tabular-nums text-[#888]">
          Gen {engine.generation} | {mode === 'weight' ? `${data?.evalSteps ?? 110}-step rollouts` : 'Live Arena Probe'}
        </span>
      </div>

      <canvas
        ref={canvasRef}
        id="landscapeCanvas"
        width={CANVAS_W}
        height={canvasHeight}
        onPointerDown={(e) => {
          const pt = toCanvasCoords(e.clientX, e.clientY);
          if (!pt) return;
          if (isIn3DRegion(pt.x, pt.y)) {
            e.currentTarget.setPointerCapture?.(e.pointerId);
            orbitDragRef.current = { lastX: e.clientX, lastY: e.clientY };
          } else {
            inspect2DAt(pt.x, pt.y);
          }
        }}
        onPointerMove={(e) => {
          if (orbitDragRef.current) {
            const dx = e.clientX - orbitDragRef.current.lastX;
            const dy = e.clientY - orbitDragRef.current.lastY;
            orbitDragRef.current = { lastX: e.clientX, lastY: e.clientY };
            setYaw((prev) => prev + dx * 0.012);
            setPitch((prev) => Math.max(0.18, Math.min(1.42, prev + dy * 0.009)));
            return;
          }
          const pt = toCanvasCoords(e.clientX, e.clientY);
          if (!pt) return;
          if (!inspect2DAt(pt.x, pt.y)) {
            setHoverCell(null);
          }
        }}
        onPointerUp={() => {
          orbitDragRef.current = null;
        }}
        onPointerCancel={() => {
          orbitDragRef.current = null;
        }}
        onPointerLeave={() => {
          if (!orbitDragRef.current) {
            setHoverCell(null);
          }
        }}
        className="w-full h-auto block bg-[#050505] border border-[#262626] select-none touch-none cursor-crosshair"
      />

      <div className="mt-[5px] flex items-center justify-between gap-[8px] flex-wrap text-[10px] text-[#888] tabular-nums">
        <span>
          {hoverReadout ??
            'Drag 3D to orbit • Hover/tap 2D contour to inspect • Yellow = Current, Cyan = Best, Red = Hit/Enemy'}
        </span>
        <button
          type="button"
          onClick={() => {
            setYaw(0.68);
            setPitch(0.58);
          }}
          className="bg-[#1a1a1a] hover:bg-[#2a2a2a] border border-[#3a3a3a] text-[#bbb] px-[5px] py-[1px] cursor-pointer"
        >
          Reset View
        </button>
      </div>
    </div>
  );
};
