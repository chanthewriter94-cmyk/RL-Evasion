import React, { useEffect, useMemo, useRef, useState } from 'react';
import { EvasionEngine } from '../simulation/EvasionEngine';

interface WeightInspectorProps {
  engine: EvasionEngine;
  tick: number;
  onToggleHiddenNodeFreeze: (hiddenLayer: number, nodeIndex: number) => void;
}

const HEATMAP_WIDTH = 546;
const HEATMAP_HEIGHT = 210;

function getInputLabel(idx: number): string {
  if (idx === 0) return 'px';
  if (idx === 1) return 'py';
  if (idx === 2) return 'vx';
  if (idx === 3) return 'vy';
  const enemyIdx = Math.floor((idx - 4) / 2) + 1;
  const axis = (idx - 4) % 2 === 0 ? 'dx' : 'dy';
  return `e${enemyIdx}.${axis}`;
}

export const WeightInspector: React.FC<WeightInspectorProps> = ({
  engine,
  tick,
  onToggleHiddenNodeFreeze,
}) => {
  const [selectedLayer, setSelectedLayer] = useState<number>(0);
  const [activeInputsOnly, setActiveInputsOnly] = useState<boolean>(true);
  const [selectedCell, setSelectedCell] = useState<{ row: number; col: number } | null>(null);
  const heatmapRef = useRef<HTMLCanvasElement | null>(null);

  const net = engine.agent;
  const numLayers = net ? net.weights.length : 0;
  const clampedLayer = numLayers > 0 ? Math.min(selectedLayer, numLayers - 1) : 0;
  const activeInputCount = 4 + engine.params.numEnemies * 2;

  useEffect(() => {
    if (selectedLayer >= numLayers && numLayers > 0) {
      setSelectedLayer(numLayers - 1);
      setSelectedCell(null);
    }
  }, [numLayers, selectedLayer]);

  const stats = useMemo(() => {
    if (!net) return null;
    let totalParams = 0;
    let minW = Infinity;
    let maxW = -Infinity;
    let sumAbs = 0;
    let totalW = 0;
    let nearZero = 0;
    let frozenNodes = 0;
    let totalHidden = 0;

    for (let l = 0; l < net.weights.length; l++) {
      const isHidden = l < net.weights.length - 1;
      for (let r = 0; r < net.weights[l].length; r++) {
        totalParams++;
        if (isHidden) {
          totalHidden++;
          if (net.frozenBiases[l]?.[r]) frozenNodes++;
        }
        for (let c = 0; c < net.weights[l][r].length; c++) {
          const w = net.weights[l][r][c];
          const absW = Math.abs(w);
          totalW++;
          totalParams++;
          sumAbs += absW;
          if (w < minW) minW = w;
          if (w > maxW) maxW = w;
          if (absW < 0.05) nearZero++;
        }
      }
    }

    return {
      totalParams,
      minW: Number.isFinite(minW) ? minW : 0,
      maxW: Number.isFinite(maxW) ? maxW : 0,
      meanAbs: totalW > 0 ? sumAbs / totalW : 0,
      nearZeroPct: totalW > 0 ? (nearZero / totalW) * 100 : 0,
      frozenNodes,
      totalHidden,
    };
  }, [net, tick]);

  const layerView = useMemo(() => {
    if (!net || !net.weights[clampedLayer]) return null;
    const matrix = net.weights[clampedLayer];
    const biases = net.biases[clampedLayer];
    const rows = matrix.length;
    const rawCols = matrix[0]?.length ?? 0;
    const isInputLayer = clampedLayer === 0;
    const isOutputLayer = clampedLayer === net.weights.length - 1;

    const colIndices: number[] = [];
    const colLimit = isInputLayer && activeInputsOnly ? Math.min(rawCols, activeInputCount) : rawCols;
    for (let c = 0; c < colLimit; c++) {
      colIndices.push(c);
    }

    let maxAbs = 0.25;
    for (let r = 0; r < rows; r++) {
      const bAbs = Math.abs(biases[r] ?? 0);
      if (bAbs > maxAbs) maxAbs = bAbs;
      for (const c of colIndices) {
        const absW = Math.abs(matrix[r][c] ?? 0);
        if (absW > maxAbs) maxAbs = absW;
      }
    }

    const rowSummaries = matrix.map((row, r) => {
      let l1 = 0;
      const drivers: { col: number; w: number }[] = [];
      for (const c of colIndices) {
        const w = row[c] ?? 0;
        l1 += Math.abs(w);
        drivers.push({ col: c, w });
      }
      drivers.sort((a, b) => Math.abs(b.w) - Math.abs(a.w));
      return {
        row: r,
        label: isOutputLayer ? (r === 0 ? 'vx' : 'vy') : `N${r + 1}`,
        bias: biases[r] ?? 0,
        act: net.layerValues[clampedLayer + 1]?.[r] ?? 0,
        l1,
        frozen: !isOutputLayer && Boolean(net.frozenBiases[clampedLayer]?.[r]),
        topDrivers: drivers.slice(0, 3),
      };
    });

    return {
      matrix,
      biases,
      rows,
      rawCols,
      colIndices,
      maxAbs,
      isInputLayer,
      isOutputLayer,
      rowSummaries,
    };
  }, [net, clampedLayer, activeInputsOnly, activeInputCount, tick]);

  useEffect(() => {
    const canvas = heatmapRef.current;
    if (!canvas || !layerView || !net) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.fillStyle = '#050505';
    ctx.fillRect(0, 0, HEATMAP_WIDTH, HEATMAP_HEIGHT);

    const { rows, colIndices, matrix, biases, maxAbs, isOutputLayer } = layerView;
    if (rows === 0 || colIndices.length === 0) return;

    const leftPad = 28;
    const topPad = 14;
    const biasColW = 16;
    const biasGap = 6;
    const gridW = HEATMAP_WIDTH - leftPad - biasColW - biasGap - 4;
    const gridH = HEATMAP_HEIGHT - topPad - 6;
    const cellW = gridW / colIndices.length;
    const cellH = gridH / rows;

    ctx.fillStyle = '#666';
    ctx.font = '9px monospace';
    ctx.textBaseline = 'middle';
    ctx.fillText('in ->', 2, 7);
    ctx.fillText('b', leftPad + gridW + biasGap + 5, 7);

    for (let r = 0; r < rows; r++) {
      const y = topPad + r * cellH;
      const isRowFrozen = !isOutputLayer && Boolean(net.frozenBiases[clampedLayer]?.[r]);

      if (rows <= 32 || r % 2 === 0) {
        ctx.fillStyle = isRowFrozen ? '#7fb3ff' : '#888';
        ctx.font = '9px monospace';
        const rLabel = isOutputLayer ? (r === 0 ? 'vx' : 'vy') : `${r + 1}`;
        ctx.fillText(rLabel, 2, y + cellH * 0.5);
      }

      for (let ci = 0; ci < colIndices.length; ci++) {
        const c = colIndices[ci];
        const w = matrix[r][c] ?? 0;
        const norm = Math.min(1, Math.abs(w) / maxAbs);
        const isFrozen = isRowFrozen || Boolean(net.frozenWeights[clampedLayer]?.[r]?.[c]);
        const x = leftPad + ci * cellW;

        if (isFrozen) {
          ctx.fillStyle =
            w >= 0
              ? `rgb(${Math.floor(20 + norm * 55)}, ${Math.floor(40 + norm * 140)}, ${Math.floor(
                  80 + norm * 175
                )})`
              : `rgb(${Math.floor(65 + norm * 140)}, ${Math.floor(35 + norm * 50)}, ${Math.floor(
                  95 + norm * 140
                )})`;
        } else {
          ctx.fillStyle =
            w >= 0
              ? `rgb(0, ${Math.floor(18 + norm * 235)}, ${Math.floor(22 + norm * 233)})`
              : `rgb(${Math.floor(28 + norm * 227)}, ${Math.floor(14 + norm * 26)}, ${Math.floor(
                  14 + norm * 26
                )})`;
        }

        ctx.fillRect(x, y, Math.max(1, cellW - (cellW > 3 ? 0.5 : 0)), Math.max(1, cellH - (cellH > 3 ? 0.5 : 0)));

        if (selectedCell && selectedCell.row === r && selectedCell.col === c) {
          ctx.strokeStyle = '#ff0';
          ctx.lineWidth = 1.5;
          ctx.strokeRect(x + 0.5, y + 0.5, Math.max(2, cellW - 1), Math.max(2, cellH - 1));
        }
      }

      const b = biases[r] ?? 0;
      const bNorm = Math.min(1, Math.abs(b) / maxAbs);
      const bx = leftPad + gridW + biasGap;
      ctx.fillStyle =
        b >= 0
          ? `rgb(0, ${Math.floor(25 + bNorm * 220)}, ${Math.floor(30 + bNorm * 220)})`
          : `rgb(${Math.floor(30 + bNorm * 220)}, 28, 28)`;
      ctx.fillRect(bx, y, biasColW, Math.max(1, cellH - (cellH > 3 ? 0.5 : 0)));

      if (selectedCell && selectedCell.row === r && selectedCell.col === -1) {
        ctx.strokeStyle = '#ff0';
        ctx.lineWidth = 1.5;
        ctx.strokeRect(bx + 0.5, y + 0.5, biasColW - 1, Math.max(2, cellH - 1));
      }
    }
  }, [layerView, net, clampedLayer, selectedCell]);

  const handleHeatmapPick = (clientX: number, clientY: number) => {
    const canvas = heatmapRef.current;
    if (!canvas || !layerView) return;
    const rect = canvas.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return;

    const px = ((clientX - rect.left) / rect.width) * HEATMAP_WIDTH;
    const py = ((clientY - rect.top) / rect.height) * HEATMAP_HEIGHT;

    const leftPad = 28;
    const topPad = 14;
    const biasColW = 16;
    const biasGap = 6;
    const gridW = HEATMAP_WIDTH - leftPad - biasColW - biasGap - 4;
    const gridH = HEATMAP_HEIGHT - topPad - 6;

    const row = Math.floor(((py - topPad) / gridH) * layerView.rows);
    if (row < 0 || row >= layerView.rows) return;

    if (px >= leftPad + gridW + biasGap && px <= leftPad + gridW + biasGap + biasColW) {
      setSelectedCell({ row, col: -1 });
      return;
    }

    const ci = Math.floor(((px - leftPad) / gridW) * layerView.colIndices.length);
    if (ci >= 0 && ci < layerView.colIndices.length) {
      setSelectedCell({ row, col: layerView.colIndices[ci] });
    }
  };

  if (!net || !layerView || !stats) {
    return (
      <div className="w-[95%] max-w-[570px] bg-[#0a0a0a] border border-[#333] p-[10px] font-mono text-[11px] text-[#888]">
        Waiting for agent weights...
      </div>
    );
  }

  const colDisplayName = (colIdx: number): string => {
    if (colIdx === -1) return 'bias';
    return layerView.isInputLayer ? getInputLabel(colIdx) : `H${clampedLayer}.${colIdx + 1}`;
  };

  const detailRow = selectedCell ? Math.min(selectedCell.row, layerView.rows - 1) : 0;
  const detailCol =
    selectedCell && (selectedCell.col === -1 || selectedCell.col < layerView.rawCols)
      ? selectedCell.col
      : 0;
  const detailVal =
    detailCol === -1
      ? layerView.biases[detailRow] ?? 0
      : layerView.matrix[detailRow]?.[detailCol] ?? 0;
  const detailInVal =
    detailCol === -1 ? 1 : net.layerValues[clampedLayer]?.[detailCol] ?? 0;
  const detailFrozen =
    !layerView.isOutputLayer && Boolean(net.frozenBiases[clampedLayer]?.[detailRow]);

  return (
    <div
      id="weightInspectorPanel"
      className="w-[95%] max-w-[570px] bg-[#0a0a0a] border border-[#333] p-[8px] font-mono text-[11px] text-[#ddd] flex flex-col gap-[6px]"
    >
      <div className="flex items-center justify-between gap-[6px] flex-wrap text-[11px] text-[#aaa] border-b border-[#222] pb-[5px]">
        <span>
          Params: <span className="text-[#eee] tabular-nums">{stats.totalParams}</span> | Range:{' '}
          <span className="text-[#f77] tabular-nums">{stats.minW.toFixed(2)}</span>..
          <span className="text-[#0ff] tabular-nums">+{stats.maxW.toFixed(2)}</span>
        </span>
        <span>
          Mean |w|: <span className="text-[#eee] tabular-nums">{stats.meanAbs.toFixed(3)}</span> |{' '}
          &lt;0.05: <span className="text-[#eee] tabular-nums">{stats.nearZeroPct.toFixed(0)}%</span>{' '}
          | Frozen:{' '}
          <span className="text-[#7fb3ff] tabular-nums">
            {stats.frozenNodes}/{stats.totalHidden}
          </span>
        </span>
      </div>

      <div className="flex items-center justify-between gap-[4px] flex-wrap">
        <div className="flex items-center gap-[3px] flex-wrap">
          {net.weights.map((mat, idx) => {
            const inCount = mat[0]?.length ?? 0;
            const outCount = mat.length;
            const isOut = idx === net.weights.length - 1;
            const label = isOut
              ? `Out (${inCount}->${outCount})`
              : `L${idx + 1} (${inCount}->${outCount})`;
            return (
              <button
                key={idx}
                type="button"
                onClick={() => {
                  setSelectedLayer(idx);
                  setSelectedCell(null);
                }}
                className={`${
                  clampedLayer === idx
                    ? 'bg-[#066] text-[#fff] border-[#0aa]'
                    : 'bg-[#1b1b1b] text-[#bbb] border-[#3a3a3a] hover:bg-[#2a2a2a]'
                } border px-[6px] py-[2px] font-mono text-[11px] cursor-pointer whitespace-nowrap`}
              >
                {label}
              </button>
            );
          })}
        </div>

        {layerView.isInputLayer && (
          <button
            type="button"
            onClick={() => setActiveInputsOnly((prev) => !prev)}
            title="Toggle between active inputs (4 + 2*E) and all 104 input slots"
            className={`${
              activeInputsOnly
                ? 'bg-[#193838] text-[#8ff] border-[#0aa]'
                : 'bg-[#1b1b1b] text-[#aaa] border-[#3a3a3a]'
            } border px-[6px] py-[2px] font-mono text-[10px] cursor-pointer whitespace-nowrap`}
          >
            {activeInputsOnly ? `Active In (${Math.min(104, activeInputCount)})` : 'All 104 In'}
          </button>
        )}
      </div>

      <canvas
        ref={heatmapRef}
        id="weightHeatmapCanvas"
        width={HEATMAP_WIDTH}
        height={HEATMAP_HEIGHT}
        onPointerMove={(e) => handleHeatmapPick(e.clientX, e.clientY)}
        onClick={(e) => handleHeatmapPick(e.clientX, e.clientY)}
        className="w-full h-auto block bg-[#050505] border border-[#2a2a2a] cursor-crosshair select-none touch-manipulation"
      />

      <div className="flex items-center justify-between gap-[6px] flex-wrap bg-[#141414] border border-[#2a2a2a] px-[6px] py-[4px]">
        <div className="tabular-nums">
          <span className="text-[#888]">
            {layerView.rowSummaries[detailRow]?.label} &larr; {colDisplayName(detailCol)}:
          </span>{' '}
          <span className={detailVal >= 0 ? 'text-[#0ff]' : 'text-[#f66]'}>
            {detailVal >= 0 ? '+' : ''}
            {detailVal.toFixed(4)}
          </span>
          {detailCol !== -1 && (
            <span className="text-[#888]">
              {' '}
              | in={detailInVal.toFixed(2)} | sig={(detailVal * detailInVal).toFixed(3)}
            </span>
          )}
        </div>

        {!layerView.isOutputLayer && (
          <button
            type="button"
            onClick={() => onToggleHiddenNodeFreeze(clampedLayer, detailRow)}
            className={`${
              detailFrozen
                ? 'bg-[#2a4b7c] border-[#5c94e0] text-[#e6f0ff]'
                : 'bg-[#222] border-[#444] text-[#ccc] hover:bg-[#333]'
            } border px-[6px] py-[1px] font-mono text-[10px] cursor-pointer whitespace-nowrap`}
          >
            {detailFrozen ? `Unfreeze ${layerView.rowSummaries[detailRow]?.label}` : `Freeze ${layerView.rowSummaries[detailRow]?.label}`}
          </button>
        )}
      </div>

      <div className="max-h-[155px] overflow-y-auto border border-[#242424] bg-[#0f0f0f]">
        <table className="w-full border-collapse text-[10px] tabular-nums">
          <thead className="sticky top-0 bg-[#181818] text-[#888] border-b border-[#2c2c2c]">
            <tr>
              <th className="text-left px-[5px] py-[3px] font-normal">Node</th>
              <th className="text-right px-[5px] py-[3px] font-normal">Act</th>
              <th className="text-right px-[5px] py-[3px] font-normal">Bias</th>
              <th className="text-right px-[5px] py-[3px] font-normal">Sum|W|</th>
              <th className="text-left px-[5px] py-[3px] font-normal">Top Drivers</th>
            </tr>
          </thead>
          <tbody>
            {layerView.rowSummaries.map((item) => (
              <tr
                key={item.row}
                onClick={() =>
                  setSelectedCell({ row: item.row, col: item.topDrivers[0]?.col ?? -1 })
                }
                className={`${
                  detailRow === item.row ? 'bg-[#1a2626]' : 'hover:bg-[#171717]'
                } border-b border-[#1c1c1c] cursor-pointer`}
              >
                <td className="px-[5px] py-[2px] whitespace-nowrap">
                  <span className={item.frozen ? 'text-[#7fb3ff]' : 'text-[#ddd]'}>
                    {item.label}
                  </span>
                  {item.frozen && <span className="text-[#5c94e0] ml-[3px]">*</span>}
                </td>
                <td
                  className={`px-[5px] py-[2px] text-right ${
                    item.act >= 0 ? 'text-[#0dd]' : 'text-[#e66]'
                  }`}
                >
                  {item.act >= 0 ? '+' : ''}
                  {item.act.toFixed(2)}
                </td>
                <td
                  className={`px-[5px] py-[2px] text-right ${
                    item.bias >= 0 ? 'text-[#8dd]' : 'text-[#d88]'
                  }`}
                >
                  {item.bias >= 0 ? '+' : ''}
                  {item.bias.toFixed(2)}
                </td>
                <td className="px-[5px] py-[2px] text-right text-[#ccc]">
                  {item.l1.toFixed(2)}
                </td>
                <td className="px-[5px] py-[2px] text-[#999] truncate max-w-[230px]">
                  {item.topDrivers
                    .map(
                      (d) =>
                        `${colDisplayName(d.col)}(${d.w >= 0 ? '+' : ''}${d.w.toFixed(2)})`
                    )
                    .join(', ')}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};
