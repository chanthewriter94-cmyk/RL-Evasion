import React, { useEffect, useRef, useState } from 'react';
import { EnemySpeedMode, SpawnMode } from '../simulation/EvasionEngine';
import { ACTIVATION_OPTIONS, ActivationType } from '../simulation/NeuralNetwork';
import { PWAInstallButton } from './PWAInstallButton';

export type RewardCategory = 'spacing' | 'flow' | 'tactical';

export interface RewardConfigItem {
  id:
    | 'R1'
    | 'R2'
    | 'R3'
    | 'R4'
    | 'R5'
    | 'R6'
    | 'R7'
    | 'R8'
    | 'R9'
    | 'R10'
    | 'R11'
    | 'R12'
    | 'R13'
    | 'R14'
    | 'R15'
    | 'R16'
    | 'R17'
    | 'R18'
    | 'R19'
    | 'R20'
    | 'R21'
    | 'R22'
    | 'R23'
    | 'R24';
  label: string;
  category: RewardCategory;
  active: boolean;
  isPositive: boolean;
  weightText: string;
  title?: string;
}

type ControlTab = 'network' | 'env' | 'rewards' | null;
type RewardFilter = 'all' | 'active' | RewardCategory;

interface ControlBarProps {
  activation: ActivationType;
  enemyCount: number;
  enemySpeedPct: number;
  enemySpeedMode: EnemySpeedMode;
  spawnMode: SpawnMode;
  simSpeed: number;
  turboMode: boolean;
  rewards: RewardConfigItem[];
  onActivationChange: (val: ActivationType) => void;
  onEnemyCountChange: (val: number) => void;
  onEnemySpeedPctChange: (val: number) => void;
  onEnemySpeedModeChange: (mode: EnemySpeedMode) => void;
  onSpawnModeChange: (mode: SpawnMode) => void;
  onSimSpeedChange: (val: number) => void;
  onToggleReward: (id: RewardConfigItem['id']) => void;
  onToggleRewardSign: (id: RewardConfigItem['id']) => void;
  onRewardWeightChange: (id: RewardConfigItem['id'], text: string) => void;
  onClearAllRewards: () => void;
  onToggleTurbo: () => void;
  isFrozen: boolean;
  onToggleFreeze: () => void;
  onPruneNetwork: () => void;
  onReversePrune: () => void;
  onDownloadWeights: () => void;
  onGetWeightsJson: () => string;
  onUploadWeights: (data: unknown) => { generation: number };
  canPrune: boolean;
  onReset: () => void;
}

const inputBaseClass =
  'bg-[#222] text-[#eee] border border-[#444] p-[4px] font-mono text-[12px] leading-normal';
const buttonBaseClass =
  'bg-[#222] text-[#eee] border border-[#444] p-[4px] font-mono text-[12px] leading-normal cursor-pointer hover:bg-[#333] whitespace-nowrap';

export const ControlBar: React.FC<ControlBarProps> = ({
  activation,
  enemyCount,
  enemySpeedPct,
  enemySpeedMode,
  spawnMode,
  simSpeed,
  turboMode,
  rewards,
  onActivationChange,
  onEnemyCountChange,
  onEnemySpeedPctChange,
  onEnemySpeedModeChange,
  onSpawnModeChange,
  onSimSpeedChange,
  onToggleReward,
  onToggleRewardSign,
  onRewardWeightChange,
  onClearAllRewards,
  onToggleTurbo,
  isFrozen,
  onToggleFreeze,
  onPruneNetwork,
  onReversePrune,
  onDownloadWeights,
  onGetWeightsJson,
  onUploadWeights,
  canPrune,
  onReset,
}) => {
  const [activeTab, setActiveTab] = useState<ControlTab>(null);
  const [rewardFilter, setRewardFilter] = useState<RewardFilter>('all');
  const [showUploadPanel, setShowUploadPanel] = useState<boolean>(false);
  const [pasteText, setPasteText] = useState<string>('');
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [statusMsg, setStatusMsg] = useState<{ text: string; isError: boolean } | null>(null);
  const [enemyCountText, setEnemyCountText] = useState<string>(String(enemyCount));
  const [enemySpeedText, setEnemySpeedText] = useState<string>(String(enemySpeedPct));
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    setEnemyCountText(String(enemyCount));
  }, [enemyCount]);

  useEffect(() => {
    setEnemySpeedText(String(enemySpeedPct));
  }, [enemySpeedPct]);

  const toggleTab = (tab: Exclude<ControlTab, null>) => {
    setActiveTab((prev) => (prev === tab ? null : tab));
  };

  const loadRawJsonString = (rawText: string) => {
    try {
      const parsed: unknown = JSON.parse(rawText);
      const result = onUploadWeights(parsed);
      setStatusMsg({ text: `Loaded Gen ${result.generation}`, isError: false });
      setPasteText('');
      setShowUploadPanel(false);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Invalid weights JSON.';
      setStatusMsg({ text: message, isError: true });
    }
  };

  const readFileAndLoad = (file: File, inputEl?: HTMLInputElement | null) => {
    const reader = new FileReader();
    reader.onload = () => {
      if (typeof reader.result === 'string') {
        loadRawJsonString(reader.result);
      } else {
        setStatusMsg({ text: 'Unable to read file as text.', isError: true });
      }
    };
    reader.onerror = () => {
      setStatusMsg({ text: 'Failed to read selected file.', isError: true });
    };
    reader.onloadend = () => {
      if (inputEl) {
        inputEl.value = '';
      }
    };
    reader.readAsText(file);
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    e.stopPropagation();
    const file = e.target.files?.[0];
    if (!file) return;
    readFileAndLoad(file, e.target);
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (!file) return;
    readFileAndLoad(file);
  };

  const handleCopyJson = async () => {
    const json = onGetWeightsJson();
    if (!json) return;
    try {
      await navigator.clipboard.writeText(json);
      setStatusMsg({ text: 'Copied JSON', isError: false });
    } catch {
      setPasteText(json);
      setShowUploadPanel(true);
      setStatusMsg({ text: 'JSON placed below', isError: false });
    }
  };

  return (
    <div className="w-full max-w-[600px] mb-[6px] text-[12px]">
      <div className="flex items-center justify-between gap-[6px] flex-wrap">
        <div className="flex items-center gap-[4px]">
          <button
            id="tabNetwork"
            type="button"
            onClick={() => toggleTab('network')}
            className={`${
              activeTab === 'network' ? 'bg-[#066]' : 'bg-[#222] hover:bg-[#333]'
            } text-[#eee] border border-[#444] px-[6px] py-[3px] font-mono text-[12px] leading-normal cursor-pointer whitespace-nowrap`}
          >
            Network {activeTab === 'network' ? '▾' : '▸'}
          </button>
          <button
            id="tabEnv"
            type="button"
            onClick={() => toggleTab('env')}
            className={`${
              activeTab === 'env' ? 'bg-[#066]' : 'bg-[#222] hover:bg-[#333]'
            } text-[#eee] border border-[#444] px-[6px] py-[3px] font-mono text-[12px] leading-normal cursor-pointer whitespace-nowrap`}
          >
            Env {activeTab === 'env' ? '▾' : '▸'}
          </button>
          <button
            id="tabRewards"
            type="button"
            onClick={() => toggleTab('rewards')}
            className={`${
              activeTab === 'rewards' ? 'bg-[#066]' : 'bg-[#222] hover:bg-[#333]'
            } text-[#eee] border border-[#444] px-[6px] py-[3px] font-mono text-[12px] leading-normal cursor-pointer whitespace-nowrap`}
          >
            Rewards{rewards.some((r) => r.active) ? ` (${rewards.filter((r) => r.active).length})` : ''}{' '}
            {activeTab === 'rewards' ? '▾' : '▸'}
          </button>
        </div>

        <div className="flex items-center gap-[4px]">
          <PWAInstallButton />
          <button
            id="btnTurbo"
            type="button"
            onClick={onToggleTurbo}
            className={`${
              turboMode ? 'bg-[#066]' : 'bg-[#222] hover:bg-[#333]'
            } text-[#eee] border border-[#444] px-[8px] py-[3px] font-mono text-[12px] leading-normal cursor-pointer whitespace-nowrap`}
          >
            MAX
          </button>
          <button
            id="btnReset"
            type="button"
            onClick={onReset}
            className="bg-[#222] text-[#eee] border border-[#444] px-[8px] py-[3px] font-mono text-[12px] leading-normal cursor-pointer hover:bg-[#333] whitespace-nowrap"
          >
            Reset
          </button>
        </div>
      </div>

      {activeTab === 'network' && (
        <div className="mt-[6px] p-[6px] bg-[#161616] border border-[#333] flex flex-col gap-[6px]">
          <div className="flex gap-[8px] flex-wrap items-center">

            <label className="inline-flex items-center gap-[2px]">
              A:
              <select
                id="activation"
                value={activation}
                onChange={(e) => onActivationChange(e.target.value as ActivationType)}
                className={inputBaseClass}
              >
                {ACTIVATION_OPTIONS.map((opt) => (
                  <option key={opt.value} value={opt.value}>
                    {opt.label}
                  </option>
                ))}
              </select>
            </label>

            <button
              id="btnPrune"
              type="button"
              disabled={!canPrune}
              onClick={onPruneNetwork}
              title="Prune the hidden node with the smallest total weight magnitude"
              className="bg-[#222] text-[#eee] border border-[#444] p-[4px] font-mono text-[12px] leading-normal cursor-pointer hover:bg-[#333] disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:bg-[#222] whitespace-nowrap"
            >
              Prune Network
            </button>

            <button
              id="btnReversePrune"
              type="button"
              disabled={!canPrune}
              onClick={onReversePrune}
              title="Prune the hidden node with the largest total weight magnitude"
              className="bg-[#222] text-[#eee] border border-[#444] p-[4px] font-mono text-[12px] leading-normal cursor-pointer hover:bg-[#333] disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:bg-[#222] whitespace-nowrap"
            >
              Reverse Prune
            </button>

            <button
              id="btnFreeze"
              type="button"
              onClick={onToggleFreeze}
              title="Freeze hidden layers, or click/drag a box on hidden neurons below"
              className={`${
                isFrozen
                  ? 'bg-[#2a4b7c] border-[#5c94e0] text-[#e6f0ff] hover:bg-[#335a94]'
                  : 'bg-[#222] border-[#444] text-[#eee] hover:bg-[#333]'
              } border p-[4px] font-mono text-[12px] leading-normal cursor-pointer whitespace-nowrap`}
            >
              {isFrozen ? 'Frozen' : 'Freeze'}
            </button>

            <button
              id="btnDownloadWeights"
              type="button"
              onClick={onDownloadWeights}
              title="Download network weights and biases as JSON"
              className={buttonBaseClass}
            >
              Download Weights
            </button>

            <button
              id="btnCopyWeights"
              type="button"
              onClick={handleCopyJson}
              title="Copy weights JSON to clipboard"
              className={buttonBaseClass}
            >
              Copy JSON
            </button>

            <button
              id="btnUploadWeights"
              type="button"
              onClick={() => setShowUploadPanel((prev) => !prev)}
              title="Load exported weights JSON via paste, drag-and-drop, or file"
              className={`${
                showUploadPanel ? 'bg-[#066]' : 'bg-[#222] hover:bg-[#333]'
              } text-[#eee] border border-[#444] p-[4px] font-mono text-[12px] leading-normal cursor-pointer whitespace-nowrap`}
            >
              Upload Weights
            </button>

            {statusMsg && (
              <span
                className={`text-[11px] ${
                  statusMsg.isError ? 'text-[#f55]' : 'text-[#0cc]'
                }`}
              >
                {statusMsg.text}
              </span>
            )}
          </div>

          {showUploadPanel && (
            <div className="pt-[6px] border-t border-[#333] flex flex-col gap-[6px]">
              <div
                onDragOver={(e) => {
                  e.preventDefault();
                  e.stopPropagation();
                  setIsDragging(true);
                }}
                onDragLeave={(e) => {
                  e.preventDefault();
                  e.stopPropagation();
                  setIsDragging(false);
                }}
                onDrop={handleDrop}
                className={`border border-dashed ${
                  isDragging ? 'border-[#0ff] bg-[#102828]' : 'border-[#444] bg-[#111]'
                } p-[6px] flex items-center justify-between gap-[8px] flex-wrap`}
              >
                <span className="text-[11px] text-[#aaa]">
                  Drop .json file here (avoids iframe file picker) or choose file:
                </span>
                <button
                  id="btnBrowseWeightsFile"
                  type="button"
                  onClick={(e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    fileInputRef.current?.click();
                  }}
                  className={buttonBaseClass}
                >
                  Choose File
                </button>
                <input
                  ref={fileInputRef}
                  id="fileUploadWeights"
                  type="file"
                  accept=".json,application/json"
                  onChange={handleFileChange}
                  className="hidden"
                />
              </div>

              <div className="flex gap-[6px] items-start">
                <textarea
                  id="pasteWeightsJson"
                  value={pasteText}
                  onChange={(e) => setPasteText(e.target.value)}
                  placeholder="Or paste downloaded weights JSON here..."
                  rows={2}
                  className="flex-1 bg-[#111] text-[#eee] border border-[#444] p-[4px] font-mono text-[11px] leading-tight resize-y"
                />
                <button
                  id="btnLoadPastedWeights"
                  type="button"
                  disabled={!pasteText.trim()}
                  onClick={() => loadRawJsonString(pasteText)}
                  className="bg-[#222] text-[#eee] border border-[#444] p-[4px] font-mono text-[12px] leading-normal cursor-pointer hover:bg-[#333] disabled:opacity-50 disabled:cursor-not-allowed whitespace-nowrap"
                >
                  Load JSON
                </button>
              </div>
            </div>
          )}
        </div>
      )}

      {activeTab === 'env' && (
        <div className="mt-[6px] p-[6px] bg-[#161616] border border-[#333] flex gap-[10px] flex-wrap items-center">
          <span className="inline-flex items-center">
            <label htmlFor="enemyCount" className="mr-[2px]">
              E:
            </label>
            <button
              id="btnDecEnemy"
              type="button"
              onClick={(e) => {
                e.preventDefault();
                onEnemyCountChange(Math.max(1, (enemyCount || 1) - 1));
              }}
              className={buttonBaseClass}
            >
              -
            </button>
            <input
              type="number"
              id="enemyCount"
              value={enemyCountText}
              min={1}
              max={50}
              onChange={(e) => {
                const raw = e.target.value;
                setEnemyCountText(raw);
                const parsed = parseInt(raw, 10);
                if (!isNaN(parsed) && parsed >= 1 && parsed <= 50) {
                  onEnemyCountChange(parsed);
                }
              }}
              onBlur={() => {
                const parsed = parseInt(enemyCountText, 10);
                const valid = !isNaN(parsed) ? Math.min(50, Math.max(1, parsed)) : enemyCount;
                setEnemyCountText(String(valid));
                if (valid !== enemyCount) onEnemyCountChange(valid);
              }}
              className={`${inputBaseClass} w-[40px] text-center`}
            />
            <button
              id="btnIncEnemy"
              type="button"
              onClick={(e) => {
                e.preventDefault();
                onEnemyCountChange(Math.min(50, (enemyCount || 1) + 1));
              }}
              className={buttonBaseClass}
            >
              +
            </button>
          </span>

          <span className="inline-flex items-center gap-[4px]">
            <span className="inline-flex border border-[#444]">
              <button
                id="btnSpeedFixed"
                type="button"
                onClick={() => onEnemySpeedModeChange('fixed')}
                title="All enemies move at the fixed % speed"
                className={`${
                  enemySpeedMode === 'fixed' ? 'bg-[#066]' : 'bg-[#222] hover:bg-[#333]'
                } text-[#eee] border-none px-[6px] py-[4px] font-mono text-[12px] leading-normal cursor-pointer whitespace-nowrap`}
              >
                Fixed
              </button>
              <button
                id="btnSpeedRarity"
                type="button"
                onClick={() => onEnemySpeedModeChange('rarity')}
                title="Each enemy rolls a random speed (1-90%) on a ~60% bell curve"
                className={`${
                  enemySpeedMode === 'rarity' ? 'bg-[#066]' : 'bg-[#222] hover:bg-[#333]'
                } text-[#eee] border-none border-l border-[#444] px-[6px] py-[4px] font-mono text-[12px] leading-normal cursor-pointer whitespace-nowrap`}
              >
                Rarity
              </button>
            </span>
            {enemySpeedMode === 'fixed' ? (
              <label className="inline-flex items-center gap-[2px]">
                %:
                <input
                  type="number"
                  id="enemySpeedPct"
                  value={enemySpeedText}
                  min={1}
                  max={200}
                  onChange={(e) => {
                    const raw = e.target.value;
                    setEnemySpeedText(raw);
                    const parsed = parseFloat(raw);
                    if (!isNaN(parsed) && parsed >= 1) {
                      onEnemySpeedPctChange(parsed);
                    }
                  }}
                  onBlur={() => {
                    const parsed = parseFloat(enemySpeedText);
                    const valid = !isNaN(parsed) && parsed >= 1 ? parsed : enemySpeedPct;
                    setEnemySpeedText(String(valid));
                    if (valid !== enemySpeedPct) onEnemySpeedPctChange(valid);
                  }}
                  className={`${inputBaseClass} w-[45px]`}
                />
              </label>
            ) : (
              <span
                className="text-[11px] text-[#0cc] whitespace-nowrap"
                title="Enemies spawn with random speeds from 1% to 90% on a bell curve averaging ~60%"
              >
                1–90% (~60%)
              </span>
            )}
          </span>

          <label className="inline-flex items-center gap-[4px]">
            S:
            <input
              type="range"
              id="speedSlider"
              min={1}
              max={100}
              value={simSpeed}
              onChange={(e) => onSimSpeedChange(parseInt(e.target.value, 10) || 1)}
              className="bg-[#222] text-[#eee] border border-[#444] p-[4px] font-mono w-[80px] cursor-pointer"
            />
            <span id="speedVal">{turboMode ? 'MAX' : `${simSpeed}x`}</span>
          </label>

          <span className="inline-flex items-center">
            <span className="mr-[4px]">Spawn:</span>
            <span className="inline-flex border border-[#444]">
              <button
                id="btnSpawnOuter"
                type="button"
                onClick={() => onSpawnModeChange('outer')}
                title="Spawn enemies off-screen from random edges with short staggered delays"
                className={`${
                  spawnMode === 'outer' ? 'bg-[#066]' : 'bg-[#222] hover:bg-[#333]'
                } text-[#eee] border-none px-[6px] py-[4px] font-mono text-[12px] leading-normal cursor-pointer whitespace-nowrap`}
              >
                Outer
              </button>
              <button
                id="btnSpawnCentral"
                type="button"
                onClick={() => onSpawnModeChange('central')}
                title="Spawn enemies at random positions inside the arena"
                className={`${
                  spawnMode === 'central' ? 'bg-[#066]' : 'bg-[#222] hover:bg-[#333]'
                } text-[#eee] border-none border-l border-[#444] px-[6px] py-[4px] font-mono text-[12px] leading-normal cursor-pointer whitespace-nowrap`}
              >
                Central
              </button>
            </span>
          </span>
        </div>
      )}

      {activeTab === 'rewards' && (() => {
        const activeCount = rewards.filter((r) => r.active).length;
        const categories: { key: RewardCategory; title: string }[] = [
          { key: 'spacing', title: 'Spacing & Territory' },
          { key: 'flow', title: 'Heading & Corridor' },
          { key: 'tactical', title: 'Tactical Evasion' },
        ];

        const renderRewardChip = (r: RewardConfigItem) => (
          <span
            key={r.id}
            className={`inline-flex items-center border ${
              r.active ? 'border-[#0aa]' : 'border-[#444]'
            }`}
          >
            <button
              id={`btn${r.id}`}
              type="button"
              title={r.title}
              onClick={() => onToggleReward(r.id)}
              className={`${
                r.active ? 'bg-[#066] text-[#fff]' : 'bg-[#222] text-[#eee] hover:bg-[#333]'
              } border-none px-[5px] py-[3px] font-mono text-[11px] leading-normal cursor-pointer`}
            >
              {r.label}
            </button>
            <button
              id={`btnSign${r.id}`}
              type="button"
              onClick={(e) => {
                e.preventDefault();
                onToggleRewardSign(r.id);
              }}
              className={`bg-[#222] hover:bg-[#333] border-none border-l border-[#444] w-[20px] py-[3px] font-mono text-[11px] leading-normal cursor-pointer ${
                r.isPositive ? 'text-[#eee]' : 'text-[#f55]'
              }`}
            >
              {r.isPositive ? '+' : '-'}
            </button>
            <input
              type="number"
              id={`w${r.id}`}
              value={r.weightText}
              step="0.1"
              title={r.title}
              onChange={(e) => onRewardWeightChange(r.id, e.target.value)}
              className="bg-[#222] text-[#eee] border-none border-l border-[#444] px-[4px] py-[3px] font-mono text-[11px] leading-normal w-[44px]"
            />
          </span>
        );

        const filterButtons: { key: RewardFilter; label: string }[] = [
          { key: 'all', label: 'All' },
          { key: 'active', label: `Active (${activeCount})` },
          { key: 'spacing', label: 'Spacing' },
          { key: 'flow', label: 'Flow' },
          { key: 'tactical', label: 'Tactical' },
        ];

        return (
          <div className="mt-[6px] p-[6px] bg-[#161616] border border-[#333] flex flex-col gap-[8px]">
            <div className="flex items-center justify-between gap-[6px] flex-wrap pb-[6px] border-b border-[#2a2a2a]">
              <div className="flex items-center gap-[3px] flex-wrap">
                {filterButtons.map((fb) => (
                  <button
                    key={fb.key}
                    type="button"
                    onClick={() => setRewardFilter(fb.key)}
                    className={`${
                      rewardFilter === fb.key
                        ? 'bg-[#066] text-[#fff] border-[#0aa]'
                        : 'bg-[#202020] text-[#bbb] border-[#3a3a3a] hover:bg-[#2d2d2d]'
                    } border px-[6px] py-[2px] font-mono text-[11px] leading-normal cursor-pointer whitespace-nowrap`}
                  >
                    {fb.label}
                  </button>
                ))}
              </div>
              <button
                id="btnClearRewards"
                type="button"
                disabled={activeCount === 0}
                onClick={onClearAllRewards}
                title="Disable all active rewards"
                className="bg-[#222] text-[#ccc] border border-[#444] px-[6px] py-[2px] font-mono text-[11px] leading-normal cursor-pointer hover:bg-[#333] disabled:opacity-40 disabled:cursor-not-allowed whitespace-nowrap"
              >
                Off All
              </button>
            </div>

            {rewardFilter === 'active' ? (
              activeCount === 0 ? (
                <div className="text-[11px] text-[#777] py-[2px]">
                  No rewards active.
                </div>
              ) : (
                <div className="flex gap-[5px] flex-wrap items-center">
                  {rewards.filter((r) => r.active).map(renderRewardChip)}
                </div>
              )
            ) : rewardFilter === 'all' ? (
              <div className="flex flex-col gap-[8px]">
                {categories.map((cat) => {
                  const items = rewards.filter((r) => r.category === cat.key);
                  const catActive = items.filter((r) => r.active).length;
                  return (
                    <div key={cat.key} className="flex flex-col gap-[4px]">
                      <div className="flex items-center justify-between text-[10px] uppercase tracking-wider text-[#888]">
                        <span>{cat.title}</span>
                        {catActive > 0 && (
                          <span className="text-[#0cc]">{catActive} on</span>
                        )}
                      </div>
                      <div className="flex gap-[5px] flex-wrap items-center">
                        {items.map(renderRewardChip)}
                      </div>
                    </div>
                  );
                })}
              </div>
            ) : (
              <div className="flex gap-[5px] flex-wrap items-center">
                {rewards.filter((r) => r.category === rewardFilter).map(renderRewardChip)}
              </div>
            )}
          </div>
        );
      })()}
    </div>
  );
};
