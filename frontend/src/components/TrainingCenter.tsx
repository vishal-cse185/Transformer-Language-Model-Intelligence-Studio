import React, { useState, useEffect } from 'react';
import {
  PlayCircle,
  PauseCircle,
  RefreshCw,
  TrendingDown,
  Activity,
  CheckCircle2,
  Clock,
  ShieldCheck,
  Save,
  Sliders,
  Sparkles,
} from 'lucide-react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { api } from '../services/api';
import { SystemStatus, TrainingState } from '../types';

interface TrainingCenterProps {
  status: SystemStatus | null;
  onTrainingUpdated?: () => void;
}

export const TrainingCenter: React.FC<TrainingCenterProps> = ({ status, onTrainingUpdated }) => {
  const [epochs, setEpochs] = useState<number>(15);
  const [batchSize, setBatchSize] = useState<number>(32);
  const [learningRate, setLearningRate] = useState<number>(0.001);
  const [weightDecay, setWeightDecay] = useState<number>(0.0001);

  const [trainingState, setTrainingState] = useState<TrainingState | null>(status?.training_state || null);
  const [isSubmitting, setIsSubmitting] = useState<boolean>(false);
  const [notification, setNotification] = useState<string | null>(null);

  // Poll training status if active
  useEffect(() => {
    let interval: any = null;
    if (trainingState?.status === 'training' || status?.is_training) {
      interval = setInterval(async () => {
        try {
          const st = await api.getTrainingStatus();
          setTrainingState(st);
          if (st.status !== 'training') {
            if (onTrainingUpdated) onTrainingUpdated();
          }
        } catch (err) {
          console.error(err);
        }
      }, 1000);
    } else {
      if (status?.training_state) {
        setTrainingState(status.training_state);
      }
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [trainingState?.status, status?.is_training]);

  const handleStartTraining = async () => {
    setIsSubmitting(true);
    try {
      await api.startTraining({
        epochs,
        batch_size: batchSize,
        learning_rate: learningRate,
        weight_decay: weightDecay,
      });
      const st = await api.getTrainingStatus();
      setTrainingState(st);
      setNotification(`Training initiated for ${epochs} epochs.`);
      if (onTrainingUpdated) onTrainingUpdated();
    } catch (err: any) {
      setNotification(`Error: ${err.message}`);
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleStopTraining = async () => {
    try {
      await api.stopTraining();
      setNotification('Stop signal sent to trainer.');
      const st = await api.getTrainingStatus();
      setTrainingState(st);
    } catch (err: any) {
      setNotification(`Error: ${err.message}`);
    }
  };

  const history = trainingState?.history || [];
  const isTraining = trainingState?.status === 'training';

  return (
    <div className="space-y-6">
      {/* Header Banner */}
      <div className="glass-panel p-6 rounded-2xl border border-slate-800">
        <div className="flex items-center space-x-2 text-indigo-400 text-xs font-semibold uppercase tracking-wider">
          <PlayCircle className="h-4 w-4 text-cyan-400" />
          <span>Model Training Center</span>
        </div>
        <h2 className="text-2xl font-bold text-white mt-1">Convergence Analytics & Optimizer Controls</h2>
        <p className="text-slate-400 text-xs sm:text-sm mt-1 max-w-3xl">
          Configure hyperparameters, train the Transformer language model using AdamW and Cross-Entropy loss, and observe real-time loss reduction, validation perplexity, and automatic checkpointing.
        </p>

        {notification && (
          <div className="mt-3 px-3 py-2 rounded-xl bg-indigo-500/10 border border-indigo-500/30 text-indigo-300 text-xs flex items-center space-x-2">
            <CheckCircle2 className="h-4 w-4 text-emerald-400 shrink-0" />
            <span>{notification}</span>
          </div>
        )}
      </div>

      {/* Hyperparameter Controls and Training Actions */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left 5 Cols: Hyperparameter Controls */}
        <div className="lg:col-span-5 glass-panel p-6 rounded-2xl border border-slate-800 space-y-5">
          <div className="flex items-center justify-between border-b border-slate-800 pb-3">
            <h3 className="text-sm font-semibold text-white uppercase tracking-wider flex items-center space-x-2">
              <Sliders className="h-4 w-4 text-cyan-400" />
              <span>Hyperparameter Tuning</span>
            </h3>
            <span className="text-xs text-slate-400">AdamW Optimizer</span>
          </div>

          <div className="space-y-4 text-xs">
            {/* Number of Epochs */}
            <div className="space-y-1.5">
              <div className="flex justify-between items-center">
                <label className="text-slate-300 font-medium">Training Epochs:</label>
                <span className="font-mono text-indigo-400 font-bold">{epochs}</span>
              </div>
              <input
                type="range"
                min={1}
                max={50}
                step={1}
                value={epochs}
                disabled={isTraining}
                onChange={(e) => setEpochs(Number(e.target.value))}
                className="w-full accent-indigo-500"
              />
              <div className="flex justify-between text-[10px] text-slate-400 font-mono">
                <span>1 epoch (Fast)</span>
                <span>25 epochs</span>
                <span>50 epochs (Deep)</span>
              </div>
            </div>

            {/* Batch Size */}
            <div className="space-y-1.5">
              <div className="flex justify-between items-center">
                <label className="text-slate-300 font-medium">Batch Size:</label>
                <span className="font-mono text-cyan-400 font-bold">{batchSize}</span>
              </div>
              <select
                value={batchSize}
                disabled={isTraining}
                onChange={(e) => setBatchSize(Number(e.target.value))}
                className="w-full rounded-xl bg-slate-950 border border-slate-800 px-3 py-2 text-xs font-mono text-slate-200 focus:outline-none focus:border-indigo-500"
              >
                <option value={8}>8 sequences / batch</option>
                <option value={16}>16 sequences / batch</option>
                <option value={32}>32 sequences / batch (Recommended)</option>
                <option value={64}>64 sequences / batch</option>
              </select>
            </div>

            {/* Learning Rate */}
            <div className="space-y-1.5">
              <div className="flex justify-between items-center">
                <label className="text-slate-300 font-medium">Learning Rate ($\eta$):</label>
                <span className="font-mono text-emerald-400 font-bold">{learningRate}</span>
              </div>
              <select
                value={learningRate}
                disabled={isTraining}
                onChange={(e) => setLearningRate(Number(e.target.value))}
                className="w-full rounded-xl bg-slate-950 border border-slate-800 px-3 py-2 text-xs font-mono text-slate-200 focus:outline-none focus:border-indigo-500"
              >
                <option value={0.0003}>0.0003 (Conservative)</option>
                <option value={0.0005}>0.0005 (Fine-tuning)</option>
                <option value={0.001}>0.0010 (Standard AdamW)</option>
                <option value={0.002}>0.0020 (Aggressive)</option>
              </select>
            </div>

            {/* Weight Decay */}
            <div className="space-y-1.5">
              <div className="flex justify-between items-center">
                <label className="text-slate-300 font-medium">Weight Decay ($\lambda$):</label>
                <span className="font-mono text-white font-bold">{weightDecay}</span>
              </div>
              <input
                type="number"
                step="0.0001"
                min="0"
                max="0.01"
                value={weightDecay}
                disabled={isTraining}
                onChange={(e) => setWeightDecay(Number(e.target.value))}
                className="w-full rounded-xl bg-slate-950 border border-slate-800 px-3 py-2 text-xs font-mono text-slate-200 focus:outline-none focus:border-indigo-500"
              />
            </div>
          </div>

          {/* Action Buttons */}
          <div className="pt-2 flex gap-3">
            {!isTraining ? (
              <button
                onClick={handleStartTraining}
                disabled={isSubmitting}
                className="flex-1 py-3 px-4 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white font-medium text-xs flex items-center justify-center space-x-2 shadow-lg shadow-indigo-600/30 transition-all hover:scale-[1.01]"
              >
                <PlayCircle className="h-4 w-4" />
                <span>Start Model Training</span>
              </button>
            ) : (
              <button
                onClick={handleStopTraining}
                className="flex-1 py-3 px-4 rounded-xl bg-red-600 hover:bg-red-500 text-white font-medium text-xs flex items-center justify-center space-x-2 shadow-lg shadow-red-600/30 transition-all"
              >
                <PauseCircle className="h-4 w-4" />
                <span>Stop Training Early</span>
              </button>
            )}
          </div>
        </div>

        {/* Right 7 Cols: Real-Time Telemetry & Checkpoint Status */}
        <div className="lg:col-span-7 glass-panel p-6 rounded-2xl border border-slate-800 space-y-5">
          <div className="flex items-center justify-between border-b border-slate-800 pb-3">
            <h3 className="text-sm font-semibold text-white uppercase tracking-wider flex items-center space-x-2">
              <Activity className="h-4 w-4 text-emerald-400" />
              <span>Live Training Progress & Checkpoint</span>
            </h3>
            <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-slate-900 border border-slate-800 text-slate-300">
              {trainingState?.status?.toUpperCase() || 'IDLE'}
            </span>
          </div>

          {/* Progress bar */}
          <div className="space-y-2">
            <div className="flex justify-between items-center text-xs text-slate-400 font-mono">
              <span>
                Progress: <strong className="text-white">{trainingState?.progress_percent ?? 0}%</strong>
              </span>
              <span>
                Epoch {trainingState?.current_epoch ?? 0} / {trainingState?.total_epochs ?? epochs}
              </span>
            </div>
            <div className="h-2.5 w-full bg-slate-900 rounded-full overflow-hidden border border-slate-800">
              <div
                className={`h-full transition-all duration-300 ${
                  isTraining
                    ? 'bg-gradient-to-r from-indigo-500 via-cyan-400 to-emerald-400'
                    : 'bg-indigo-600'
                }`}
                style={{ width: `${Math.max(2, trainingState?.progress_percent ?? 0)}%` }}
              />
            </div>
          </div>

          {/* Real-Time Stats Grid */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs font-mono">
            <div className="p-3 rounded-xl bg-slate-950 border border-slate-800">
              <span className="text-slate-400 text-[10px] block">Current Train Loss</span>
              <span className="text-base font-bold text-white">
                {trainingState?.current_train_loss?.toFixed(4) ?? 'N/A'}
              </span>
            </div>

            <div className="p-3 rounded-xl bg-slate-950 border border-slate-800">
              <span className="text-slate-400 text-[10px] block">Current Val Loss</span>
              <span className="text-base font-bold text-cyan-300">
                {trainingState?.current_val_loss?.toFixed(4) ?? 'N/A'}
              </span>
            </div>

            <div className="p-3 rounded-xl bg-slate-950 border border-slate-800">
              <span className="text-slate-400 text-[10px] block">Best Val Loss</span>
              <span className="text-base font-bold text-emerald-400">
                {trainingState?.best_val_loss !== Infinity
                  ? trainingState?.best_val_loss?.toFixed(4)
                  : 'N/A'}
              </span>
            </div>

            <div className="p-3 rounded-xl bg-slate-950 border border-slate-800">
              <span className="text-slate-400 text-[10px] block">Elapsed Time</span>
              <span className="text-base font-bold text-amber-300">
                {trainingState?.elapsed_seconds ?? 0}s
              </span>
            </div>
          </div>

          {/* Model Checkpoint Status */}
          <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="p-2 rounded-lg bg-emerald-500/10 border border-emerald-500/20 text-emerald-400">
                <Save className="h-4 w-4" />
              </div>
              <div>
                <span className="text-xs font-semibold text-white block">
                  Model Checkpoint (`backend/outputs/best_model.pt`)
                </span>
                <span className="text-[11px] text-slate-400">
                  Automatically persists best weights & vocabulary mapping
                </span>
              </div>
            </div>
            <span className="px-2.5 py-1 rounded-full bg-emerald-500/10 text-emerald-300 border border-emerald-500/30 text-[11px] font-mono">
              Active Checkpoint
            </span>
          </div>
        </div>
      </div>

      {/* Loss Curves & Convergence Analytics (Recharts) */}
      <div className="glass-panel p-6 rounded-2xl border border-slate-800 space-y-4">
        <div className="flex items-center justify-between border-b border-slate-800 pb-3">
          <div>
            <h3 className="text-sm font-semibold text-white uppercase tracking-wider flex items-center space-x-2">
              <TrendingDown className="h-4 w-4 text-indigo-400" />
              <span>Training & Validation Loss Curves</span>
            </h3>
            <p className="text-xs text-slate-400 mt-0.5">
              Cross-Entropy loss minimized across training epochs
            </p>
          </div>
        </div>

        {history.length > 0 ? (
          <div className="h-72 w-full pt-2">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={history} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="epoch" stroke="#64748b" tick={{ fill: '#64748b', fontSize: 11 }} />
                <YAxis stroke="#64748b" tick={{ fill: '#64748b', fontSize: 11 }} domain={['auto', 'auto']} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#0f172a',
                    borderColor: '#334155',
                    borderRadius: '0.75rem',
                    color: '#f8fafc',
                    fontSize: '11px',
                    fontFamily: 'monospace',
                  }}
                />
                <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '10px' }} />
                <Line
                  type="monotone"
                  dataKey="train_loss"
                  name="Training Loss"
                  stroke="#818cf8"
                  strokeWidth={2.5}
                  dot={{ r: 3, fill: '#818cf8' }}
                  activeDot={{ r: 5 }}
                />
                <Line
                  type="monotone"
                  dataKey="val_loss"
                  name="Validation Loss"
                  stroke="#06b6d4"
                  strokeWidth={2.5}
                  dot={{ r: 3, fill: '#06b6d4' }}
                  activeDot={{ r: 5 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        ) : (
          <div className="h-48 flex items-center justify-center text-slate-400 text-xs">
            Start training above to visualize real loss curves.
          </div>
        )}
      </div>

      {/* Epochs History Table */}
      {history.length > 0 && (
        <div className="glass-panel p-6 rounded-2xl border border-slate-800 space-y-4">
          <div className="border-b border-slate-800 pb-3">
            <h3 className="text-sm font-semibold text-white uppercase tracking-wider">
              Complete Epochs Execution Log
            </h3>
          </div>

          <div className="overflow-x-auto max-h-80 overflow-y-auto">
            <table className="w-full text-left text-xs font-mono">
              <thead className="sticky top-0 bg-slate-950 text-slate-400 border-b border-slate-800 text-[11px]">
                <tr>
                  <th className="pb-2.5">Epoch</th>
                  <th className="pb-2.5">Train Loss</th>
                  <th className="pb-2.5">Val Loss</th>
                  <th className="pb-2.5">Perplexity</th>
                  <th className="pb-2.5">Top-1 Accuracy</th>
                  <th className="pb-2.5">Learning Rate</th>
                  <th className="pb-2.5">Elapsed Time</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/40 text-slate-300">
                {history.map((h) => (
                  <tr key={h.epoch} className="hover:bg-slate-800/30">
                    <td className="py-2.5 font-bold text-indigo-400">Epoch #{h.epoch}</td>
                    <td className="py-2.5">{h.train_loss.toFixed(4)}</td>
                    <td className="py-2.5 text-cyan-300 font-semibold">{h.val_loss.toFixed(4)}</td>
                    <td className="py-2.5">{h.val_perplexity.toFixed(1)}</td>
                    <td className="py-2.5 text-emerald-400">{h.val_top1_acc.toFixed(1)}%</td>
                    <td className="py-2.5 text-slate-400">{h.learning_rate}</td>
                    <td className="py-2.5 text-slate-400">{h.elapsed_time}s</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};
