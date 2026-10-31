import React from 'react';
import {
  Activity,
  Zap,
  BookOpen,
  Layers,
  Database,
  ArrowRight,
  TrendingDown,
  Sparkles,
  ShieldCheck,
  CheckCircle2,
  Clock,
  Cpu,
} from 'lucide-react';
import { SystemStatus } from '../types';

interface DashboardProps {
  status: SystemStatus | null;
  setActiveTab: (tab: string) => void;
}

export const Dashboard: React.FC<DashboardProps> = ({ status, setActiveTab }) => {
  const evalMetrics = status?.latest_evaluation;
  const trainState = status?.training_state;
  const modelParams = status?.model_parameters;

  const totalEpochsTrained = trainState?.history?.length ?? 0;
  const testLoss = evalMetrics?.cross_entropy_loss ?? (trainState?.best_val_loss !== Infinity ? trainState?.best_val_loss : null);
  const perplexity = evalMetrics?.perplexity ?? (testLoss ? Math.round(Math.exp(Math.min(testLoss, 20)) * 100) / 100 : null);
  const top1Accuracy = evalMetrics?.top1_accuracy ?? null;
  const top5Accuracy = evalMetrics?.top5_accuracy ?? null;

  return (
    <div className="space-y-6">
      {/* Hero Welcome Banner */}
      <div className="relative overflow-hidden rounded-2xl p-6 sm:p-8 bg-gradient-to-r from-slate-900 via-indigo-950/60 to-slate-900 border border-indigo-500/20 shadow-2xl">
        <div className="absolute -right-12 -top-12 h-64 w-64 rounded-full bg-indigo-500/10 blur-3xl pointer-events-none" />
        <div className="absolute right-32 -bottom-12 h-64 w-64 rounded-full bg-cyan-500/10 blur-3xl pointer-events-none" />

        <div className="relative z-10 flex flex-col md:flex-row md:items-center justify-between gap-6">
          <div className="max-w-2xl">
            <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-indigo-500/10 border border-indigo-500/30 text-indigo-300 text-xs font-semibold mb-3">
              <Sparkles className="h-3.5 w-3.5 text-cyan-400" />
              <span>Autoregressive Decoder Language Model</span>
            </div>
            <h1 className="text-3xl sm:text-4xl font-extrabold text-white tracking-tight leading-tight">
              Transformer Intelligence Studio
            </h1>
            <p className="mt-2 text-sm sm:text-base text-slate-300 leading-relaxed">
              Research and analyze self-attention mechanisms, causal masking, and language modeling from scratch. Built with pure PyTorch tensor operations—no external LLM APIs, no black boxes.
            </p>
          </div>

          <div className="flex flex-wrap sm:flex-nowrap gap-3">
            <button
              onClick={() => setActiveTab('prediction')}
              className="flex items-center space-x-2 px-5 py-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white font-medium text-xs sm:text-sm shadow-lg shadow-indigo-600/30 transition-all hover:scale-[1.02] active:scale-[0.98]"
            >
              <Zap className="h-4 w-4" />
              <span>Token Playground</span>
              <ArrowRight className="h-4 w-4" />
            </button>
            <button
              onClick={() => setActiveTab('attention')}
              className="flex items-center space-x-2 px-4 py-2.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 font-medium text-xs sm:text-sm transition-all"
            >
              <Layers className="h-4 w-4 text-cyan-400" />
              <span>Inspect Attention</span>
            </button>
          </div>
        </div>
      </div>

      {/* Primary KPI Grid (8 Key Metrics) */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {/* Metric 1: Model Status */}
        <div className="glass-card p-4 rounded-xl border border-slate-800/80">
          <div className="flex items-center justify-between text-slate-400 text-xs font-medium">
            <span>Model Status</span>
            <ShieldCheck className="h-4 w-4 text-emerald-400" />
          </div>
          <div className="mt-2 flex items-baseline space-x-2">
            <span className="text-xl font-bold text-white capitalize">
              {trainState?.status === 'training' ? 'Training...' : totalEpochsTrained > 0 ? 'Trained' : 'Initialized'}
            </span>
            <span className="text-xs text-emerald-400 font-medium flex items-center">
              <CheckCircle2 className="h-3 w-3 mr-0.5 inline" /> Ready
            </span>
          </div>
          <p className="mt-1 text-[11px] text-slate-400 font-mono">
            {modelParams?.total_parameters?.toLocaleString() ?? '185,000'} params
          </p>
        </div>

        {/* Metric 2: Vocabulary Size */}
        <div className="glass-card p-4 rounded-xl border border-slate-800/80">
          <div className="flex items-center justify-between text-slate-400 text-xs font-medium">
            <span>Vocabulary Size</span>
            <Database className="h-4 w-4 text-indigo-400" />
          </div>
          <div className="mt-2 flex items-baseline space-x-2">
            <span className="text-xl font-bold text-white font-mono">
              {status?.vocab_size ?? 0}
            </span>
            <span className="text-xs text-slate-400">tokens</span>
          </div>
          <p className="mt-1 text-[11px] text-slate-400">
            Includes &lt;pad&gt;, &lt;unk&gt;, &lt;eos&gt;
          </p>
        </div>

        {/* Metric 3: Training Dataset Size */}
        <div className="glass-card p-4 rounded-xl border border-slate-800/80">
          <div className="flex items-center justify-between text-slate-400 text-xs font-medium">
            <span>Training Dataset</span>
            <BookOpen className="h-4 w-4 text-cyan-400" />
          </div>
          <div className="mt-2 flex items-baseline space-x-2">
            <span className="text-xl font-bold text-white font-mono">
              {status?.dataset_sizes?.train ?? 0}
            </span>
            <span className="text-xs text-slate-400">sequences</span>
          </div>
          <p className="mt-1 text-[11px] text-slate-400">
            Val: {status?.dataset_sizes?.val ?? 0} | Test: {status?.dataset_sizes?.test ?? 0}
          </p>
        </div>

        {/* Metric 4: Epochs Trained */}
        <div className="glass-card p-4 rounded-xl border border-slate-800/80">
          <div className="flex items-center justify-between text-slate-400 text-xs font-medium">
            <span>Epochs Trained</span>
            <Clock className="h-4 w-4 text-amber-400" />
          </div>
          <div className="mt-2 flex items-baseline space-x-2">
            <span className="text-xl font-bold text-white font-mono">
              {totalEpochsTrained}
            </span>
            <span className="text-xs text-amber-400 font-medium">
              {trainState?.status === 'training' ? `of ${trainState.total_epochs}` : 'completed'}
            </span>
          </div>
          <p className="mt-1 text-[11px] text-slate-400">
            AdamW optimizer &bull; lr 0.001
          </p>
        </div>

        {/* Metric 5: Test Loss */}
        <div className="glass-card p-4 rounded-xl border border-slate-800/80">
          <div className="flex items-center justify-between text-slate-400 text-xs font-medium">
            <span>Cross-Entropy Loss</span>
            <TrendingDown className="h-4 w-4 text-indigo-400" />
          </div>
          <div className="mt-2 flex items-baseline space-x-2">
            <span className="text-xl font-bold text-white font-mono">
              {testLoss !== null ? testLoss.toFixed(3) : 'Pending'}
            </span>
            <span className="text-xs text-slate-400">nats/token</span>
          </div>
          <p className="mt-1 text-[11px] text-slate-400">
            Held-out validation loss
          </p>
        </div>

        {/* Metric 6: Perplexity */}
        <div className="glass-card p-4 rounded-xl border border-slate-800/80">
          <div className="flex items-center justify-between text-slate-400 text-xs font-medium">
            <span>Perplexity (PPL)</span>
            <Activity className="h-4 w-4 text-cyan-400" />
          </div>
          <div className="mt-2 flex items-baseline space-x-2">
            <span className="text-xl font-bold text-white font-mono">
              {perplexity !== null ? perplexity.toFixed(1) : 'Pending'}
            </span>
            <span className="text-xs text-cyan-400 font-mono text-[10px]">exp(loss)</span>
          </div>
          <p className="mt-1 text-[11px] text-slate-400">
            Effective branching factor
          </p>
        </div>

        {/* Metric 7: Top-1 Accuracy */}
        <div className="glass-card p-4 rounded-xl border border-slate-800/80">
          <div className="flex items-center justify-between text-slate-400 text-xs font-medium">
            <span>Top-1 Accuracy</span>
            <Sparkles className="h-4 w-4 text-emerald-400" />
          </div>
          <div className="mt-2 flex items-baseline space-x-2">
            <span className="text-xl font-bold text-emerald-400 font-mono">
              {top1Accuracy !== null ? `${top1Accuracy.toFixed(1)}%` : 'Evaluated'}
            </span>
            <span className="text-xs text-slate-400">exact match</span>
          </div>
          <p className="mt-1 text-[11px] text-slate-400">
            Target token = argmax(logits)
          </p>
        </div>

        {/* Metric 8: Top-5 Accuracy */}
        <div className="glass-card p-4 rounded-xl border border-slate-800/80">
          <div className="flex items-center justify-between text-slate-400 text-xs font-medium">
            <span>Top-5 Accuracy</span>
            <CheckCircle2 className="h-4 w-4 text-violet-400" />
          </div>
          <div className="mt-2 flex items-baseline space-x-2">
            <span className="text-xl font-bold text-violet-400 font-mono">
              {top5Accuracy !== null ? `${top5Accuracy.toFixed(1)}%` : 'Evaluated'}
            </span>
            <span className="text-xs text-slate-400">in top 5</span>
          </div>
          <p className="mt-1 text-[11px] text-slate-400">
            Target in top 5 candidates
          </p>
        </div>
      </div>

      {/* Grid: Architecture Specs & Latest Experiment */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left 2 Cols: Latest Training Experiment & Live Progress */}
        <div className="lg:col-span-2 glass-panel p-6 rounded-2xl border border-slate-800 space-y-5">
          <div className="flex items-center justify-between border-b border-slate-800 pb-4">
            <div>
              <h2 className="text-lg font-bold text-white flex items-center space-x-2">
                <Activity className="h-5 w-5 text-indigo-400" />
                <span>Latest Training Experiment & Telemetry</span>
              </h2>
              <p className="text-xs text-slate-400 mt-0.5">
                Real-time tracking of convergence, loss minimization, and model checkpointing
              </p>
            </div>
            <button
              onClick={() => setActiveTab('training')}
              className="text-xs text-indigo-400 hover:text-indigo-300 font-medium flex items-center space-x-1"
            >
              <span>Training Studio</span>
              <ArrowRight className="h-3.5 w-3.5" />
            </button>
          </div>

          {trainState && (
            <div className="space-y-4">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 p-4 rounded-xl bg-slate-900/80 border border-slate-800">
                <div>
                  <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
                    Current Experiment Status
                  </div>
                  <div className="text-sm font-medium text-slate-200 mt-1">
                    {trainState.message || 'Model ready for training.'}
                  </div>
                </div>
                <div className="flex items-center space-x-3 text-xs font-mono">
                  <div className="text-right">
                    <span className="text-slate-400 block text-[10px]">Best Val Loss</span>
                    <span className="text-emerald-400 font-bold text-sm">
                      {trainState.best_val_loss !== Infinity ? trainState.best_val_loss.toFixed(4) : 'N/A'}
                    </span>
                  </div>
                  <div className="text-right border-l border-slate-800 pl-3">
                    <span className="text-slate-400 block text-[10px]">Best Epoch</span>
                    <span className="text-indigo-300 font-bold text-sm">
                      #{trainState.best_epoch || 1}
                    </span>
                  </div>
                </div>
              </div>

              {/* Progress bar if training or history exists */}
              <div>
                <div className="flex items-center justify-between text-xs text-slate-400 mb-1.5 font-medium">
                  <span>Training Progress ({trainState.progress_percent}%)</span>
                  <span>
                    Epoch {trainState.current_epoch} / {trainState.total_epochs || totalEpochsTrained || 15}
                  </span>
                </div>
                <div className="h-2 w-full bg-slate-800 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-indigo-500 to-cyan-400 transition-all duration-300"
                    style={{ width: `${Math.max(5, trainState.progress_percent)}%` }}
                  />
                </div>
              </div>

              {/* Recent Epochs Quick Table */}
              {trainState.history && trainState.history.length > 0 && (
                <div className="overflow-x-auto">
                  <table className="w-full text-left text-xs font-mono">
                    <thead>
                      <tr className="text-slate-400 border-b border-slate-800 text-[11px]">
                        <th className="pb-2">Epoch</th>
                        <th className="pb-2">Train Loss</th>
                        <th className="pb-2">Val Loss</th>
                        <th className="pb-2">Perplexity</th>
                        <th className="pb-2">Top-1 Acc</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800/40 text-slate-300">
                      {trainState.history.slice(-4).map((h) => (
                        <tr key={h.epoch} className="hover:bg-slate-800/30">
                          <td className="py-2 text-indigo-400 font-semibold">#{h.epoch}</td>
                          <td className="py-2">{h.train_loss.toFixed(4)}</td>
                          <td className="py-2 text-cyan-300">{h.val_loss.toFixed(4)}</td>
                          <td className="py-2">{h.val_perplexity.toFixed(1)}</td>
                          <td className="py-2 text-emerald-400">{h.val_top1_acc.toFixed(1)}%</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Right 1 Col: Model Specifications & Quick Links */}
        <div className="glass-panel p-6 rounded-2xl border border-slate-800 space-y-5">
          <div className="border-b border-slate-800 pb-4">
            <h2 className="text-lg font-bold text-white flex items-center space-x-2">
              <Cpu className="h-5 w-5 text-cyan-400" />
              <span>Model Architecture</span>
            </h2>
            <p className="text-xs text-slate-400 mt-0.5">
              PyTorch Decoder-Only Architecture
            </p>
          </div>

          <div className="space-y-3 text-xs">
            <div className="flex justify-between items-center py-1.5 border-b border-slate-800/60">
              <span className="text-slate-400">Embedding Dimension ($d_{'model'}$):</span>
              <span className="font-mono text-white font-semibold">128</span>
            </div>
            <div className="flex justify-between items-center py-1.5 border-b border-slate-800/60">
              <span className="text-slate-400">Attention Heads ($H$):</span>
              <span className="font-mono text-cyan-300 font-semibold">4 ($d_k = 32$)</span>
            </div>
            <div className="flex justify-between items-center py-1.5 border-b border-slate-800/60">
              <span className="text-slate-400">Transformer Blocks ($N$):</span>
              <span className="font-mono text-indigo-300 font-semibold">2 Layers</span>
            </div>
            <div className="flex justify-between items-center py-1.5 border-b border-slate-800/60">
              <span className="text-slate-400">Feed-Forward Dim ($d_{'ff'}$):</span>
              <span className="font-mono text-white font-semibold">512 (GELU)</span>
            </div>
            <div className="flex justify-between items-center py-1.5 border-b border-slate-800/60">
              <span className="text-slate-400">Max Sequence Window:</span>
              <span className="font-mono text-white font-semibold">32 tokens</span>
            </div>
            <div className="flex justify-between items-center py-1.5 border-b border-slate-800/60">
              <span className="text-slate-400">Causal Masking:</span>
              <span className="font-mono text-emerald-400 font-semibold">Strict Lower-Tri</span>
            </div>
            <div className="flex justify-between items-center py-1.5 border-b border-slate-800/60">
              <span className="text-slate-400">Optimizer:</span>
              <span className="font-mono text-white font-semibold">AdamW (lr=1e-3)</span>
            </div>
          </div>

          <div className="pt-2">
            <button
              onClick={() => setActiveTab('architecture')}
              className="w-full flex items-center justify-center space-x-2 py-2 px-3 rounded-xl bg-slate-800 hover:bg-slate-700 text-indigo-300 border border-slate-700 text-xs font-medium transition-all"
            >
              <span>Explore Interactive Architecture</span>
              <ArrowRight className="h-3.5 w-3.5" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
