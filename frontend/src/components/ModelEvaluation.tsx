import React, { useState, useEffect } from 'react';
import {
  BarChart3,
  TrendingDown,
  Activity,
  CheckCircle2,
  XCircle,
  RefreshCw,
  Sparkles,
  HelpCircle,
  ShieldCheck,
} from 'lucide-react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts';
import { api } from '../services/api';
import { ModelEvaluation } from '../types';

interface ModelEvaluationProps {
  evaluation: ModelEvaluation | null;
  onEvaluationUpdated?: (evalData: ModelEvaluation) => void;
}

export const ModelEvaluationView: React.FC<ModelEvaluationProps> = ({
  evaluation: initialEvaluation,
  onEvaluationUpdated,
}) => {
  const [evaluation, setEvaluation] = useState<ModelEvaluation | null>(initialEvaluation);
  const [evaluating, setEvaluating] = useState<boolean>(false);
  const [notification, setNotification] = useState<string | null>(null);

  useEffect(() => {
    if (initialEvaluation) {
      setEvaluation(initialEvaluation);
    } else {
      runEvaluation();
    }
  }, [initialEvaluation]);

  const runEvaluation = async () => {
    setEvaluating(true);
    try {
      const res = await api.evaluateModel();
      setEvaluation(res);
      setNotification('Held-out test split evaluation completed.');
      if (onEvaluationUpdated) onEvaluationUpdated(res);
    } catch (err: any) {
      setNotification(`Error: ${err.message}`);
    } finally {
      setEvaluating(false);
    }
  };

  const accuracyData = evaluation
    ? [
        { name: 'Top-1 Acc', value: evaluation.top1_accuracy, fill: '#818cf8' },
        { name: 'Top-3 Acc', value: evaluation.top3_accuracy, fill: '#06b6d4' },
        { name: 'Top-5 Acc', value: evaluation.top5_accuracy, fill: '#10b981' },
      ]
    : [];

  return (
    <div className="space-y-6">
      {/* Header Banner */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 glass-panel p-6 rounded-2xl border border-slate-800">
        <div>
          <div className="flex items-center space-x-2 text-emerald-400 text-xs font-semibold uppercase tracking-wider">
            <BarChart3 className="h-4 w-4 text-cyan-400" />
            <span>Held-Out Model Evaluation</span>
          </div>
          <h2 className="text-2xl font-bold text-white mt-1">Quantitative Generalization Benchmarks</h2>
          <p className="text-slate-400 text-xs sm:text-sm mt-1 max-w-2xl">
            Calculated strictly on the held-out test split (10% of sequences unseen during training gradient updates) to measure true linguistic generalization, perplexity, and ranking accuracy.
          </p>
        </div>

        <button
          onClick={runEvaluation}
          disabled={evaluating}
          className="px-5 py-2.5 rounded-xl bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-medium flex items-center justify-center space-x-2 shadow-lg shadow-emerald-600/20 shrink-0 transition-all hover:scale-[1.02]"
        >
          {evaluating ? <RefreshCw className="h-4 w-4 animate-spin" /> : <RefreshCw className="h-4 w-4" />}
          <span>Re-Evaluate Test Set</span>
        </button>
      </div>

      {notification && (
        <div className="px-3 py-2 rounded-xl bg-emerald-500/10 border border-emerald-500/30 text-emerald-300 text-xs flex items-center space-x-2">
          <CheckCircle2 className="h-4 w-4 text-emerald-400 shrink-0" />
          <span>{notification}</span>
        </div>
      )}

      {evaluation ? (
        <>
          {/* Key Metrics Grid (5 KPI Cards) */}
          <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
            {/* Card 1: Cross-Entropy Loss */}
            <div className="glass-card p-4 rounded-xl border border-slate-800">
              <div className="flex items-center justify-between text-slate-400 text-xs font-medium">
                <span>Test Loss (CE)</span>
                <TrendingDown className="h-4 w-4 text-indigo-400" />
              </div>
              <div className="mt-2 text-xl font-bold text-white font-mono">
                {evaluation.cross_entropy_loss.toFixed(4)}
              </div>
              <p className="mt-1 text-[11px] text-slate-400 font-mono">nats / token</p>
            </div>

            {/* Card 2: Perplexity */}
            <div className="glass-card p-4 rounded-xl border border-slate-800">
              <div className="flex items-center justify-between text-slate-400 text-xs font-medium">
                <span>Perplexity</span>
                <Activity className="h-4 w-4 text-cyan-400" />
              </div>
              <div className="mt-2 text-xl font-bold text-cyan-300 font-mono">
                {evaluation.perplexity.toFixed(2)}
              </div>
              <p className="mt-1 text-[11px] text-slate-400 font-mono">exp(CE loss)</p>
            </div>

            {/* Card 3: Top-1 Accuracy */}
            <div className="glass-card p-4 rounded-xl border border-slate-800">
              <div className="flex items-center justify-between text-slate-400 text-xs font-medium">
                <span>Top-1 Accuracy</span>
                <Sparkles className="h-4 w-4 text-indigo-400" />
              </div>
              <div className="mt-2 text-xl font-bold text-indigo-400 font-mono">
                {evaluation.top1_accuracy.toFixed(1)}%
              </div>
              <p className="mt-1 text-[11px] text-slate-400">Exact next token</p>
            </div>

            {/* Card 4: Top-3 Accuracy */}
            <div className="glass-card p-4 rounded-xl border border-slate-800">
              <div className="flex items-center justify-between text-slate-400 text-xs font-medium">
                <span>Top-3 Accuracy</span>
                <CheckCircle2 className="h-4 w-4 text-cyan-400" />
              </div>
              <div className="mt-2 text-xl font-bold text-cyan-400 font-mono">
                {evaluation.top3_accuracy.toFixed(1)}%
              </div>
              <p className="mt-1 text-[11px] text-slate-400">In top 3 candidates</p>
            </div>

            {/* Card 5: Top-5 Accuracy */}
            <div className="glass-card p-4 rounded-xl border border-slate-800">
              <div className="flex items-center justify-between text-slate-400 text-xs font-medium">
                <span>Top-5 Accuracy</span>
                <ShieldCheck className="h-4 w-4 text-emerald-400" />
              </div>
              <div className="mt-2 text-xl font-bold text-emerald-400 font-mono">
                {evaluation.top5_accuracy.toFixed(1)}%
              </div>
              <p className="mt-1 text-[11px] text-slate-400">In top 5 candidates</p>
            </div>
          </div>

          {/* Performance Graph & Accuracy Breakdown */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            {/* Left 6 Cols: Top-K Accuracy Bar Chart */}
            <div className="lg:col-span-6 glass-panel p-6 rounded-2xl border border-slate-800 space-y-4">
              <div className="border-b border-slate-800 pb-3">
                <h3 className="text-sm font-semibold text-white uppercase tracking-wider">
                  Top-K Next-Token Retrieval Accuracy (%)
                </h3>
                <p className="text-xs text-slate-400 mt-0.5">
                  Percentage of true ground truth tokens appearing within the model's top K predictions
                </p>
              </div>

              <div className="h-64 w-full pt-2">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={accuracyData} margin={{ top: 10, right: 20, left: 0, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                    <XAxis dataKey="name" stroke="#64748b" tick={{ fill: '#64748b', fontSize: 11 }} />
                    <YAxis domain={[0, 100]} stroke="#64748b" tick={{ fill: '#64748b', fontSize: 11 }} />
                    <Tooltip
                      formatter={(val: any) => [`${val}%`, 'Accuracy']}
                      contentStyle={{
                        backgroundColor: '#0f172a',
                        borderColor: '#334155',
                        borderRadius: '0.75rem',
                        color: '#f8fafc',
                        fontSize: '11px',
                        fontFamily: 'monospace',
                      }}
                    />
                    <Bar dataKey="value" radius={[6, 6, 0, 0]}>
                      {accuracyData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.fill} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Right 6 Cols: Metrics Explanation */}
            <div className="lg:col-span-6 glass-panel p-6 rounded-2xl border border-slate-800 space-y-4">
              <div className="border-b border-slate-800 pb-3">
                <h3 className="text-sm font-semibold text-white uppercase tracking-wider flex items-center space-x-2">
                  <HelpCircle className="h-4 w-4 text-indigo-400" />
                  <span>Evaluation Metric Interpretation</span>
                </h3>
              </div>

              <div className="space-y-3 text-xs leading-relaxed text-slate-300">
                <div className="p-3 rounded-xl bg-slate-900/60 border border-slate-800 space-y-1">
                  <strong className="text-white block font-mono">Cross-Entropy Loss:</strong>
                  <span>
                    Measures the negative log-likelihood of the ground truth next token: $\mathcal{'{L}'} = -\frac{'{1}'}{'{N}'} \sum \log P(w_t \mid w_{'{<t}'})$. Lower is strictly better.
                  </span>
                </div>

                <div className="p-3 rounded-xl bg-slate-900/60 border border-slate-800 space-y-1">
                  <strong className="text-cyan-300 block font-mono">Perplexity ($\mathcal{'{PPL}'}$):</strong>
                  <span>
                    Calculated as $\exp(\text{'{loss}'})$. A perplexity of {evaluation.perplexity.toFixed(1)} means that at each token position, the model is as uncertain as choosing uniformly among ~{Math.round(evaluation.perplexity)} equally likely candidate words.
                  </span>
                </div>

                <div className="p-3 rounded-xl bg-slate-900/60 border border-slate-800 space-y-1">
                  <strong className="text-emerald-300 block font-mono">Top-1 vs Top-5 Accuracy:</strong>
                  <span>
                    Language often has multiple valid synonyms or grammatical continuations. Top-5 accuracy ({evaluation.top5_accuracy.toFixed(1)}%) captures whether the model assigns high probability mass to plausible candidates even if another synonym was in the text.
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* Actual Test Split Prediction Examples Table */}
          {evaluation.prediction_examples && (
            <div className="glass-panel p-6 rounded-2xl border border-slate-800 space-y-4">
              <div className="border-b border-slate-800 pb-3">
                <h3 className="text-sm font-semibold text-white uppercase tracking-wider">
                  Held-Out Test Sequence Predictions (Actual vs Model Top-5)
                </h3>
                <p className="text-xs text-slate-400 mt-0.5">
                  Inspection of actual held-out sequences comparing Expected Target Token vs Model's Top-1 prediction and top-5 probability distribution
                </p>
              </div>

              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs font-mono">
                  <thead>
                    <tr className="text-slate-400 border-b border-slate-800 text-[11px]">
                      <th className="pb-2.5 w-16">Sample</th>
                      <th className="pb-2.5">Context Window ($X$)</th>
                      <th className="pb-2.5">Ground Truth ($Y$)</th>
                      <th className="pb-2.5">Model Top-1 Prediction</th>
                      <th className="pb-2.5">Top-5 Candidate Probabilities</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/40 text-slate-300">
                    {evaluation.prediction_examples.map((ex) => (
                      <tr key={ex.sample_index} className="hover:bg-slate-800/30">
                        <td className="py-3 font-bold text-indigo-400">#{ex.sample_index}</td>
                        <td className="py-3 text-slate-300">
                          <span className="px-2 py-1 rounded bg-slate-950 border border-slate-800 text-slate-300 inline-block">
                            ... {ex.context_preview}
                          </span>
                        </td>
                        <td className="py-3 font-bold text-emerald-400">
                          "{ex.expected_target}"
                        </td>
                        <td className="py-3">
                          <div className="inline-flex items-center space-x-1.5 px-2 py-1 rounded-md bg-slate-900 border border-slate-800">
                            {ex.is_correct ? (
                              <CheckCircle2 className="h-3.5 w-3.5 text-emerald-400" />
                            ) : ex.in_top5 ? (
                              <CheckCircle2 className="h-3.5 w-3.5 text-cyan-400" />
                            ) : (
                              <XCircle className="h-3.5 w-3.5 text-slate-500" />
                            )}
                            <span className={ex.is_correct ? 'text-emerald-300 font-bold' : 'text-slate-300'}>
                              "{ex.top1_prediction}"
                            </span>
                          </div>
                        </td>
                        <td className="py-3">
                          <div className="flex flex-wrap gap-1.5">
                            {ex.top5_candidates.map((c) => (
                              <span
                                key={c.id}
                                className={`px-2 py-0.5 rounded text-[10px] border ${
                                  c.token === ex.expected_target
                                    ? 'bg-emerald-500/15 border-emerald-500/40 text-emerald-300 font-bold'
                                    : 'bg-slate-900 border-slate-800 text-slate-400'
                                }`}
                              >
                                {c.token}: {c.probability}%
                              </span>
                            ))}
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </>
      ) : (
        <div className="text-center py-12 text-slate-400 text-sm glass-panel rounded-2xl border border-slate-800">
          Click above to compute held-out test evaluation.
        </div>
      )}
    </div>
  );
};
