import React, { useState, useEffect } from 'react';
import {
  Sparkles,
  Zap,
  RotateCcw,
  Sliders,
  ChevronRight,
  TrendingUp,
  Hash,
  ArrowRight,
  Play,
  Check,
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
import { NextTokenPredictionResult, TokenCandidate } from '../types';

export const PredictionPlayground: React.FC = () => {
  const [prompt, setPrompt] = useState<string>('Once upon a time in a bright green');
  const [temperature, setTemperature] = useState<number>(0.8);
  const [topK, setTopK] = useState<number>(5);
  const [greedy, setGreedy] = useState<boolean>(false);

  const [prediction, setPrediction] = useState<NextTokenPredictionResult | null>(null);
  const [predicting, setPredicting] = useState<boolean>(false);
  const [autoStepRunning, setAutoStepRunning] = useState<boolean>(false);

  const presets = [
    'Once upon a time in a bright green',
    'Oliver loved to explore the sunny',
    'A transformer is a deep neural',
    'Shall I compare thee to a summer',
    'Every small act of kindness makes',
  ];

  useEffect(() => {
    runPrediction(prompt);
  }, []);

  const runPrediction = async (text: string) => {
    if (!text.trim()) return;
    setPredicting(true);
    try {
      const res = await api.predictNextToken({
        prompt: text,
        temperature,
        top_k: topK,
        greedy,
      });
      setPrediction(res);
    } catch (err) {
      console.error(err);
    } finally {
      setPredicting(false);
    }
  };

  const handleAppendToken = (tokenToAppend?: string) => {
    const nextTok = tokenToAppend || prediction?.selected_token;
    if (!nextTok) return;

    // Append token intelligently with punctuation handling
    let newPrompt = prompt.trim();
    if (['.', ',', '!', '?', ';', ':'].includes(nextTok)) {
      newPrompt += nextTok;
    } else {
      newPrompt += ' ' + nextTok;
    }
    setPrompt(newPrompt);
    runPrediction(newPrompt);
  };

  const handleReset = () => {
    setPrompt('Once upon a time');
    runPrediction('Once upon a time');
  };

  const chartData = prediction
    ? prediction.top_candidates.map((c, idx) => ({
        token: c.token,
        probability: c.probability,
        fill: idx === 0 ? '#818cf8' : idx === 1 ? '#06b6d4' : '#64748b',
        rank: c.rank,
        is_selected: c.is_selected,
      }))
    : [];

  return (
    <div className="space-y-6">
      {/* Header Banner */}
      <div className="glass-panel p-6 rounded-2xl border border-slate-800">
        <div className="flex items-center space-x-2 text-indigo-400 text-xs font-semibold uppercase tracking-wider">
          <Sparkles className="h-4 w-4 text-cyan-400" />
          <span>Interactive Inference Studio</span>
        </div>
        <h2 className="text-2xl font-bold text-white mt-1">Next-Token Prediction Playground</h2>
        <p className="text-slate-400 text-xs sm:text-sm mt-1 max-w-3xl">
          Enter any context sentence to observe how the Transformer computes logit activations, projects through the LM output head, normalizes via Softmax, and outputs candidate probabilities.
        </p>
      </div>

      {/* Main Playground: Prompt Editor & Controls */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left 7 Cols: Prompt Editor & Preset Shortcuts */}
        <div className="lg:col-span-7 glass-panel p-6 rounded-2xl border border-slate-800 space-y-4">
          <div className="flex items-center justify-between border-b border-slate-800 pb-3">
            <h3 className="text-sm font-semibold text-white uppercase tracking-wider flex items-center space-x-2">
              <Zap className="h-4 w-4 text-amber-400" />
              <span>Context Sentence Editor</span>
            </h3>
            <button
              onClick={handleReset}
              className="text-xs text-slate-400 hover:text-slate-200 flex items-center space-x-1"
            >
              <RotateCcw className="h-3 w-3" />
              <span>Reset</span>
            </button>
          </div>

          {/* Interactive Textarea */}
          <div className="relative">
            <textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="Enter a starting sentence..."
              rows={4}
              className="w-full rounded-xl bg-slate-950 border border-slate-800 p-4 text-sm font-mono text-slate-100 focus:outline-none focus:border-indigo-500 leading-relaxed"
            />
            <div className="flex items-center justify-between px-2 pt-1 text-[11px] text-slate-400 font-mono">
              <span>Tokens in context: {prompt.trim().split(/\s+/).filter(Boolean).length} / 32</span>
              <span>Autoregressive window active</span>
            </div>
          </div>

          {/* Presets */}
          <div className="space-y-1.5 pt-1">
            <span className="text-xs text-slate-400 font-medium">Quick Prompt Presets:</span>
            <div className="flex flex-wrap gap-2">
              {presets.map((preset, idx) => (
                <button
                  key={idx}
                  onClick={() => {
                    setPrompt(preset);
                    runPrediction(preset);
                  }}
                  className="px-2.5 py-1 rounded-lg bg-slate-900 border border-slate-800 text-slate-300 hover:text-cyan-300 hover:border-slate-700 text-xs truncate max-w-[260px]"
                >
                  {preset}...
                </button>
              ))}
            </div>
          </div>

          {/* Action Row */}
          <div className="flex flex-wrap gap-3 pt-2">
            <button
              onClick={() => runPrediction(prompt)}
              disabled={predicting}
              className="px-5 py-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white font-medium text-xs flex items-center space-x-2 shadow-lg shadow-indigo-600/30 transition-all hover:scale-[1.01]"
            >
              <Zap className="h-3.5 w-3.5" />
              <span>Predict Next Token</span>
            </button>

            <button
              onClick={() => handleAppendToken()}
              disabled={!prediction || predicting}
              className="px-4 py-2.5 rounded-xl bg-cyan-600 hover:bg-cyan-500 text-white font-medium text-xs flex items-center space-x-2 shadow-md shadow-cyan-600/20 transition-all"
            >
              <span>Append Next Token:</span>
              <strong className="font-mono text-cyan-200">
                "{prediction?.selected_token || '...'}"
              </strong>
              <ArrowRight className="h-3.5 w-3.5" />
            </button>
          </div>
        </div>

        {/* Right 5 Cols: Sampling Parameters */}
        <div className="lg:col-span-5 glass-panel p-6 rounded-2xl border border-slate-800 space-y-4">
          <div className="border-b border-slate-800 pb-3">
            <h3 className="text-sm font-semibold text-white uppercase tracking-wider flex items-center space-x-2">
              <Sliders className="h-4 w-4 text-cyan-400" />
              <span>Decoding & Sampling Strategy</span>
            </h3>
          </div>

          <div className="space-y-4 text-xs">
            {/* Sampling Mode Toggle */}
            <div className="space-y-1.5">
              <label className="text-slate-300 font-medium block">Decoding Method:</label>
              <div className="grid grid-cols-2 gap-2">
                <button
                  onClick={() => {
                    setGreedy(true);
                    runPrediction(prompt);
                  }}
                  className={`p-2.5 rounded-xl border text-center transition-all ${
                    greedy
                      ? 'bg-indigo-600/20 border-indigo-500 text-white font-bold'
                      : 'bg-slate-900 border-slate-800 text-slate-400 hover:text-slate-200'
                  }`}
                >
                  Greedy Search (Argmax)
                </button>
                <button
                  onClick={() => {
                    setGreedy(false);
                    runPrediction(prompt);
                  }}
                  className={`p-2.5 rounded-xl border text-center transition-all ${
                    !greedy
                      ? 'bg-cyan-600/20 border-cyan-500 text-white font-bold'
                      : 'bg-slate-900 border-slate-800 text-slate-400 hover:text-slate-200'
                  }`}
                >
                  Temperature Sampling
                </button>
              </div>
            </div>

            {/* Temperature Slider */}
            {!greedy && (
              <div className="space-y-1.5">
                <div className="flex justify-between items-center">
                  <span className="text-slate-300 font-medium">Temperature ($T$):</span>
                  <span className="font-mono text-cyan-400 font-bold">{temperature.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min={0.1}
                  max={2.0}
                  step={0.05}
                  value={temperature}
                  onChange={(e) => {
                    setTemperature(Number(e.target.value));
                    runPrediction(prompt);
                  }}
                  className="w-full accent-cyan-500"
                />
                <div className="flex justify-between text-[10px] text-slate-400 font-mono">
                  <span>0.1 (Focused & Sharp)</span>
                  <span>1.0 (Standard)</span>
                  <span>2.0 (Creative & Flat)</span>
                </div>
              </div>
            )}

            {/* Top-K Filter */}
            {!greedy && (
              <div className="space-y-1.5">
                <div className="flex justify-between items-center">
                  <span className="text-slate-300 font-medium">Top-K Candidate Cutoff:</span>
                  <span className="font-mono text-emerald-400 font-bold">{topK}</span>
                </div>
                <input
                  type="range"
                  min={1}
                  max={20}
                  step={1}
                  value={topK}
                  onChange={(e) => {
                    setTopK(Number(e.target.value));
                    runPrediction(prompt);
                  }}
                  className="w-full accent-emerald-500"
                />
                <div className="flex justify-between text-[10px] text-slate-400 font-mono">
                  <span>Top-1 (Greedy)</span>
                  <span>Top-5</span>
                  <span>Top-20</span>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Top 5 Candidate Tokens & Probability Distribution */}
      {prediction && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Left 6 Cols: Top-5 Cards */}
          <div className="lg:col-span-6 glass-panel p-6 rounded-2xl border border-slate-800 space-y-4">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <h3 className="text-sm font-semibold text-white uppercase tracking-wider flex items-center space-x-2">
                <TrendingUp className="h-4 w-4 text-emerald-400" />
                <span>Top-5 Candidate Next Tokens</span>
              </h3>
              <span className="text-xs font-mono text-slate-400">
                Mode: <span className="text-indigo-300">{prediction.sampling_mode}</span>
              </span>
            </div>

            <div className="space-y-2.5">
              {prediction.top_candidates.map((candidate) => {
                const isWinner = candidate.is_selected;
                return (
                  <div
                    key={candidate.rank}
                    onClick={() => handleAppendToken(candidate.token)}
                    className={`p-3.5 rounded-xl border flex items-center justify-between cursor-pointer transition-all ${
                      isWinner
                        ? 'bg-indigo-600/20 border-indigo-500 shadow-md shadow-indigo-500/15'
                        : 'bg-slate-900/60 border-slate-800 hover:border-slate-700 hover:bg-slate-800/40'
                    }`}
                  >
                    <div className="flex items-center space-x-3">
                      <span
                        className={`h-6 w-6 rounded-full flex items-center justify-center text-xs font-bold font-mono ${
                          candidate.rank === 1
                            ? 'bg-amber-400/20 text-amber-300 border border-amber-400/40'
                            : candidate.rank === 2
                            ? 'bg-slate-400/20 text-slate-300 border border-slate-400/40'
                            : 'bg-slate-800 text-slate-400'
                        }`}
                      >
                        #{candidate.rank}
                      </span>
                      <div>
                        <span className="text-base font-bold font-mono text-white">
                          "{candidate.token}"
                        </span>
                        <div className="flex items-center space-x-2 text-[10px] text-slate-400 font-mono mt-0.5">
                          <span>ID: {candidate.token_id}</span>
                          <span>&bull;</span>
                          <span>Logit: {candidate.raw_logit}</span>
                        </div>
                      </div>
                    </div>

                    <div className="text-right">
                      <span className="text-base font-bold font-mono text-cyan-300">
                        {candidate.probability}%
                      </span>
                      {isWinner && (
                        <span className="text-[10px] font-sans block text-emerald-400 font-medium">
                          Selected Candidate
                        </span>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Right 6 Cols: Probability Distribution Bar Chart */}
          <div className="lg:col-span-6 glass-panel p-6 rounded-2xl border border-slate-800 space-y-4">
            <div className="border-b border-slate-800 pb-3">
              <h3 className="text-sm font-semibold text-white uppercase tracking-wider">
                Probability Distribution ($\text{'{softmax}'}(z / T)$)
              </h3>
              <p className="text-xs text-slate-400 mt-0.5">
                Relative likelihood allocated across candidate vocabulary words
              </p>
            </div>

            <div className="h-64 w-full pt-2">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chartData} margin={{ top: 10, right: 20, left: 0, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="token" stroke="#64748b" tick={{ fill: '#64748b', fontSize: 11 }} />
                  <YAxis domain={[0, 'auto']} stroke="#64748b" tick={{ fill: '#64748b', fontSize: 11 }} />
                  <Tooltip
                    formatter={(val: any) => [`${val}%`, 'Probability']}
                    contentStyle={{
                      backgroundColor: '#0f172a',
                      borderColor: '#334155',
                      borderRadius: '0.75rem',
                      color: '#f8fafc',
                      fontSize: '11px',
                      fontFamily: 'monospace',
                    }}
                  />
                  <Bar dataKey="probability" radius={[6, 6, 0, 0]}>
                    {chartData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.fill} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
