import React, { useState } from 'react';
import {
  Zap,
  Sliders,
  Sparkles,
  RefreshCw,
  Copy,
  Check,
  RotateCcw,
  Clock,
  CheckCircle2,
  ListOrdered,
} from 'lucide-react';
import { api } from '../services/api';
import { GenerationResult } from '../types';

export const GenerationWorkspace: React.FC = () => {
  const [prompt, setPrompt] = useState<string>('Oliver the curious rabbit hopped');
  const [maxNewTokens, setMaxNewTokens] = useState<number>(20);
  const [temperature, setTemperature] = useState<number>(0.8);
  const [topK, setTopK] = useState<number>(10);

  const [generationResult, setGenerationResult] = useState<GenerationResult | null>(null);
  const [generating, setGenerating] = useState<boolean>(false);
  const [copied, setCopied] = useState<boolean>(false);

  const presets = [
    'Once upon a time in a bright green forest',
    'Oliver the curious rabbit hopped',
    'A transformer is a deep neural network',
    'Shall I compare thee to a summer day',
    'Every morning the golden sun rose over',
  ];

  const handleGenerate = async () => {
    if (!prompt.trim()) return;
    setGenerating(true);
    try {
      const res = await api.generateText({
        prompt,
        max_new_tokens: maxNewTokens,
        temperature,
        top_k: topK,
      });
      setGenerationResult(res);
    } catch (err) {
      console.error(err);
    } finally {
      setGenerating(false);
    }
  };

  const handleCopy = () => {
    if (!generationResult) return;
    navigator.clipboard.writeText(generationResult.generated_text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="space-y-6">
      {/* Header Banner */}
      <div className="glass-panel p-6 rounded-2xl border border-slate-800">
        <div className="flex items-center space-x-2 text-cyan-400 text-xs font-semibold uppercase tracking-wider">
          <Zap className="h-4 w-4 text-indigo-400" />
          <span>Autoregressive Text Generation</span>
        </div>
        <h2 className="text-2xl font-bold text-white mt-1">Autonomous Language Generation Workspace</h2>
        <p className="text-slate-400 text-xs sm:text-sm mt-1 max-w-3xl">
          Generate complete linguistic passages token-by-token. The model feeds its own emitted predictions back into its context window, illustrating autoregressive language generation in real-time.
        </p>
      </div>

      {/* Main Settings & Controls */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left 7 Cols: Prompt and Presets */}
        <div className="lg:col-span-7 glass-panel p-6 rounded-2xl border border-slate-800 space-y-4">
          <div className="flex items-center justify-between border-b border-slate-800 pb-3">
            <h3 className="text-sm font-semibold text-white uppercase tracking-wider flex items-center space-x-2">
              <Sparkles className="h-4 w-4 text-cyan-400" />
              <span>Prompt Seed</span>
            </h3>
            <span className="text-xs text-slate-400">Context initiator</span>
          </div>

          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            rows={3}
            placeholder="Enter starting prompt to seed generation..."
            className="w-full rounded-xl bg-slate-950 border border-slate-800 p-3.5 text-xs sm:text-sm font-mono text-slate-100 focus:outline-none focus:border-cyan-500"
          />

          <div className="space-y-1.5 pt-1">
            <span className="text-xs text-slate-400 font-medium">Quick Prompts:</span>
            <div className="flex flex-wrap gap-2">
              {presets.map((preset, idx) => (
                <button
                  key={idx}
                  onClick={() => setPrompt(preset)}
                  className="px-2.5 py-1 rounded-lg bg-slate-900 border border-slate-800 text-slate-300 hover:text-cyan-300 hover:border-slate-700 text-xs truncate max-w-[280px]"
                >
                  {preset}...
                </button>
              ))}
            </div>
          </div>

          <div className="pt-2">
            <button
              onClick={handleGenerate}
              disabled={generating || !prompt.trim()}
              className="w-full py-3 px-5 rounded-xl bg-cyan-600 hover:bg-cyan-500 disabled:opacity-50 text-white font-medium text-xs flex items-center justify-center space-x-2 shadow-lg shadow-cyan-600/30 transition-all hover:scale-[1.01]"
            >
              {generating ? <RefreshCw className="h-4 w-4 animate-spin" /> : <Zap className="h-4 w-4" />}
              <span>Generate Autonomous Sequence</span>
            </button>
          </div>
        </div>

        {/* Right 5 Cols: Generation Parameters */}
        <div className="lg:col-span-5 glass-panel p-6 rounded-2xl border border-slate-800 space-y-4">
          <div className="border-b border-slate-800 pb-3">
            <h3 className="text-sm font-semibold text-white uppercase tracking-wider flex items-center space-x-2">
              <Sliders className="h-4 w-4 text-indigo-400" />
              <span>Generation Parameters</span>
            </h3>
          </div>

          <div className="space-y-4 text-xs">
            {/* Max New Tokens */}
            <div className="space-y-1.5">
              <div className="flex justify-between items-center">
                <span className="text-slate-300 font-medium">Tokens to Generate:</span>
                <span className="font-mono text-cyan-400 font-bold">{maxNewTokens} tokens</span>
              </div>
              <input
                type="range"
                min={5}
                max={50}
                step={5}
                value={maxNewTokens}
                onChange={(e) => setMaxNewTokens(Number(e.target.value))}
                className="w-full accent-cyan-500"
              />
              <div className="flex justify-between text-[10px] text-slate-400 font-mono">
                <span>5 tokens (Short)</span>
                <span>25 tokens</span>
                <span>50 tokens (Long)</span>
              </div>
            </div>

            {/* Temperature */}
            <div className="space-y-1.5">
              <div className="flex justify-between items-center">
                <span className="text-slate-300 font-medium">Temperature ($T$):</span>
                <span className="font-mono text-indigo-400 font-bold">{temperature.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min={0.1}
                max={1.8}
                step={0.05}
                value={temperature}
                onChange={(e) => setTemperature(Number(e.target.value))}
                className="w-full accent-indigo-500"
              />
              <div className="flex justify-between text-[10px] text-slate-400 font-mono">
                <span>0.1 (Deterministic)</span>
                <span>0.8 (Balanced)</span>
                <span>1.8 (Creative)</span>
              </div>
            </div>

            {/* Top-K Sampling */}
            <div className="space-y-1.5">
              <div className="flex justify-between items-center">
                <span className="text-slate-300 font-medium">Top-K Candidate Pool:</span>
                <span className="font-mono text-emerald-400 font-bold">{topK}</span>
              </div>
              <input
                type="range"
                min={1}
                max={30}
                step={1}
                value={topK}
                onChange={(e) => setTopK(Number(e.target.value))}
                className="w-full accent-emerald-500"
              />
              <div className="flex justify-between text-[10px] text-slate-400 font-mono">
                <span>Top-1 (Greedy)</span>
                <span>Top-10</span>
                <span>Top-30</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Generated Output Showcase */}
      {generationResult && (
        <div className="glass-panel p-6 rounded-2xl border border-slate-800 space-y-4">
          <div className="flex items-center justify-between border-b border-slate-800 pb-3">
            <h3 className="text-sm font-semibold text-white uppercase tracking-wider flex items-center space-x-2">
              <CheckCircle2 className="h-4 w-4 text-emerald-400" />
              <span>Autoregressively Generated Text</span>
            </h3>
            <button
              onClick={handleCopy}
              className="text-xs text-slate-400 hover:text-white flex items-center space-x-1.5 px-2.5 py-1 rounded-lg bg-slate-900 border border-slate-800"
            >
              {copied ? <Check className="h-3.5 w-3.5 text-emerald-400" /> : <Copy className="h-3.5 w-3.5" />}
              <span>{copied ? 'Copied' : 'Copy Text'}</span>
            </button>
          </div>

          <div className="p-5 rounded-xl bg-slate-950 border border-slate-800 text-sm sm:text-base font-mono leading-relaxed text-slate-200">
            <span className="text-slate-400 bg-slate-900/60 px-1 py-0.5 rounded border border-slate-800 mr-1.5">
              {generationResult.prompt}
            </span>
            <span className="text-cyan-300 font-semibold">
              {generationResult.generated_text.slice(generationResult.prompt.length)}
            </span>
          </div>

          <div className="flex items-center space-x-4 text-xs font-mono text-slate-400 pt-1">
            <span>Tokens Emitted: <strong className="text-white">{generationResult.tokens_added}</strong></span>
            <span>&bull;</span>
            <span>Temperature: <strong className="text-white">{generationResult.hyperparameters.temperature}</strong></span>
            <span>&bull;</span>
            <span>Top-K: <strong className="text-white">{generationResult.hyperparameters.top_k}</strong></span>
          </div>
        </div>
      )}

      {/* Step-by-Step Generation History Table */}
      {generationResult?.history && generationResult.history.length > 0 && (
        <div className="glass-panel p-6 rounded-2xl border border-slate-800 space-y-4">
          <div className="border-b border-slate-800 pb-3">
            <h3 className="text-sm font-semibold text-white uppercase tracking-wider flex items-center space-x-2">
              <ListOrdered className="h-4 w-4 text-indigo-400" />
              <span>Step-by-Step Autoregressive Trace Log</span>
            </h3>
            <p className="text-xs text-slate-400 mt-0.5">
              Detailed breakdown of each generated token, probability assigned, and top alternatives considered
            </p>
          </div>

          <div className="overflow-x-auto max-h-80 overflow-y-auto">
            <table className="w-full text-left text-xs font-mono">
              <thead className="sticky top-0 bg-slate-950 text-slate-400 border-b border-slate-800 text-[11px]">
                <tr>
                  <th className="pb-2.5 w-16">Step</th>
                  <th className="pb-2.5">Emitted Token</th>
                  <th className="pb-2.5">Token ID</th>
                  <th className="pb-2.5">Selected Probability</th>
                  <th className="pb-2.5">Alternative Candidates Considered</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/40 text-slate-300">
                {generationResult.history.map((step) => (
                  <tr key={step.step} className="hover:bg-slate-800/30">
                    <td className="py-2.5 font-bold text-cyan-400">#{step.step}</td>
                    <td className="py-2.5 text-white font-bold text-sm">"{step.token}"</td>
                    <td className="py-2.5 text-slate-400">ID: {step.token_id}</td>
                    <td className="py-2.5 text-emerald-400 font-semibold">{step.probability}%</td>
                    <td className="py-2.5">
                      <div className="flex flex-wrap gap-1.5">
                        {step.alternatives.map((alt, idx) => (
                          <span
                            key={idx}
                            className="px-2 py-0.5 rounded bg-slate-900 border border-slate-800 text-slate-300 text-[10px]"
                          >
                            {alt.token}: {alt.prob}%
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
    </div>
  );
};
