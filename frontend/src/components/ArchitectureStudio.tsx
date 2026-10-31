import React, { useState, useEffect } from 'react';
import {
  Layers,
  Cpu,
  Code2,
  BookOpen,
  ArrowDown,
  CheckCircle,
  HelpCircle,
  Sparkles,
  Zap,
} from 'lucide-react';
import { api } from '../services/api';
import { ArchitectureComponent } from '../types';

export const ArchitectureStudio: React.FC = () => {
  const [components, setComponents] = useState<ArchitectureComponent[]>([]);
  const [selectedCompId, setSelectedCompId] = useState<string>('qkv_projections');
  const [paramInfo, setParamInfo] = useState<any>(null);
  const [config, setConfig] = useState<any>(null);

  useEffect(() => {
    loadArchitecture();
  }, []);

  const loadArchitecture = async () => {
    try {
      const data = await api.getArchitecture();
      setComponents(data.components);
      setParamInfo(data.parameters);
      setConfig(data.config);
    } catch (err) {
      console.error(err);
    }
  };

  const selectedComponent = components.find((c) => c.id === selectedCompId) || components[0];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="glass-panel p-6 rounded-2xl border border-slate-800">
        <div className="flex items-center space-x-2 text-indigo-400 text-xs font-semibold uppercase tracking-wider">
          <Layers className="h-4 w-4 text-cyan-400" />
          <span>Manual Transformer Architecture</span>
        </div>
        <h2 className="text-2xl font-bold text-white mt-1">PyTorch Decoder-Only Model Topology</h2>
        <p className="text-slate-400 text-xs sm:text-sm mt-1 max-w-3xl">
          Engineered completely from scratch using primitive PyTorch tensor operations. Click any component below to inspect its mathematical formulation, tensor shapes, parameters, and live PyTorch implementation.
        </p>

        {/* Global Configuration Badges */}
        <div className="flex flex-wrap gap-2.5 mt-4 pt-3 border-t border-slate-800 text-xs font-mono">
          <span className="px-3 py-1 rounded-lg bg-slate-900 border border-slate-800 text-slate-300">
            <span className="text-slate-400">Embedding ($d_{'model'}$): </span>
            <strong className="text-white">128</strong>
          </span>
          <span className="px-3 py-1 rounded-lg bg-slate-900 border border-slate-800 text-slate-300">
            <span className="text-slate-400">Heads ($H$): </span>
            <strong className="text-cyan-400">4 ($d_k = 32$)</strong>
          </span>
          <span className="px-3 py-1 rounded-lg bg-slate-900 border border-slate-800 text-slate-300">
            <span className="text-slate-400">Layers ($N$): </span>
            <strong className="text-indigo-400">2 Blocks</strong>
          </span>
          <span className="px-3 py-1 rounded-lg bg-slate-900 border border-slate-800 text-slate-300">
            <span className="text-slate-400">Context Window ($T$): </span>
            <strong className="text-white">32 Tokens</strong>
          </span>
          <span className="px-3 py-1 rounded-lg bg-slate-900 border border-slate-800 text-slate-300">
            <span className="text-slate-400">Feed-Forward ($d_{'ff'}$): </span>
            <strong className="text-white">512 (GELU)</strong>
          </span>
          <span className="px-3 py-1 rounded-lg bg-slate-900 border border-slate-800 text-slate-300">
            <span className="text-slate-400">Total Params: </span>
            <strong className="text-emerald-400">{paramInfo?.total_parameters?.toLocaleString() ?? '185,000'}</strong>
          </span>
        </div>
      </div>

      {/* Main Interactive Diagram and Component Inspector */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left 5 Cols: Interactive Sequential Diagram */}
        <div className="lg:col-span-5 glass-panel p-6 rounded-2xl border border-slate-800 space-y-4">
          <div className="flex items-center justify-between border-b border-slate-800 pb-3">
            <h3 className="text-sm font-semibold text-white uppercase tracking-wider flex items-center space-x-2">
              <Cpu className="h-4 w-4 text-cyan-400" />
              <span>Computational Dataflow</span>
            </h3>
            <span className="text-[11px] text-slate-400">Click component to inspect</span>
          </div>

          <div className="space-y-2">
            {/* Input tokens */}
            <div className="p-2.5 rounded-xl bg-slate-950 border border-slate-800/80 text-center text-xs font-mono text-slate-400">
              Input Token IDs: <code className="text-indigo-300 font-semibold">[B, T]</code>
            </div>

            <div className="flex justify-center text-slate-600">
              <ArrowDown className="h-3.5 w-3.5 animate-bounce" />
            </div>

            {/* Step 1 & 2: Embedding + Positional */}
            <div className="grid grid-cols-2 gap-2">
              <button
                onClick={() => setSelectedCompId('token_embedding')}
                className={`p-3 rounded-xl border text-left transition-all ${
                  selectedCompId === 'token_embedding'
                    ? 'bg-indigo-600/20 border-indigo-500 shadow-md shadow-indigo-500/20 text-white'
                    : 'bg-slate-900/80 border-slate-800 hover:border-slate-700 text-slate-300'
                }`}
              >
                <div className="text-[10px] text-indigo-400 font-mono uppercase font-semibold">Step 1</div>
                <div className="text-xs font-bold mt-0.5">Token Embedding</div>
                <div className="text-[10px] font-mono text-slate-400 mt-1">[B, T] &rarr; [B, T, 128]</div>
              </button>

              <button
                onClick={() => setSelectedCompId('positional_encoding')}
                className={`p-3 rounded-xl border text-left transition-all ${
                  selectedCompId === 'positional_encoding'
                    ? 'bg-indigo-600/20 border-indigo-500 shadow-md shadow-indigo-500/20 text-white'
                    : 'bg-slate-900/80 border-slate-800 hover:border-slate-700 text-slate-300'
                }`}
              >
                <div className="text-[10px] text-cyan-400 font-mono uppercase font-semibold">Step 2</div>
                <div className="text-xs font-bold mt-0.5">Positional Encoding</div>
                <div className="text-[10px] font-mono text-slate-400 mt-1">Sinusoidal PE [1, T, 128]</div>
              </button>
            </div>

            <div className="flex justify-center text-slate-600">
              <ArrowDown className="h-3.5 w-3.5" />
            </div>

            {/* Transformer Block Container (2x Repeat) */}
            <div className="p-3.5 rounded-2xl bg-indigo-950/20 border-2 border-dashed border-indigo-500/30 space-y-2">
              <div className="flex justify-between items-center text-[10px] font-mono uppercase tracking-wider text-indigo-300 font-bold px-1">
                <span>Transformer Decoder Block &times; 2</span>
                <span className="px-1.5 py-0.5 rounded bg-indigo-500/20 text-indigo-300">Pre-LN Residual</span>
              </div>

              {/* Sub-step 1: Q, K, V */}
              <button
                onClick={() => setSelectedCompId('qkv_projections')}
                className={`w-full p-2.5 rounded-xl border text-left transition-all ${
                  selectedCompId === 'qkv_projections'
                    ? 'bg-indigo-600/25 border-indigo-500 text-white'
                    : 'bg-slate-900/90 border-slate-800 hover:border-slate-700 text-slate-300'
                }`}
              >
                <div className="flex justify-between items-center text-xs font-semibold">
                  <span>Q, K, V Projections (4 Heads)</span>
                  <span className="text-[10px] font-mono text-cyan-400">d_k=32</span>
                </div>
                <div className="text-[10px] font-mono text-slate-400 mt-0.5">
                  $W_q, W_k, W_v \in \mathbb{'{R}'}^{'{128 \\times 128}'}$
                </div>
              </button>

              {/* Sub-step 2: Scaled Dot-Product & Causal Mask */}
              <div className="grid grid-cols-2 gap-2">
                <button
                  onClick={() => setSelectedCompId('scaled_dot_product_attention')}
                  className={`p-2.5 rounded-xl border text-left transition-all ${
                    selectedCompId === 'scaled_dot_product_attention'
                      ? 'bg-indigo-600/25 border-indigo-500 text-white'
                      : 'bg-slate-900/90 border-slate-800 hover:border-slate-700 text-slate-300'
                  }`}
                >
                  <div className="text-xs font-semibold">Scaled Dot-Product</div>
                  <div className="text-[10px] font-mono text-slate-400 mt-0.5">$QK^T / \sqrt{'{32}'}$</div>
                </button>

                <button
                  onClick={() => setSelectedCompId('causal_masking')}
                  className={`p-2.5 rounded-xl border text-left transition-all ${
                    selectedCompId === 'causal_masking'
                      ? 'bg-indigo-600/25 border-indigo-500 text-white'
                      : 'bg-slate-900/90 border-slate-800 hover:border-slate-700 text-slate-300'
                  }`}
                >
                  <div className="text-xs font-semibold">Causal Mask</div>
                  <div className="text-[10px] font-mono text-amber-400 mt-0.5">Lower-triangular</div>
                </button>
              </div>

              {/* Sub-step 3: Feed Forward */}
              <button
                onClick={() => setSelectedCompId('feed_forward')}
                className={`w-full p-2.5 rounded-xl border text-left transition-all ${
                  selectedCompId === 'feed_forward'
                    ? 'bg-indigo-600/25 border-indigo-500 text-white'
                    : 'bg-slate-900/90 border-slate-800 hover:border-slate-700 text-slate-300'
                }`}
              >
                <div className="flex justify-between items-center text-xs font-semibold">
                  <span>Feed-Forward Network (FFN)</span>
                  <span className="text-[10px] font-mono text-emerald-400">128 &rarr; 512 &rarr; 128</span>
                </div>
                <div className="text-[10px] font-mono text-slate-400 mt-0.5">
                  $\text{'{GELU}'}(x W_1) W_2$
                </div>
              </button>

              {/* Sub-step 4: Residual & Norm */}
              <button
                onClick={() => setSelectedCompId('residual_and_norm')}
                className={`w-full p-2.5 rounded-xl border text-left transition-all ${
                  selectedCompId === 'residual_and_norm'
                    ? 'bg-indigo-600/25 border-indigo-500 text-white'
                    : 'bg-slate-900/90 border-slate-800 hover:border-slate-700 text-slate-300'
                }`}
              >
                <div className="flex justify-between items-center text-xs font-semibold">
                  <span>Residuals & LayerNorm</span>
                  <span className="text-[10px] font-mono text-indigo-300">Pre-LN</span>
                </div>
                <div className="text-[10px] font-mono text-slate-400 mt-0.5">
                  $x + \text{'{SubLayer}'}(\text{'{LN}'}(x))$
                </div>
              </button>
            </div>

            <div className="flex justify-center text-slate-600">
              <ArrowDown className="h-3.5 w-3.5" />
            </div>

            {/* Output LM Head */}
            <button
              onClick={() => setSelectedCompId('lm_head')}
              className={`w-full p-3 rounded-xl border text-left transition-all ${
                selectedCompId === 'lm_head'
                  ? 'bg-cyan-600/25 border-cyan-500 text-white shadow-md shadow-cyan-500/20'
                  : 'bg-slate-900/80 border-slate-800 hover:border-slate-700 text-slate-300'
              }`}
            >
              <div className="flex justify-between items-center">
                <span className="text-xs font-bold">Linear Output Projection (LM Head)</span>
                <span className="text-[10px] font-mono text-cyan-400 uppercase">Logits</span>
              </div>
              <div className="text-[10px] font-mono text-slate-400 mt-1">
                [B, T, 128] &rarr; [B, T, Vocab Size] &rarr; Softmax Next Token
              </div>
            </button>
          </div>
        </div>

        {/* Right 7 Cols: Selected Component Deep-Dive Inspector */}
        <div className="lg:col-span-7 glass-panel p-6 rounded-2xl border border-slate-800 space-y-6">
          {selectedComponent ? (
            <>
              {/* Component Title & Shapes */}
              <div className="border-b border-slate-800 pb-4">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <h3 className="text-xl font-bold text-white flex items-center space-x-2">
                    <Sparkles className="h-5 w-5 text-indigo-400" />
                    <span>{selectedComponent.name}</span>
                  </h3>
                  <span className="px-2.5 py-1 rounded-md bg-indigo-500/10 border border-indigo-500/30 text-indigo-300 font-mono text-xs">
                    {selectedComponent.layer_type}
                  </span>
                </div>

                <div className="grid grid-cols-3 gap-3 mt-4 text-xs font-mono">
                  <div className="p-2.5 rounded-lg bg-slate-950 border border-slate-800/80">
                    <span className="text-slate-400 block text-[10px]">Input Shape</span>
                    <span className="text-cyan-300 font-semibold">{selectedComponent.input_shape}</span>
                  </div>
                  <div className="p-2.5 rounded-lg bg-slate-950 border border-slate-800/80">
                    <span className="text-slate-400 block text-[10px]">Output Shape</span>
                    <span className="text-indigo-300 font-semibold">{selectedComponent.output_shape}</span>
                  </div>
                  <div className="p-2.5 rounded-lg bg-slate-950 border border-slate-800/80">
                    <span className="text-slate-400 block text-[10px]">Parameters</span>
                    <span className="text-emerald-300 font-semibold">
                      {selectedComponent.parameters.toLocaleString()}
                    </span>
                  </div>
                </div>
              </div>

              {/* Mathematical Formulation */}
              <div className="space-y-2">
                <h4 className="text-xs font-semibold text-slate-300 uppercase tracking-wider flex items-center space-x-2">
                  <BookOpen className="h-3.5 w-3.5 text-cyan-400" />
                  <span>Mathematical Formulation</span>
                </h4>
                <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 text-center font-mono text-sm sm:text-base text-cyan-300 shadow-inner">
                  {selectedComponent.math_formula}
                </div>
              </div>

              {/* Plain-English Explanation */}
              <div className="space-y-2">
                <h4 className="text-xs font-semibold text-slate-300 uppercase tracking-wider flex items-center space-x-2">
                  <HelpCircle className="h-3.5 w-3.5 text-indigo-400" />
                  <span>Architecture & NLP Intuition</span>
                </h4>
                <p className="text-xs sm:text-sm text-slate-300 leading-relaxed bg-slate-900/50 p-4 rounded-xl border border-slate-800/60">
                  {selectedComponent.explanation}
                </p>
              </div>

              {/* PyTorch Tensor Code Snippet */}
              <div className="space-y-2">
                <h4 className="text-xs font-semibold text-slate-300 uppercase tracking-wider flex items-center space-x-2">
                  <Code2 className="h-3.5 w-3.5 text-emerald-400" />
                  <span>PyTorch Tensor Implementation</span>
                </h4>
                <pre className="p-4 rounded-xl bg-slate-950 border border-slate-800 text-xs font-mono text-emerald-300 overflow-x-auto">
                  <code>{selectedComponent.pytorch_code}</code>
                </pre>
              </div>
            </>
          ) : (
            <div className="text-center py-12 text-slate-400 text-sm">
              Select a component on the left to inspect its details.
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
