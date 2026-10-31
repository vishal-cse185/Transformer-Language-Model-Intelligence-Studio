import React, { useState, useEffect } from 'react';
import {
  Eye,
  Sliders,
  Sparkles,
  HelpCircle,
  Layers,
  ArrowRight,
  ShieldAlert,
  Search,
} from 'lucide-react';
import { api } from '../services/api';
import { AttentionInspectionResult } from '../types';

export const AttentionVisualization: React.FC = () => {
  const [inputText, setInputText] = useState<string>('The curious rabbit loved to explore the sunny forest.');
  const [attentionData, setAttentionData] = useState<AttentionInspectionResult | null>(null);
  const [selectedLayer, setSelectedLayer] = useState<number>(1);
  const [selectedHead, setSelectedHead] = useState<number | 'avg'>('avg');
  const [focusedTokenIndex, setFocusedTokenIndex] = useState<number>(0);
  const [loading, setLoading] = useState<boolean>(false);

  const presets = [
    'The curious rabbit loved to explore the sunny forest.',
    'A transformer processes all tokens in parallel using self attention.',
    'To be, or not to be, that is the question.',
    'Every small act of kindness makes our forest a brighter place.',
  ];

  useEffect(() => {
    runInspect(inputText);
  }, []);

  const runInspect = async (text: string) => {
    if (!text.trim()) return;
    setLoading(true);
    try {
      const res = await api.inspectAttention(text);
      setAttentionData(res);
      setFocusedTokenIndex(Math.min(res.tokens.length - 1, 3));
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  // Get current active attention matrix (T x T)
  const currentLayerData = attentionData?.layers.find((l) => l.layer_idx === selectedLayer);
  const currentMatrix: number[][] = currentLayerData
    ? selectedHead === 'avg'
      ? currentLayerData.average_head
      : currentLayerData.heads[selectedHead]
    : [];

  const tokens = attentionData?.tokens || [];
  const seqLen = tokens.length;

  // Helper for heatmap cell color
  const getCellColor = (val: number, i: number, j: number) => {
    if (j > i) {
      // Future token: masked
      return 'bg-slate-950/60 text-slate-700 border-dashed border-slate-800/60';
    }
    // Past or current token: intensity based on attention weight
    if (val < 0.05) return 'bg-slate-900/80 text-slate-400';
    if (val < 0.15) return 'bg-indigo-950 text-indigo-300';
    if (val < 0.3) return 'bg-indigo-900/80 text-indigo-200';
    if (val < 0.5) return 'bg-indigo-700 text-white font-semibold';
    if (val < 0.75) return 'bg-indigo-600 text-white font-bold';
    return 'bg-cyan-500 text-slate-950 font-extrabold shadow-sm shadow-cyan-400/30';
  };

  return (
    <div className="space-y-6">
      {/* Header Banner */}
      <div className="glass-panel p-6 rounded-2xl border border-slate-800">
        <div className="flex items-center space-x-2 text-cyan-400 text-xs font-semibold uppercase tracking-wider">
          <Eye className="h-4 w-4 text-indigo-400" />
          <span>Multi-Head Attention Studio</span>
        </div>
        <h2 className="text-2xl font-bold text-white mt-1">Self-Attention Heatmaps & Causal Flow</h2>
        <p className="text-slate-400 text-xs sm:text-sm mt-1 max-w-3xl">
          Visualize real $Q K^T / \sqrt{'{d_k}'}$ attention weights across 4 parallel heads and 2 layers. Inspect how every query token dynamically attends to preceding context tokens while future tokens are strictly masked by $-\infty$.
        </p>

        {/* Input prompt & presets */}
        <div className="mt-5 space-y-3">
          <div className="flex flex-col sm:flex-row gap-2">
            <input
              type="text"
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              placeholder="Enter text to compute attention..."
              className="flex-1 rounded-xl bg-slate-950 border border-slate-800 px-4 py-2.5 text-xs sm:text-sm font-mono text-slate-200 focus:outline-none focus:border-cyan-500"
            />
            <button
              onClick={() => runInspect(inputText)}
              disabled={loading}
              className="px-5 py-2.5 rounded-xl bg-cyan-600 hover:bg-cyan-500 text-white text-xs font-medium flex items-center justify-center space-x-2 shadow-md shadow-cyan-600/20 shrink-0"
            >
              <Search className="h-3.5 w-3.5" />
              <span>Compute Attention</span>
            </button>
          </div>

          <div className="flex flex-wrap items-center gap-1.5 text-xs">
            <span className="text-slate-400 text-[11px] mr-1">Presets:</span>
            {presets.map((preset, idx) => (
              <button
                key={idx}
                onClick={() => {
                  setInputText(preset);
                  runInspect(preset);
                }}
                className="px-2.5 py-1 rounded-lg bg-slate-900 border border-slate-800 text-slate-300 hover:text-cyan-300 hover:border-slate-700 text-[11px] truncate max-w-[240px]"
              >
                {preset}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Layer and Head Selector Toolbar */}
      <div className="flex flex-wrap items-center justify-between gap-4 glass-panel p-4 rounded-xl border border-slate-800">
        {/* Layer Selector */}
        <div className="flex items-center space-x-2">
          <span className="text-xs text-slate-400 font-medium">Layer:</span>
          <div className="flex rounded-lg bg-slate-950 p-1 border border-slate-800 text-xs">
            <button
              onClick={() => setSelectedLayer(1)}
              className={`px-3 py-1 rounded-md transition-all font-medium ${
                selectedLayer === 1 ? 'bg-indigo-600 text-white' : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Block 1
            </button>
            <button
              onClick={() => setSelectedLayer(2)}
              className={`px-3 py-1 rounded-md transition-all font-medium ${
                selectedLayer === 2 ? 'bg-indigo-600 text-white' : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Block 2
            </button>
          </div>
        </div>

        {/* Head Selector */}
        <div className="flex items-center space-x-2">
          <span className="text-xs text-slate-400 font-medium">Attention Head:</span>
          <div className="flex rounded-lg bg-slate-950 p-1 border border-slate-800 text-xs">
            <button
              onClick={() => setSelectedHead('avg')}
              className={`px-3 py-1 rounded-md transition-all font-medium ${
                selectedHead === 'avg' ? 'bg-cyan-600 text-white' : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Average Head
            </button>
            {[0, 1, 2, 3].map((h) => (
              <button
                key={h}
                onClick={() => setSelectedHead(h)}
                className={`px-2.5 py-1 rounded-md transition-all font-medium ${
                  selectedHead === h ? 'bg-indigo-600 text-white' : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                Head {h + 1}
              </button>
            ))}
          </div>
        </div>

        {/* Causal Mask Indicator */}
        <div className="flex items-center space-x-2 text-xs text-amber-400 font-medium">
          <ShieldAlert className="h-4 w-4" />
          <span>Causal Mask: Upper triangle = $-\infty$</span>
        </div>
      </div>

      {/* Main Heatmap Matrix and Token Focus Inspector */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left 8 Cols: Interactive Attention Heatmap */}
        <div className="lg:col-span-8 glass-panel p-6 rounded-2xl border border-slate-800 space-y-4">
          <div className="flex items-center justify-between border-b border-slate-800 pb-3">
            <div>
              <h3 className="text-sm font-semibold text-white uppercase tracking-wider">
                Attention Weight Heatmap ($A = \\text{'{softmax}'}(QK^T / \\sqrt{'{32}'} + M)$)
              </h3>
              <p className="text-[11px] text-slate-400 mt-0.5">
                Rows = Query Token ($i$) &bull; Columns = Key Token ($j$) &bull; Values sum to 1.0 per row
              </p>
            </div>
            <div className="text-[11px] text-slate-400 font-mono">
              Seq Length: <span className="text-cyan-300 font-bold">{seqLen}</span>
            </div>
          </div>

          {currentMatrix.length > 0 ? (
            <div className="overflow-x-auto pb-4">
              <div className="inline-block min-w-full">
                {/* Column Headers (Keys) */}
                <div className="flex items-end pl-24 pb-2">
                  {tokens.map((token, j) => (
                    <div
                      key={j}
                      className="w-10 sm:w-12 text-center text-[10px] font-mono text-slate-400 truncate -rotate-45 origin-bottom-left"
                      title={`Key Token #${j}: "${token}"`}
                    >
                      {token}
                    </div>
                  ))}
                </div>

                {/* Matrix Rows (Queries) */}
                <div className="space-y-1">
                  {currentMatrix.map((row, i) => (
                    <div
                      key={i}
                      onClick={() => setFocusedTokenIndex(i)}
                      className={`flex items-center space-x-1 cursor-pointer p-0.5 rounded transition-all ${
                        focusedTokenIndex === i ? 'bg-indigo-500/10' : 'hover:bg-slate-900/40'
                      }`}
                    >
                      {/* Row Label (Query Token) */}
                      <div className="w-24 text-right pr-2 text-xs font-mono truncate flex items-center justify-end space-x-1.5">
                        <span className="text-slate-400 text-[10px]">#{i}</span>
                        <span className={`font-semibold ${focusedTokenIndex === i ? 'text-cyan-300' : 'text-slate-300'}`}>
                          {tokens[i]}
                        </span>
                      </div>

                      {/* Row Cells */}
                      {row.map((val, j) => {
                        const isMasked = j > i;
                        return (
                          <div
                            key={j}
                            className={`w-10 sm:w-12 h-9 rounded flex flex-col items-center justify-center text-[10px] font-mono matrix-cell border ${getCellColor(
                              val,
                              i,
                              j
                            )}`}
                            title={`Query "${tokens[i]}" (#${i}) attending to Key "${tokens[j]}" (#${j}): ${
                              isMasked ? 'Masked (-inf -> 0.00)' : `${(val * 100).toFixed(1)}%`
                            }`}
                          >
                            {isMasked ? (
                              <span className="text-[9px] text-slate-600 font-sans">0.00</span>
                            ) : (
                              <span>{(val * 100).toFixed(0)}%</span>
                            )}
                          </div>
                        );
                      })}
                    </div>
                  ))}
                </div>
              </div>
            </div>
          ) : (
            <div className="text-center py-12 text-slate-400 text-sm">
              Computing attention matrix...
            </div>
          )}
        </div>

        {/* Right 4 Cols: Token Focus Inspector */}
        <div className="lg:col-span-4 glass-panel p-6 rounded-2xl border border-slate-800 space-y-4">
          <div className="border-b border-slate-800 pb-3">
            <h3 className="text-sm font-semibold text-white uppercase tracking-wider flex items-center space-x-2">
              <Sparkles className="h-4 w-4 text-cyan-400" />
              <span>Token Focus Inspector</span>
            </h3>
            <p className="text-[11px] text-slate-400 mt-0.5">
              Attention distribution for token at position #{focusedTokenIndex}:
            </p>
          </div>

          {tokens.length > 0 && currentMatrix[focusedTokenIndex] && (
            <div className="space-y-4">
              <div className="p-3 rounded-xl bg-slate-950 border border-slate-800 flex items-center justify-between">
                <div>
                  <span className="text-[10px] text-slate-400 font-mono block">Selected Query:</span>
                  <span className="text-sm font-bold text-cyan-300 font-mono">
                    "{tokens[focusedTokenIndex]}"
                  </span>
                </div>
                <div className="text-right">
                  <span className="text-[10px] text-slate-400 font-mono block">Context Span:</span>
                  <span className="text-xs font-semibold text-slate-300 font-mono">
                    Tokens 0 &rarr; {focusedTokenIndex}
                  </span>
                </div>
              </div>

              {/* Attention Bar Chart for preceding tokens */}
              <div className="space-y-2">
                <span className="text-xs text-slate-400 font-medium block">
                  Attention Weights on Preceding Tokens:
                </span>
                <div className="space-y-1.5 max-h-72 overflow-y-auto pr-1">
                  {tokens.slice(0, focusedTokenIndex + 1).map((tok, j) => {
                    const weight = currentMatrix[focusedTokenIndex][j] || 0;
                    const pct = Math.round(weight * 1000) / 10;
                    return (
                      <div key={j} className="space-y-0.5">
                        <div className="flex justify-between items-center text-xs font-mono">
                          <span className="text-slate-300">
                            #{j} <strong className="text-white">{tok}</strong>
                          </span>
                          <span className="text-cyan-300 font-semibold">{pct}%</span>
                        </div>
                        <div className="h-2 w-full bg-slate-900 rounded-full overflow-hidden border border-slate-800">
                          <div
                            className="h-full bg-gradient-to-r from-indigo-500 to-cyan-400 transition-all duration-300"
                            style={{ width: `${Math.max(2, pct)}%` }}
                          />
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Autoregressive causality reminder */}
              <div className="p-3 rounded-xl bg-amber-500/10 border border-amber-500/20 text-[11px] text-amber-300 space-y-1">
                <div className="font-semibold flex items-center space-x-1.5">
                  <ShieldAlert className="h-3.5 w-3.5 shrink-0" />
                  <span>Why future tokens receive zero weight</span>
                </div>
                <p className="text-slate-300 text-[10px] leading-relaxed">
                  In next-token prediction, token #{focusedTokenIndex} must predict token #{focusedTokenIndex + 1}. If it could attend to future positions $j &gt; {focusedTokenIndex}$, it would trivially memorize the target token without learning linguistic structure.
                </p>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Conceptual Deep Dive: Queries, Keys, Values & Scaling */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="glass-card p-5 rounded-2xl border border-slate-800 space-y-2">
          <div className="text-xs font-bold text-indigo-400 uppercase tracking-wider font-mono">
            Query Vector ($Q = X W_q$)
          </div>
          <h4 className="text-sm font-semibold text-white">What am I looking for?</h4>
          <p className="text-xs text-slate-300 leading-relaxed">
            The Query vector acts as an information retrieval request. It encapsulates what syntactic or semantic context the current token requires to predict its next token.
          </p>
        </div>

        <div className="glass-card p-5 rounded-2xl border border-slate-800 space-y-2">
          <div className="text-xs font-bold text-cyan-400 uppercase tracking-wider font-mono">
            Key Vector ($K = X W_k$)
          </div>
          <h4 className="text-sm font-semibold text-white">What do I have to offer?</h4>
          <p className="text-xs text-slate-300 leading-relaxed">
            The Key vector acts as a searchable index label for preceding tokens. The dot product $Q_i \cdot K_j$ determines the compatibility score between token $i$ and token $j$.
          </p>
        </div>

        <div className="glass-card p-5 rounded-2xl border border-slate-800 space-y-2">
          <div className="text-xs font-bold text-emerald-400 uppercase tracking-wider font-mono">
            Value Vector ($V = X W_v$)
          </div>
          <h4 className="text-sm font-semibold text-white">The actual semantic payload</h4>
          <p className="text-xs text-slate-300 leading-relaxed">
            The Value vector contains the actual representation information. Once attention weights are calculated via $\text{'{softmax}'}(QK^T/\sqrt{'{d_k}'})$, they weight the values: $\sum_j A_{'ij'} V_j$.
          </p>
        </div>
      </div>
    </div>
  );
};
