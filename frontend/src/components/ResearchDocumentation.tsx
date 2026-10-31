import React, { useState } from 'react';
import {
  BookOpen,
  Layers,
  Sparkles,
  HelpCircle,
  Eye,
  TrendingDown,
  ShieldAlert,
  Code2,
  ChevronRight,
} from 'lucide-react';

export const ResearchDocumentation: React.FC = () => {
  const [activeSection, setActiveSection] = useState<string>('what_is_transformer');

  const sections = [
    { id: 'what_is_transformer', title: '1. What is a Transformer?' },
    { id: 'self_attention', title: '2. What is Self-Attention?' },
    { id: 'scaled_dot_product', title: '3. Scaled Dot-Product Attention' },
    { id: 'causal_masking', title: '4. Causal Attention Masking' },
    { id: 'positional_encoding', title: '5. Positional Encoding' },
    { id: 'next_token_prediction', title: '6. Next-Token Prediction Objective' },
    { id: 'cross_entropy_perplexity', title: '7. Cross-Entropy Loss & Perplexity' },
    { id: 'autoregressive_generation', title: '8. Autoregressive Generation & Sampling' },
  ];

  return (
    <div className="space-y-6">
      {/* Header Banner */}
      <div className="glass-panel p-6 rounded-2xl border border-slate-800">
        <div className="flex items-center space-x-2 text-indigo-400 text-xs font-semibold uppercase tracking-wider">
          <BookOpen className="h-4 w-4 text-cyan-400" />
          <span>Research & Theory Encyclopedia</span>
        </div>
        <h2 className="text-2xl font-bold text-white mt-1">Transformer Theoretical Foundations</h2>
        <p className="text-slate-400 text-xs sm:text-sm mt-1 max-w-3xl">
          Comprehensive scientific documentation covering mathematical foundations, tensor mechanics, and information-theoretic principles of self-attention language models.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left 4 Cols: Table of Contents */}
        <div className="lg:col-span-4 glass-panel p-4 rounded-2xl border border-slate-800 space-y-2 h-fit">
          <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider px-3 py-2 border-b border-slate-800">
            Table of Contents
          </div>
          <div className="space-y-1">
            {sections.map((s) => (
              <button
                key={s.id}
                onClick={() => setActiveSection(s.id)}
                className={`w-full text-left px-3 py-2 rounded-xl text-xs font-medium flex items-center justify-between transition-all ${
                  activeSection === s.id
                    ? 'bg-indigo-600 text-white shadow-sm shadow-indigo-600/30'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900/60'
                }`}
              >
                <span>{s.title}</span>
                <ChevronRight className="h-3.5 w-3.5 opacity-60" />
              </button>
            ))}
          </div>
        </div>

        {/* Right 8 Cols: Documentation Content */}
        <div className="lg:col-span-8 glass-panel p-6 sm:p-8 rounded-2xl border border-slate-800 space-y-6 leading-relaxed">
          {activeSection === 'what_is_transformer' && (
            <div className="space-y-4">
              <h3 className="text-xl font-bold text-white flex items-center space-x-2">
                <Layers className="h-5 w-5 text-indigo-400" />
                <span>1. What is a Transformer?</span>
              </h3>
              <p className="text-sm text-slate-300">
                Introduced by Vaswani et al. in 2017 (<em>"Attention Is All You Need"</em>), the Transformer is a deep neural network architecture that replaced recurrent neural networks (RNNs, LSTMs, GRUs) by processing sequential data entirely through attention mechanisms without sequential recurrence.
              </p>
              <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-2 text-xs">
                <div className="font-semibold text-indigo-300">Why Transformers Overcame RNNs:</div>
                <ul className="list-disc list-inside space-y-1 text-slate-400">
                  <li><strong className="text-slate-200">Parallel Training:</strong> RNNs require sequential step-by-step recurrence (h_t = f(h_t-1, x_t)), preventing GPU parallelization. Transformers compute representations for all tokens across the sequence length simultaneously in parallel.</li>
                  <li><strong className="text-slate-200">Constant Path Length O(1):</strong> In RNNs, information must travel through N sequential recurrence steps, leading to vanishing or exploding gradients across long contexts. In Transformers, any token can directly attend to any preceding token in a single layer.</li>
                </ul>
              </div>
            </div>
          )}

          {activeSection === 'self_attention' && (
            <div className="space-y-4">
              <h3 className="text-xl font-bold text-white flex items-center space-x-2">
                <Eye className="h-5 w-5 text-cyan-400" />
                <span>2. What is Self-Attention?</span>
              </h3>
              <p className="text-sm text-slate-300">
                Self-attention (intra-attention) is an attention mechanism relating different positions of a single sequence to compute a contextual representation of that same sequence.
              </p>
              <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 text-center font-mono text-sm text-cyan-300">
                Context_i = &Sigma;_{'{j &le; i}'} &alpha;_{'{ij}'} V_j
              </div>
              <p className="text-xs text-slate-400">
                Instead of static word embeddings (where the word <em>"bank"</em> has the same vector whether referring to a river bank or financial bank), self-attention blends the representations of surrounding tokens dynamically, creating context-aware token representations.
              </p>
            </div>
          )}

          {activeSection === 'scaled_dot_product' && (
            <div className="space-y-4">
              <h3 className="text-xl font-bold text-white flex items-center space-x-2">
                <Sparkles className="h-5 w-5 text-emerald-400" />
                <span>3. Scaled Dot-Product Attention & Scaling Derivation</span>
              </h3>
              <p className="text-sm text-slate-300">
                The scaled dot-product attention computes compatibility scores between Query vectors Q and Key vectors K, normalized by the square root of key dimension d_k:
              </p>
              <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 text-center font-mono text-sm sm:text-base text-emerald-300">
                Attention(Q, K, V) = softmax( (Q &middot; K^T) / &radic;d_k + M ) V
              </div>

              <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 space-y-2 text-xs text-slate-300">
                <strong className="text-white block font-mono">Mathematical Proof: Why Scale by 1 / &radic;d_k?</strong>
                <p>
                  Assume elements of q and k are independent random variables with mean 0 and variance 1:
                </p>
                <div className="font-mono text-cyan-300 pl-4 py-1">
                  q &middot; k = &Sigma;_{'{i=1}'}^{'{d_k}'} q_i k_i &implies; E[q &middot; k] = 0, &nbsp; Var(q &middot; k) = &Sigma; Var(q_i k_i) = d_k
                </div>
                <p>
                  For large d_k (e.g. d_k = 32 or 64), the dot products grow large in magnitude, pushing the softmax function into regions with extremely small gradients (gradient vanishing). Dividing by &radic;d_k scales the variance back to Var( (q &middot; k) / &radic;d_k ) = 1, stabilizing training.
                </p>
              </div>
            </div>
          )}

          {activeSection === 'causal_masking' && (
            <div className="space-y-4">
              <h3 className="text-xl font-bold text-white flex items-center space-x-2">
                <ShieldAlert className="h-5 w-5 text-amber-400" />
                <span>4. Causal Attention Masking (Autoregressive Property)</span>
              </h3>
              <p className="text-sm text-slate-300">
                In language modeling, the task is autoregressive: predict token w_t given only previous tokens w_{'{<t}'} = (w_1, ..., w_{'{t-1}'}).
              </p>
              <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 text-center font-mono text-xs sm:text-sm text-amber-300">
                M_{'{i,j}'} = 0 if j &le; i, else -&infin; if j &gt; i
              </div>
              <p className="text-xs text-slate-300">
                When added to the raw dot product scores before softmax, any future token position j &gt; i becomes e^&#123;-&infin;&#125; = 0 in the softmax denominator and numerator. This guarantees that position i receives strictly zero information from position j &gt; i, maintaining causality.
              </p>
            </div>
          )}

          {activeSection === 'positional_encoding' && (
            <div className="space-y-4">
              <h3 className="text-xl font-bold text-white flex items-center space-x-2">
                <Code2 className="h-5 w-5 text-indigo-400" />
                <span>5. Positional Encoding</span>
              </h3>
              <p className="text-sm text-slate-300">
                Self-attention is mathematically permutation-invariant: if you shuffle the input tokens, the set of output vectors will be identically shuffled. Because word order is critical to syntax and semantics, order must be explicitly injected.
              </p>
              <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 font-mono text-xs text-indigo-300 space-y-1 text-center">
                <div>PE_{'{(pos, 2i)}'} = sin( pos / 10000^{'{2i/d}'} )</div>
                <div>PE_{'{(pos, 2i+1)}'} = cos( pos / 10000^{'{2i/d}'} )</div>
              </div>
              <p className="text-xs text-slate-400">
                These sinusoidal geometric frequencies allow the model to easily attend to relative positions because for any fixed offset k, PE_{'{pos+k}'} can be represented as a linear function of PE_{'{pos}'}.
              </p>
            </div>
          )}

          {activeSection === 'next_token_prediction' && (
            <div className="space-y-4">
              <h3 className="text-xl font-bold text-white flex items-center space-x-2">
                <Sparkles className="h-5 w-5 text-cyan-400" />
                <span>6. Next-Token Prediction Objective</span>
              </h3>
              <p className="text-sm text-slate-300">
                The core principle behind generative foundation models is learning the joint probability distribution of natural language by factorizing it using the chain rule of probability:
              </p>
              <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 text-center font-mono text-sm text-cyan-300">
                P(w_1, w_2, ..., w_T) = &prod;_{'{t=1}'}^T P(w_t | w_1, ..., w_{'{t-1}'})
              </div>
              <p className="text-xs text-slate-300">
                By training a neural network to maximize the log-likelihood of predicting the single immediate next token over thousands of sequences, the network is forced to learn grammar, facts, reasoning patterns, and thematic consistency.
              </p>
            </div>
          )}

          {activeSection === 'cross_entropy_perplexity' && (
            <div className="space-y-4">
              <h3 className="text-xl font-bold text-white flex items-center space-x-2">
                <TrendingDown className="h-5 w-5 text-indigo-400" />
                <span>7. Cross-Entropy Loss & Perplexity</span>
              </h3>
              <p className="text-sm text-slate-300">
                Training minimizes the Cross-Entropy loss between the model's predicted probability distribution and the true one-hot target token:
              </p>
              <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 font-mono text-xs sm:text-sm text-indigo-300 text-center">
                L_{'{CE}'} = - (1 / T) &Sigma;_{'{t=1}'}^T log P_&theta;(w_t | w_{'{<t}'})
              </div>
              <p className="text-sm text-slate-300">
                Perplexity (PPL) is the exponential of the cross-entropy loss:
              </p>
              <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 font-mono text-xs sm:text-sm text-cyan-300 text-center">
                PPL = exp( L_{'{CE}'} ) = 2^{'{L_{CE} / ln 2}'}
              </div>
              <p className="text-xs text-slate-400">
                Perplexity represents the effective branching factor: if a model has a perplexity of 50, it predicts the next token with the same uncertainty as choosing uniformly from 50 equally probable words. A perfect model has PPL = 1.
              </p>
            </div>
          )}

          {activeSection === 'autoregressive_generation' && (
            <div className="space-y-4">
              <h3 className="text-xl font-bold text-white flex items-center space-x-2">
                <Zap className="h-5 w-5 text-emerald-400" />
                <span>8. Autoregressive Generation & Sampling Strategies</span>
              </h3>
              <p className="text-sm text-slate-300">
                During inference, the model generates text in an autoregressive feedback loop:
              </p>
              <ol className="list-decimal list-inside space-y-1 text-xs text-slate-300 pl-2">
                <li>Pass context prompt tokens into Transformer forward pass.</li>
                <li>Extract output logits for the final sequence position: z &isin; R^V.</li>
                <li>Apply decoding strategy to select next token w_{'{t+1}'}.</li>
                <li>Append w_{'{t+1}'} to the prompt context and repeat until EOS or maximum length.</li>
              </ol>

              <div className="space-y-2 pt-2 text-xs">
                <div className="p-3 rounded-xl bg-slate-950 border border-slate-800 space-y-1">
                  <strong className="text-white block font-mono">1. Greedy Decoding (Argmax):</strong>
                  <span className="text-slate-400">Always picks w_{'{t+1}'} = argmax_i z_i. Completely deterministic, but prone to repetitive loops.</span>
                </div>
                <div className="p-3 rounded-xl bg-slate-950 border border-slate-800 space-y-1">
                  <strong className="text-cyan-300 block font-mono">2. Temperature Scaling (T):</strong>
                  <span className="text-slate-400">Scales logits by z / T. Lower T &lt; 1.0 peaks the distribution (more deterministic). Higher T &gt; 1.0 flattens the distribution (more creative).</span>
                </div>
                <div className="p-3 rounded-xl bg-slate-950 border border-slate-800 space-y-1">
                  <strong className="text-emerald-300 block font-mono">3. Top-K Sampling:</strong>
                  <span className="text-slate-400">Filters vocabulary to only the K highest-probability tokens, redistributing remaining probability mass. Prevents low-probability gibberish tokens.</span>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
