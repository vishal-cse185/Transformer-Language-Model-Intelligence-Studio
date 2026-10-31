import React, { useState, useEffect } from 'react';
import {
  Database,
  Search,
  BookOpen,
  Hash,
  AlertCircle,
  CheckCircle2,
  RefreshCw,
  FileText,
  Sliders,
  ChevronRight,
  Layers,
} from 'lucide-react';
import { api } from '../services/api';
import { DatasetDetails } from '../types';

interface DatasetTokenizerStudioProps {
  onCorpusChanged?: () => void;
}

export const DatasetTokenizerStudio: React.FC<DatasetTokenizerStudioProps> = ({ onCorpusChanged }) => {
  const [datasetDetails, setDatasetDetails] = useState<DatasetDetails | null>(null);
  const [corporaList, setCorporaList] = useState<Record<string, any>>({});
  const [selectedCorpusId, setSelectedCorpusId] = useState<string>('combined');
  const [customText, setCustomText] = useState<string>('');
  const [isCustomMode, setIsCustomMode] = useState<boolean>(false);
  const [loading, setLoading] = useState<boolean>(false);
  const [notification, setNotification] = useState<string | null>(null);

  // Interactive tokenizer testing state
  const [testInput, setTestInput] = useState<string>(
    'Oliver the rabbit hopped happily through the field.'
  );
  const [tokenizeResult, setTokenizeResult] = useState<any>(null);
  const [tokenizing, setTokenizing] = useState<boolean>(false);

  useEffect(() => {
    loadDatasetInfo();
    loadCorporaList();
    runTokenizeTest(testInput);
  }, []);

  const loadDatasetInfo = async () => {
    try {
      const details = await api.getDatasetDetails();
      setDatasetDetails(details);
      setSelectedCorpusId(details.corpus_id);
    } catch (err: any) {
      console.error(err);
    }
  };

  const loadCorporaList = async () => {
    try {
      const res = await api.getCorpora();
      setCorporaList(res.available_corpora);
    } catch (err: any) {
      console.error(err);
    }
  };

  const runTokenizeTest = async (text: string) => {
    if (!text.trim()) return;
    setTokenizing(true);
    try {
      const res = await api.testTokenizer(text);
      setTokenizeResult(res);
    } catch (err) {
      console.error(err);
    } finally {
      setTokenizing(false);
    }
  };

  const handleSelectCorpus = async (cid: string) => {
    setLoading(true);
    setIsCustomMode(cid === 'custom');
    try {
      if (cid === 'custom') {
        setSelectedCorpusId('custom');
        setLoading(false);
        return;
      }
      setSelectedCorpusId(cid);
      await api.selectCorpus(cid);
      await loadDatasetInfo();
      runTokenizeTest(testInput);
      setNotification(`Switched corpus to "${corporaList[cid]?.name || cid}"`);
      if (onCorpusChanged) onCorpusChanged();
    } catch (err: any) {
      setNotification(`Error: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleApplyCustomCorpus = async () => {
    if (!customText.trim()) return;
    setLoading(true);
    try {
      await api.selectCorpus('custom', customText);
      await loadDatasetInfo();
      runTokenizeTest(testInput);
      setNotification('Custom corpus tokenized and initialized successfully!');
      if (onCorpusChanged) onCorpusChanged();
    } catch (err: any) {
      setNotification(`Error: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Header Info */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 glass-panel p-6 rounded-2xl border border-slate-800">
        <div>
          <div className="flex items-center space-x-2 text-indigo-400 text-xs font-semibold uppercase tracking-wider">
            <Database className="h-4 w-4 text-cyan-400" />
            <span>Dataset & Vocabulary Studio</span>
          </div>
          <h2 className="text-2xl font-bold text-white mt-1">Corpus Preprocessing & Tokenizer</h2>
          <p className="text-slate-400 text-xs sm:text-sm mt-1 max-w-2xl">
            Clean text, construct integer token vocabularies, assign special tokens (&lt;pad&gt;, &lt;unk&gt;, &lt;bos&gt;, &lt;eos&gt;), and partition continuous sequences into autoregressive sliding window pairs ($X, Y$).
          </p>
        </div>

        {notification && (
          <div className="px-3 py-2 rounded-xl bg-indigo-500/10 border border-indigo-500/30 text-indigo-300 text-xs flex items-center space-x-2 animate-fadeIn">
            <CheckCircle2 className="h-4 w-4 text-emerald-400 shrink-0" />
            <span>{notification}</span>
          </div>
        )}
      </div>

      {/* Corpus Selector Tabs & Stats */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left 2 Cols: Corpus Presets & Custom Upload */}
        <div className="lg:col-span-2 glass-panel p-6 rounded-2xl border border-slate-800 space-y-5">
          <div className="flex items-center justify-between border-b border-slate-800 pb-3">
            <h3 className="text-sm font-semibold text-white uppercase tracking-wider flex items-center space-x-2">
              <BookOpen className="h-4 w-4 text-indigo-400" />
              <span>Documented Corpus Library</span>
            </h3>
            <span className="text-xs text-slate-400 font-mono">
              Active: <span className="text-cyan-300 font-bold">{selectedCorpusId}</span>
            </span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {Object.entries(corporaList).map(([cid, meta]) => {
              const isSelected = selectedCorpusId === cid && !isCustomMode;
              return (
                <button
                  key={cid}
                  onClick={() => handleSelectCorpus(cid)}
                  disabled={loading}
                  className={`p-4 rounded-xl text-left border transition-all ${
                    isSelected
                      ? 'bg-indigo-600/15 border-indigo-500/50 shadow-md shadow-indigo-500/10'
                      : 'bg-slate-900/60 border-slate-800 hover:border-slate-700 hover:bg-slate-800/40'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-bold text-white">{meta.name}</span>
                    {isSelected && (
                      <span className="h-2 w-2 rounded-full bg-indigo-400 shadow-sm shadow-indigo-400" />
                    )}
                  </div>
                  <p className="text-[11px] text-slate-400 mt-1 line-clamp-2">
                    {meta.description}
                  </p>
                  <div className="mt-3 flex items-center space-x-3 text-[10px] text-slate-400 font-mono">
                    <span>{meta.word_count.toLocaleString()} words</span>
                    <span>&bull;</span>
                    <span>{meta.character_count.toLocaleString()} chars</span>
                  </div>
                </button>
              );
            })}

            {/* Custom Corpus Card */}
            <button
              onClick={() => handleSelectCorpus('custom')}
              className={`p-4 rounded-xl text-left border transition-all ${
                isCustomMode
                  ? 'bg-cyan-600/15 border-cyan-500/50 shadow-md shadow-cyan-500/10'
                  : 'bg-slate-900/60 border-slate-800 hover:border-slate-700 hover:bg-slate-800/40'
              }`}
            >
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold text-white">Custom Corpus Editor</span>
                {isCustomMode && <span className="h-2 w-2 rounded-full bg-cyan-400" />}
              </div>
              <p className="text-[11px] text-slate-400 mt-1">
                Paste your own English text corpus to train a domain-specific model.
              </p>
              <div className="mt-3 text-[10px] text-cyan-300 font-mono">
                Supports unlimited plain text
              </div>
            </button>
          </div>

          {/* Custom Corpus Textarea when selected */}
          {isCustomMode && (
            <div className="space-y-3 pt-2">
              <label className="block text-xs font-medium text-slate-300">
                Paste Raw Text Corpus:
              </label>
              <textarea
                value={customText}
                onChange={(e) => setCustomText(e.target.value)}
                placeholder="Paste English paragraphs here (e.g. news, stories, technical documentation)..."
                rows={6}
                className="w-full rounded-xl bg-slate-950 border border-slate-800 p-3 text-xs font-mono text-slate-200 focus:outline-none focus:border-cyan-500"
              />
              <button
                onClick={handleApplyCustomCorpus}
                disabled={loading || !customText.trim()}
                className="px-4 py-2 rounded-xl bg-cyan-600 hover:bg-cyan-500 disabled:opacity-50 text-white text-xs font-medium flex items-center space-x-2 shadow-lg shadow-cyan-600/20"
              >
                {loading ? <RefreshCw className="h-3.5 w-3.5 animate-spin" /> : <FileText className="h-3.5 w-3.5" />}
                <span>Tokenize and Build Custom Dataset</span>
              </button>
            </div>
          )}
        </div>

        {/* Right 1 Col: Special Tokens & Dataset Splits */}
        <div className="glass-panel p-6 rounded-2xl border border-slate-800 space-y-5">
          <div className="border-b border-slate-800 pb-3">
            <h3 className="text-sm font-semibold text-white uppercase tracking-wider flex items-center space-x-2">
              <Hash className="h-4 w-4 text-cyan-400" />
              <span>Special Tokens & Splits</span>
            </h3>
          </div>

          {/* Special Tokens */}
          <div className="space-y-2">
            <div className="text-xs text-slate-400 font-medium">Reserved Special Tokens:</div>
            <div className="grid grid-cols-2 gap-2 text-xs font-mono">
              <div className="p-2.5 rounded-lg bg-slate-900 border border-slate-800">
                <div className="flex justify-between items-center">
                  <span className="text-amber-400 font-bold">&lt;pad&gt;</span>
                  <span className="text-slate-400 text-[10px]">ID: 0</span>
                </div>
                <p className="text-[10px] text-slate-400 mt-0.5">Padding mask</p>
              </div>

              <div className="p-2.5 rounded-lg bg-slate-900 border border-slate-800">
                <div className="flex justify-between items-center">
                  <span className="text-red-400 font-bold">&lt;unk&gt;</span>
                  <span className="text-slate-400 text-[10px]">ID: 1</span>
                </div>
                <p className="text-[10px] text-slate-400 mt-0.5">Unknown word</p>
              </div>

              <div className="p-2.5 rounded-lg bg-slate-900 border border-slate-800">
                <div className="flex justify-between items-center">
                  <span className="text-emerald-400 font-bold">&lt;bos&gt;</span>
                  <span className="text-slate-400 text-[10px]">ID: 2</span>
                </div>
                <p className="text-[10px] text-slate-400 mt-0.5">Beginning of seq</p>
              </div>

              <div className="p-2.5 rounded-lg bg-slate-900 border border-slate-800">
                <div className="flex justify-between items-center">
                  <span className="text-indigo-400 font-bold">&lt;eos&gt;</span>
                  <span className="text-slate-400 text-[10px]">ID: 3</span>
                </div>
                <p className="text-[10px] text-slate-400 mt-0.5">End of sequence</p>
              </div>
            </div>
          </div>

          {/* Dataset Splits Breakdown */}
          <div className="space-y-2 pt-2 border-t border-slate-800">
            <div className="text-xs text-slate-400 font-medium">Dataset Partitioning:</div>
            <div className="space-y-2 text-xs">
              <div className="flex justify-between items-center p-2 rounded-lg bg-slate-900/60 border border-slate-800">
                <span className="text-slate-300">Training Split (80%):</span>
                <span className="font-mono text-indigo-300 font-bold">
                  {datasetDetails?.splits_count?.train ?? 0} sequences
                </span>
              </div>
              <div className="flex justify-between items-center p-2 rounded-lg bg-slate-900/60 border border-slate-800">
                <span className="text-slate-300">Validation Split (10%):</span>
                <span className="font-mono text-cyan-300 font-bold">
                  {datasetDetails?.splits_count?.val ?? 0} sequences
                </span>
              </div>
              <div className="flex justify-between items-center p-2 rounded-lg bg-slate-900/60 border border-slate-800">
                <span className="text-slate-300">Held-Out Test (10%):</span>
                <span className="font-mono text-emerald-300 font-bold">
                  {datasetDetails?.splits_count?.test ?? 0} sequences
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Interactive Tokenizer Inspector */}
      <div className="glass-panel p-6 rounded-2xl border border-slate-800 space-y-4">
        <div className="flex items-center justify-between border-b border-slate-800 pb-3">
          <h3 className="text-sm font-semibold text-white uppercase tracking-wider flex items-center space-x-2">
            <Search className="h-4 w-4 text-indigo-400" />
            <span>Interactive Tokenizer Inspector</span>
          </h3>
          <span className="text-xs text-slate-400">
            Test arbitrary text to inspect tokenization & unknown word handling
          </span>
        </div>

        <div className="flex flex-col sm:flex-row gap-3">
          <input
            type="text"
            value={testInput}
            onChange={(e) => {
              setTestInput(e.target.value);
              runTokenizeTest(e.target.value);
            }}
            placeholder="Type any sentence here..."
            className="flex-1 rounded-xl bg-slate-950 border border-slate-800 px-4 py-2.5 text-xs sm:text-sm font-mono text-slate-200 focus:outline-none focus:border-indigo-500"
          />
          <button
            onClick={() => runTokenizeTest(testInput)}
            disabled={tokenizing}
            className="px-4 py-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-medium flex items-center justify-center space-x-2 shrink-0 shadow-md shadow-indigo-600/20"
          >
            {tokenizing ? <RefreshCw className="h-3.5 w-3.5 animate-spin" /> : <Search className="h-3.5 w-3.5" />}
            <span>Tokenize</span>
          </button>
        </div>

        {tokenizeResult && (
          <div className="space-y-4 pt-2">
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-xs text-slate-400 mr-2">Token Chips (ID & Span):</span>
              {tokenizeResult.token_details.map((item: any) => (
                <div
                  key={item.position}
                  className={`inline-flex items-center space-x-1.5 px-3 py-1.5 rounded-lg border text-xs font-mono transition-all ${
                    item.is_unk
                      ? 'bg-amber-500/15 border-amber-500/40 text-amber-300'
                      : 'bg-slate-900 border-slate-800 text-slate-200 hover:border-indigo-500/50'
                  }`}
                >
                  <span className="font-semibold">{item.token}</span>
                  <span className="text-[10px] px-1 py-0.2 rounded bg-black/40 text-slate-400">
                    ID:{item.id}
                  </span>
                  {item.is_unk && (
                    <span className="text-[9px] px-1 rounded bg-amber-500/20 text-amber-300 font-sans">
                      UNK
                    </span>
                  )}
                </div>
              ))}
            </div>

            <div className="flex flex-wrap gap-4 text-xs font-mono text-slate-400 pt-1">
              <div>
                <span>Total Tokens: </span>
                <span className="text-white font-bold">{tokenizeResult.tokens_count}</span>
              </div>
              <div>
                <span>Unknown Tokens: </span>
                <span className={tokenizeResult.unknown_tokens_count > 0 ? 'text-amber-400 font-bold' : 'text-emerald-400 font-bold'}>
                  {tokenizeResult.unknown_tokens_count}
                </span>
              </div>
              <div className="truncate max-w-md">
                <span>Encoded Tensor: </span>
                <span className="text-indigo-300">[{tokenizeResult.encoded_ids.join(', ')}]</span>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Sliding Window Sequence Pairs (X, Y) Table */}
      <div className="glass-panel p-6 rounded-2xl border border-slate-800 space-y-4">
        <div className="border-b border-slate-800 pb-3">
          <h3 className="text-sm font-semibold text-white uppercase tracking-wider flex items-center space-x-2">
            <Layers className="h-4 w-4 text-cyan-400" />
            <span>Sample Autoregressive Sequence Pairs ($X \\to Y$)</span>
          </h3>
          <p className="text-xs text-slate-400 mt-1">
            Input sequence $X = [w_0, \\dots, w_{'{T-1}'}]$ and Target sequence $Y = [w_1, \\dots, w_T]$ shifted by 1 token for next-token prediction training.
          </p>
        </div>

        {datasetDetails?.sample_sequence_pairs && (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs font-mono">
              <thead>
                <tr className="text-slate-400 border-b border-slate-800 text-[11px]">
                  <th className="pb-2 w-16">Pair #</th>
                  <th className="pb-2">Input Context $X$ (First 10 tokens)</th>
                  <th className="pb-2">Target Sequence $Y$ (Shifted +1)</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/40 text-slate-300">
                {datasetDetails.sample_sequence_pairs.map((pair) => (
                  <tr key={pair.sample_index} className="hover:bg-slate-800/30">
                    <td className="py-2.5 text-indigo-400 font-semibold">#{pair.sample_index}</td>
                    <td className="py-2.5 text-slate-200">
                      <span className="px-2 py-1 rounded bg-slate-900 border border-slate-800 text-indigo-300 inline-block">
                        {pair.input_preview} ...
                      </span>
                    </td>
                    <td className="py-2.5 text-slate-200">
                      <span className="px-2 py-1 rounded bg-slate-900 border border-slate-800 text-cyan-300 inline-block">
                        {pair.target_preview} ...
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Vocabulary Top Words Grid */}
      {datasetDetails?.vocab_stats?.top_words && (
        <div className="glass-panel p-6 rounded-2xl border border-slate-800 space-y-4">
          <div className="border-b border-slate-800 pb-3">
            <h3 className="text-sm font-semibold text-white uppercase tracking-wider flex items-center space-x-2">
              <Hash className="h-4 w-4 text-emerald-400" />
              <span>Top Vocabulary Words & Frequencies</span>
            </h3>
            <p className="text-xs text-slate-400 mt-1">
              Top 25 most frequent tokens in the current corpus
            </p>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-2 text-xs font-mono">
            {datasetDetails.vocab_stats.top_words.map((w, idx) => (
              <div
                key={w.token}
                className="p-2 rounded-lg bg-slate-900/80 border border-slate-800 flex justify-between items-center"
              >
                <div className="flex items-center space-x-1.5 truncate">
                  <span className="text-slate-400 text-[10px]">#{idx + 1}</span>
                  <span className="font-semibold text-white truncate">{w.token}</span>
                </div>
                <span className="text-[10px] px-1.5 py-0.5 rounded bg-indigo-500/15 text-indigo-300">
                  {w.count}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
