import React from 'react';
import { Cpu, Zap, Activity, BookOpen, Layers, Eye, PlayCircle, BarChart3, Sparkles, Database } from 'lucide-react';
import { SystemStatus } from '../types';

interface HeaderProps {
  activeTab: string;
  setActiveTab: (tab: string) => void;
  status: SystemStatus | null;
}

export const Header: React.FC<HeaderProps> = ({ activeTab, setActiveTab, status }) => {
  const tabs = [
    { id: 'dashboard', label: 'Main Dashboard', icon: Activity },
    { id: 'dataset', label: 'Dataset & Tokenizer', icon: Database },
    { id: 'architecture', label: 'Transformer Studio', icon: Layers },
    { id: 'attention', label: 'Self-Attention Viz', icon: Eye },
    { id: 'training', label: 'Training Center', icon: PlayCircle },
    { id: 'evaluation', label: 'Model Evaluation', icon: BarChart3 },
    { id: 'prediction', label: 'Token Playground', icon: Sparkles },
    { id: 'generation', label: 'Text Generation', icon: Zap },
    { id: 'docs', label: 'Research & Docs', icon: BookOpen },
  ];

  const isTraining = status?.is_training ?? false;
  const isTrained = (status?.training_state?.history?.length ?? 0) > 0;

  return (
    <header className="sticky top-0 z-50 glass-panel border-b border-slate-800/80 bg-slate-950/80 backdrop-blur-xl">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16 border-b border-slate-800/40">
          {/* Logo & Subtitle */}
          <div className="flex items-center space-x-3">
            <div className="h-10 w-10 rounded-xl bg-gradient-to-tr from-indigo-600 via-indigo-500 to-cyan-400 p-[1px] shadow-lg shadow-indigo-500/20">
              <div className="h-full w-full bg-slate-950 rounded-[11px] flex items-center justify-center">
                <Zap className="h-5 w-5 text-indigo-400 animate-pulse" />
              </div>
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <span className="text-lg font-bold tracking-tight text-white">
                  TRANSFORMER<span className="text-indigo-400 font-extrabold ml-1">STUDIO</span>
                </span>
                <span className="px-2 py-0.5 text-[10px] font-semibold bg-indigo-500/20 text-indigo-300 border border-indigo-500/30 rounded-full">
                  PyTorch Tensor Core
                </span>
              </div>
              <p className="text-xs text-slate-400 font-medium hidden sm:block">
                Self-Attention Based Next-Token Prediction & Language Modeling Platform
              </p>
            </div>
          </div>

          {/* System Telemetry Badges */}
          <div className="flex items-center space-x-3 text-xs">
            {/* Training indicator */}
            {isTraining ? (
              <div className="flex items-center space-x-1.5 px-2.5 py-1 rounded-full bg-amber-500/10 border border-amber-500/30 text-amber-300 font-medium animate-pulse">
                <span className="h-2 w-2 rounded-full bg-amber-400 animate-ping" />
                <span>Training in progress...</span>
              </div>
            ) : isTrained ? (
              <div className="flex items-center space-x-1.5 px-2.5 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/30 text-emerald-300 font-medium">
                <span className="h-2 w-2 rounded-full bg-emerald-400" />
                <span>Model Trained & Ready</span>
              </div>
            ) : (
              <div className="flex items-center space-x-1.5 px-2.5 py-1 rounded-full bg-slate-800 border border-slate-700 text-slate-300">
                <span className="h-2 w-2 rounded-full bg-slate-400" />
                <span>Initialized</span>
              </div>
            )}

            {/* Device info */}
            <div className="hidden md:flex items-center space-x-1 px-2.5 py-1 rounded-md bg-slate-900 border border-slate-800 text-slate-300">
              <Cpu className="h-3.5 w-3.5 text-cyan-400" />
              <span className="uppercase text-[11px] font-mono">{status?.device || 'CPU'}</span>
            </div>

            {/* Vocab Size */}
            <div className="hidden lg:flex items-center space-x-1 px-2.5 py-1 rounded-md bg-slate-900 border border-slate-800 text-slate-300 font-mono text-[11px]">
              <span className="text-slate-400">Vocab:</span>
              <span className="text-indigo-300 font-semibold">{status?.vocab_size ?? 0}</span>
            </div>
          </div>
        </div>

        {/* Navigation Tabs */}
        <nav className="flex space-x-1 overflow-x-auto py-2.5 no-scrollbar" aria-label="Tabs">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center space-x-2 px-3 py-1.5 rounded-lg text-xs font-medium whitespace-nowrap transition-all duration-150 ${
                  isActive
                    ? 'bg-indigo-600 text-white shadow-sm shadow-indigo-500/30'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
                }`}
              >
                <Icon className={`h-3.5 w-3.5 ${isActive ? 'text-white' : 'text-slate-400'}`} />
                <span>{tab.label}</span>
              </button>
            );
          })}
        </nav>
      </div>
    </header>
  );
};
