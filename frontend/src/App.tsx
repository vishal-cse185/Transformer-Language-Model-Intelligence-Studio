import React, { useState, useEffect } from 'react';
import { Header } from './components/Header';
import { Dashboard } from './components/Dashboard';
import { DatasetTokenizerStudio } from './components/DatasetTokenizerStudio';
import { ArchitectureStudio } from './components/ArchitectureStudio';
import { AttentionVisualization } from './components/AttentionVisualization';
import { TrainingCenter } from './components/TrainingCenter';
import { ModelEvaluationView } from './components/ModelEvaluation';
import { PredictionPlayground } from './components/PredictionPlayground';
import { GenerationWorkspace } from './components/GenerationWorkspace';
import { ResearchDocumentation } from './components/ResearchDocumentation';
import { api } from './services/api';
import { SystemStatus } from './types';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<string>('dashboard');
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [backendError, setBackendError] = useState<string | null>(null);

  const fetchStatus = async () => {
    try {
      const data = await api.getSystemStatus();
      setStatus(data);
      setBackendError(null);
    } catch (err: any) {
      setBackendError(err.message || 'Connecting to Transformer API...');
    }
  };

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 3000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans selection:bg-indigo-500/30 selection:text-indigo-200">
      {/* Top Navigation */}
      <Header activeTab={activeTab} setActiveTab={setActiveTab} status={status} />

      {/* Backend connection warning if API is not yet running */}
      {backendError && (
        <div className="bg-amber-500/10 border-b border-amber-500/20 text-amber-300 text-xs px-4 py-2 text-center font-medium">
          Backend notice: {backendError} (Ensure FastAPI dev server is running on port 8000)
        </div>
      )}

      {/* Main Container */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6 sm:py-8">
        {activeTab === 'dashboard' && (
          <Dashboard status={status} setActiveTab={setActiveTab} />
        )}

        {activeTab === 'dataset' && (
          <DatasetTokenizerStudio onCorpusChanged={fetchStatus} />
        )}

        {activeTab === 'architecture' && (
          <ArchitectureStudio />
        )}

        {activeTab === 'attention' && (
          <AttentionVisualization />
        )}

        {activeTab === 'training' && (
          <TrainingCenter status={status} onTrainingUpdated={fetchStatus} />
        )}

        {activeTab === 'evaluation' && (
          <ModelEvaluationView
            evaluation={status?.latest_evaluation || null}
            onEvaluationUpdated={() => fetchStatus()}
          />
        )}

        {activeTab === 'prediction' && (
          <PredictionPlayground />
        )}

        {activeTab === 'generation' && (
          <GenerationWorkspace />
        )}

        {activeTab === 'docs' && (
          <ResearchDocumentation />
        )}
      </main>

      {/* Studio Footer */}
      <footer className="border-t border-slate-900 bg-slate-950/80 py-6 text-center text-xs text-slate-500 font-mono">
        <div className="max-w-7xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-2">
          <div>
            TRANSFORMER INTELLIGENCE STUDIO &bull; Self-Attention Language Modeling Platform
          </div>
          <div className="text-slate-400">
            Pure PyTorch Tensor Operations &bull; Manual Multi-Head Attention &bull; React + Vite
          </div>
        </div>
      </footer>
    </div>
  );
};

export default App;
