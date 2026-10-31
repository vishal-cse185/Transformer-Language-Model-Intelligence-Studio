import {
  ArchitectureComponent,
  AttentionInspectionResult,
  DatasetDetails,
  GenerationResult,
  ModelEvaluation,
  NextTokenPredictionResult,
  SystemStatus,
  TrainingState,
} from '../types';

const API_BASE = '/api';

export const api = {
  async getSystemStatus(): Promise<SystemStatus> {
    const res = await fetch(`${API_BASE}/status`);
    if (!res.ok) throw new Error(`Failed to fetch status: ${res.statusText}`);
    return res.json();
  },

  async getCorpora(): Promise<{ available_corpora: Record<string, any>; active_corpus_id: string }> {
    const res = await fetch(`${API_BASE}/corpora`);
    if (!res.ok) throw new Error('Failed to fetch corpora');
    return res.json();
  },

  async selectCorpus(corpus_id: string, custom_text?: string): Promise<{ message: string; stats: any }> {
    const res = await fetch(`${API_BASE}/dataset/select`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ corpus_id, custom_text, seq_len: 32 }),
    });
    if (!res.ok) throw new Error('Failed to select corpus');
    return res.json();
  },

  async getDatasetDetails(): Promise<DatasetDetails> {
    const res = await fetch(`${API_BASE}/dataset/details`);
    if (!res.ok) throw new Error('Failed to fetch dataset details');
    return res.json();
  },

  async testTokenizer(text: string): Promise<any> {
    const res = await fetch(`${API_BASE}/tokenizer/test`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    });
    if (!res.ok) throw new Error('Failed to test tokenizer');
    return res.json();
  },

  async getArchitecture(): Promise<{
    parameters: any;
    config: any;
    components: ArchitectureComponent[];
  }> {
    const res = await fetch(`${API_BASE}/architecture`);
    if (!res.ok) throw new Error('Failed to fetch architecture');
    return res.json();
  },

  async startTraining(params: {
    epochs: number;
    batch_size: number;
    learning_rate: number;
    weight_decay: number;
  }): Promise<any> {
    const res = await fetch(`${API_BASE}/train/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || 'Failed to start training');
    }
    return res.json();
  },

  async stopTraining(): Promise<any> {
    const res = await fetch(`${API_BASE}/train/stop`, {
      method: 'POST',
    });
    if (!res.ok) throw new Error('Failed to stop training');
    return res.json();
  },

  async getTrainingStatus(): Promise<TrainingState> {
    const res = await fetch(`${API_BASE}/train/status`);
    if (!res.ok) throw new Error('Failed to get training status');
    return res.json();
  },

  async evaluateModel(): Promise<ModelEvaluation> {
    const res = await fetch(`${API_BASE}/evaluate`, {
      method: 'POST',
    });
    if (!res.ok) throw new Error('Failed to evaluate model');
    return res.json();
  },

  async predictNextToken(params: {
    prompt: string;
    temperature: number;
    top_k: number;
    greedy: boolean;
  }): Promise<NextTokenPredictionResult> {
    const res = await fetch(`${API_BASE}/predict/next-token`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    });
    if (!res.ok) throw new Error('Failed to predict next token');
    return res.json();
  },

  async generateText(params: {
    prompt: string;
    max_new_tokens: number;
    temperature: number;
    top_k: number;
  }): Promise<GenerationResult> {
    const res = await fetch(`${API_BASE}/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    });
    if (!res.ok) throw new Error('Failed to generate text');
    return res.json();
  },

  async inspectAttention(text: string): Promise<AttentionInspectionResult> {
    const res = await fetch(`${API_BASE}/attention/inspect`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    });
    if (!res.ok) throw new Error('Failed to inspect attention');
    return res.json();
  },
};
