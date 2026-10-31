export interface SystemStatus {
  status: string;
  device: string;
  active_corpus: string;
  is_training: boolean;
  training_state: TrainingState;
  vocab_size: number;
  dataset_sizes: {
    train: number;
    val: number;
    test: number;
  };
  model_config: {
    d_model: number;
    num_heads: number;
    num_layers: number;
    d_ff: number;
    max_seq_len: number;
    dropout: number;
    vocab_size?: number;
  };
  model_parameters: ModelParameters;
  latest_evaluation: ModelEvaluation | null;
}

export interface ModelParameters {
  total_parameters: number;
  trainable_parameters: number;
  embedding_parameters: number;
  d_model: number;
  num_heads: number;
  d_k: number;
  num_layers: number;
  d_ff: number;
  max_seq_len: number;
  vocab_size: number;
  blocks: Array<{
    block_idx: number;
    total: number;
    attention: number;
    feed_forward: number;
  }>;
}

export interface TrainingState {
  status: 'idle' | 'training' | 'completed' | 'stopped' | 'error';
  current_epoch: number;
  total_epochs: number;
  current_step: number;
  total_steps: number;
  progress_percent: number;
  current_train_loss: number;
  current_val_loss: number;
  history: TrainingEpochHistory[];
  best_val_loss: number;
  best_epoch: number;
  elapsed_seconds: number;
  message: string;
}

export interface TrainingEpochHistory {
  epoch: number;
  train_loss: number;
  train_perplexity: number;
  train_top1_acc: number;
  val_loss: number;
  val_perplexity: number;
  val_top1_acc: number;
  learning_rate: number;
  elapsed_time: number;
}

export interface ModelEvaluation {
  cross_entropy_loss: number;
  perplexity: number;
  top1_accuracy: number;
  top3_accuracy: number;
  top5_accuracy: number;
  total_tokens_evaluated: number;
  prediction_examples: Array<{
    sample_index: number;
    context_preview: string;
    expected_target: string;
    top1_prediction: string;
    is_correct: boolean;
    in_top5: boolean;
    top5_candidates: Array<{
      token: string;
      id: number;
      probability: number;
    }>;
  }>;
  evaluated_at: string;
}

export interface TokenCandidate {
  rank: number;
  token: string;
  token_id: number;
  probability: number;
  raw_logit: number;
  is_selected: boolean;
}

export interface NextTokenPredictionResult {
  prompt: string;
  input_tokens: string[];
  input_ids: number[];
  selected_token: string;
  selected_token_id: number;
  top_candidates: TokenCandidate[];
  temperature: number;
  sampling_mode: string;
  attentions: Array<{
    layer: number;
    heads: number[][][]; // (H, T, T)
  }>;
}

export interface GenerationStep {
  step: number;
  token: string;
  token_id: number;
  probability: number;
  alternatives: Array<{
    token: string;
    prob: number;
  }>;
}

export interface GenerationResult {
  prompt: string;
  generated_text: string;
  tokens_added: number;
  history: GenerationStep[];
  hyperparameters: {
    max_new_tokens: number;
    temperature: number;
    top_k: number;
  };
}

export interface AttentionInspectionResult {
  tokens: string[];
  token_ids: number[];
  seq_len: number;
  num_layers: number;
  num_heads: number;
  d_k: number;
  layers: Array<{
    layer_idx: number;
    heads: number[][][];
    average_head: number[][];
  }>;
  causal_mask: number[][];
  math_explanation: {
    formula: string;
    scale_factor: number;
    causal_property: string;
  };
}

export interface ArchitectureComponent {
  id: string;
  name: string;
  layer_type: string;
  input_shape: string;
  output_shape: string;
  parameters: number;
  math_formula: string;
  explanation: string;
  pytorch_code: string;
}

export interface DatasetDetails {
  vocab_stats: {
    vocab_size: number;
    special_tokens_count: number;
    total_corpus_tokens: number;
    unique_raw_tokens: number;
    top_words: Array<{
      token: string;
      id: number;
      count: number;
    }>;
    special_tokens: Array<{
      token: string;
      id: number;
    }>;
  };
  corpus_id: string;
  sample_sequence_pairs: Array<{
    sample_index: number;
    input_preview: string;
    target_preview: string;
    input_ids_preview: number[];
    target_ids_preview: number[];
    full_length: number;
  }>;
  splits_count: {
    train: number;
    val: number;
    test: number;
  };
}
