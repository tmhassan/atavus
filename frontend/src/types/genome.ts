export interface GenomeAnalysisResponse {
  analysis_id: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  created_at: string;
  file_info: {
    filename: string;
    size: number;
    content_type: string;
  };
  results?: AnalysisResults;
  error_message?: string;
  progress: number;
}

export interface AnalysisResults {
  ancestry_analysis: {
    harappa_world: Record<string, number>;
    dodecad_k12b: Record<string, number>;
    eurogenes_k13: Record<string, number>;
    puntdnal: Record<string, number>;
    regional_breakdowns: {
      south_asian: Record<string, number>;
      west_eurasian: Record<string, number>;
      east_eurasian: Record<string, number>;
    };
  };
  g25_coordinates: {
    coordinates: number[];
    magnitude: number;
    quality_score: number;
  };
  quality_metrics: Record<string, number>;
  confidence_scores: Record<string, number>;
  snps_analyzed: number;
  analysis_metadata: {
    analysis_type: string;
    processing_time: string;
    timestamp: string;
  };
}

export interface UploadProgress {
  progress: number;
  status: string;
  message: string;
}
