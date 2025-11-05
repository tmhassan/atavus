import axios from 'axios';
import { GenomeAnalysisResponse, AnalysisResults } from '@/types/genome';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 300000, // 5 minutes for large file uploads
});

// Request interceptor for logging
api.interceptors.request.use((config) => {
  console.log(`🚀 API Request: ${config.method?.toUpperCase()} ${config.url}`);
  return config;
});

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => {
    console.log(`✅ API Response: ${response.status} ${response.config.url}`);
    return response;
  },
  (error) => {
    console.error(`❌ API Error: ${error.response?.status} ${error.config?.url}`, error.response?.data);
    return Promise.reject(error);
  }
);

export const genomeAPI = {
  // Health check
  healthCheck: async (): Promise<any> => {
    const response = await api.get('/health');
    return response.data;
  },

  // Upload genome file
  uploadGenomeFile: async (
    file: File,
    onUploadProgress?: (progress: number) => void
  ): Promise<GenomeAnalysisResponse> => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('analysis_type', 'ultimate_ancestry');
    formData.append('include_health_traits', 'false');
    formData.append('include_haplogroups', 'false');

    const response = await api.post('/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      onUploadProgress: (progressEvent) => {
        if (progressEvent.total) {
          const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          onUploadProgress?.(progress);
        }
      },
    });

    return response.data;
  },

  // Get analysis status
  getAnalysisStatus: async (analysisId: string): Promise<GenomeAnalysisResponse> => {
    const response = await api.get(`/analysis/${analysisId}`);
    return response.data;
  },

  // Get analysis results
  getAnalysisResults: async (analysisId: string): Promise<AnalysisResults> => {
    const response = await api.get(`/analysis/${analysisId}/results`);
    return response.data;
  },

  // Download results
  downloadResults: async (analysisId: string, format: 'json' | 'txt' = 'json'): Promise<Blob> => {
    const response = await api.get(`/analysis/${analysisId}/download`, {
      params: { format },
      responseType: 'blob',
    });
    return response.data;
  },

  // List analyses
  listAnalyses: async (limit = 10, offset = 0): Promise<any> => {
    const response = await api.get('/analysis', {
      params: { limit, offset },
    });
    return response.data;
  },

  // Delete analysis
  deleteAnalysis: async (analysisId: string): Promise<void> => {
    await api.delete(`/analysis/${analysisId}`);
  },
};

export default api;
