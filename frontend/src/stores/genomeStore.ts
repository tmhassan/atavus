import { create } from 'zustand';
import { GenomeAnalysisResponse, AnalysisResults } from '@/types/genome';

interface GenomeStore {
  // Current analysis
  currentAnalysis: GenomeAnalysisResponse | null;
  analysisResults: AnalysisResults | null;
  
  // Upload state
  isUploading: boolean;
  uploadProgress: number;
  
  // Analysis state
  isAnalyzing: boolean;
  analysisProgress: number;
  
  // UI state
  activeTab: string;
  showConfetti: boolean;
  
  // Actions
  setCurrentAnalysis: (analysis: GenomeAnalysisResponse | null) => void;
  setAnalysisResults: (results: AnalysisResults | null) => void;
  setUploadProgress: (progress: number) => void;
  setAnalysisProgress: (progress: number) => void;
  setIsUploading: (uploading: boolean) => void;
  setIsAnalyzing: (analyzing: boolean) => void;
  setActiveTab: (tab: string) => void;
  setShowConfetti: (show: boolean) => void;
  
  // Reset state
  resetState: () => void;
}

export const useGenomeStore = create<GenomeStore>((set) => ({
  // Initial state
  currentAnalysis: null,
  analysisResults: null,
  isUploading: false,
  uploadProgress: 0,
  isAnalyzing: false,
  analysisProgress: 0,
  activeTab: 'harappa',
  showConfetti: false,

  // Actions
  setCurrentAnalysis: (analysis) => set({ currentAnalysis: analysis }),
  setAnalysisResults: (results) => set({ analysisResults: results }),
  setUploadProgress: (progress) => set({ uploadProgress: progress }),
  setAnalysisProgress: (progress) => set({ analysisProgress: progress }),
  setIsUploading: (uploading) => set({ isUploading: uploading }),
  setIsAnalyzing: (analyzing) => set({ isAnalyzing: analyzing }),
  setActiveTab: (tab) => set({ activeTab: tab }),
  setShowConfetti: (show) => set({ showConfetti: show }),

  // Reset state
  resetState: () => set({
    currentAnalysis: null,
    analysisResults: null,
    isUploading: false,
    uploadProgress: 0,
    isAnalyzing: false,
    analysisProgress: 0,
    activeTab: 'harappa',
    showConfetti: false,
  }),
}));
