import React, { useEffect, useState } from 'react';
import { QueryClient, QueryClientProvider } from 'react-query';
import { BrowserRouter as Router, Routes, Route, Navigate, useParams } from 'react-router-dom';
import { Toaster } from 'react-hot-toast';
import { motion, AnimatePresence } from 'framer-motion';
import { useGenomeStore } from '@/stores/genomeStore';
import { FileUpload } from '@/components/upload/FileUpload';
import { AnalysisProgress } from '@/components/analysis/AnalysisProgress';
import { ResultsDashboard } from '@/components/results/ResultsDashboard';
import { Header } from '@/components/layout/Header';
import { Footer } from '@/components/layout/Footer';
import { LoadingScreen } from '@/components/ui/LoadingScreen';
import { ErrorBoundary } from '@/components/ui/ErrorBoundary';
import { genomeAPI } from '@/services/api';
import { AlertCircle } from 'lucide-react';
import toast from 'react-hot-toast';

// Configure React Query
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 3,
      retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 30000),
      staleTime: 5 * 60 * 1000, // 5 minutes
      cacheTime: 10 * 60 * 1000, // 10 minutes
    },
  },
});

export const App: React.FC = () => {
  const {
    currentAnalysis,
    analysisResults,
    isUploading,
    isAnalyzing,
    setCurrentAnalysis,
    setAnalysisResults,
    setIsAnalyzing,
    resetState,
  } = useGenomeStore();

  const [isInitializing, setIsInitializing] = useState(true);
  const [apiHealthy, setApiHealthy] = useState(false);

  // Initialize app and check API health
  useEffect(() => {
    const initializeApp = async () => {
      try {
        // Check API health
        await genomeAPI.healthCheck();
        setApiHealthy(true);
        
        // Check for existing analysis in URL or localStorage
        const urlParams = new URLSearchParams(window.location.search);
        const analysisId = urlParams.get('analysis') || localStorage.getItem('currentAnalysisId');
        
        if (analysisId) {
          try {
            const analysis = await genomeAPI.getAnalysisStatus(analysisId);
            setCurrentAnalysis(analysis);
            
            if (analysis.status === 'completed') {
              const results = await genomeAPI.getAnalysisResults(analysisId);
              setAnalysisResults(results);
            } else if (analysis.status === 'processing') {
              setIsAnalyzing(true);
              startPolling(analysisId);
            }
          } catch (error) {
            console.warn('Could not restore previous analysis:', error);
            localStorage.removeItem('currentAnalysisId');
          }
        }
      } catch (error) {
        console.error('API health check failed:', error);
        setApiHealthy(false);
        toast.error('Unable to connect to analysis server. Please check your connection.');
      } finally {
        setIsInitializing(false);
      }
    };

    initializeApp();
  }, [setCurrentAnalysis, setAnalysisResults, setIsAnalyzing]);

  // Save current analysis ID to localStorage
  useEffect(() => {
    if (currentAnalysis?.analysis_id) {
      localStorage.setItem('currentAnalysisId', currentAnalysis.analysis_id);
      
      // Update URL without page reload
      const url = new URL(window.location.href);
      url.searchParams.set('analysis', currentAnalysis.analysis_id);
      window.history.replaceState({}, '', url.toString());
    }
  }, [currentAnalysis?.analysis_id]);

  // Polling function for analysis progress
  const startPolling = (analysisId: string) => {
    const pollInterval = setInterval(async () => {
      try {
        const status = await genomeAPI.getAnalysisStatus(analysisId);
        setCurrentAnalysis(status);

        if (status.status === 'completed') {
          clearInterval(pollInterval);
          setIsAnalyzing(false);
          
          const results = await genomeAPI.getAnalysisResults(analysisId);
          setAnalysisResults(results);
          
          toast.success('🎉 Analysis completed successfully!');
          useGenomeStore.getState().setShowConfetti(true);
          
        } else if (status.status === 'failed') {
          clearInterval(pollInterval);
          setIsAnalyzing(false);
          toast.error(status.error_message || 'Analysis failed. Please try again.');
        }
      } catch (error) {
        console.error('Polling error:', error);
        clearInterval(pollInterval);
        setIsAnalyzing(false);
        toast.error('Error checking analysis status.');
      }
    }, 2000);

    return () => clearInterval(pollInterval);
  };

  // Handle new analysis start
  const handleNewAnalysis = () => {
    resetState();
    localStorage.removeItem('currentAnalysisId');
    
    // Clear URL parameters
    const url = new URL(window.location.href);
    url.searchParams.delete('analysis');
    window.history.replaceState({}, '', url.toString());
  };

  // Determine current app state
  const getAppState = () => {
    if (isInitializing) return 'initializing';
    if (!apiHealthy) return 'api-error';
    if (isUploading) return 'uploading';
    if (isAnalyzing || (currentAnalysis && currentAnalysis.status === 'processing')) return 'analyzing';
    if (analysisResults && currentAnalysis?.status === 'completed') return 'results';
    return 'upload';
  };

  const appState = getAppState();

  // Loading screen component
  if (isInitializing) {
    return <LoadingScreen />;
  }

  // API error screen
  if (!apiHealthy) {
    return (
      <div className="min-h-screen bg-dark-bg-primary flex items-center justify-center">
        <div className="max-w-md w-full mx-4">
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            className="dark-card p-8 text-center"
          >
            <AlertCircle className="w-16 h-16 text-dark-accent-error mx-auto mb-4" />
            <h2 className="text-2xl font-bold text-dark-text-primary mb-2 font-sans">
              Service Unavailable
            </h2>
            <p className="text-dark-text-secondary mb-6">
              Unable to connect to the genome analysis server. Please check your connection and try again.
            </p>
            <button
              onClick={() => window.location.reload()}
              className="dark-btn px-6 py-3 rounded-lg font-medium"
            >
              Retry Connection
            </button>
          </motion.div>
        </div>
      </div>
    );
  }

  return (
    <ErrorBoundary>
      <QueryClientProvider client={queryClient}>
        <Router>
          <div className="min-h-screen bg-dark-bg-primary text-dark-text-primary">
            {/* Toast Notifications */}
            <Toaster
              position="top-right"
              toastOptions={{
                duration: 4000,
                style: {
                  background: '#161616',
                  color: '#ffffff',
                  boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.3), 0 4px 6px -2px rgba(0, 0, 0, 0.2)',
                  borderRadius: '12px',
                  border: '1px solid #262626',
                },
                success: {
                  iconTheme: {
                    primary: '#10b981',
                    secondary: '#ffffff',
                  },
                },
                error: {
                  iconTheme: {
                    primary: '#ef4444',
                    secondary: '#ffffff',
                  },
                },
              }}
            />

            {/* Header */}
            <Header 
              currentState={appState}
              onNewAnalysis={handleNewAnalysis}
              analysisId={currentAnalysis?.analysis_id}
            />

            {/* Main Content */}
            <main className="flex-1">
              <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
                <AnimatePresence mode="wait">
                  <motion.div
                    key={appState}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -20 }}
                    transition={{ duration: 0.3 }}
                    className="dark-slide-up"
                  >
                    <Routes>
                      {/* Upload Route */}
                      <Route 
                        path="/" 
                        element={
                          appState === 'upload' ? (
                            <FileUpload />
                          ) : appState === 'analyzing' ? (
                            <AnalysisProgress />
                          ) : appState === 'results' ? (
                            <Navigate to="/results" replace />
                          ) : (
                            <FileUpload />
                          )
                        } 
                      />
                      
                      {/* Results Route */}
                      <Route 
                        path="/results" 
                        element={
                          analysisResults && currentAnalysis ? (
                            <ResultsDashboard
                              results={analysisResults}
                              analysisId={currentAnalysis.analysis_id}
                            />
                          ) : (
                            <Navigate to="/" replace />
                          )
                        } 
                      />
                      
                      {/* Analysis Route */}
                      <Route 
                        path="/analysis/:id" 
                        element={<AnalysisRoute />}
                      />
                      
                      {/* Catch-all redirect */}
                      <Route path="*" element={<Navigate to="/" replace />} />
                    </Routes>
                  </motion.div>
                </AnimatePresence>
              </div>
            </main>

            {/* Footer */}
            <Footer />
          </div>
        </Router>
      </QueryClientProvider>
    </ErrorBoundary>
  );
};

// Analysis route component for direct links with proper TypeScript
interface RouteParams {
  id: string;
}

const AnalysisRoute: React.FC = () => {
  const { id } = useParams<keyof RouteParams>() as RouteParams;
  const { setCurrentAnalysis, setAnalysisResults, setIsAnalyzing } = useGenomeStore();
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const loadAnalysis = async () => {
      if (!id) return;

      try {
        const analysis = await genomeAPI.getAnalysisStatus(id);
        setCurrentAnalysis(analysis);

        if (analysis.status === 'completed') {
          const results = await genomeAPI.getAnalysisResults(id);
          setAnalysisResults(results);
        } else if (analysis.status === 'processing') {
          setIsAnalyzing(true);
        }
      } catch (error) {
        console.error('Failed to load analysis:', error);
        setError('Analysis not found or has expired.');
      } finally {
        setLoading(false);
      }
    };

    loadAnalysis();
  }, [id, setCurrentAnalysis, setAnalysisResults, setIsAnalyzing]);

  if (loading) {
    return <LoadingScreen />;
  }

  if (error) {
    return (
      <div className="text-center py-12">
        <AlertCircle className="w-16 h-16 text-dark-accent-error mx-auto mb-4" />
        <h2 className="text-2xl font-bold text-dark-text-primary mb-2 font-sans">Analysis Not Found</h2>
        <p className="text-dark-text-secondary mb-6">{error}</p>
        <Navigate to="/" replace />
      </div>
    );
  }

  return <Navigate to="/results" replace />;
};

export default App;
