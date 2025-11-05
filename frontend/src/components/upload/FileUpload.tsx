import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { motion, AnimatePresence } from 'framer-motion';
import { Upload, File, AlertCircle, CheckCircle, Dna } from 'lucide-react';
import { Button } from '@/components/ui/Button';
import { Card } from '@/components/ui/Card';
import { genomeAPI } from '@/services/api';
import { useGenomeStore } from '@/stores/genomeStore';
import toast from 'react-hot-toast';

export const FileUpload: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [dragActive, setDragActive] = useState(false);
  
  const {
    setCurrentAnalysis,
    setIsUploading,
    setUploadProgress,
    setIsAnalyzing,
    uploadProgress,
    isUploading,
  } = useGenomeStore();

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (file) {
      setSelectedFile(file);
      toast.success(`File "${file.name}" selected successfully!`);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'text/plain': ['.txt'],
      'text/csv': ['.csv'],
    },
    maxFiles: 1,
    maxSize: 50 * 1024 * 1024,
    onDragEnter: () => setDragActive(true),
    onDragLeave: () => setDragActive(false),
    onDropRejected: (rejectedFiles) => {
      const error = rejectedFiles[0]?.errors[0];
      if (error?.code === 'file-too-large') {
        toast.error('File too large. Maximum size is 50MB.');
      } else if (error?.code === 'file-invalid-type') {
        toast.error('Invalid file type. Please upload a .txt or .csv file.');
      } else {
        toast.error('File upload rejected. Please try again.');
      }
    },
  });

  const handleUpload = async () => {
    if (!selectedFile) {
      toast.error('Please select a file first.');
      return;
    }

    try {
      setIsUploading(true);
      setUploadProgress(0);

      const analysis = await genomeAPI.uploadGenomeFile(
        selectedFile,
        (progress) => setUploadProgress(progress)
      );

      setCurrentAnalysis(analysis);
      setIsAnalyzing(true);
      
      toast.success('File uploaded successfully! Analysis started.');
      
      pollAnalysisProgress(analysis.analysis_id);
      
    } catch (error: any) {
      console.error('Upload error:', error);
      toast.error(error.response?.data?.detail || 'Upload failed. Please try again.');
    } finally {
      setIsUploading(false);
    }
  };

  const pollAnalysisProgress = async (analysisId: string) => {
    const pollInterval = setInterval(async () => {
      try {
        const status = await genomeAPI.getAnalysisStatus(analysisId);
        setCurrentAnalysis(status);

        if (status.status === 'completed') {
          clearInterval(pollInterval);
          setIsAnalyzing(false);
          toast.success('🎉 Analysis completed successfully!');
          
          const results = await genomeAPI.getAnalysisResults(analysisId);
          useGenomeStore.getState().setAnalysisResults(results);
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
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <div className="min-h-screen bg-dark-bg-primary">
      {/* Simplified Modern Header - Matches the overall dark theme */}
      <div className="relative overflow-hidden bg-dark-bg-primary py-12">
        <motion.div
          initial={{ opacity: 0, y: -30 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center max-w-4xl mx-auto px-4"
        >
          <motion.div
            className="inline-block mb-6"
            animate={{ rotate: 360 }}
            transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
          >
            <Dna className="w-16 h-16 text-dark-accent-primary mx-auto" />
          </motion.div>
          
          <h1 className="text-4xl font-bold text-dark-text-primary mb-4 font-sans">
            Discover Your Genetic Heritage
          </h1>
          <p className="text-lg text-dark-text-secondary max-w-2xl mx-auto leading-relaxed">
            Uncover the stories written in your DNA. Journey through time to explore your ancestral 
            roots with our advanced genomic analysis platform featuring multiple calculators and 
            G25 coordinates.
          </p>
        </motion.div>
      </div>

      {/* Main Upload Section */}
      <div className="max-w-4xl mx-auto px-4 pb-16">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="dark-slide-up"
        >
          <div className="dark-card dark-card-interactive">
            <div className="px-8 py-12">
              {/* Upload Section Header */}
              <div className="text-center mb-12">
                <motion.div
                  className="dark-pulse inline-block mb-6"
                >
                  <div className="relative">
                    <Dna className="w-20 h-20 text-dark-accent-primary mx-auto" />
                    <div className="absolute inset-0 bg-dark-accent-primary opacity-20 rounded-full blur-xl dark-glow"></div>
                  </div>
                </motion.div>
                
                <h2 className="text-3xl font-bold text-dark-text-primary mb-4 font-sans">
                  Upload Your Genetic Data
                </h2>
                <p className="text-dark-text-secondary max-w-2xl mx-auto leading-relaxed">
                  Share your 23andMe raw data to begin an extraordinary journey through your ancestral 
                  timeline. Discover the civilizations, migrations, and stories that shaped your genetic heritage.
                </p>
              </div>

              {/* Enhanced File Drop Zone */}
              <div
                {...getRootProps()}
                className={`
                  relative dark-border-dashed rounded-2xl p-12 text-center cursor-pointer 
                  transition-all duration-500 transform dark-interactive
                  ${isDragActive || dragActive 
                    ? 'dark-border-accent bg-dark-bg-hover shadow-dark-glow scale-105 dark-glass' 
                    : 'border-dark-border-secondary hover:border-dark-accent-primary hover:bg-dark-bg-hover hover:shadow-dark-lg dark-glass'
                  }
                `}
              >
                <motion.div
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  className="pointer-events-none"
                >
                  <input {...getInputProps()} />
                  
                  <AnimatePresence mode="wait">
                    {selectedFile ? (
                      <motion.div
                        key="file-selected"
                        initial={{ opacity: 0, scale: 0.8 }}
                        animate={{ opacity: 1, scale: 1 }}
                        exit={{ opacity: 0, scale: 0.8 }}
                        className="space-y-6 dark-slide-up"
                      >
                        <div className="relative">
                          <CheckCircle className="w-16 h-16 text-dark-accent-secondary mx-auto dark-pulse" />
                          <div className="absolute inset-0 bg-dark-accent-secondary opacity-20 rounded-full blur-xl"></div>
                        </div>
                        
                        <div className="dark-glass-strong rounded-xl p-6 dark-border">
                          <p className="text-2xl font-semibold text-dark-text-primary mb-2 font-sans">
                            {selectedFile.name}
                          </p>
                          <p className="text-dark-accent-primary font-medium text-lg">
                            {formatFileSize(selectedFile.size)}
                          </p>
                          <div className="mt-4 flex items-center justify-center text-dark-text-secondary">
                            <CheckCircle className="w-5 h-5 mr-2" />
                            <span className="font-medium">Ready for genetic analysis</span>
                          </div>
                        </div>
                        
                        <Button
                          onClick={(e) => {
                            e.stopPropagation();
                            setSelectedFile(null);
                          }}
                          variant="outline"
                          className="dark-btn-secondary px-6 py-2 dark-interactive"
                        >
                          <Upload className="w-4 h-4 mr-2" />
                          Choose Different File
                        </Button>
                      </motion.div>
                    ) : (
                      <motion.div
                        key="no-file"
                        initial={{ opacity: 0, scale: 0.8 }}
                        animate={{ opacity: 1, scale: 1 }}
                        exit={{ opacity: 0, scale: 0.8 }}
                        className="space-y-6 dark-slide-up"
                      >
                        <div className="relative">
                          <Upload className="w-16 h-16 text-dark-accent-primary mx-auto dark-float" />
                          <div className="absolute inset-0 bg-dark-accent-primary opacity-20 rounded-full blur-xl"></div>
                        </div>
                        
                        <div>
                          <p className="text-2xl font-semibold text-dark-text-primary mb-3 font-sans">
                            {isDragActive ? 'Drop your file here' : 'Choose Your Genetic Data'}
                          </p>
                          <p className="text-dark-accent-primary font-medium">
                            23andMe raw data files (.txt, .csv) • Maximum 50MB
                          </p>
                          <div className="mt-4 flex items-center justify-center text-dark-text-secondary">
                            <Dna className="w-5 h-5 mr-2 text-dark-accent-secondary" />
                            <span>Discover your ancestral origins</span>
                          </div>
                        </div>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </motion.div>
              </div>

              {/* Upload Progress */}
              <AnimatePresence>
                {isUploading && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    exit={{ opacity: 0, height: 0 }}
                    className="mt-8 dark-slide-up"
                  >
                    <div className="dark-glass-strong rounded-full h-4 overflow-hidden dark-border">
                      <motion.div
                        className="bg-dark-gradient-accent h-full rounded-full"
                        initial={{ width: 0 }}
                        animate={{ width: `${uploadProgress}%` }}
                        transition={{ duration: 0.3 }}
                      />
                    </div>
                    <p className="text-dark-text-secondary font-medium mt-3 text-center">
                      Uploading your genetic data... {uploadProgress}%
                    </p>
                  </motion.div>
                )}
              </AnimatePresence>

              {/* Modern Upload Button */}
              <div className="mt-8">
                <button
                  onClick={handleUpload}
                  disabled={!selectedFile || isUploading}
                  className="w-full dark-btn text-xl py-4 font-semibold rounded-dark shadow-dark-xl hover:shadow-dark-glow transform hover:scale-105 transition-all duration-300 font-sans disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none"
                >
                  {isUploading ? (
                    <>
                      <Dna className="w-6 h-6 mr-3 animate-spin" />
                      Analyzing Your Genome...
                    </>
                  ) : (
                    <>
                      <Dna className="w-6 h-6 mr-3" />
                      Begin Genetic Analysis
                    </>
                  )}
                </button>
              </div>

              {/* File Requirements */}
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.4 }}
                className="mt-8 dark-glass-strong p-6 rounded-dark dark-border dark-slide-up"
              >
                <div className="flex items-start">
                  <AlertCircle className="w-6 h-6 text-dark-accent-primary mt-1 mr-4 flex-shrink-0" />
                  <div className="text-dark-text-secondary">
                    <p className="font-semibold mb-3 text-lg font-sans text-dark-text-primary">File Requirements:</p>
                    <ul className="space-y-2 text-dark-text-secondary">
                      <li className="flex items-center">
                        <span className="w-2 h-2 bg-dark-accent-primary rounded-full mr-3"></span>
                        23andMe raw data file (.txt or .csv format)
                      </li>
                      <li className="flex items-center">
                        <span className="w-2 h-2 bg-dark-accent-primary rounded-full mr-3"></span>
                        Maximum file size: 50MB
                      </li>
                      <li className="flex items-center">
                        <span className="w-2 h-2 bg-dark-accent-primary rounded-full mr-3"></span>
                        File should contain SNP data with rsIDs
                      </li>
                      <li className="flex items-center">
                        <span className="w-2 h-2 bg-dark-accent-primary rounded-full mr-3"></span>
                        Analysis typically takes 15-30 seconds
                      </li>
                    </ul>
                  </div>
                </div>
              </motion.div>
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  );
};
