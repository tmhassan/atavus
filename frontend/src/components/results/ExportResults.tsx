import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Button } from '@/components/ui/Button';
import { Card, CardContent } from '@/components/ui/Card';
import { Download, FileText, Image, Share2, Mail, X } from 'lucide-react';
import { AnalysisResults } from '@/types/genome';
import { genomeAPI } from '@/services/api';
import toast from 'react-hot-toast';

interface ExportResultsProps {
  results: AnalysisResults;
  analysisId: string;
}

export const ExportResults: React.FC<ExportResultsProps> = ({ results, analysisId }) => {
  const [showExportModal, setShowExportModal] = useState(false);
  const [isExporting, setIsExporting] = useState(false);

  const exportOptions = [
    {
      id: 'json',
      name: 'JSON Report',
      description: 'Complete data in JSON format',
      icon: FileText,
      format: 'json' as const
    },
    {
      id: 'txt',
      name: 'Text Report',
      description: 'Human-readable summary',
      icon: FileText,
      format: 'txt' as const
    }
  ];

  const handleExport = async (format: 'json' | 'txt') => {
    try {
      setIsExporting(true);
      const blob = await genomeAPI.downloadResults(analysisId, format);
      
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `genome_analysis_${analysisId}.${format}`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      
      toast.success(`${format.toUpperCase()} report downloaded successfully!`);
      setShowExportModal(false);
    } catch (error) {
      console.error('Export error:', error);
      toast.error('Export failed. Please try again.');
    } finally {
      setIsExporting(false);
    }
  };

  return (
    <>
      <Button
        onClick={() => setShowExportModal(true)}
        icon={<Download className="w-4 h-4" />}
      >
        Export Results
      </Button>

      <AnimatePresence>
        {showExportModal && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4"
            onClick={() => setShowExportModal(false)}
          >
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              onClick={(e) => e.stopPropagation()}
              className="bg-white rounded-xl shadow-2xl max-w-md w-full"
            >
              <Card className="border-0 shadow-none">
                <CardContent className="p-6">
                  <div className="flex items-center justify-between mb-6">
                    <h3 className="text-xl font-semibold text-gray-900">Export Results</h3>
                    <button
                      onClick={() => setShowExportModal(false)}
                      className="text-gray-400 hover:text-gray-600 transition-colors"
                    >
                      <X className="w-6 h-6" />
                    </button>
                  </div>

                  <div className="space-y-4">
                    {exportOptions.map((option) => {
                      const Icon = option.icon;
                      return (
                        <motion.button
                          key={option.id}
                          onClick={() => handleExport(option.format)}
                          disabled={isExporting}
                          className="w-full p-4 border-2 border-gray-200 rounded-lg hover:border-blue-300 hover:bg-blue-50 transition-all duration-200 text-left disabled:opacity-50 disabled:cursor-not-allowed"
                          whileHover={{ scale: 1.02 }}
                          whileTap={{ scale: 0.98 }}
                        >
                          <div className="flex items-center">
                            <Icon className="w-8 h-8 text-blue-600 mr-4" />
                            <div>
                              <p className="font-medium text-gray-900">{option.name}</p>
                              <p className="text-sm text-gray-600">{option.description}</p>
                            </div>
                          </div>
                        </motion.button>
                      );
                    })}
                  </div>

                  <div className="mt-6 pt-6 border-t border-gray-200">
                    <p className="text-sm text-gray-600 text-center">
                      Your analysis includes {results.snps_analyzed.toLocaleString()} SNPs across multiple calculators
                    </p>
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
};
