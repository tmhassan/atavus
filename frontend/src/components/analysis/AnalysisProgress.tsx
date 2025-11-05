import React from 'react';
import { motion } from 'framer-motion';
import { CircularProgressbar, buildStyles } from 'react-circular-progressbar';
import { Dna, Brain, Globe, BarChart3 } from 'lucide-react';
import { Card, CardContent } from '@/components/ui/Card';
import { useGenomeStore } from '@/stores/genomeStore';
import 'react-circular-progressbar/dist/styles.css';

export const AnalysisProgress: React.FC = () => {
  const { currentAnalysis, isAnalyzing } = useGenomeStore();

  if (!currentAnalysis || !isAnalyzing) return null;

  const progress = currentAnalysis.progress;
  const status = currentAnalysis.status;

  const getStatusMessage = () => {
    if (progress < 30) return 'Parsing your genome data...';
    if (progress < 60) return 'Running ancestry calculations...';
    if (progress < 90) return 'Generating G25 coordinates...';
    return 'Finalizing results...';
  };

  const getStatusIcon = () => {
    if (progress < 30) return <Dna className="w-8 h-8 text-dark-accent-primary" />;
    if (progress < 60) return <Brain className="w-8 h-8 text-dark-accent-secondary" />;
    if (progress < 90) return <Globe className="w-8 h-8 text-dark-accent-tertiary" />;
    return <BarChart3 className="w-8 h-8 text-dark-accent-warning" />;
  };

  return (
    <div className="max-w-2xl mx-auto">
      <Card className="dark-card text-center">
        <CardContent className="py-12">
          <motion.div
            initial={{ scale: 0.8, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ duration: 0.5 }}
            className="space-y-8"
          >
            {/* Progress Circle */}
            <div className="w-32 h-32 mx-auto">
              <CircularProgressbar
                value={progress}
                text={`${progress}%`}
                styles={buildStyles({
                  textSize: '16px',
                  pathColor: '#3b82f6',
                  textColor: '#ffffff',
                  trailColor: '#202020',
                  pathTransitionDuration: 0.5,
                })}
              />
            </div>

            {/* Status Icon */}
            <motion.div
              animate={{ rotate: 360 }}
              transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
              className="flex justify-center"
            >
              {getStatusIcon()}
            </motion.div>

            {/* Status Message */}
            <div>
              <h3 className="text-xl font-semibold text-dark-text-primary mb-2 font-sans">
                Analyzing Your Genome
              </h3>
              <p className="text-dark-text-secondary text-lg">
                {getStatusMessage()}
              </p>
            </div>

            {/* Analysis Steps */}
            <div className="grid grid-cols-4 gap-4 mt-8">
              {[
                { step: 1, label: 'Parse', icon: Dna, completed: progress > 30 },
                { step: 2, label: 'Analyze', icon: Brain, completed: progress > 60 },
                { step: 3, label: 'Calculate', icon: Globe, completed: progress > 90 },
                { step: 4, label: 'Results', icon: BarChart3, completed: progress >= 100 },
              ].map(({ step, label, icon: Icon, completed }) => (
                <motion.div
                  key={step}
                  initial={{ opacity: 0.3 }}
                  animate={{ opacity: completed ? 1 : 0.3 }}
                  className={`p-3 rounded-lg border border-dark-border-primary ${completed ? 'bg-dark-bg-hover' : 'bg-dark-bg-card'}`}
                >
                  <Icon className={`w-6 h-6 mx-auto mb-2 ${completed ? 'text-dark-accent-primary' : 'text-dark-text-muted'}`} />
                  <p className={`text-sm font-medium ${completed ? 'text-dark-text-primary' : 'text-dark-text-muted'}`}>
                    {label}
                  </p>
                </motion.div>
              ))}
            </div>

            {/* File Info */}
            <div className="bg-dark-bg-card rounded-lg p-4 mt-6 border border-dark-border-primary">
              <p className="text-sm text-dark-text-secondary">
                <span className="font-medium text-dark-text-primary">File:</span> {currentAnalysis.file_info.filename}
              </p>
              <p className="text-sm text-dark-text-secondary">
                <span className="font-medium text-dark-text-primary">Size:</span> {(currentAnalysis.file_info.size / (1024 * 1024)).toFixed(1)} MB
              </p>
              <p className="text-sm text-dark-text-secondary">
                <span className="font-medium text-dark-text-primary">Started:</span> {new Date(currentAnalysis.created_at).toLocaleTimeString()}
              </p>
            </div>
          </motion.div>
        </CardContent>
      </Card>
    </div>
  );
};
