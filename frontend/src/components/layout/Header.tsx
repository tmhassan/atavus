import React from 'react';
import { motion } from 'framer-motion';
import { Button } from '@/components/ui/Button';
import { Home, RefreshCw, BarChart3 } from 'lucide-react';

interface HeaderProps {
  currentState: string;
  onNewAnalysis: () => void;
  analysisId?: string;
}

export const Header: React.FC<HeaderProps> = ({ currentState, onNewAnalysis, analysisId }) => {
  return (
    <header className="bg-dark-bg-secondary border-b border-dark-border-primary sticky top-0 z-40 dark-glass-strong">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16 relative">
          {/* Left spacer - invisible but takes same space as right nav */}
          <div className="flex items-center space-x-3 opacity-0 pointer-events-none">
            {currentState === 'results' && (
              <>
                <Button
                  variant="outline"
                  size="sm"
                  className="dark-btn-secondary dark-interactive"
                  icon={<RefreshCw className="w-4 h-4" />}
                >
                  New Analysis
                </Button>
                {analysisId && (
                  <Button
                    variant="outline"
                    size="sm"
                    className="dark-btn-secondary dark-interactive"
                    icon={<BarChart3 className="w-4 h-4" />}
                  >
                    Share
                  </Button>
                )}
              </>
            )}
            
            {currentState !== 'upload' && (
              <Button
                variant="ghost"
                size="sm"
                className="dark-btn-secondary dark-interactive"
                icon={<Home className="w-4 h-4" />}
              >
                Home
              </Button>
            )}
          </div>

          {/* Centered atavus Logo */}
          <div className="absolute left-1/2 transform -translate-x-1/2">
            <motion.h1 
              className="text-2xl font-bold text-dark-text-primary font-sans relative"
              style={{
                textShadow: '0 0 20px rgba(59, 130, 246, 0.6), 0 0 40px rgba(59, 130, 246, 0.4), 0 0 60px rgba(59, 130, 246, 0.2)',
                filter: 'drop-shadow(0 0 10px rgba(59, 130, 246, 0.8))'
              }}
              animate={{
                textShadow: [
                  '0 0 20px rgba(59, 130, 246, 0.6), 0 0 40px rgba(59, 130, 246, 0.4), 0 0 60px rgba(59, 130, 246, 0.2)',
                  '0 0 25px rgba(59, 130, 246, 0.8), 0 0 50px rgba(59, 130, 246, 0.6), 0 0 75px rgba(59, 130, 246, 0.4)',
                  '0 0 20px rgba(59, 130, 246, 0.6), 0 0 40px rgba(59, 130, 246, 0.4), 0 0 60px rgba(59, 130, 246, 0.2)'
                ]
              }}
              transition={{
                duration: 3,
                repeat: Infinity,
                ease: "easeInOut"
              }}
            >
              atavus
            </motion.h1>
          </div>

          {/* Right Navigation Actions - Visible */}
          <div className="flex items-center space-x-3">
            {currentState === 'results' && (
              <>
                <Button
                  onClick={onNewAnalysis}
                  variant="outline"
                  size="sm"
                  className="dark-btn-secondary dark-interactive"
                  icon={<RefreshCw className="w-4 h-4" />}
                >
                  New Analysis
                </Button>
                {analysisId && (
                  <Button
                    onClick={() => window.open(`/analysis/${analysisId}`, '_blank')}
                    variant="outline"
                    size="sm"
                    className="dark-btn-secondary dark-interactive"
                    icon={<BarChart3 className="w-4 h-4" />}
                  >
                    Share
                  </Button>
                )}
              </>
            )}
            
            {currentState !== 'upload' && (
              <Button
                onClick={onNewAnalysis}
                variant="ghost"
                size="sm"
                className="dark-btn-secondary dark-interactive"
                icon={<Home className="w-4 h-4" />}
              >
                Home
              </Button>
            )}
          </div>
        </div>
      </div>
    </header>
  );
};
