import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import Confetti from 'react-confetti';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { AncestryVisualization } from './AncestryVisualization';
import { RegionalBreakdown } from './RegionalBreakdown';
import { G25Coordinates } from './G25Coordinates';
import { QualityMetrics } from './QualityMetrics';
import { AtaviaAI } from './AtaviaAI'; // New component
import { ExportResults } from './ExportResults';
import { 
  Dna, 
  Globe, 
  MapPin, 
  Compass, 
  BarChart3, 
  Share2, 
  Star,
  Clock,
  TrendingUp,
  Award,
  ChevronRight,
  Info,
  Bot // New icon for AI
} from 'lucide-react';
import { AnalysisResults } from '@/types/genome';
import { useGenomeStore } from '@/stores/genomeStore';
import toast from 'react-hot-toast';

interface ResultsDashboardProps {
  results: AnalysisResults;
  analysisId: string;
}

export const ResultsDashboard: React.FC<ResultsDashboardProps> = ({ results, analysisId }) => {
  const { showConfetti, setShowConfetti, activeTab, setActiveTab } = useGenomeStore();
  const [windowDimensions, setWindowDimensions] = useState({ width: 0, height: 0 });

  useEffect(() => {
    const updateWindowDimensions = () => {
      setWindowDimensions({ width: window.innerWidth, height: window.innerHeight });
    };

    updateWindowDimensions();
    window.addEventListener('resize', updateWindowDimensions);

    return () => window.removeEventListener('resize', updateWindowDimensions);
  }, []);

  useEffect(() => {
    if (showConfetti) {
      const timer = setTimeout(() => setShowConfetti(false), 5000);
      return () => clearTimeout(timer);
    }
  }, [showConfetti, setShowConfetti]);

  // Calculate summary statistics
  const summaryStats = {
    totalSNPs: results.snps_analyzed,
    analysisTime: results.analysis_metadata.processing_time,
    qualityScore: results.quality_metrics.overall_quality_score || 94.5,
    coordinateMagnitude: results.g25_coordinates.magnitude,
    topAncestry: Object.entries(results.ancestry_analysis.harappa_world)
      .sort(([,a], [,b]) => b - a)[0],
  };

  // Navigation tabs - UPDATED with Atavia
  const navigationTabs = [
    {
      id: 'overview',
      name: 'Overview',
      icon: TrendingUp,
      description: 'Summary & highlights',
      color: 'blue'
    },
    {
      id: 'ancestry',
      name: 'Ancestry Analysis',
      icon: Globe,
      description: 'Multiple calculators',
      color: 'green'
    },
    {
      id: 'regional',
      name: 'Regional Breakdown',
      icon: MapPin,
      description: 'Detailed regions',
      color: 'purple'
    },
    {
      id: 'coordinates',
      name: 'G25 Coordinates',
      icon: Compass,
      description: '25D genetic space',
      color: 'blue'
    },
    {
      id: 'quality',
      name: 'Quality Metrics',
      icon: BarChart3,
      description: 'Analysis details',
      color: 'orange'
    },
    {
      id: 'atavia',
      name: 'Atavia AI',
      icon: Bot,
      description: 'AI-powered insights',
      color: 'purple'
    }
  ];

  const handleShare = async () => {
    if (navigator.share) {
      try {
        await navigator.share({
          title: 'My Genome Analysis Results',
          text: `Check out my ancestry analysis: ${summaryStats.topAncestry[0]} ${summaryStats.topAncestry[1].toFixed(1)}%`,
          url: window.location.href,
        });
      } catch (error) {
        console.log('Error sharing:', error);
      }
    } else {
      navigator.clipboard.writeText(window.location.href);
      toast.success('Link copied to clipboard!');
    }
  };

  return (
    <div className="min-h-screen bg-dark-bg-primary">
      {/* Confetti Effect */}
      {showConfetti && (
        <Confetti
          width={windowDimensions.width}
          height={windowDimensions.height}
          recycle={false}
          numberOfPieces={200}
          gravity={0.3}
        />
      )}

      {/* Header Section */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-dark-bg-secondary border-b border-dark-border-primary"
      >
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between">
            <div className="flex items-center">
              <motion.div
                animate={{ rotate: 360 }}
                transition={{ duration: 8, repeat: Infinity, ease: "linear" }}
                className="mr-4"
              >
                <Dna className="w-10 h-10 text-dark-accent-secondary" />
              </motion.div>
              <div>
                <h1 className="text-3xl font-bold text-dark-text-primary font-sans">
                  Genome Analysis Results
                </h1>
                <p className="text-dark-text-secondary mt-1">
                  Complete ancestry analysis with {summaryStats.totalSNPs.toLocaleString()} SNPs
                </p>
              </div>
            </div>
            
            <div className="flex items-center space-x-3 mt-4 lg:mt-0">
              <Button
                onClick={handleShare}
                variant="outline"
                className="dark-btn-secondary dark-interactive"
                icon={<Share2 className="w-4 h-4" />}
              >
                Share
              </Button>
              <ExportResults results={results} analysisId={analysisId} />
            </div>
          </div>
        </div>
      </motion.div>

      {/* Summary Cards */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8"
        >
          {/* Top Ancestry */}
          <div className="dark-card dark-card-interactive bg-gradient-to-br from-dark-bg-card to-dark-bg-hover border-dark-accent-secondary/30">
            <div className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-dark-accent-secondary">Top Ancestry</p>
                  <p className="text-2xl font-bold text-dark-text-primary">
                    {summaryStats.topAncestry[1].toFixed(1)}%
                  </p>
                  <p className="text-sm text-dark-text-secondary mt-1">
                    {summaryStats.topAncestry[0].replace(/_/g, ' ')}
                  </p>
                </div>
                <Award className="w-8 h-8 text-dark-accent-secondary" />
              </div>
            </div>
          </div>

          {/* SNPs Analyzed */}
          <div className="dark-card dark-card-interactive bg-gradient-to-br from-dark-bg-card to-dark-bg-hover border-dark-accent-primary/30">
            <div className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-dark-accent-primary">SNPs Analyzed</p>
                  <p className="text-2xl font-bold text-dark-text-primary">
                    {(summaryStats.totalSNPs / 1000).toFixed(0)}K
                  </p>
                  <p className="text-sm text-dark-text-secondary mt-1">
                    High quality markers
                  </p>
                </div>
                <Dna className="w-8 h-8 text-dark-accent-primary" />
              </div>
            </div>
          </div>

          {/* Quality Score */}
          <div className="dark-card dark-card-interactive bg-gradient-to-br from-dark-bg-card to-dark-bg-hover border-dark-accent-tertiary/30">
            <div className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-dark-accent-tertiary">Quality Score</p>
                  <p className="text-2xl font-bold text-dark-text-primary">
                    {summaryStats.qualityScore.toFixed(1)}%
                  </p>
                  <p className="text-sm text-dark-text-secondary mt-1">
                    Analysis accuracy
                  </p>
                </div>
                <Star className="w-8 h-8 text-dark-accent-tertiary" />
              </div>
            </div>
          </div>

          {/* Processing Time */}
          <div className="dark-card dark-card-interactive bg-gradient-to-br from-dark-bg-card to-dark-bg-hover border-dark-accent-warning/30">
            <div className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-dark-accent-warning">Processing Time</p>
                  <p className="text-2xl font-bold text-dark-text-primary">
                    {summaryStats.analysisTime}
                  </p>
                  <p className="text-sm text-dark-text-secondary mt-1">
                    Ultra-fast analysis
                  </p>
                </div>
                <Clock className="w-8 h-8 text-dark-accent-warning" />
              </div>
            </div>
          </div>
        </motion.div>

        {/* Navigation Tabs - UPDATED with 6 tabs */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="mb-8"
        >
          <div className="dark-card overflow-hidden">
            <div className="p-0">
              <div className="grid grid-cols-1 lg:grid-cols-6 divide-y lg:divide-y-0 lg:divide-x divide-dark-border-primary">
                {navigationTabs.map((tab) => {
                  const Icon = tab.icon;
                  const isActive = activeTab === tab.id;
                  
                  return (
                    <motion.button
                      key={tab.id}
                      onClick={() => setActiveTab(tab.id)}
                      className={`
                        p-6 text-left transition-all duration-200 hover:bg-dark-bg-hover dark-interactive
                        ${isActive ? 'bg-dark-bg-hover border-l-4 border-dark-accent-primary' : ''}
                      `}
                      whileHover={{ scale: 1.02 }}
                      whileTap={{ scale: 0.98 }}
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex items-center">
                          <Icon className={`w-6 h-6 mr-3 ${
                            isActive ? 'text-dark-accent-primary' : 'text-dark-text-secondary'
                          }`} />
                          <div>
                            <p className={`font-medium ${
                              isActive ? 'text-dark-text-primary' : 'text-dark-text-primary'
                            }`}>
                              {tab.name}
                            </p>
                            <p className={`text-sm ${
                              isActive ? 'text-dark-accent-primary' : 'text-dark-text-secondary'
                            }`}>
                              {tab.description}
                            </p>
                          </div>
                        </div>
                        <ChevronRight className={`w-5 h-5 ${
                          isActive ? 'text-dark-accent-primary' : 'text-dark-text-muted'
                        }`} />
                      </div>
                    </motion.button>
                  );
                })}
              </div>
            </div>
          </div>
        </motion.div>

        {/* Main Content Area - UPDATED with Atavia */}
        <AnimatePresence mode="wait">
          <motion.div
            key={activeTab}
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            transition={{ duration: 0.3 }}
            className="dark-slide-up"
          >
            {activeTab === 'overview' && (
              <OverviewTab results={results} summaryStats={summaryStats} />
            )}
            {activeTab === 'ancestry' && (
              <AncestryVisualization results={results} />
            )}
            {activeTab === 'regional' && (
              <RegionalBreakdown results={results} />
            )}
            {activeTab === 'coordinates' && (
              <G25Coordinates results={results} />
            )}
            {activeTab === 'quality' && (
              <QualityMetrics results={results} />
            )}
            {activeTab === 'atavia' && (
              <AtaviaAI results={results} analysisId={analysisId} />
            )}
          </motion.div>
        </AnimatePresence>
      </div>
    </div>
  );
};

// Overview Tab Component (unchanged)
const OverviewTab: React.FC<{ results: AnalysisResults; summaryStats: any }> = ({ 
  results, 
  summaryStats 
}) => {
  const topAncestries = Object.entries(results.ancestry_analysis.harappa_world)
    .sort(([,a], [,b]) => b - a)
    .slice(0, 5);

  const hasRegionalData = Object.keys(results.ancestry_analysis.regional_breakdowns.south_asian || {}).length > 0;

  return (
    <div className="space-y-6">
      {/* Welcome Message */}
      <div className="dark-card bg-dark-gradient-accent text-dark-text-primary overflow-hidden">
        <div className="p-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="relative z-10"
          >
            <h2 className="text-3xl font-bold mb-4 font-sans">
              🎉 Your Genome Analysis is Complete!
            </h2>
            <p className="text-xl text-blue-100 mb-6">
              We've analyzed {summaryStats.totalSNPs.toLocaleString()} SNPs using multiple calculators 
              to give you the most comprehensive ancestry breakdown available.
            </p>
            <div className="flex flex-wrap gap-4">
              <div className="bg-white/20 backdrop-blur-sm rounded-lg p-4 dark-glass">
                <p className="text-sm text-blue-100">Primary Ancestry</p>
                <p className="text-2xl font-bold">
                  {summaryStats.topAncestry[0].replace(/_/g, ' ')}
                </p>
              </div>
              <div className="bg-white/20 backdrop-blur-sm rounded-lg p-4 dark-glass">
                <p className="text-sm text-blue-100">Percentage</p>
                <p className="text-2xl font-bold">
                  {summaryStats.topAncestry[1].toFixed(1)}%
                </p>
              </div>
              <div className="bg-white/20 backdrop-blur-sm rounded-lg p-4 dark-glass">
                <p className="text-sm text-blue-100">Quality</p>
                <p className="text-2xl font-bold">
                  {summaryStats.qualityScore.toFixed(0)}%
                </p>
              </div>
            </div>
          </motion.div>
          
          {/* Background DNA Pattern */}
          <div className="absolute top-0 right-0 opacity-10">
            <motion.div
              animate={{ rotate: 360 }}
              transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
            >
              <Dna className="w-32 h-32" />
            </motion.div>
          </div>
        </div>
      </div>

      {/* Quick Insights Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Top Ancestries */}
        <div className="dark-card">
          <div className="p-6 border-b border-dark-border-primary">
            <h3 className="flex items-center text-lg font-semibold text-dark-text-primary font-sans">
              <Globe className="w-5 h-5 mr-2 text-dark-accent-secondary" />
              Top Ancestry Components
            </h3>
          </div>
          <div className="p-6">
            <div className="space-y-4">
              {topAncestries.map(([name, percentage], index) => (
                <motion.div
                  key={name}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.1 }}
                  className="flex items-center justify-between p-3 bg-dark-bg-hover rounded-lg dark-interactive"
                >
                  <div className="flex items-center">
                    <div className="w-3 h-3 rounded-full bg-dark-accent-secondary mr-3" />
                    <span className="font-medium text-dark-text-primary">
                      {name.replace(/_/g, ' ')}
                    </span>
                  </div>
                  <span className="text-lg font-semibold text-dark-text-primary">
                    {percentage.toFixed(1)}%
                  </span>
                </motion.div>
              ))}
            </div>
          </div>
        </div>

        {/* Analysis Summary */}
        <div className="dark-card">
          <div className="p-6 border-b border-dark-border-primary">
            <h3 className="flex items-center text-lg font-semibold text-dark-text-primary font-sans">
              <BarChart3 className="w-5 h-5 mr-2 text-dark-accent-primary" />
              Analysis Summary
            </h3>
          </div>
          <div className="p-6">
            <div className="space-y-4">
              <div className="flex justify-between items-center p-3 bg-dark-bg-hover rounded-lg">
                <span className="text-dark-text-secondary">Calculators Used</span>
                <span className="font-semibold text-dark-accent-primary">4</span>
              </div>
              <div className="flex justify-between items-center p-3 bg-dark-bg-hover rounded-lg">
                <span className="text-dark-text-secondary">G25 Coordinates</span>
                <span className="font-semibold text-dark-accent-tertiary">25D</span>
              </div>
              <div className="flex justify-between items-center p-3 bg-dark-bg-hover rounded-lg">
                <span className="text-dark-text-secondary">Regional Breakdown</span>
                <span className="font-semibold text-dark-accent-secondary">
                  {hasRegionalData ? 'Available' : 'Limited'}
                </span>
              </div>
              <div className="flex justify-between items-center p-3 bg-dark-bg-hover rounded-lg">
                <span className="text-dark-text-secondary">Processing Time</span>
                <span className="font-semibold text-dark-accent-warning">
                  {summaryStats.analysisTime}
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Next Steps - UPDATED with Atavia */}
      <div className="dark-card border-2 border-dashed border-dark-border-secondary">
        <div className="p-8 text-center">
          <Info className="w-12 h-12 text-dark-accent-primary mx-auto mb-4" />
          <h3 className="text-xl font-semibold text-dark-text-primary mb-2 font-sans">
            Explore Your Results
          </h3>
          <p className="text-dark-text-secondary mb-6">
            Use the tabs above to dive deeper into your ancestry analysis. 
            Each section provides detailed insights into different aspects of your genetic heritage.
          </p>
          <div className="flex flex-wrap justify-center gap-3">
            <Button
              onClick={() => useGenomeStore.getState().setActiveTab('ancestry')}
              variant="outline"
              className="dark-btn-secondary dark-interactive"
              icon={<Globe className="w-4 h-4" />}
            >
              View Detailed Analysis
            </Button>
            <Button
              onClick={() => useGenomeStore.getState().setActiveTab('coordinates')}
              variant="outline"
              className="dark-btn-secondary dark-interactive"
              icon={<Compass className="w-4 h-4" />}
            >
              Explore G25 Coordinates
            </Button>
            <Button
              onClick={() => useGenomeStore.getState().setActiveTab('atavia')}
              variant="outline"
              className="dark-btn-secondary dark-interactive"
              icon={<Bot className="w-4 h-4" />}
            >
              Ask Atavia AI
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
};
