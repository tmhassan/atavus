import React, { useState, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar } from 'recharts';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Globe, BarChart3, Radar as RadarIcon, Map, TrendingUp, AlertCircle } from 'lucide-react';
import { AnalysisResults } from '@/types/genome';

interface AncestryVisualizationProps {
  results: AnalysisResults;
}

export const AncestryVisualization: React.FC<AncestryVisualizationProps> = ({ results }) => {
  const [selectedCalculator, setSelectedCalculator] = useState<'harappa_world' | 'dodecad_k12b' | 'eurogenes_k13' | 'puntdnal'>('harappa_world');
  const [viewMode, setViewMode] = useState<'pie' | 'bar' | 'radar'>('pie');

  // Color palettes for different calculators
  const colorPalettes = {
    harappa_world: ['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6', '#06B6D4', '#84CC16', '#F97316'],
    dodecad_k12b: ['#1E40AF', '#059669', '#D97706', '#DC2626', '#7C3AED', '#0891B2', '#65A30D', '#EA580C'],
    eurogenes_k13: ['#2563EB', '#047857', '#B45309', '#B91C1C', '#6D28D9', '#0E7490', '#4D7C0F', '#C2410C'],
    puntdnal: ['#1D4ED8', '#065F46', '#92400E', '#991B1B', '#581C87', '#155E75', '#365314', '#9A3412'],
  };

  // Prepare data for different calculators with proper error handling
  const getCalculatorData = (calculator: string) => {
    // Defensive programming: check if results and ancestry_analysis exist
    if (!results || !results.ancestry_analysis) {
      console.warn('Results or ancestry_analysis is undefined');
      return [];
    }

    const data = results.ancestry_analysis[calculator as keyof typeof results.ancestry_analysis];
    
    // Check if data exists and is an object
    if (!data || typeof data !== 'object') {
      console.warn(`Calculator data for ${calculator} is undefined or not an object:`, data);
      return [];
    }

    try {
      return Object.entries(data)
        .map(([name, value]) => ({
          name: name.replace(/_/g, ' '),
          value: Number(value.toFixed(1)),
          percentage: value,
        }))
        .sort((a, b) => b.value - a.value)
        .slice(0, 8); // Top 8 components
    } catch (error) {
      console.error('Error processing calculator data:', error);
      return [];
    }
  };

  const currentData = useMemo(() => getCalculatorData(selectedCalculator), [selectedCalculator, results]);

  // Custom tooltip for charts
  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <motion.div
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          className="bg-dark-bg-card p-3 rounded-lg shadow-dark-lg border border-dark-border-primary"
        >
          <p className="font-medium text-dark-text-primary">{label}</p>
          <p className="text-dark-accent-primary">
            <span className="font-semibold">{payload[0].value}%</span>
          </p>
        </motion.div>
      );
    }
    return null;
  };

  // Calculator tabs
  const calculatorTabs = [
    { id: 'harappa_world', name: 'HarappaWorld', icon: Globe, description: 'K=17 Global Analysis' },
    { id: 'dodecad_k12b', name: 'Dodecad K12b', icon: BarChart3, description: '12 Population Model' },
    { id: 'eurogenes_k13', name: 'Eurogenes K13', icon: RadarIcon, description: '13 Component Analysis' },
    { id: 'puntdnal', name: 'PuntDNAL', icon: Map, description: 'Ancient DNA Focus' },
  ];

  // View mode buttons
  const viewModes = [
    { id: 'pie', name: 'Pie Chart', icon: '🥧' },
    { id: 'bar', name: 'Bar Chart', icon: '📊' },
    { id: 'radar', name: 'Radar Chart', icon: '🎯' },
  ];

  // Check if we have valid data to display
  const hasValidData = currentData && currentData.length > 0;

  // Early return if no results
  if (!results || !results.ancestry_analysis) {
    return (
      <div className="space-y-6">
        <Card className="dark-card overflow-hidden">
          <CardContent className="p-12 text-center">
            <AlertCircle className="w-16 h-16 text-dark-accent-error mx-auto mb-4" />
            <h3 className="text-xl font-semibold text-dark-text-primary mb-2 font-sans">
              No Analysis Data Available
            </h3>
            <p className="text-dark-text-secondary">
              Unable to load ancestry analysis results. Please try running the analysis again.
            </p>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Calculator Selection */}
      <Card className="dark-card overflow-hidden">
        <CardHeader className="p-6 border-b border-dark-border-primary bg-gradient-to-r from-dark-bg-card to-dark-bg-hover">
          <CardTitle className="flex items-center text-lg font-semibold text-white font-sans">
            <TrendingUp className="w-6 h-6 mr-2 text-dark-accent-primary" />
            Multiple Calculator Results
          </CardTitle>
        </CardHeader>
        <CardContent className="p-6">
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
            {calculatorTabs.map((tab) => {
              const Icon = tab.icon;
              const hasData = results.ancestry_analysis[tab.id as keyof typeof results.ancestry_analysis];
              
              return (
                <motion.button
                  key={tab.id}
                  onClick={() => hasData && setSelectedCalculator(tab.id as any)}
                  disabled={!hasData}
                  className={`
                    p-4 rounded-xl border-2 transition-all duration-200 text-left dark-interactive
                    ${selectedCalculator === tab.id && hasData
                      ? 'border-dark-accent-primary bg-dark-bg-hover shadow-dark-glow'
                      : hasData
                      ? 'border-dark-border-secondary hover:border-dark-accent-primary hover:bg-dark-bg-hover'
                      : 'border-dark-border-primary bg-dark-bg-tertiary opacity-50 cursor-not-allowed'
                    }
                  `}
                  whileHover={hasData ? { scale: 1.02 } : undefined}
                  whileTap={hasData ? { scale: 0.98 } : undefined}
                >
                  <div className="flex items-center mb-2">
                    <Icon className={`w-5 h-5 mr-2 ${
                      selectedCalculator === tab.id && hasData ? 'text-dark-accent-primary' : 
                      hasData ? 'text-dark-text-secondary' : 'text-dark-text-muted'
                    }`} />
                    <span className={`font-medium ${
                      selectedCalculator === tab.id && hasData ? 'text-dark-text-primary' : 
                      hasData ? 'text-dark-text-primary' : 'text-dark-text-muted'
                    }`}>
                      {tab.name}
                    </span>
                  </div>
                  <p className={`text-sm ${
                    selectedCalculator === tab.id && hasData ? 'text-dark-accent-primary' : 
                    hasData ? 'text-dark-text-secondary' : 'text-dark-text-muted'
                  }`}>
                    {hasData ? tab.description : 'No data available'}
                  </p>
                </motion.button>
              );
            })}
          </div>

          {/* View Mode Selection */}
          {hasValidData && (
            <div className="flex flex-wrap gap-2 mb-6">
              {viewModes.map((mode) => (
                <Button
                  key={mode.id}
                  onClick={() => setViewMode(mode.id as any)}
                  variant={viewMode === mode.id ? 'primary' : 'outline'}
                  size="sm"
                  className={viewMode === mode.id ? 'dark-btn' : 'dark-btn-secondary dark-interactive'}
                >
                  <span className="mr-2">{mode.icon}</span>
                  {mode.name}
                </Button>
              ))}
            </div>
          )}

          {/* Chart Display - FIXED: Removed extra container and proper height */}
          {hasValidData ? (
            <AnimatePresence mode="wait">
              <motion.div
                key={`${selectedCalculator}-${viewMode}`}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                transition={{ duration: 0.3 }}
                className="h-96 dark-slide-up"
                style={{ height: '400px', width: '100%' }}
              >
                <ResponsiveContainer width="100%" height="100%">
                  {viewMode === 'pie' ? (
                    <PieChart>
                      <Pie
                        data={currentData}
                        cx="50%"
                        cy="50%"
                        outerRadius={120}
                        innerRadius={40}
                        paddingAngle={2}
                        dataKey="value"
                        animationBegin={0}
                        animationDuration={800}
                      >
                        {currentData.map((entry, index) => (
                          <Cell
                            key={`cell-${index}`}
                            fill={colorPalettes[selectedCalculator][index % colorPalettes[selectedCalculator].length]}
                          />
                        ))}
                      </Pie>
                      <Tooltip content={<CustomTooltip />} />
                    </PieChart>
                  ) : viewMode === 'bar' ? (
                    <BarChart data={currentData} margin={{ top: 20, right: 30, left: 20, bottom: 60 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#404040" />
                      <XAxis
                        dataKey="name"
                        angle={-45}
                        textAnchor="end"
                        height={80}
                        fontSize={12}
                        stroke="#a3a3a3"
                      />
                      <YAxis stroke="#a3a3a3" fontSize={12} />
                      <Tooltip content={<CustomTooltip />} />
                      <Bar
                        dataKey="value"
                        radius={[4, 4, 0, 0]}
                        animationDuration={800}
                      >
                        {currentData.map((entry, index) => (
                          <Cell
                            key={`cell-${index}`}
                            fill={colorPalettes[selectedCalculator][index % colorPalettes[selectedCalculator].length]}
                          />
                        ))}
                      </Bar>
                    </BarChart>
                  ) : (
                    <RadarChart data={currentData.slice(0, 6)} margin={{ top: 20, right: 80, bottom: 20, left: 80 }}>
                      <PolarGrid stroke="#404040" />
                      <PolarAngleAxis dataKey="name" fontSize={12} stroke="#a3a3a3" />
                      <PolarRadiusAxis
                        angle={90}
                        domain={[0, Math.max(...currentData.map(d => d.value))]}
                        fontSize={10}
                        stroke="#a3a3a3"
                      />
                      <Radar
                        dataKey="value"
                        stroke={colorPalettes[selectedCalculator][0]}
                        fill={colorPalettes[selectedCalculator][0]}
                        fillOpacity={0.3}
                        strokeWidth={2}
                        animationDuration={800}
                      />
                      <Tooltip content={<CustomTooltip />} />
                    </RadarChart>
                  )}
                </ResponsiveContainer>
              </motion.div>
            </AnimatePresence>
          ) : (
            <div className="h-96 flex items-center justify-center bg-dark-bg-tertiary rounded-lg border border-dark-border-primary">
              <div className="text-center">
                <AlertCircle className="w-12 h-12 text-dark-text-muted mx-auto mb-4" />
                <p className="text-dark-text-secondary font-medium">No data available for {selectedCalculator}</p>
                <p className="text-dark-text-muted text-sm">Try selecting a different calculator</p>
              </div>
            </div>
          )}

          {/* Legend */}
          {hasValidData && (
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 mt-6">
              {currentData.map((item, index) => (
                <motion.div
                  key={item.name}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.1 }}
                  className="flex items-center p-2 rounded-lg bg-dark-bg-hover hover:bg-dark-bg-tertiary transition-colors dark-interactive border border-dark-border-primary"
                >
                  <div
                    className="w-4 h-4 rounded-full mr-3 flex-shrink-0"
                    style={{
                      backgroundColor: colorPalettes[selectedCalculator][index % colorPalettes[selectedCalculator].length]
                    }}
                  />
                  <div className="min-w-0 flex-1">
                    <p className="text-sm font-medium text-dark-text-primary truncate">{item.name}</p>
                    <p className="text-sm text-dark-text-secondary">{item.value}%</p>
                  </div>
                </motion.div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
};
