import React, { useState, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, LineChart, Line } from 'recharts';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Compass, Download, Eye, BarChart3, TrendingUp } from 'lucide-react';
import { AnalysisResults } from '@/types/genome';

interface G25CoordinatesProps {
  results: AnalysisResults;
}

export const G25Coordinates: React.FC<G25CoordinatesProps> = ({ results }) => {
  const [viewMode, setViewMode] = useState<'scatter' | 'line' | 'table'>('scatter');
  const [selectedPCs, setSelectedPCs] = useState<{ x: number; y: number }>({ x: 1, y: 2 });

  const coordinates = results.g25_coordinates.coordinates;
  const magnitude = results.g25_coordinates.magnitude;
  const qualityScore = results.g25_coordinates.quality_score;

  // Prepare data for different visualizations
  const scatterData = useMemo(() => {
    return [{
      x: coordinates[selectedPCs.x - 1] || 0,
      y: coordinates[selectedPCs.y - 1] || 0,
      name: 'Your Sample',
    }];
  }, [coordinates, selectedPCs]);

  const lineData = useMemo(() => {
    return coordinates.slice(0, 10).map((value, index) => ({
      pc: `PC${index + 1}`,
      value: Number(value.toFixed(6)),
      index: index + 1,
    }));
  }, [coordinates]);

  // PC selection options
  const pcOptions = Array.from({ length: Math.min(10, coordinates.length) }, (_, i) => ({
    value: i + 1,
    label: `PC${i + 1}`,
  }));

  const downloadCoordinates = () => {
    const coordinateText = coordinates
      .map((coord, index) => `PC${index + 1}: ${coord.toFixed(6)}`)
      .join('\n');
    
    const blob = new Blob([coordinateText], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'g25_coordinates.txt';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-6">
      <Card className="dark-card overflow-hidden">
        <CardHeader className="p-6 border-b border-dark-border-primary bg-gradient-to-r from-dark-bg-card to-dark-bg-hover">
          <CardTitle className="flex items-center justify-between text-lg font-semibold text-white font-sans">
            <div className="flex items-center">
              <Compass className="w-6 h-6 mr-2 text-dark-accent-primary" />
              Global25 Coordinates
            </div>
            <Button
              onClick={downloadCoordinates}
              variant="outline"
              size="sm"
              className="dark-btn-secondary dark-interactive"
              icon={<Download className="w-4 h-4" />}
            >
              Download
            </Button>
          </CardTitle>
        </CardHeader>
        <CardContent className="p-6">
          {/* Quality Metrics */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 mb-6">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="bg-gradient-to-br from-dark-bg-hover to-dark-bg-tertiary p-4 rounded-lg border border-dark-border-primary"
            >
              <div className="flex items-center mb-2">
                <Compass className="w-5 h-5 text-dark-accent-primary mr-2" />
                <span className="font-medium text-dark-text-primary">Coordinate Magnitude</span>
              </div>
              <p className="text-2xl font-bold text-dark-accent-primary">{magnitude.toFixed(6)}</p>
              <p className="text-sm text-dark-text-secondary">Vector length in 25D space</p>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 }}
              className="bg-gradient-to-br from-dark-bg-hover to-dark-bg-tertiary p-4 rounded-lg border border-dark-border-primary"
            >
              <div className="flex items-center mb-2">
                <TrendingUp className="w-5 h-5 text-dark-accent-secondary mr-2" />
                <span className="font-medium text-dark-text-primary">Quality Score</span>
              </div>
              <p className="text-2xl font-bold text-dark-accent-secondary">{qualityScore.toFixed(1)}%</p>
              <p className="text-sm text-dark-text-secondary">Coordinate accuracy</p>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
              className="bg-gradient-to-br from-dark-bg-hover to-dark-bg-tertiary p-4 rounded-lg border border-dark-border-primary"
            >
              <div className="flex items-center mb-2">
                <BarChart3 className="w-5 h-5 text-dark-accent-tertiary mr-2" />
                <span className="font-medium text-dark-text-primary">Dimensions</span>
              </div>
              <p className="text-2xl font-bold text-dark-accent-tertiary">{coordinates.length}</p>
              <p className="text-sm text-dark-text-secondary">Principal components</p>
            </motion.div>
          </div>

          {/* View Mode Selection */}
          <div className="flex flex-wrap gap-2 mb-6">
            <Button
              onClick={() => setViewMode('scatter')}
              variant={viewMode === 'scatter' ? 'primary' : 'outline'}
              size="sm"
              className={viewMode === 'scatter' ? 'dark-btn' : 'dark-btn-secondary dark-interactive'}
              icon={<Eye className="w-4 h-4" />}
            >
              2D Plot
            </Button>
            <Button
              onClick={() => setViewMode('line')}
              variant={viewMode === 'line' ? 'primary' : 'outline'}
              size="sm"
              className={viewMode === 'line' ? 'dark-btn' : 'dark-btn-secondary dark-interactive'}
              icon={<TrendingUp className="w-4 h-4" />}
            >
              PC Values
            </Button>
            <Button
              onClick={() => setViewMode('table')}
              variant={viewMode === 'table' ? 'primary' : 'outline'}
              size="sm"
              className={viewMode === 'table' ? 'dark-btn' : 'dark-btn-secondary dark-interactive'}
              icon={<BarChart3 className="w-4 h-4" />}
            >
              Table View
            </Button>
          </div>

          {/* Visualization */}
          <AnimatePresence mode="wait">
            {viewMode === 'scatter' && (
              <motion.div
                key="scatter"
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.95 }}
                className="space-y-4 dark-slide-up"
              >
                {/* PC Selection */}
                <div className="flex items-center gap-4 p-4 bg-dark-bg-hover rounded-lg border border-dark-border-primary">
                  <span className="font-medium text-dark-text-primary">Plot:</span>
                  <select
                    value={selectedPCs.x}
                    onChange={(e) => setSelectedPCs(prev => ({ ...prev, x: Number(e.target.value) }))}
                    className="px-3 py-1 bg-dark-bg-card border border-dark-border-secondary rounded-md text-dark-text-primary focus:ring-2 focus:ring-dark-accent-primary focus:border-dark-accent-primary"
                  >
                    {pcOptions.map(option => (
                      <option key={option.value} value={option.value}>{option.label}</option>
                    ))}
                  </select>
                  <span className="text-dark-text-secondary">vs</span>
                  <select
                    value={selectedPCs.y}
                    onChange={(e) => setSelectedPCs(prev => ({ ...prev, y: Number(e.target.value) }))}
                    className="px-3 py-1 bg-dark-bg-card border border-dark-border-secondary rounded-md text-dark-text-primary focus:ring-2 focus:ring-dark-accent-primary focus:border-dark-accent-primary"
                  >
                    {pcOptions.map(option => (
                      <option key={option.value} value={option.value}>{option.label}</option>
                    ))}
                  </select>
                </div>

                {/* Scatter Plot */}
                <div className="h-96" style={{ height: '400px', width: '100%' }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#404040" />
                      <XAxis 
                        type="number" 
                        dataKey="x" 
                        name={`PC${selectedPCs.x}`}
                        stroke="#a3a3a3"
                        fontSize={12}
                      />
                      <YAxis 
                        type="number" 
                        dataKey="y" 
                        name={`PC${selectedPCs.y}`}
                        stroke="#a3a3a3"
                        fontSize={12}
                      />
                      <Tooltip 
                        cursor={{ strokeDasharray: '3 3' }}
                        content={({ active, payload }) => {
                          if (active && payload && payload.length) {
                            return (
                              <div className="bg-dark-bg-card p-3 rounded-lg shadow-dark-lg border border-dark-border-primary">
                                <p className="font-medium text-dark-text-primary">Your Sample</p>
                                <p className="text-dark-accent-primary">PC{selectedPCs.x}: {payload[0].value}</p>
                                <p className="text-dark-accent-primary">PC{selectedPCs.y}: {payload[1].value}</p>
                              </div>
                            );
                          }
                          return null;
                        }}
                      />
                      <Scatter 
                        data={scatterData} 
                        fill="#3B82F6"
                        strokeWidth={2}
                        stroke="#ffffff"
                      />
                    </ScatterChart>
                  </ResponsiveContainer>
                </div>
              </motion.div>
            )}

            {viewMode === 'line' && (
              <motion.div
                key="line"
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.95 }}
                className="dark-slide-up"
              >
                <div className="h-96" style={{ height: '400px', width: '100%' }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={lineData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#404040" />
                      <XAxis 
                        dataKey="pc" 
                        stroke="#a3a3a3"
                        fontSize={12}
                      />
                      <YAxis 
                        stroke="#a3a3a3"
                        fontSize={12}
                      />
                      <Tooltip 
                        content={({ active, payload, label }) => {
                          if (active && payload && payload.length) {
                            return (
                              <div className="bg-dark-bg-card p-3 rounded-lg shadow-dark-lg border border-dark-border-primary">
                                <p className="font-medium text-dark-text-primary">{label}</p>
                                <p className="text-dark-accent-primary">
                                  Value: <span className="font-semibold">{payload[0].value}</span>
                                </p>
                              </div>
                            );
                          }
                          return null;
                        }}
                      />
                      <Line 
                        type="monotone" 
                        dataKey="value" 
                        stroke="#3B82F6" 
                        strokeWidth={3}
                        dot={{ fill: '#3B82F6', strokeWidth: 2, r: 6 }}
                        activeDot={{ r: 8, stroke: '#ffffff', strokeWidth: 2 }}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </motion.div>
            )}

            {viewMode === 'table' && (
              <motion.div
                key="table"
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.95 }}
                className="dark-slide-up"
              >
                <div className="bg-dark-bg-tertiary rounded-lg border border-dark-border-primary overflow-hidden">
                  <div className="overflow-x-auto">
                    <table className="w-full">
                      <thead className="bg-dark-bg-hover border-b border-dark-border-primary">
                        <tr>
                          <th className="px-4 py-3 text-left text-sm font-medium text-dark-text-primary">Component</th>
                          <th className="px-4 py-3 text-left text-sm font-medium text-dark-text-primary">Value</th>
                          <th className="px-4 py-3 text-left text-sm font-medium text-dark-text-primary">Magnitude</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-dark-border-primary">
                        {coordinates.slice(0, 25).map((value, index) => (
                          <motion.tr
                            key={index}
                            initial={{ opacity: 0, x: -20 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{ delay: index * 0.05 }}
                            className="hover:bg-dark-bg-hover transition-colors"
                          >
                            <td className="px-4 py-3 text-sm font-medium text-dark-accent-primary">
                              PC{index + 1}
                            </td>
                            <td className="px-4 py-3 text-sm text-dark-text-primary font-mono">
                              {value.toFixed(6)}
                            </td>
                            <td className="px-4 py-3 text-sm text-dark-text-secondary">
                              {Math.abs(value).toFixed(3)}
                            </td>
                          </motion.tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </CardContent>
      </Card>
    </div>
  );
};
