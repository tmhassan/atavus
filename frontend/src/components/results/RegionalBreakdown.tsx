import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Treemap, ResponsiveContainer, Tooltip } from 'recharts';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card';
import { MapPin, Users, Globe2, Layers } from 'lucide-react';
import { AnalysisResults } from '@/types/genome';

interface RegionalBreakdownProps {
  results: AnalysisResults;
}

export const RegionalBreakdown: React.FC<RegionalBreakdownProps> = ({ results }) => {
  const [selectedRegion, setSelectedRegion] = useState<'south_asian' | 'west_eurasian' | 'east_eurasian'>('south_asian');

  const regionData = {
    south_asian: {
      title: 'South Asian Regions',
      icon: MapPin,
      color: '#8B5CF6',
      data: results.ancestry_analysis.regional_breakdowns.south_asian || {},
      description: 'Detailed breakdown of South Asian ancestry components'
    },
    west_eurasian: {
      title: 'West Eurasian Regions',
      icon: Globe2,
      color: '#3B82F6',
      data: results.ancestry_analysis.regional_breakdowns.west_eurasian || {},
      description: 'West Eurasian and Middle Eastern components'
    },
    east_eurasian: {
      title: 'East Eurasian Regions',
      icon: Users,
      color: '#10B981',
      data: results.ancestry_analysis.regional_breakdowns.east_eurasian || {},
      description: 'East Asian and Siberian components'
    }
  };

  // Prepare treemap data - EXACT SAME LOGIC AS ORIGINAL
  const getTreemapData = (data: Record<string, number>) => {
    return Object.entries(data)
      .filter(([_, value]) => value > 0.5) // Only show significant components
      .map(([name, value]) => ({
        name: name.replace(/_/g, ' '),
        value: Number(value.toFixed(1)),
        percentage: value,
      }))
      .sort((a, b) => b.value - a.value);
  };

  const currentTreemapData = getTreemapData(regionData[selectedRegion].data);

  // Custom treemap content - EXACT SAME LOGIC AS ORIGINAL
  const CustomTreemapContent = (props: any) => {
    const { x, y, width, height, name, value } = props;
    
    if (width < 50 || height < 30) return null; // Don't render if too small
    
    return (
      <g>
        <rect
          x={x}
          y={y}
          width={width}
          height={height}
          fill={regionData[selectedRegion].color}
          fillOpacity={0.8}
          stroke="#fff"
          strokeWidth={2}
          rx={4}
        />
        <text
          x={x + width / 2}
          y={y + height / 2 - 8}
          textAnchor="middle"
          fill="#fff"
          fontSize={Math.min(width / 8, height / 4, 14)}
          fontWeight="600"
        >
          {name}
        </text>
        <text
          x={x + width / 2}
          y={y + height / 2 + 8}
          textAnchor="middle"
          fill="#fff"
          fontSize={Math.min(width / 10, height / 5, 12)}
        >
          {value}%
        </text>
      </g>
    );
  };

  // Region selection tabs
  const regionTabs = Object.entries(regionData).map(([key, data]) => ({
    id: key,
    ...data
  }));

  return (
    <div className="space-y-6">
      <Card className="dark-card overflow-hidden">
        <CardHeader className="p-6 border-b border-dark-border-primary bg-gradient-to-r from-dark-bg-card to-dark-bg-hover">
          <CardTitle className="flex items-center text-lg font-semibold text-white font-sans">
            <Layers className="w-6 h-6 mr-2 text-dark-accent-tertiary" />
            Regional Ancestry Breakdown
          </CardTitle>
        </CardHeader>
        <CardContent className="p-6">
          {/* Region Selection - Updated with Dark Theme */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 mb-6">
            {regionTabs.map((tab) => {
              const Icon = tab.icon;
              const hasData = Object.keys(tab.data).length > 0;
              
              return (
                <motion.button
                  key={tab.id}
                  onClick={() => hasData && setSelectedRegion(tab.id as any)}
                  disabled={!hasData}
                  className={`
                    p-4 rounded-xl border-2 transition-all duration-200 text-left dark-interactive
                    ${selectedRegion === tab.id && hasData
                      ? 'border-dark-accent-tertiary bg-dark-bg-hover shadow-dark-glow'
                      : hasData
                      ? 'border-dark-border-secondary hover:border-dark-accent-tertiary hover:bg-dark-bg-hover'
                      : 'border-dark-border-primary bg-dark-bg-tertiary opacity-50 cursor-not-allowed'
                    }
                  `}
                  whileHover={hasData ? { scale: 1.02 } : undefined}
                  whileTap={hasData ? { scale: 0.98 } : undefined}
                >
                  <div className="flex items-center mb-2">
                    <Icon className={`w-5 h-5 mr-2 ${
                      selectedRegion === tab.id && hasData ? 'text-dark-accent-tertiary' : 
                      hasData ? 'text-dark-text-secondary' : 'text-dark-text-muted'
                    }`} />
                    <span className={`font-medium ${
                      selectedRegion === tab.id && hasData ? 'text-dark-text-primary' : 
                      hasData ? 'text-dark-text-primary' : 'text-dark-text-muted'
                    }`}>
                      {tab.title}
                    </span>
                  </div>
                  <p className={`text-sm ${
                    selectedRegion === tab.id && hasData ? 'text-dark-accent-tertiary' : 
                    hasData ? 'text-dark-text-secondary' : 'text-dark-text-muted'
                  }`}>
                    {hasData ? tab.description : 'No significant components'}
                  </p>
                  {hasData && (
                    <div className="mt-2">
                      <span className="inline-block px-2 py-1 text-xs font-medium bg-dark-bg-card text-dark-text-primary rounded-full border border-dark-border-primary">
                        {Object.keys(tab.data).length} regions
                      </span>
                    </div>
                  )}
                </motion.button>
              );
            })}
          </div>

          {/* Treemap Visualization - FIXED: Removed strokeWidth prop */}
          {currentTreemapData.length > 0 ? (
            <AnimatePresence mode="wait">
              <motion.div
                key={selectedRegion}
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.95 }}
                transition={{ duration: 0.3 }}
                className="h-80 mb-6 dark-slide-up"
              >
                <ResponsiveContainer width="100%" height="100%">
                  <Treemap
                    data={currentTreemapData}
                    dataKey="value"
                    aspectRatio={4/3}
                    stroke="#fff"
                    content={<CustomTreemapContent />}
                    animationDuration={800}
                  >
                    <Tooltip
                      content={({ active, payload }) => {
                        if (active && payload && payload.length) {
                          const data = payload[0].payload;
                          return (
                            <div className="bg-dark-bg-card p-3 rounded-lg shadow-dark-lg border border-dark-border-primary">
                              <p className="font-medium text-dark-text-primary">{data.name}</p>
                              <p className="text-dark-accent-tertiary">
                                <span className="font-semibold">{data.value}%</span> ancestry
                              </p>
                            </div>
                          );
                        }
                        return null;
                      }}
                    />
                  </Treemap>
                </ResponsiveContainer>
              </motion.div>
            </AnimatePresence>
          ) : (
            <div className="h-80 flex items-center justify-center bg-dark-bg-tertiary rounded-lg border border-dark-border-primary">
              <div className="text-center">
                <Globe2 className="w-12 h-12 text-dark-text-muted mx-auto mb-4" />
                <p className="text-dark-text-secondary font-medium">No regional data available</p>
                <p className="text-dark-text-muted text-sm">This region has no significant ancestry components</p>
              </div>
            </div>
          )}

          {/* Detailed List - Updated with Dark Theme */}
          {currentTreemapData.length > 0 && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              {currentTreemapData.map((item, index) => (
                <motion.div
                  key={item.name}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.1 }}
                  className="flex items-center justify-between p-4 bg-dark-bg-hover rounded-lg hover:bg-dark-bg-tertiary transition-colors dark-interactive border border-dark-border-primary"
                >
                  <div className="flex items-center">
                    <div
                      className="w-4 h-4 rounded-full mr-3"
                      style={{ backgroundColor: regionData[selectedRegion].color }}
                    />
                    <span className="font-medium text-dark-text-primary">{item.name}</span>
                  </div>
                  <div className="text-right">
                    <span className="text-lg font-semibold text-dark-text-primary">{item.value}%</span>
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
