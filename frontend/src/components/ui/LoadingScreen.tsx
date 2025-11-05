import React from 'react';
import { motion } from 'framer-motion';
import { Dna } from 'lucide-react';

export const LoadingScreen: React.FC = () => {
  return (
    <div className="min-h-screen bg-dark-bg-primary flex items-center justify-center">
      <div className="text-center">
        <motion.div
          animate={{ 
            rotate: 360,
            scale: [1, 1.2, 1]
          }}
          transition={{ 
            rotate: { duration: 2, repeat: Infinity, ease: "linear" },
            scale: { duration: 1, repeat: Infinity, ease: "easeInOut" }
          }}
          className="mb-6"
        >
          <Dna className="w-16 h-16 text-dark-accent-primary mx-auto" />
        </motion.div>
        
        <motion.h2
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-2xl font-bold text-dark-text-primary mb-2 font-sans"
        >
          Initializing atavus
        </motion.h2>
        
        <motion.p
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="text-dark-text-secondary"
        >
          Setting up your analysis environment...
        </motion.p>
        
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: "100%" }}
          transition={{ duration: 2, ease: "easeInOut" }}
          className="w-64 h-1 bg-dark-bg-hover rounded-full mx-auto mt-6 overflow-hidden"
        >
          <motion.div
            animate={{ x: ["0%", "100%"] }}
            transition={{ duration: 1.5, repeat: Infinity, ease: "easeInOut" }}
            className="h-full w-1/3 bg-dark-accent-primary rounded-full"
          />
        </motion.div>
      </div>
    </div>
  );
};
