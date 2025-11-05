import React from 'react';
import { motion } from 'framer-motion';
import { Dna, Github, Mail, Globe } from 'lucide-react';

export const Footer: React.FC = () => {
  const currentYear = new Date().getFullYear();

  return (
    <footer className="bg-dark-bg-secondary border-t border-dark-border-primary mt-auto">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          {/* Brand */}
          <div className="flex items-start">
            <motion.div
              animate={{ rotate: 360 }}
              transition={{ duration: 12, repeat: Infinity, ease: "linear" }}
              className="mr-3 mt-1"
            >
              <Dna className="w-6 h-6 text-dark-accent-primary" />
            </motion.div>
            <div>
              <h3 className="text-lg font-semibold text-dark-text-primary font-sans">
                atavus
              </h3>
              <p className="text-dark-text-secondary text-sm mt-1">
                Professional ancestry analysis with multiple calculators and G25 coordinates.
              </p>
            </div>
          </div>

          {/* Features */}
          <div>
            <h4 className="text-sm font-semibold text-dark-text-primary uppercase tracking-wide mb-4 font-sans">
              Features
            </h4>
            <ul className="space-y-2 text-sm text-dark-text-secondary">
              <li className="dark-interactive hover:text-dark-accent-primary transition-colors cursor-pointer">
                • HarappaWorld K=17 Analysis
              </li>
              <li className="dark-interactive hover:text-dark-accent-primary transition-colors cursor-pointer">
                • Dodecad K12b Calculator
              </li>
              <li className="dark-interactive hover:text-dark-accent-primary transition-colors cursor-pointer">
                • Eurogenes K13 Analysis
              </li>
              <li className="dark-interactive hover:text-dark-accent-primary transition-colors cursor-pointer">
                • PuntDNAL Ancient DNA
              </li>
              <li className="dark-interactive hover:text-dark-accent-primary transition-colors cursor-pointer">
                • Global25 Coordinates
              </li>
              <li className="dark-interactive hover:text-dark-accent-primary transition-colors cursor-pointer">
                • Regional Breakdowns
              </li>
            </ul>
          </div>

          {/* Links */}
          <div>
            <h4 className="text-sm font-semibold text-dark-text-primary uppercase tracking-wide mb-4 font-sans">
              Connect
            </h4>
            <div className="flex space-x-4">
              <motion.a
                href="https://github.com"
                target="_blank"
                rel="noopener noreferrer"
                className="text-dark-text-muted hover:text-dark-accent-primary transition-colors dark-interactive p-2 rounded-lg hover:bg-dark-bg-hover"
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.95 }}
              >
                <Github className="w-5 h-5" />
              </motion.a>
              <motion.a
                href="mailto:contact@atavus.com"
                className="text-dark-text-muted hover:text-dark-accent-primary transition-colors dark-interactive p-2 rounded-lg hover:bg-dark-bg-hover"
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.95 }}
              >
                <Mail className="w-5 h-5" />
              </motion.a>
              <motion.a
                href="https://atavus.com"
                target="_blank"
                rel="noopener noreferrer"
                className="text-dark-text-muted hover:text-dark-accent-primary transition-colors dark-interactive p-2 rounded-lg hover:bg-dark-bg-hover"
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.95 }}
              >
                <Globe className="w-5 h-5" />
              </motion.a>
            </div>
          </div>
        </div>

        <div className="border-t border-dark-border-primary mt-8 pt-6 flex flex-col sm:flex-row justify-between items-center">
          <p className="text-sm text-dark-text-muted">
            © {currentYear} atavus. All rights reserved.
          </p>
          <p className="text-sm text-dark-text-muted mt-2 sm:mt-0">
            Built with React, TypeScript, and modern web technologies.
          </p>
        </div>
      </div>
    </footer>
  );
};
