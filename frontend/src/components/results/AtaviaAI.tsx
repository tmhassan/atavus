import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { 
  Bot, 
  Send, 
  User, 
  Sparkles, 
  Download, 
  RefreshCw,
  MessageSquare,
  Lightbulb,
  Globe,
  MapPin,
  BarChart3,
  Users,
  Dna
} from 'lucide-react';
import { AnalysisResults } from '@/types/genome';
import toast from 'react-hot-toast';

interface AtaviaAIProps {
  results: AnalysisResults;
  analysisId: string;
}

interface Message {
  id: string;
  type: 'user' | 'ai';
  content: string;
  timestamp: Date;
  isLoading?: boolean;
}

interface QuickPrompt {
  id: string;
  title: string;
  prompt: string;
  icon: React.ComponentType<any>;
  description: string;
  category: string;
}

// Custom markdown components for beautiful rendering
const MarkdownComponents = {
  // Headers with dark theme styling
  h1: ({ children, ...props }: any) => (
    <h1 className="text-3xl font-bold text-dark-text-primary mb-6 mt-8 border-b border-dark-border-primary pb-2" {...props}>
      {children}
    </h1>
  ),
  h2: ({ children, ...props }: any) => (
    <h2 className="text-2xl font-semibold text-dark-text-primary mb-4 mt-6" {...props}>
      {children}
    </h2>
  ),
  h3: ({ children, ...props }: any) => (
    <h3 className="text-xl font-semibold text-dark-text-primary mb-3 mt-4" {...props}>
      {children}
    </h3>
  ),
  h4: ({ children, ...props }: any) => (
    <h4 className="text-lg font-semibold text-dark-text-primary mb-2 mt-3" {...props}>
      {children}
    </h4>
  ),
  
  // Paragraphs with proper spacing
  p: ({ children, ...props }: any) => (
    <p className="text-dark-text-primary mb-4 leading-relaxed" {...props}>
      {children}
    </p>
  ),
  
  // Bold text with accent color
  strong: ({ children, ...props }: any) => (
    <strong className="font-semibold text-dark-accent-primary" {...props}>
      {children}
    </strong>
  ),
  
  // Italic text
  em: ({ children, ...props }: any) => (
    <em className="italic text-dark-text-secondary" {...props}>
      {children}
    </em>
  ),
  
  // Unordered lists with custom bullets
  ul: ({ children, ...props }: any) => (
    <ul className="mb-4 space-y-2" {...props}>
      {children}
    </ul>
  ),
  
  // Ordered lists
  ol: ({ children, ...props }: any) => (
    <ol className="mb-4 space-y-2 list-decimal list-inside" {...props}>
      {children}
    </ol>
  ),
  
  // List items with custom styling
  li: ({ children, ...props }: any) => (
    <li className="flex items-start text-dark-text-primary" {...props}>
      <span className="w-2 h-2 bg-dark-accent-tertiary rounded-full mt-2 mr-3 flex-shrink-0"></span>
      <span className="flex-1">{children}</span>
    </li>
  ),
  
  // Links with hover effects
  a: ({ children, href, ...props }: any) => (
    <a 
      href={href} 
      className="text-dark-accent-primary hover:text-dark-accent-secondary transition-colors underline" 
      target="_blank" 
      rel="noopener noreferrer"
      {...props}
    >
      {children}
    </a>
  ),
  
  // Code blocks and inline code
  code: ({ inline, children, className, ...props }: any) => {
    if (inline) {
      return (
        <code className="bg-dark-bg-tertiary text-dark-accent-primary px-2 py-1 rounded text-sm font-mono" {...props}>
          {children}
        </code>
      );
    }
    return (
      <pre className="bg-dark-bg-tertiary border border-dark-border-primary rounded-lg p-4 overflow-x-auto mb-4" {...props}>
        <code className="text-dark-text-primary font-mono text-sm">{children}</code>
      </pre>
    );
  },
  
  // Blockquotes with accent border
  blockquote: ({ children, ...props }: any) => (
    <blockquote className="border-l-4 border-dark-accent-tertiary bg-dark-bg-hover pl-6 py-4 my-4 rounded-r-lg" {...props}>
      <div className="text-dark-text-primary italic">{children}</div>
    </blockquote>
  ),
  
  // Tables with dark theme
  table: ({ children, ...props }: any) => (
    <div className="overflow-x-auto mb-4">
      <table className="w-full border-collapse border border-dark-border-primary rounded-lg" {...props}>
        {children}
      </table>
    </div>
  ),
  
  thead: ({ children, ...props }: any) => (
    <thead className="bg-dark-bg-hover" {...props}>
      {children}
    </thead>
  ),
  
  tbody: ({ children, ...props }: any) => (
    <tbody {...props}>
      {children}
    </tbody>
  ),
  
  tr: ({ children, ...props }: any) => (
    <tr className="border-b border-dark-border-primary hover:bg-dark-bg-hover transition-colors" {...props}>
      {children}
    </tr>
  ),
  
  th: ({ children, ...props }: any) => (
    <th className="text-left p-3 font-semibold text-dark-text-primary border-r border-dark-border-primary last:border-r-0" {...props}>
      {children}
    </th>
  ),
  
  td: ({ children, ...props }: any) => (
    <td className="p-3 text-dark-text-primary border-r border-dark-border-primary last:border-r-0" {...props}>
      {children}
    </td>
  ),
  
  // Horizontal rules
  hr: ({ ...props }: any) => (
    <hr className="border-dark-border-primary my-6" {...props} />
  ),
};

export const AtaviaAI: React.FC<AtaviaAIProps> = ({ results, analysisId }) => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [sessionId] = useState(() => `session_${Date.now()}`);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Quick prompt templates
  const quickPrompts: QuickPrompt[] = [
    {
      id: 'overview',
      title: 'Comprehensive Overview',
      prompt: 'Provide a comprehensive summary of my genome ancestry results, highlighting key populations and their proportions with historical context.',
      icon: Globe,
      description: 'Get a detailed overview of your ancestry',
      category: 'Overview'
    },
    {
      id: 'calculator_explanation',
      title: 'Calculator Meanings',
      prompt: 'Explain what each ancestry calculator (HarappaWorld, Dodecad K12b, Eurogenes K13, PuntDNAL) means and how to interpret my results from each one.',
      icon: BarChart3,
      description: 'Understand what each calculator tells you',
      category: 'Analysis'
    },
    {
      id: 'regional_breakdown',
      title: 'Regional Deep Dive',
      prompt: 'Give me a detailed explanation of my regional ancestry breakdown, including historical migration patterns and genetic context for each region.',
      icon: MapPin,
      description: 'Explore your regional ancestry in detail',
      category: 'Regional'
    },
    {
      id: 'population_proximity',
      title: 'Population Matching',
      prompt: 'Based on my G25 coordinates and ancestry percentages, identify the closest modern and ancient population groups to my genetic profile.',
      icon: Users,
      description: 'Find your closest genetic populations',
      category: 'Matching'
    },
    {
      id: 'detailed_analysis',
      title: 'AI Deep Analysis',
      prompt: 'Perform an in-depth AI analysis of my genome data, integrating all calculators and regional data to provide insights into ancestry, migration patterns, and genetic traits.',
      icon: Sparkles,
      description: 'Get the most comprehensive AI analysis',
      category: 'Advanced'
    },
    {
      id: 'g25_interpretation',
      title: 'G25 Coordinates',
      prompt: 'Explain my G25 coordinates in detail, what they mean in terms of genetic positioning, and how they relate to different world populations.',
      icon: Dna,
      description: 'Understand your genetic coordinates',
      category: 'Technical'
    }
  ];

  // Scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Initial welcome message
  useEffect(() => {
    const welcomeMessage: Message = {
      id: 'welcome',
      type: 'ai',
      content: `# Welcome to Atavia AI! 👋

I'm **Atavia**, your AI-powered genome analysis assistant. I have access to your complete ancestry analysis results and can help you explore and understand your genetic heritage in detail.

## I can help you with:

- **Comprehensive ancestry breakdowns** with historical context
- **Calculator explanations** for HarappaWorld, Dodecad, Eurogenes, and PuntDNAL
- **Regional ancestry insights** and migration patterns
- **Population matching** to find your closest genetic relatives
- **G25 coordinate interpretation** and genetic positioning
- **Custom questions** about any aspect of your results

Choose a quick prompt below or ask me anything about your genome analysis!`,
      timestamp: new Date()
    };
    setMessages([welcomeMessage]);
  }, []);

  const sendMessage = async (content: string) => {
    if (!content.trim() || isLoading) return;

    const userMessage: Message = {
      id: `user_${Date.now()}`,
      type: 'user',
      content: content.trim(),
      timestamp: new Date()
    };

    const loadingMessage: Message = {
      id: `ai_${Date.now()}`,
      type: 'ai',
      content: '',
      timestamp: new Date(),
      isLoading: true
    };

    setMessages(prev => [...prev, userMessage, loadingMessage]);
    setInputValue('');
    setIsLoading(true);

    try {
      const response = await fetch('/api/atavia/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          sessionId,
          prompt: content,
          genomeData: results,
          analysisId,
          messageHistory: messages.slice(-10) // Last 10 messages for context
        })
      });

      if (!response.ok) {
        throw new Error('Failed to get AI response');
      }

      const data = await response.json();

      const aiMessage: Message = {
        id: `ai_${Date.now()}`,
        type: 'ai',
        content: data.response,
        timestamp: new Date()
      };

      setMessages(prev => prev.slice(0, -1).concat(aiMessage));
    } catch (error) {
      console.error('Error getting AI response:', error);
      const errorMessage: Message = {
        id: `ai_error_${Date.now()}`,
        type: 'ai',
        content: 'I apologize, but I encountered an error processing your request. Please try again or rephrase your question.',
        timestamp: new Date()
      };
      setMessages(prev => prev.slice(0, -1).concat(errorMessage));
      toast.error('Failed to get AI response. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleQuickPrompt = (prompt: QuickPrompt) => {
    sendMessage(prompt.prompt);
  };

  const exportConversation = () => {
    const conversation = messages
      .filter(m => !m.isLoading)
      .map(m => `${m.type.toUpperCase()}: ${m.content}`)
      .join('\n\n');
    
    const blob = new Blob([conversation], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `atavia_conversation_${analysisId}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    toast.success('Conversation exported successfully!');
  };

  const clearConversation = () => {
    setMessages([]);
    toast.success('Conversation cleared!');
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <Card className="dark-card bg-gradient-to-r from-dark-bg-card to-dark-bg-hover border-dark-accent-tertiary/30">
        <CardHeader className="p-6 border-b border-dark-border-primary">
          <CardTitle className="flex items-center justify-between text-lg font-semibold text-white font-sans">
            <div className="flex items-center">
              <motion.div
                animate={{ rotate: [0, 10, -10, 0] }}
                transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
              >
                <Bot className="w-6 h-6 mr-2 text-dark-accent-tertiary" />
              </motion.div>
              Atavia AI - Your Genome Analysis Assistant
            </div>
            <div className="flex items-center space-x-2">
              <Button
                onClick={exportConversation}
                variant="outline"
                size="sm"
                className="dark-btn-secondary dark-interactive"
                icon={<Download className="w-4 h-4" />}
              >
                Export
              </Button>
              <Button
                onClick={clearConversation}
                variant="outline"
                size="sm"
                className="dark-btn-secondary dark-interactive"
                icon={<RefreshCw className="w-4 h-4" />}
              >
                Clear
              </Button>
            </div>
          </CardTitle>
        </CardHeader>
      </Card>

      {/* Quick Prompts */}
      <Card className="dark-card">
        <CardHeader className="p-6 border-b border-dark-border-primary">
          <CardTitle className="flex items-center text-lg font-semibold text-white font-sans">
            <Lightbulb className="w-5 h-5 mr-2 text-dark-accent-warning" />
            Quick Analysis Prompts
          </CardTitle>
        </CardHeader>
        <CardContent className="p-6">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {quickPrompts.map((prompt) => {
              const Icon = prompt.icon;
              return (
                <motion.button
                  key={prompt.id}
                  onClick={() => handleQuickPrompt(prompt)}
                  disabled={isLoading}
                  className="p-4 rounded-xl border border-dark-border-secondary hover:border-dark-accent-tertiary bg-dark-bg-hover hover:bg-dark-bg-tertiary transition-all duration-200 text-left dark-interactive disabled:opacity-50"
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                >
                  <div className="flex items-center mb-2">
                    <Icon className="w-5 h-5 mr-2 text-dark-accent-tertiary" />
                    <span className="font-medium text-dark-text-primary text-sm">{prompt.title}</span>
                  </div>
                  <p className="text-xs text-dark-text-secondary">{prompt.description}</p>
                  <div className="mt-2">
                    <span className="inline-block px-2 py-1 text-xs bg-dark-accent-tertiary/20 text-dark-accent-tertiary rounded-full">
                      {prompt.category}
                    </span>
                  </div>
                </motion.button>
              );
            })}
          </div>
        </CardContent>
      </Card>

      {/* Chat Interface */}
      <Card className="dark-card">
        <CardHeader className="p-6 border-b border-dark-border-primary">
          <CardTitle className="flex items-center text-lg font-semibold text-white font-sans">
            <MessageSquare className="w-5 h-5 mr-2 text-dark-accent-primary" />
            Conversation with Atavia
          </CardTitle>
        </CardHeader>
        <CardContent className="p-0">
          {/* Messages */}
          <div className="h-96 overflow-y-auto p-6 space-y-6">
            <AnimatePresence>
              {messages.map((message) => (
                <motion.div
                  key={message.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  <div className={`max-w-4xl ${message.type === 'user' ? 'order-2' : 'order-1'}`}>
                    <div className={`flex items-start space-x-3 ${message.type === 'user' ? 'flex-row-reverse space-x-reverse' : ''}`}>
                      <div className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${
                        message.type === 'user' 
                          ? 'bg-dark-accent-primary' 
                          : 'bg-dark-accent-tertiary'
                      }`}>
                        {message.type === 'user' ? (
                          <User className="w-4 h-4 text-white" />
                        ) : (
                          <Bot className="w-4 h-4 text-white" />
                        )}
                      </div>
                      <div className={`rounded-2xl px-6 py-4 ${
                        message.type === 'user'
                          ? 'bg-dark-accent-primary text-white'
                          : 'bg-dark-bg-hover border border-dark-border-primary'
                      }`}>
                        {message.isLoading ? (
                          <div className="flex items-center space-x-3">
                            <div className="flex space-x-1">
                              <div className="w-2 h-2 bg-dark-accent-tertiary rounded-full animate-bounce"></div>
                              <div className="w-2 h-2 bg-dark-accent-tertiary rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                              <div className="w-2 h-2 bg-dark-accent-tertiary rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                            </div>
                            <span className="text-dark-text-secondary">Atavia is thinking...</span>
                          </div>
                        ) : (
                          <div className="prose prose-sm max-w-none">
                            {message.type === 'user' ? (
                              <p className="text-white m-0">{message.content}</p>
                            ) : (
                              <ReactMarkdown
                                remarkPlugins={[remarkGfm]}
                                components={MarkdownComponents}
                              >
                                {message.content}
                              </ReactMarkdown>
                            )}
                          </div>
                        )}
                        <div className={`text-xs mt-3 ${
                          message.type === 'user' ? 'text-blue-100' : 'text-dark-text-muted'
                        }`}>
                          {message.timestamp.toLocaleTimeString()}
                        </div>
                      </div>
                    </div>
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>
            <div ref={messagesEndRef} />
          </div>

          {/* Input */}
          <div className="border-t border-dark-border-primary p-6">
            <div className="flex space-x-4">
              <input
                type="text"
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && sendMessage(inputValue)}
                placeholder="Ask Atavia about your genome analysis..."
                disabled={isLoading}
                className="flex-1 px-4 py-3 bg-dark-bg-card border border-dark-border-secondary rounded-xl text-dark-text-primary placeholder-dark-text-muted focus:ring-2 focus:ring-dark-accent-primary focus:border-dark-accent-primary disabled:opacity-50"
              />
              <Button
                onClick={() => sendMessage(inputValue)}
                disabled={!inputValue.trim() || isLoading}
                className="dark-btn px-6 py-3"
                icon={<Send className="w-4 h-4" />}
              >
                Send
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};
