import React, { useState } from 'react';
import { Eye, Moon, Sun } from 'lucide-react';

export default function App() {
  const [darkMode, setDarkMode] = useState(false);

  return (
    <div className={`${darkMode ? 'dark' : ''}`}>
      <div className="min-h-screen bg-gradient-to-br from-blue-200 via-white to-purple-200 dark:from-gray-900 dark:to-gray-800 relative overflow-hidden">
        {/* Floating Background Elements */}
        <div className="absolute inset-0 overflow-hidden pointer-events-none">
          <div className="absolute top-20 left-20 w-32 h-32 bg-blue-400/20 rounded-full blur-xl animate-pulse"></div>
          <div className="absolute bottom-32 right-32 w-48 h-48 bg-purple-400/20 rounded-full blur-xl animate-pulse delay-1000"></div>
          <div className="absolute top-1/2 left-1/3 w-24 h-24 bg-pink-400/20 rounded-full blur-xl animate-pulse delay-2000"></div>
        </div>

        {/* Top Navigation Bar */}
        <div className="relative z-10 bg-white/80 dark:bg-gray-900/80 backdrop-blur-lg border-b border-gray-200/50 dark:border-gray-700/50">
          <div className="max-w-7xl mx-auto px-6 py-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-4">
                <div className="text-2xl font-bold text-blue-600 dark:text-blue-400 flex items-center">
                  <Eye className="mr-2" />
                  CrowdVision AI
                </div>
              </div>
              
              <div className="flex items-center space-x-4">
                <button
                  onClick={() => setDarkMode(!darkMode)}
                  className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
                >
                  {darkMode ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Main Content - Live Vision Feed */}
        <div className="relative z-10 max-w-7xl mx-auto p-6">
          <div className="bg-white/80 dark:bg-gray-900/80 backdrop-blur-lg rounded-2xl shadow-xl border border-gray-200/50 dark:border-gray-700/50 overflow-hidden transform hover:scale-[1.01] transition-all duration-300">
            <div className="p-6 border-b border-gray-200/50 dark:border-gray-700/50">
              <div className="flex items-center justify-between">
                <h3 className="text-xl font-bold text-gray-900 dark:text-white flex items-center">
                  <Eye className="mr-2 text-blue-500" />
                  Live Vision Feed
                </h3>
                <div className="flex items-center space-x-2">
                  <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                  <span className="text-sm text-gray-600 dark:text-gray-300">Live</span>
                </div>
              </div>
            </div>
            
            <div className="relative">
              <iframe
                src="http://localhost:8501"
                title="CrowdVision AI - Live Feed"
                className="w-full h-[70vh] border-none"
              ></iframe>
              
              {/* Loading Overlay */}
              <div className="absolute inset-0 bg-gray-100/50 dark:bg-gray-800/50 flex items-center justify-center opacity-0 hover:opacity-100 transition-opacity duration-300 pointer-events-none">
                <div className="text-center">
                  <div className="w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-2"></div>
                  <p className="text-sm text-gray-600 dark:text-gray-300">Processing AI Analysis...</p>
                </div>
              </div>
            </div>

            <div className="p-4 bg-gray-50/50 dark:bg-gray-800/50 border-t border-gray-200/50 dark:border-gray-700/50">
              <div className="flex items-center justify-between text-sm text-gray-600 dark:text-gray-300">
                <span>Resolution: 1920x1080 | FPS: 30</span>
                <span>Latency: 45ms</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}