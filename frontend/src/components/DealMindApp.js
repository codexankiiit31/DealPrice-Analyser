import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';

const DealMindApp = () => {
  // State management - mirrors Gradio state
  const [categories, setCategories] = useState([]);
  const [selectedCategories, setSelectedCategories] = useState([]);
  const [isSearching, setIsSearching] = useState(false);
  const [statusMessage, setStatusMessage] = useState('');
  const [statusType, setStatusType] = useState(''); // 'error', 'success', 'warning'
  const [searchResults, setSearchResults] = useState([]);
  const [logs, setLogs] = useState([]);
  const [currentJobId, setCurrentJobId] = useState(null);
  
  // WebSocket connection
  const wsRef = useRef(null);
  const logsContainerRef = useRef(null);
  
  // POLLING REFS - Prevent memory leaks
  const pollIntervalRef = useRef(null);
  const pollTimeoutRef = useRef(null);

  // Base API URL
  const API_BASE = process.env.REACT_APP_API_BASE || 'http://localhost:8000';

  // Initialize - Load categories on component mount
  useEffect(() => {
    loadCategories();
    setupWebSocket();
    
    return () => {
      // CLEANUP - Prevent memory leaks
      cleanupPolling();
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  // Auto-scroll logs to bottom
  useEffect(() => {
    if (logsContainerRef.current) {
      logsContainerRef.current.scrollTop = logsContainerRef.current.scrollHeight;
    }
  }, [logs]);

  // CLEANUP FUNCTION - Prevent memory leaks
  const cleanupPolling = () => {
    if (pollIntervalRef.current) {
      clearInterval(pollIntervalRef.current);
      pollIntervalRef.current = null;
    }
    if (pollTimeoutRef.current) {
      clearTimeout(pollTimeoutRef.current);
      pollTimeoutRef.current = null;
    }
  };

  // Load available categories from API
  const loadCategories = async () => {
    try {
      const response = await axios.get(`${API_BASE}/api/categories`);
      setCategories(response.data.categories);
    } catch (error) {
      console.error('Error loading categories:', error);
      setStatusMessage('Error loading categories');
      setStatusType('error');
    }
  };

  // Setup WebSocket connection for real-time logs
  const setupWebSocket = () => {
    const wsUrl = `${API_BASE.replace('http', 'ws')}/ws/logs`;
    wsRef.current = new WebSocket(wsUrl);
    
    wsRef.current.onopen = () => {
      console.log('WebSocket connected');
      addLog('Connected to DealMind AI logs', 'info');
    };
    
    wsRef.current.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data);
        if (message.type === 'log') {
          addLog(message.data.formatted_message, message.data.level.toLowerCase());
        } else if (message.type === 'status') {
          console.log('Status update:', message.data);
        }
      } catch (error) {
        console.error('Error parsing WebSocket message:', error);
      }
    };
    
    wsRef.current.onclose = () => {
      console.log('WebSocket disconnected');
      addLog('Disconnected from logs', 'warning');
    };
    
    wsRef.current.onerror = (error) => {
      console.error('WebSocket error:', error);
      addLog('WebSocket connection error', 'error');
    };
  };

  // Add log entry (mirrors Gradio log display)
  const addLog = (message, level = 'info') => {
    const timestamp = new Date().toLocaleTimeString();
    const formattedMessage = `[${timestamp}] ${message}`;
    
    setLogs(prevLogs => {
      const newLogs = [...prevLogs, { message: formattedMessage, level }];
      // Keep only last 50 logs to prevent memory issues
      return newLogs.slice(-50);
    });
  };

  // Handle category selection
  const handleCategoryChange = (categoryName, isChecked) => {
    if (isChecked) {
      if (selectedCategories.length >= 3) {
        setStatusMessage('⚠️ You can select up to 3 categories only.');
        setStatusType('warning');
        return;
      }
      setSelectedCategories(prev => [...prev, categoryName]);
    } else {
      setSelectedCategories(prev => prev.filter(cat => cat !== categoryName));
    }
    
    // Clear status message when categories are changed
    if (statusMessage) {
      setStatusMessage('');
      setStatusType('');
    }
  };

  // Validate categories (mirrors Gradio validation)
  const validateCategories = () => {
    if (selectedCategories.length === 0) {
      setStatusMessage('⚠️ Please select at least one category before running.');
      setStatusType('warning');
      return false;
    }
    if (selectedCategories.length > 3) {
      setStatusMessage('⚠️ You can select up to 3 categories only.');
      setStatusType('warning');
      return false;
    }
    return true;
  };

  // Start deal search (mirrors Gradio button click)
  const handleFindDeals = async () => {
    if (!validateCategories()) {
      return;
    }

    // CLEANUP PREVIOUS POLLING - Prevent multiple polls
    cleanupPolling();

    setIsSearching(true);
    setStatusMessage('Starting deal search...');
    setStatusType('success');
    setSearchResults([]); // Clear previous results
    addLog(`Starting search for categories: ${selectedCategories.join(', ')}`, 'info');

    try {
      // Start search
      const response = await axios.post(`${API_BASE}/api/search`, {
        selected_categories: selectedCategories
      });
      
      const jobId = response.data.job_id;
      setCurrentJobId(jobId);
      
      setStatusMessage('Search in progress... Please wait');
      setStatusType('success');
      
      // Poll for results with optimized timing
      pollForResults(jobId);
      
    } catch (error) {
      console.error('Error starting search:', error);
      const errorMessage = error.response?.data?.detail || 'Error starting search';
      setStatusMessage(errorMessage);
      setStatusType('error');
      setIsSearching(false);
      addLog(`Error: ${errorMessage}`, 'error');
    }
  };

  // OPTIMIZED POLLING - Smart intervals with exponential backoff
  const pollForResults = async (jobId) => {
    let pollCount = 0;
    const maxPolls = 60; // Maximum 60 polls (6 minutes with final interval)
    
    const poll = async () => {
      try {
        const response = await axios.get(`${API_BASE}/api/results/${jobId}`);
        const { status, results, error_message, total_count } = response.data;
        
        if (status === 'completed') {
          cleanupPolling();
          setSearchResults(results);
          setStatusMessage(`Search completed! Found ${total_count} deals.`);
          setStatusType('success');
          setIsSearching(false);
          addLog(`Search completed with ${total_count} results`, 'info');
          return;
        } 
        
        if (status === 'error') {
          cleanupPolling();
          setStatusMessage(error_message || 'Search failed');
          setStatusType('error');
          setIsSearching(false);
          addLog(`Search failed: ${error_message}`, 'error');
          return;
        }
        
        // Continue polling if status is 'running'
        pollCount++;
        
        if (pollCount >= maxPolls) {
          cleanupPolling();
          setStatusMessage('Search timeout - please try again');
          setStatusType('error');
          setIsSearching(false);
          addLog('Search timeout', 'error');
          return;
        }
        
        // SMART INTERVALS - Start fast, then slow down
        let nextInterval;
        if (pollCount <= 5) {
          nextInterval = 3000; // First 5 polls: every 3 seconds (15 seconds total)
        } else if (pollCount <= 15) {
          nextInterval = 5000; // Next 10 polls: every 5 seconds (50 seconds more)
        } else {
          nextInterval = 10000; // Remaining polls: every 10 seconds
        }
        
        pollIntervalRef.current = setTimeout(poll, nextInterval);
        
      } catch (error) {
        console.error('Error polling results:', error);
        cleanupPolling();
        setStatusMessage('Error retrieving results');
        setStatusType('error');
        setIsSearching(false);
        addLog('Error retrieving results', 'error');
      }
    };
    
    // Start first poll immediately
    poll();
  };

  // Handle table row click (mirrors Gradio select)
  const handleRowClick = (index) => {
    console.log('Row clicked:', index);
    // Add any row selection logic here if needed
  };

  // Format logs for display (mirrors Gradio HTML formatting)
  const formatLogsForDisplay = () => {
    return logs.map((log, index) => (
      <div key={index} className={`log-entry ${log.level}`}>
        <span dangerouslySetInnerHTML={{ __html: log.message }} />
      </div>
    ));
  };

  return (
    <div className="dealmind-container">
      {/* Header Section - mirrors Gradio gr.Markdown */}
      <div className="header-section">
        <div className="app-title">🏷️ DealSense AI</div>
        <div className="app-subtitle">Multi-Agent AI Hunting & Evaluating the Best Deals Online</div>
      </div>

      {/* Description Section */}
      <div className="app-description">
        🏷️ <strong>DealSense AI, a Multi-Agent System</strong>, fetches deals in real time, predicts fair prices, 
      and filters only the best bargains — smart, fast, and fair. Let AI do the work so you never overpay.
      </div>

      {/* How It Works Section */}
      <div className="how-it-works">
        <h3>🤖 How It Works:</h3>
        <p>
          <strong>1</strong> Choose up to 3 categories to search.<br />
          <strong>2</strong> Click "Find Deals" — AI agents scan feeds, estimate fair prices, and filter top discounts.<br />
          <strong>3</strong> See the best deals in a table with prices, discounts, and direct links.
        </p>
      </div>


      {/* Category Selection - mirrors Gradio CheckboxGroup */}
      <div className="category-section">
        <div className="section-title">🎯 Select up to 3 Deal Categories</div>
        <div className="checkbox-group">
          {categories.map((category) => (
            <div key={category.name} className="checkbox-item">
              <input
                type="checkbox"
                id={category.name}
                checked={selectedCategories.includes(category.name)}
                onChange={(e) => handleCategoryChange(category.name, e.target.checked)}
                disabled={isSearching}
              />
              <label htmlFor={category.name}>{category.display_name}</label>
            </div>
          ))}
        </div>
      </div>

      {/* Find Deals Button - mirrors Gradio Button */}
      <div className="button-section">
        <button 
          className="find-deals-btn" 
          onClick={handleFindDeals}
          disabled={isSearching}
        >
          {isSearching ? (
            <>
              <span className="spinner"></span>
              Searching...
            </>
          ) : (
            '🔍 Find Deals'
          )}
        </button>
      </div>

      {/* Status Message */}
      {statusMessage && (
        <div className={`status-message ${statusType}`}>
          {statusMessage}
        </div>
      )}

      {/* Results Section - mirrors Gradio Dataframe */}
      <div className="results-section">
        <div className="section-title">📋 Best Deals Found</div>
        {searchResults.length > 0 ? (
          <table className="results-table">
            <thead>
              <tr>
                <th style={{width: '50%'}}>Description</th>
                <th style={{width: '12%'}}>Price</th>
                <th style={{width: '12%'}}>AI Estimate</th>
                <th style={{width: '12%'}}>Discount</th>
                <th style={{width: '14%'}}>Link</th>
              </tr>
            </thead>
            <tbody>
              {searchResults.map((row, index) => (
                <tr key={index} onClick={() => handleRowClick(index)}>
                  <td>{row[0]}</td>
                  <td>{row[1]}</td>
                  <td>{row[2]}</td>
                  <td>{row[3]}</td>
                  <td dangerouslySetInnerHTML={{ __html: row[4] }}></td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <div className="empty-results">
            {isSearching ? (
              <div className="loading">
                <span className="spinner"></span>
                Searching for deals...
              </div>
            ) : (
              'No deals found yet. Click "Find Deals" to start searching.'
            )}
          </div>
        )}
      </div>

      {/* Live Agent Logs Section - mirrors Gradio HTML */}
      <div className="logs-section">
        <div className="section-title">📊 Live Agent Logs</div>
        <div className="logs-container" ref={logsContainerRef}>
          {logs.length > 0 ? formatLogsForDisplay() : (
            <div className="log-entry">Waiting for logs...</div>
          )}
        </div>
      </div>
    </div>
  );
};

export default DealMindApp;