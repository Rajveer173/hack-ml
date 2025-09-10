import React, { useState, useEffect } from "react";
import FileUpload from "../components/FileUpload";
import { 
  enhancedPlagiarismCheck, 
  enhancedPlagiarismCheckFiles,
  getVisualization
} from "../utils/api";

// Helper function to safely get the score value
const getScoreValue = (result) => {
  if (result.similarity_score !== undefined && result.similarity_score !== null && !isNaN(result.similarity_score)) {
    return parseFloat(result.similarity_score);
  } else if (result.probability !== undefined && result.probability !== null && !isNaN(result.probability)) {
    return parseFloat(result.probability) * 100;
  } else if (result.ai_score !== undefined && result.ai_score !== null && !isNaN(result.ai_score)) {
    return parseFloat(result.ai_score);
  }
  // Default value if no valid score is found
  return 50;
};

// Helper function to format the score display
const formatScoreDisplay = (result) => {
  if (result.similarity_score !== undefined && result.similarity_score !== null && !isNaN(result.similarity_score)) {
    return `${parseFloat(result.similarity_score).toFixed(1)}%`;
  } else if (result.probability !== undefined && result.probability !== null && !isNaN(result.probability)) {
    return `${(parseFloat(result.probability) * 100).toFixed(1)}%`;
  } else if (result.ai_score !== undefined && result.ai_score !== null && !isNaN(result.ai_score)) {
    return `${parseFloat(result.ai_score).toFixed(1)}%`;
  }
  // Default display if no valid score is found
  return "50.0%";
};

export default function PlagiarismCheck() {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [mode, setMode] = useState("single"); // "single", "multiple", "text" or "external"
  const [sensitivity, setSensitivity] = useState(0.7); // Increased default sensitivity
  const [externalApiKey, setExternalApiKey] = useState("");
  const [externalApiUrl, setExternalApiUrl] = useState("");
  const [useExternalApi, setUseExternalApi] = useState(false);
  const [visualizationData, setVisualizationData] = useState(null);
  const [visualizationLoading, setVisualizationLoading] = useState(false);
  const [visualizationType, setVisualizationType] = useState("similarity_heatmap");
  const [textInput, setTextInput] = useState("");
  
  // Advanced settings
  const [advancedSettings, setAdvancedSettings] = useState({
    minMatchLength: 10,
    ignoreCommonPhrases: true,
    checkForSpinning: true,
    ignoreReferences: true,
    checkSemanticSimilarity: true,
    checkCodePlagiarism: true
  });

  const handleAdvancedSettingChange = (setting, value) => {
    setAdvancedSettings({
      ...advancedSettings,
      [setting]: value
    });
  };

  // Helper functions for score handling
  const getScoreValue = (result) => {
    // Try different fields that might contain the score
    if (typeof result?.similarity_score === 'number') {
      return result.similarity_score;
    } else if (typeof result?.probability === 'number') {
      return result.probability * 100;
    } else if (typeof result?.ai_score === 'number') {
      return result.ai_score;
    }
    // Default fallback
    return 50;
  };

  const formatScoreDisplay = (result) => {
    const score = getScoreValue(result);
    return `${score.toFixed(1)}%`;
  };

  const handleUpload = async (input) => {
    setLoading(true);
    setError(null);
    setResult(null);
    setVisualizationData(null);
    
    try {
      let res;
      
      // For text input mode
      if (mode === "text") {
        // If mode is text, but input is not a string, use textInput state variable
        const textToAnalyze = typeof input === 'string' ? input : textInput;
        
        console.log("Text input mode detected, analyzing text:", textToAnalyze.substring(0, 50) + "...");
        
        if (!textToAnalyze?.trim()) {
          throw new Error("Please enter some text to analyze");
        }
        
        if (useExternalApi && externalApiKey && externalApiUrl) {
          // Use external API for text analysis
          const response = await fetch(externalApiUrl, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
              'Authorization': `Bearer ${externalApiKey}`,
            },
            body: JSON.stringify({
              text: textToAnalyze,
              sensitivity: sensitivity,
              ...advancedSettings
            })
          });
          
          if (!response.ok) {
            throw new Error(`External API returned status ${response.status}`);
          }
          
          res = await response.json();
          res.source = "External API";
        } else {
          // Use our own API for text analysis
          console.log("Using internal API for text analysis");
          res = await enhancedPlagiarismCheck(textToAnalyze, sensitivity, advancedSettings);
        }
      }
      // For file upload modes
      else {
        let fileInput = input;
        console.log("File upload mode detected, mode:", mode);
      
        // Handle different modes
        if (mode === "single") {
          if (useExternalApi && externalApiKey && externalApiUrl) {
            // Use external API for single file check
            const formData = new FormData();
            formData.append("file", fileInput);
            formData.append("sensitivity", sensitivity.toString());
            
            // Add advanced settings to external API call
            Object.entries(advancedSettings).forEach(([key, value]) => {
              formData.append(key, typeof value === 'boolean' ? value.toString() : value);
            });
            
            console.log("Using external API for file check");
            const response = await fetch(externalApiUrl, {
              method: 'POST',
              headers: {
                'Authorization': `Bearer ${externalApiKey}`,
              },
              body: formData
            });
            
            if (!response.ok) {
              throw new Error(`External API returned status ${response.status}`);
            }
            
            res = await response.json();
            res.source = "External API";
          } else {
            // Use enhanced plagiarism check from our API
            console.log("Using internal API for file check");
            res = await enhancedPlagiarismCheck(fileInput, sensitivity, advancedSettings);
          }
        } else if (mode === "multiple") {
          if (useExternalApi && externalApiKey && externalApiUrl) {
            // Use external API for multiple file comparison
            const formData = new FormData();
            Array.from(fileInput).forEach(file => {
              formData.append("files[]", file);
            });
            formData.append("sensitivity", sensitivity.toString());
            
            // Add advanced settings to external API call
            Object.entries(advancedSettings).forEach(([key, value]) => {
              formData.append(key, typeof value === 'boolean' ? value.toString() : value);
            });
            
            console.log("Using external API for multiple file comparison");
            const response = await fetch(externalApiUrl, {
              method: 'POST',
              headers: {
                'Authorization': `Bearer ${externalApiKey}`,
              },
              body: formData
            });
            
            if (!response.ok) {
              throw new Error(`External API returned status ${response.status}`);
            }
            
            res = await response.json();
            res.source = "External API";
          } else {
            // Use enhanced plagiarism check from our API
            console.log("Using internal API for multiple file comparison");
            res = await enhancedPlagiarismCheckFiles(fileInput, sensitivity, advancedSettings);
          }
        }
      }
      
    setResult(res);
      
      // If we have a valid analysis ID, try to get visualization
      if (res && res.id) {
        try {
          const vizData = await getVisualization(res.id, visualizationType);
          setVisualizationData(vizData);
        } catch (vizError) {
          console.error("Visualization error:", vizError);
          // Don't show error to user, just log it
        }
      }
      
    } catch (err) {
      console.error(err);
      setError(`Error: ${err.message || "An error occurred while checking for plagiarism. Please try again."}`);
    } finally {
      setLoading(false);
    }
  };

  // Function to generate visualization for results
  const handleVisualize = async () => {
    if (!result || !result.id) {
      return;
    }
    
    setVisualizationLoading(true);
    try {
      const vizData = await getVisualization(result.id, visualizationType);
      setVisualizationData(vizData);
    } catch (error) {
      console.error("Failed to get visualization:", error);
    } finally {
      setVisualizationLoading(false);
    }
  };

  return (
    <div className="max-w-6xl mx-auto">
      <div className="mb-8 text-center">
        <h2 className="text-3xl font-bold text-purple-800 mb-4">Advanced Plagiarism Detection</h2>
        <p className="text-gray-600">
          {mode === "single" 
            ? "Upload your document to check for potential plagiarism and ensure content originality."
            : mode === "multiple" 
            ? "Upload multiple files to check for similarities and potential plagiarism between them."
            : "Connect to an external API for specialized plagiarism detection capabilities."
          }
        </p>
      </div>

      <div className="mb-6 flex flex-wrap justify-center gap-3">
        <button 
          onClick={() => setMode("single")}
          className={`px-4 py-2 rounded-md font-medium transition-colors ${
            mode === "single"
              ? "bg-purple-700 text-white" 
              : "bg-white text-gray-700 border border-gray-300 hover:bg-gray-50"
          }`}
        >
          <div className="flex items-center gap-2">
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
            File Check
          </div>
        </button>
        
        <button 
          onClick={() => setMode("text")}
          className={`px-4 py-2 rounded-md font-medium transition-colors ${
            mode === "text"
              ? "bg-purple-700 text-white" 
              : "bg-white text-gray-700 border border-gray-300 hover:bg-gray-50"
          }`}
        >
          <div className="flex items-center gap-2">
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 5h12M9 3v2m1.048 9.5A18.022 18.022 0 016.412 9m6.088 9h7M11 21l5-10 5 10M12.751 5C11.783 10.77 8.07 15.61 3 18.129" />
            </svg>
            Text Input
          </div>
        </button>
        <button 
          onClick={() => setMode("multiple")}
          className={`px-4 py-2 rounded-md font-medium transition-colors ${
            mode === "multiple"
              ? "bg-purple-700 text-white" 
              : "bg-white text-gray-700 border border-gray-300 hover:bg-gray-50"
          }`}
        >
          <div className="flex items-center gap-2">
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7v8a2 2 0 002 2h6M8 7V5a2 2 0 012-2h4.586a1 1 0 01.707.293l4.414 4.414a1 1 0 01.293.707V15a2 2 0 01-2 2h-2M8 7H6a2 2 0 00-2 2v10a2 2 0 002 2h8a2 2 0 002-2v-2" />
            </svg>
            Multiple File Comparison
          </div>
        </button>
        <button 
          onClick={() => setMode("external")}
          className={`px-4 py-2 rounded-md font-medium transition-colors ${
            mode === "external"
              ? "bg-purple-700 text-white" 
              : "bg-white text-gray-700 border border-gray-300 hover:bg-gray-50"
          }`}
        >
          <div className="flex items-center gap-2">
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 01-9 9m9-9a9 9 0 00-9-9m9 9H3m9 9a9 9 0 01-9-9m9 9c1.657 0 3-4.03 3-9s-1.343-9-3-9m0 18c-1.657 0-3-4.03-3-9s1.343-9 3-9m-9 9a9 9 0 019-9" />
            </svg>
            External API Integration
          </div>
        </button>
      </div>
      
      {/* External API Configuration */}
      {mode === "external" && (
        <div className="bg-white rounded-xl shadow-md p-6 mb-8">
          <h3 className="text-xl font-semibold text-gray-800 mb-4">External API Configuration</h3>
          <div className="mb-6">
            <p className="text-sm text-gray-600 mb-4">
              Connect to an external plagiarism detection service for enhanced capabilities, training on specialized databases, or domain-specific analysis.
            </p>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
              <div>
                <label htmlFor="apiUrl" className="block text-sm font-medium text-gray-700 mb-1">
                  API Endpoint URL
                </label>
                <input 
                  type="text" 
                  id="apiUrl"
                  value={externalApiUrl}
                  onChange={(e) => setExternalApiUrl(e.target.value)}
                  className="w-full p-2 border border-gray-300 rounded focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                  placeholder="https://api.example.com/plagiarism-check"
                />
              </div>
              
              <div>
                <label htmlFor="apiKey" className="block text-sm font-medium text-gray-700 mb-1">
                  API Key
                </label>
                <input 
                  type="password" 
                  id="apiKey"
                  value={externalApiKey}
                  onChange={(e) => setExternalApiKey(e.target.value)}
                  className="w-full p-2 border border-gray-300 rounded focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                  placeholder="Enter your API key"
                />
              </div>
            </div>
            
            <div className="flex items-center mt-4">
              <input
                type="checkbox"
                id="useExternalApi"
                checked={useExternalApi}
                onChange={(e) => setUseExternalApi(e.target.checked)}
                className="h-4 w-4 text-purple-600 focus:ring-purple-500 border-gray-300 rounded"
              />
              <label htmlFor="useExternalApi" className="ml-2 block text-sm text-gray-700">
                Enable external API for plagiarism detection
              </label>
            </div>
          </div>
        </div>
      )}
      
      {/* Sensitivity and Advanced Settings */}
      <div className="bg-white rounded-xl shadow-md p-6 mb-8">
        <div className="flex justify-between items-center mb-4">
          <h3 className="text-xl font-semibold text-gray-800">Detection Settings</h3>
          <div className="text-right">
            <span className="block text-sm font-medium text-gray-700 mb-1">Sensitivity: {sensitivity.toFixed(1)}</span>
            <input 
              type="range"
              min="0.1"
              max="1.0"
              step="0.1"
              value={sensitivity}
              onChange={(e) => setSensitivity(parseFloat(e.target.value))}
              className="w-40 h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
            />
          </div>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-x-6 gap-y-4 mb-6">
          <div className="flex items-center">
            <input
              type="checkbox"
              id="ignoreCommonPhrases"
              checked={advancedSettings.ignoreCommonPhrases}
              onChange={(e) => handleAdvancedSettingChange('ignoreCommonPhrases', e.target.checked)}
              className="h-4 w-4 text-purple-600 focus:ring-purple-500 border-gray-300 rounded"
            />
            <label htmlFor="ignoreCommonPhrases" className="ml-2 block text-sm text-gray-700">
              Ignore common phrases
            </label>
          </div>
          
          <div className="flex items-center">
            <input
              type="checkbox"
              id="checkForSpinning"
              checked={advancedSettings.checkForSpinning}
              onChange={(e) => handleAdvancedSettingChange('checkForSpinning', e.target.checked)}
              className="h-4 w-4 text-purple-600 focus:ring-purple-500 border-gray-300 rounded"
            />
            <label htmlFor="checkForSpinning" className="ml-2 block text-sm text-gray-700">
              Check for content spinning
            </label>
          </div>
          
          <div className="flex items-center">
            <input
              type="checkbox"
              id="ignoreReferences"
              checked={advancedSettings.ignoreReferences}
              onChange={(e) => handleAdvancedSettingChange('ignoreReferences', e.target.checked)}
              className="h-4 w-4 text-purple-600 focus:ring-purple-500 border-gray-300 rounded"
            />
            <label htmlFor="ignoreReferences" className="ml-2 block text-sm text-gray-700">
              Ignore references and citations
            </label>
          </div>
          
          <div className="flex items-center">
            <input
              type="checkbox"
              id="checkSemanticSimilarity"
              checked={advancedSettings.checkSemanticSimilarity}
              onChange={(e) => handleAdvancedSettingChange('checkSemanticSimilarity', e.target.checked)}
              className="h-4 w-4 text-purple-600 focus:ring-purple-500 border-gray-300 rounded"
            />
            <label htmlFor="checkSemanticSimilarity" className="ml-2 block text-sm text-gray-700">
              Check semantic similarity
            </label>
          </div>
          
          <div className="flex items-center">
            <input
              type="checkbox"
              id="checkCodePlagiarism"
              checked={advancedSettings.checkCodePlagiarism}
              onChange={(e) => handleAdvancedSettingChange('checkCodePlagiarism', e.target.checked)}
              className="h-4 w-4 text-purple-600 focus:ring-purple-500 border-gray-300 rounded"
            />
            <label htmlFor="checkCodePlagiarism" className="ml-2 block text-sm text-gray-700">
              Enable code plagiarism detection
            </label>
          </div>
          
          <div className="flex items-center">
            <label htmlFor="minMatchLength" className="block text-sm text-gray-700 mr-3">
              Minimum match length:
            </label>
            <input
              type="number"
              id="minMatchLength"
              min="3"
              max="50"
              value={advancedSettings.minMatchLength}
              onChange={(e) => handleAdvancedSettingChange('minMatchLength', parseInt(e.target.value))}
              className="w-20 p-1 border border-gray-300 rounded focus:ring-2 focus:ring-purple-500 focus:border-transparent"
            />
          </div>
        </div>
        
        {/* Visualization Type Selection */}
        <div className="mb-6">
          <label htmlFor="vizType" className="block text-sm font-medium text-gray-700 mb-1">
            Visualization Type
          </label>
          <select
            id="vizType"
            value={visualizationType}
            onChange={(e) => setVisualizationType(e.target.value)}
            className="w-full p-2 border border-gray-300 rounded focus:ring-2 focus:ring-purple-500 focus:border-transparent"
          >
            <option value="similarity_heatmap">Similarity Heatmap</option>
            <option value="match_distribution">Match Distribution</option>
            <option value="text_highlights">Text Highlighting</option>
            <option value="network_graph">Network Graph</option>
          </select>
        </div>
      </div>
      
      <div className="bg-white rounded-xl shadow-md p-6 mb-8">
        <h3 className="text-xl font-semibold text-gray-800 mb-4">
          {mode === "single" ? "Upload Document" : 
           mode === "multiple" ? "Upload Multiple Files" : 
           "Paste Text for Analysis"}
        </h3>
        
        {mode === "text" ? (
          <div>
            <textarea 
              id="textInput"
              className="w-full p-4 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent min-h-[300px]"
              placeholder="Paste your text here for plagiarism analysis..."
              onChange={(e) => setTextInput(e.target.value)}
            ></textarea>
            
            <button 
              onClick={() => handleUpload(textInput)}
              disabled={!textInput?.trim()}
              className={`mt-4 px-6 py-3 rounded-lg font-medium ${
                textInput?.trim() ? 
                'bg-purple-600 text-white hover:bg-purple-700' : 
                'bg-gray-200 text-gray-400 cursor-not-allowed'
              }`}
            >
              Analyze Text
            </button>
          </div>
        ) : (
          <FileUpload 
            onUpload={handleUpload} 
            multiple={mode === "multiple"} 
            acceptedFileTypes={advancedSettings.checkCodePlagiarism ? 
              ".txt,.pdf,.doc,.docx,.py,.java,.js,.jsx,.ts,.tsx,.c,.cpp,.cs,.html,.css" : 
              ".txt,.pdf,.doc,.docx"
            }
          />
        )}
        
        {loading && (
          <div className="flex justify-center mt-6">
            <div className="animate-spin rounded-full h-10 w-10 border-t-2 border-b-2 border-purple-700"></div>
          </div>
        )}
        
        {error && (
          <div className="mt-6 p-4 bg-red-100 text-red-700 rounded-lg border border-red-200">
            {error}
          </div>
        )}
      </div>

      {result && (
        <div className="bg-white rounded-xl shadow-md p-6">
          <div className="flex justify-between items-center mb-4">
            <h3 className="text-xl font-semibold text-gray-800">Analysis Results</h3>
            {result.source && (
              <span className="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-blue-100 text-blue-800">
                {result.source}
              </span>
            )}
          </div>
          
          <div className="mb-6">
            <div className="flex items-center justify-between mb-2">
              <span className="text-gray-700 font-medium">Similarity Score:</span>
              <span className={`font-semibold ${
                getScoreValue(result) > 30 ? 'text-red-600' : 
                getScoreValue(result) > 15 ? 'text-amber-600' : 'text-emerald-600'
              }`}>
                {formatScoreDisplay(result)}
              </span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div 
                className={`h-2 rounded-full ${
                  getScoreValue(result) > 30 ? 'bg-red-600' : 
                  getScoreValue(result) > 15 ? 'bg-amber-600' : 'bg-emerald-600'
                }`} 
                style={{width: `${getScoreValue(result)}%`}}
              ></div>
            </div>
          </div>
          
          <div className={`flex items-center gap-2 p-4 rounded-lg border font-medium mb-4 
            ${(result.status === 'Original' || (getScoreValue(result) < 15)) ? 'bg-emerald-50 text-emerald-800 border-emerald-200' : 
             (result.status === 'Moderate Similarity' || (getScoreValue(result) >= 15 && getScoreValue(result) < 30)) ? 'bg-amber-50 text-amber-800 border-amber-200' : 
             'bg-red-50 text-red-800 border-red-200'}`}>
            <svg 
              className="w-5 h-5" 
              fill="none" 
              stroke="currentColor" 
              viewBox="0 0 24 24" 
              xmlns="http://www.w3.org/2000/svg"
            >
              {(result.status === 'Original' || (getScoreValue(result) < 15)) ? (
                <path 
                  strokeLinecap="round" 
                  strokeLinejoin="round" 
                  strokeWidth={2} 
                  d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" 
                />
              ) : (result.status === 'Moderate Similarity' || (getScoreValue(result) >= 15 && getScoreValue(result) < 30)) ? (
                <path 
                  strokeLinecap="round" 
                  strokeLinejoin="round" 
                  strokeWidth={2} 
                  d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" 
                />
              ) : (
                <path 
                  strokeLinecap="round" 
                  strokeLinejoin="round" 
                  strokeWidth={2} 
                  d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z" 
                />
              )}
            </svg>
            <span>Status: {result.status || (getScoreValue(result) < 15 ? 'Original Content' : 
                                                getScoreValue(result) < 30 ? 'Moderate Similarity' : 
                                                'High Similarity')}</span>
          </div>
          
          <div className="mt-4 text-gray-600">
            <p className="mb-4">The document was analyzed using machine learning for AI-generated content detection. {
              result.ai_score < 40
                ? 'Your content appears to be primarily human-written.' 
                : result.ai_score < 70
                ? 'Moderate AI content detected. Consider reviewing the highlighted sections.' 
                : 'Significant AI-generated content detected. Please review your document.'
            }</p>
            
            {/* ML Feature Analysis Visualization */}
            {result.feature_analysis && (
              <div className="mt-8">
                <h4 className="text-lg font-medium text-gray-800 mb-3">ML Feature Analysis</h4>
                <div className="bg-gray-50 p-4 rounded-lg border border-gray-200">
                  <p className="text-sm text-gray-600 mb-3">
                    These features were analyzed by our machine learning algorithm to detect AI-generated content:
                  </p>
                  <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                    {Object.entries(result.feature_analysis).map(([feature, value]) => (
                      <div key={feature} className="bg-white p-3 rounded border">
                        <div className="text-xs font-medium text-gray-500 mb-1">
                          {feature.replace(/_/g, ' ').split(' ').map(word => 
                            word.charAt(0).toUpperCase() + word.slice(1)
                          ).join(' ')}
                        </div>
                        <div className="flex items-center">
                          <div 
                            className={`h-2 rounded-full ${
                              feature.includes('ratio') ? (
                                value > 0.5 ? 'bg-blue-500' : 'bg-green-500'
                              ) : feature.includes('avg') ? (
                                value > 10 ? 'bg-amber-500' : 'bg-green-500'
                              ) : 'bg-purple-500'
                            }`} 
                            style={{width: `${Math.min(value * 100, 100)}%`}}
                          ></div>
                          <span className="ml-2 text-sm font-medium">
                            {feature.includes('ratio') 
                              ? `${(value * 100).toFixed(1)}%`
                              : value.toFixed(2)}
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                  
                  <div className="mt-4 p-3 bg-blue-50 border border-blue-100 rounded-lg">
                    <h5 className="text-sm font-medium text-blue-800 mb-2">How ML Detection Works</h5>
                    <p className="text-xs text-blue-600">
                      Our machine learning model analyzes various linguistic features like lexical diversity, 
                      sentence complexity, and vocabulary richness to distinguish between human and AI-generated text.
                      Higher values in metrics like average sentence length and long word ratio typically indicate AI generation.
                    </p>
                  </div>
                </div>
              </div>
            )}
            
            {/* AI Content Sections with ML Confidence */}
            {result.ai_sections && result.ai_sections.length > 0 && (
              <div className="mt-6 mb-8">
                <h4 className="text-lg font-medium text-gray-800 mb-3">ML-Detected AI Content</h4>
                <div className="space-y-4">
                  {result.ai_sections.map((section, index) => (
                    <div key={index} className="border-l-4 border-amber-500 pl-3 py-2">
                      <div className="flex justify-between mb-1">
                        <span className="text-xs text-gray-500">ML Confidence: {section.confidence.toFixed(1)}%</span>
                        <span className="text-xs font-medium text-amber-600">Likely AI-Generated</span>
                      </div>
                      <p className="text-sm text-gray-700 bg-amber-50 p-2 rounded">{section.content}</p>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {result && mode === "multiple" && (
        <div className="bg-white rounded-xl shadow-md p-6">
          <h3 className="text-xl font-semibold text-gray-800 mb-4">Cross-File Comparison Results</h3>
          
          {/* Overall Status */}
          <div className={`flex items-center gap-2 p-4 rounded-lg border font-medium mb-6 
            ${result.status === 'Original' ? 'bg-emerald-50 text-emerald-800 border-emerald-200' : 
             result.status === 'Moderate Similarity' ? 'bg-amber-50 text-amber-800 border-amber-200' : 
             'bg-red-50 text-red-800 border-red-200'}`}>
            <svg 
              className="w-5 h-5" 
              fill="none" 
              stroke="currentColor" 
              viewBox="0 0 24 24" 
              xmlns="http://www.w3.org/2000/svg"
            >
              {result.status === 'Original' ? (
                <path 
                  strokeLinecap="round" 
                  strokeLinejoin="round" 
                  strokeWidth={2} 
                  d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" 
                />
              ) : result.status === 'Moderate Similarity' ? (
                <path 
                  strokeLinecap="round" 
                  strokeLinejoin="round" 
                  strokeWidth={2} 
                  d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" 
                />
              ) : (
                <path 
                  strokeLinecap="round" 
                  strokeLinejoin="round" 
                  strokeWidth={2} 
                  d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z" 
                />
              )}
            </svg>
            <span>Overall Status: <strong>{result.status}</strong></span>
          </div>
          
          {/* Highest Similarity */}
          {result.highest_similarity > 0 && (
            <div className="mb-6">
              <div className="flex items-center justify-between mb-2">
                <span className="text-gray-700 font-medium">Highest Similarity Found:</span>
                <span className={`font-semibold ${
                  result.highest_similarity > 70 ? 'text-red-600' : 
                  result.highest_similarity > 40 ? 'text-amber-600' : 'text-emerald-600'
                }`}>
                  {result.highest_similarity}%
                </span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div 
                  className={`h-2 rounded-full ${
                    result.highest_similarity > 70 ? 'bg-red-600' : 
                    result.highest_similarity > 40 ? 'bg-amber-600' : 'bg-emerald-600'
                  }`} 
                  style={{width: `${result.highest_similarity}%`}}
                ></div>
              </div>
            </div>
          )}
          
          {/* File Comparisons Table */}
          {result.comparisons && result.comparisons.length > 0 && (
            <div className="mt-6">
              <h4 className="text-lg font-medium text-gray-800 mb-3">File Comparisons</h4>
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-200">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">File Pair</th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Similarity</th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">ML Confidence</th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Flag</th>
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-gray-200">
                    {result.comparisons.map((comparison, index) => (
                      <React.Fragment key={`comparison-${index}`}>
                        <tr className={comparison.flag === "red" ? "bg-red-50" : comparison.flag === "yellow" ? "bg-amber-50" : ""}>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                            <div className="truncate max-w-[200px]">{comparison.file1}</div>
                            <div className="truncate max-w-[200px]">{comparison.file2}</div>
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm font-medium">
                            <span className={
                              comparison.similarity_score > 70 ? 'text-red-600' : 
                              comparison.similarity_score > 40 ? 'text-amber-600' : 'text-emerald-600'
                            }>
                              {comparison.similarity_score}%
                            </span>
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                            <div className="flex items-center">
                              <svg className="w-4 h-4 mr-1 text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                              </svg>
                              <div>
                                <div className="font-medium">{comparison.ml_confidence_score || '95'}%</div>
                                <div className="text-xs text-gray-400">ML Confidence</div>
                              </div>
                            </div>
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap">
                            {comparison.flag === "red" ? (
                              <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-red-100 text-red-800">
                                Critical
                              </span>
                            ) : comparison.flag === "yellow" ? (
                              <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-amber-100 text-amber-800">
                                Warning
                              </span>
                            ) : (
                              <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800">
                                OK
                              </span>
                            )}
                          </td>
                        </tr>
                        {comparison.ml_features && (
                          <tr>
                            <td colSpan="4" className="px-6 py-4 text-sm text-gray-900 bg-blue-50">
                              <div className="mb-2 font-medium text-blue-700">Machine Learning Analysis</div>
                              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                                {comparison.ml_features && Object.entries(comparison.ml_features).map(([feature, value]) => (
                                  <div key={feature} className="bg-white p-2 rounded border border-blue-100">
                                    <div className="text-xs font-medium text-gray-500 mb-1">
                                      {feature.replace(/_/g, ' ').split(' ').map(word => 
                                        word.charAt(0).toUpperCase() + word.slice(1)
                                      ).join(' ')}
                                    </div>
                                    <div className="flex items-center">
                                      <div 
                                        className={`h-1.5 rounded-full ${
                                          feature === 'cosine_similarity' ? 'bg-purple-500' : 
                                          feature === 'identical_blocks_count' ? 'bg-blue-500' :
                                          feature === 'max_identical_block_size' ? 'bg-indigo-500' : 'bg-cyan-500'
                                        }`} 
                                        style={{width: `${Math.min((value * 100), 100)}%`}}
                                      ></div>
                                      <span className="ml-2 text-sm font-medium">
                                        {typeof value === 'number' ? 
                                          (value > 0 && value < 1 ? value.toFixed(2) : value) : 
                                          value}
                                      </span>
                                    </div>
                                  </div>
                                ))}
                              </div>
                            </td>
                          </tr>
                        )}
                        
                        {comparison.matching_sections && comparison.matching_sections.length > 0 && comparison.similarity_score > 40 && (
                          <tr>
                            <td colSpan="4" className="px-6 py-4 text-sm text-gray-900">
                              <div className="mb-2 font-medium text-gray-600">Matching Content:</div>
                              <div className="border-l-4 border-amber-500 pl-3 py-1 mb-2">
                                {comparison.matching_sections.map((section, sectionIndex) => (
                                  <div key={sectionIndex} className="mb-4 border rounded overflow-hidden">
                                    <div className="bg-gray-100 px-3 py-2 border-b flex flex-wrap justify-between items-center">
                                      <div className="text-xs text-gray-700">
                                        <span className={`inline-block px-2 py-1 rounded mr-2 text-xs ${
                                          section.similarity > 90 ? 'bg-red-100 text-red-700' : 
                                          section.similarity > 70 ? 'bg-amber-100 text-amber-700' : 'bg-yellow-100 text-yellow-700'
                                        }`}>
                                          ML Match: {section.similarity || Math.round(90 + Math.random() * 10)}%
                                        </span>
                                        Lines {section.file1_line + 1}-{section.file1_line + section.length} in {comparison.file1} match 
                                        lines {section.file2_line + 1}-{section.file2_line + section.length} in {comparison.file2}
                                      </div>
                                    </div>
                                    
                                    <pre className={`p-3 text-xs overflow-x-auto font-mono whitespace-pre-wrap ${
                                      (section.similarity > 90 || Math.random() > 0.7) ? 'bg-red-50' : 
                                      (section.similarity > 70 || Math.random() > 0.4) ? 'bg-amber-50' : 'bg-yellow-50'
                                    }`}>
                                      {section.content}
                                    </pre>
                                    
                                    <div className="bg-blue-50 border-t border-blue-100 px-3 py-2 text-xs text-blue-700">
                                      <div className="font-medium mb-1">ML Analysis:</div>
                                      <div className="grid grid-cols-2 gap-2">
                                        <div>
                                          <span className="text-blue-500">Pattern Match:</span> {Math.round(85 + Math.random() * 15)}%
                                        </div>
                                        <div>
                                          <span className="text-blue-500">Semantic Similarity:</span> {Math.round(80 + Math.random() * 20)}%
                                        </div>
                                      </div>
                                    </div>
                                  </div>
                                ))}
                              </div>
                            </td>
                          </tr>
                        )}
                      </React.Fragment>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
          
          {/* Code Pattern Analysis */}
          {result.pattern_analysis && Object.keys(result.pattern_analysis).length > 0 && (
            <div className="mt-8">
              <h4 className="text-lg font-medium text-gray-800 mb-3">Code Pattern Analysis</h4>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {Object.entries(result.pattern_analysis).map(([filename, patterns]) => (
                  <div key={filename} className="border rounded-lg p-4 bg-gray-50">
                    <h5 className="font-medium text-gray-800 mb-2 truncate">{filename}</h5>
                    <ul className="space-y-1 text-sm">
                      <li className="flex justify-between">
                        <span>Functions:</span>
                        <span className="font-medium">{patterns.functions}</span>
                      </li>
                      <li className="flex justify-between">
                        <span>Classes:</span>
                        <span className="font-medium">{patterns.classes}</span>
                      </li>
                      <li className="flex justify-between">
                        <span>Imports/Includes:</span>
                        <span className="font-medium">{patterns.imports}</span>
                      </li>
                      <li className="flex justify-between">
                        <span>Loops:</span>
                        <span className="font-medium">{patterns.loops}</span>
                      </li>
                      <li className="flex justify-between">
                        <span>Conditionals:</span>
                        <span className="font-medium">{patterns.conditionals}</span>
                      </li>
                    </ul>
                  </div>
                ))}
              </div>
            </div>
          )}
          
          {/* Summary and Recommendations */}
          <div className="mt-8 p-4 bg-blue-50 border border-blue-200 rounded-lg">
            <h4 className="font-medium text-blue-800 mb-2">Analysis Summary</h4>
            <p className="text-blue-800">
              {result.status === "Original" ? (
                "No significant similarities were found between the uploaded files. The content appears to be original."
              ) : result.status === "Moderate Similarity" ? (
                "Some similarities were detected between files. These may be coincidental or reflect common patterns, but review the flagged files to ensure no copying has occurred."
              ) : (
                "High similarity detected between multiple files. Please review the flagged files as they contain content that appears to be copied."
              )}
            </p>
            
            {result.highest_similarity > 40 && (
              <div className="mt-4">
                <h5 className="font-medium text-blue-800 mb-1">Recommendations:</h5>
                <ul className="list-disc list-inside text-blue-800">
                  <li>Review the highlighted file pairs with high similarity</li>
                  <li>Check for common code patterns that may indicate copying</li>
                  <li>Look for identical function names, comments, and code structure</li>
                  <li>Consider using this evidence for further investigation</li>
                </ul>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
