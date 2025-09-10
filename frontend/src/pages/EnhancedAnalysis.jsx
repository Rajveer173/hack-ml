import React, { useState, useEffect } from "react";
import { 
  enhancedAIDetection, 
  enhancedAIDetectionFile,
  enhancedPlagiarismCheck,
  enhancedPlagiarismCheckFiles,
  getAnalysisHistory, 
  getVisualization,
  exportResults,
  updateSettings,
  provideFeedback,
  getSettings,
  testApiConnection,
  callCustomApi
} from "../utils/api";
import { useAuth } from "../context/AuthContext";
import { Link } from "react-router-dom";
import FileUpload from "../components/FileUpload";

export default function EnhancedAnalysis() {
  const { isAuthenticated, user, userSettings, updateUserSettings, login, logout } = useAuth();
  
  const [text, setText] = useState("");
  const [files, setFiles] = useState([]);
  const [analysisType, setAnalysisType] = useState("ai-detection"); // "ai-detection" or "plagiarism"
  const [inputMethod, setInputMethod] = useState("text"); // "text" or "file"
  const [sensitivity, setSensitivity] = useState(0.5);
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [visualizationType, setVisualizationType] = useState("features");
  const [visualizationData, setVisualizationData] = useState(null);
  const [exportFormat, setExportFormat] = useState("pdf");
  const [feedback, setFeedback] = useState({ rating: 0, comments: "" });
  const [activeTab, setActiveTab] = useState("analysis");
  const [settings, setSettings] = useState({});
  const [apiConnections, setApiConnections] = useState([]);
  const [selectedApi, setSelectedApi] = useState(null);
  const [showApiModal, setShowApiModal] = useState(false);
  const [newApiConnection, setNewApiConnection] = useState({
    name: "",
    endpoint: "",
    apiKey: "",
    type: "plagiarism" // or "ai-detection"
  });

  // Load settings and history on component mount
  useEffect(() => {
    fetchSettings();
    fetchHistory();
    fetchApiConnections();
  }, []);

  // Update local settings when user settings change
  useEffect(() => {
    if (userSettings) {
      setSensitivity(userSettings.defaultSensitivity || 0.5);
      setAnalysisType(userSettings.defaultAnalysisType || "ai-detection");
      setExportFormat(userSettings.defaultExportFormat || "pdf");
      setVisualizationType(userSettings.defaultVisualizationType || "features");
      setSettings(userSettings);
    }
  }, [userSettings]);

  const fetchSettings = async () => {
    try {
      // This will try user-specific settings first, then fall back to general settings
      const settingsData = await getSettings();
      if (!userSettings) {
        // Only set from API if we don't have user settings from auth context
        setSensitivity(settingsData.defaultSensitivity || 0.5);
        setAnalysisType(settingsData.defaultAnalysisType || "ai-detection");
        setExportFormat(settingsData.defaultExportFormat || "pdf");
        setVisualizationType(settingsData.defaultVisualizationType || "features");
        setSettings(settingsData);
      }
    } catch (error) {
      console.error("Failed to fetch settings:", error);
    }
  };

  const fetchHistory = async () => {
    try {
      const historyData = await getAnalysisHistory();
      setHistory(historyData.history || []);
    } catch (error) {
      console.error("Failed to fetch history:", error);
    }
  };

  const handleAnalyze = async () => {
    // Validate input based on input method
    if (inputMethod === "text" && !text.trim()) {
      alert("Please enter text to analyze");
      return;
    }
    
    if (inputMethod === "file" && files.length === 0) {
      alert("Please upload file(s) to analyze");
      return;
    }
    
    // For plagiarism with file uploads, we need at least 2 files
    if (inputMethod === "file" && analysisType === "plagiarism" && files.length < 2) {
      alert("Please upload at least 2 files for plagiarism comparison");
      return;
    }

    setLoading(true);
    setResult(null);
    setVisualizationData(null);
    
    // Check if user has selected a custom API that matches the current analysis type
    const useCustomApi = selectedApi && selectedApi.type === analysisType;

    try {
      let response;
      
      // If using a custom API
      if (useCustomApi) {
        try {
          // Create a FormData object for file uploads
          const formData = new FormData();
          
          if (inputMethod === "text") {
            formData.append("text", text);
          } else {
            for (let i = 0; i < files.length; i++) {
              formData.append("files", files[i]);
            }
          }
          
          formData.append("sensitivity", sensitivity);
          
          // Add API key to headers if provided
          const headers = {};
          if (selectedApi.apiKey) {
            headers["Authorization"] = `Bearer ${selectedApi.apiKey}`;
          }
          
          // Make the API request to the custom endpoint
          const apiResponse = await fetch(selectedApi.endpoint, {
            method: "POST",
            headers,
            body: formData,
          });
          
          if (!apiResponse.ok) {
            throw new Error(`API responded with status: ${apiResponse.status}`);
          }
          
          response = await apiResponse.json();
          
          // Add a notification about the external API usage
          response.source = `External API: ${selectedApi.name}`;
        } catch (apiError) {
          console.error("Error with custom API:", apiError);
          alert(`Failed to use custom API: ${apiError.message}. Falling back to default service.`);
          
          // Fall back to the default service
          useCustomApi = false;
        }
      }
      
      // If not using a custom API or if custom API failed
      if (!useCustomApi) {
        // Handle different combinations of analysis type and input method
        if (analysisType === "ai-detection") {
          // AI detection can use either text or file input
          if (inputMethod === "text") {
            response = await enhancedAIDetection(text, sensitivity);
          } else {
            // For file-based AI detection, we only analyze one file at a time
            if (files.length === 0) {
              alert("Please select a file to analyze");
              setLoading(false);
              return;
            }
            
            // Use the first file for AI detection
            response = await enhancedAIDetectionFile(files[0], sensitivity);
          }
        } else {
          // Plagiarism detection can use either text or files
          if (inputMethod === "text") {
            response = await enhancedPlagiarismCheck(text, sensitivity);
          } else {
            response = await enhancedPlagiarismCheckFiles(files, sensitivity);
          }
        }
      }
      
      setResult(response);
      // Refresh history after analysis
      fetchHistory();
    } catch (error) {
      console.error(`Error during ${analysisType}:`, error);
      alert(`An error occurred during ${analysisType}. Please try again.`);
    } finally {
      setLoading(false);
    }
  };

  const [visualizationLoading, setVisualizationLoading] = useState(false);
  const [visualizationError, setVisualizationError] = useState(null);
  
  const handleVisualize = async (analysisId = result?.id) => {
    if (!analysisId) {
      setVisualizationError("No analysis selected. Please run an analysis first.");
      return;
    }

    setVisualizationLoading(true);
    setVisualizationError(null);
    try {
      const data = await getVisualization(analysisId, visualizationType);
      if (!data || !data.image) {
        throw new Error("No visualization data returned from server");
      }
      setVisualizationData(data);
    } catch (error) {
      console.error("Error generating visualization:", error);
      setVisualizationError(`Failed to generate visualization: ${error.message || "Unknown error"}. Please try a different visualization type.`);
      setVisualizationData(null);
    } finally {
      setVisualizationLoading(false);
    }
  };

  const handleExport = async (analysisId = result?.id) => {
    if (!analysisId) return;

    try {
      const blob = await exportResults(analysisId, exportFormat);
      
      // Create a download link and trigger the download
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.style.display = 'none';
      a.href = url;
      a.download = `analysis-${analysisId}.${exportFormat}`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
    } catch (error) {
      console.error("Error exporting results:", error);
    }
  };

  const handleSubmitFeedback = async () => {
    if (!result?.id || feedback.rating === 0) return;

    try {
      await provideFeedback(result.id, feedback);
      alert("Thank you for your feedback!");
      setFeedback({ rating: 0, comments: "" });
    } catch (error) {
      console.error("Error submitting feedback:", error);
    }
  };
  
  const handleFileUpload = (uploadedFiles) => {
    setFiles(uploadedFiles);
  };
  
  const clearFiles = () => {
    setFiles([]);
  };

  // API Connection functions
  const fetchApiConnections = async () => {
    try {
      // This would normally call an API endpoint to fetch the user's API connections
      // For now, we'll use some mock data stored in local storage
      const storedConnections = localStorage.getItem('apiConnections');
      if (storedConnections) {
        setApiConnections(JSON.parse(storedConnections));
      }
    } catch (error) {
      console.error("Failed to fetch API connections:", error);
    }
  };

  const handleSaveApiConnection = () => {
    if (!newApiConnection.name || !newApiConnection.endpoint) {
      alert("Please provide a name and endpoint for your API connection");
      return;
    }

    const updatedConnections = [
      ...apiConnections, 
      { ...newApiConnection, id: Date.now().toString() }
    ];
    
    setApiConnections(updatedConnections);
    localStorage.setItem('apiConnections', JSON.stringify(updatedConnections));
    
    setNewApiConnection({
      name: "",
      endpoint: "",
      apiKey: "",
      type: "plagiarism"
    });
    
    setShowApiModal(false);
  };

  const handleDeleteApiConnection = (id) => {
    const updatedConnections = apiConnections.filter(conn => conn.id !== id);
    setApiConnections(updatedConnections);
    localStorage.setItem('apiConnections', JSON.stringify(updatedConnections));
    
    if (selectedApi && selectedApi.id === id) {
      setSelectedApi(null);
    }
  };

  const handleUseApi = async (api) => {
    try {
      // Test the API connection before setting it as active
      const result = await testApiConnection(api);
      
      if (result.success) {
        setSelectedApi(api);
        alert(`Successfully connected to ${api.name}. Now using it for ${api.type === "plagiarism" ? "plagiarism" : "AI detection"} checks.`);
      } else {
        alert(`Connection to ${api.name} failed: ${result.message}`);
      }
    } catch (error) {
      console.error("Error testing API connection:", error);
      alert(`Failed to test connection to ${api.name}. Error: ${error.message}`);
    }
  };

  const handleSaveSettings = async () => {
    try {
      const newSettings = {
        defaultSensitivity: sensitivity,
        defaultAnalysisType: analysisType,
        defaultExportFormat: exportFormat,
        defaultVisualizationType: visualizationType
      };
      
      if (!isAuthenticated) {
        // If not authenticated, warn user that settings will only be temporary
        const confirmed = window.confirm(
          "You are not logged in. Settings will only be saved for this session and won't be available next time. Would you like to continue?"
        );
        
        if (!confirmed) return;
      }
      
      const response = await updateSettings(newSettings);
      
      // Update both local state and auth context (if authenticated)
      setSettings({...settings, ...newSettings});
      if (isAuthenticated && updateUserSettings) {
        updateUserSettings({...userSettings, ...newSettings});
      }
      
      if (isAuthenticated) {
        alert("Settings saved successfully to your account!");
      } else {
        alert("Settings saved for this session only. Please login to save permanently.");
      }
    } catch (error) {
      console.error("Error saving settings:", error);
      alert("Failed to save settings. Please try again.");
    }
  };

  return (
    <div className="max-w-6xl mx-auto px-4 py-8">
      {/* Authentication Status Bar */}
      <div className="mb-4 p-3 bg-gray-100 rounded-md flex justify-between items-center">
        <div>
          {isAuthenticated ? (
            <span className="text-green-600 font-medium">
              <i className="fas fa-user-check mr-2"></i>
              Logged in as: <span className="font-bold">{user?.username}</span>
            </span>
          ) : (
            <span className="text-orange-500">
              <i className="fas fa-user-slash mr-2"></i>
              Not logged in - settings will not be saved
            </span>
          )}
        </div>
        <div>
          {isAuthenticated ? (
            <button
              onClick={() => logout()}
              className="bg-red-500 hover:bg-red-600 text-white px-4 py-1 rounded-md text-sm"
            >
              Logout
            </button>
          ) : (
            <Link to="/login" className="bg-blue-500 hover:bg-blue-600 text-white px-4 py-1 rounded-md text-sm">
              Login / Register
            </Link>
          )}
        </div>
      </div>

      <div className="mb-8 text-center">
        <h2 className="text-3xl font-bold text-purple-800 mb-2">Enhanced Content Analysis</h2>
        <p className="text-gray-600">
          Advanced AI detection and plagiarism checking with visualization tools and customizable settings
        </p>
      </div>

      <div className="mb-6 flex flex-wrap justify-center gap-2">
        <button 
          onClick={() => setActiveTab("analysis")}
          className={`px-4 py-2 rounded-md font-medium transition-colors ${
            activeTab === "analysis" 
              ? "bg-purple-700 text-white" 
              : "bg-white text-gray-700 border border-gray-300 hover:bg-gray-50"
          }`}
        >
          Analysis
        </button>
        <button 
          onClick={() => setActiveTab("history")}
          className={`px-4 py-2 rounded-md font-medium transition-colors ${
            activeTab === "history" 
              ? "bg-purple-700 text-white" 
              : "bg-white text-gray-700 border border-gray-300 hover:bg-gray-50"
          }`}
        >
          History
        </button>
        <button 
          onClick={() => setActiveTab("visualization")}
          className={`px-4 py-2 rounded-md font-medium transition-colors ${
            activeTab === "visualization" 
              ? "bg-purple-700 text-white" 
              : "bg-white text-gray-700 border border-gray-300 hover:bg-gray-50"
          }`}
        >
          Visualization
        </button>
        <button 
          onClick={() => setActiveTab("api-connections")}
          className={`px-4 py-2 rounded-md font-medium transition-colors flex items-center ${
            activeTab === "api-connections" 
              ? "bg-purple-700 text-white" 
              : "bg-white text-gray-700 border border-gray-300 hover:bg-gray-50"
          }`}
        >
          <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-1" viewBox="0 0 20 20" fill="currentColor">
            <path fillRule="evenodd" d="M12.316 3.051a1 1 0 01.633 1.265l-4 12a1 1 0 11-1.898-.632l4-12a1 1 0 011.265-.633zM5.707 6.293a1 1 0 010 1.414L3.414 10l2.293 2.293a1 1 0 11-1.414 1.414l-3-3a1 1 0 010-1.414l3-3a1 1 0 011.414 0zm8.586 0a1 1 0 011.414 0l3 3a1 1 0 010 1.414l-3 3a1 1 0 11-1.414-1.414L16.586 10l-2.293-2.293a1 1 0 010-1.414z" clipRule="evenodd" />
          </svg>
          API Connections
        </button>
        <button 
          onClick={() => setActiveTab("settings")}
          className={`px-4 py-2 rounded-md font-medium transition-colors ${
            activeTab === "settings" 
              ? "bg-purple-700 text-white" 
              : "bg-white text-gray-700 border border-gray-300 hover:bg-gray-50"
          }`}
        >
          Settings
        </button>
      </div>

      {activeTab === "analysis" && (
        <div className="bg-white rounded-xl shadow-md p-6 mb-8">
          <div className="mb-6">
            <div className="flex space-x-4 mb-4">
              <button 
                onClick={() => setAnalysisType("ai-detection")}
                className={`flex-1 px-4 py-2 rounded-md font-medium ${
                  analysisType === "ai-detection"
                    ? "bg-blue-600 text-white"
                    : "bg-gray-100 text-gray-800 hover:bg-gray-200"
                }`}
              >
                AI Content Detection
              </button>
              <button 
                onClick={() => setAnalysisType("plagiarism")}
                className={`flex-1 px-4 py-2 rounded-md font-medium ${
                  analysisType === "plagiarism"
                    ? "bg-blue-600 text-white"
                    : "bg-gray-100 text-gray-800 hover:bg-gray-200"
                }`}
              >
                Plagiarism Check
              </button>
            </div>

            <div className="mb-4">
              <label className="block text-gray-700 font-medium mb-2">Sensitivity</label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.1"
                value={sensitivity}
                onChange={(e) => setSensitivity(parseFloat(e.target.value))}
                className="w-full"
              />
              <div className="flex justify-between text-xs text-gray-500">
                <span>More Permissive</span>
                <span>Balanced</span>
                <span>More Strict</span>
              </div>
            </div>
            
            {/* Input Method Toggle */}
            <div className="mb-4">
              <div className="flex space-x-4">
                <button 
                  onClick={() => setInputMethod("text")}
                  className={`flex-1 px-4 py-2 rounded-md font-medium ${
                    inputMethod === "text"
                      ? "bg-gray-700 text-white"
                      : "bg-gray-100 text-gray-800 hover:bg-gray-200"
                  }`}
                >
                  <i className="fas fa-keyboard mr-2"></i> Enter Text
                </button>
                <button 
                  onClick={() => setInputMethod("file")}
                  className={`flex-1 px-4 py-2 rounded-md font-medium ${
                    inputMethod === "file"
                      ? "bg-gray-700 text-white"
                      : "bg-gray-100 text-gray-800 hover:bg-gray-200"
                  }`}
                >
                  <i className="fas fa-file-upload mr-2"></i> Upload Document{analysisType === "plagiarism" ? "s" : ""}
                </button>
              </div>
              
              {inputMethod === "file" && (
                <p className="text-sm text-gray-500 mt-2">
                  <i className="fas fa-info-circle mr-1"></i>
                  {analysisType === "plagiarism" 
                    ? "For plagiarism check, upload at least 2 files to compare."
                    : "For AI detection, upload a text document (.txt, .docx, etc.) to analyze."}
                </p>
              )}
            </div>

            {inputMethod === "text" ? (
              <textarea
                value={text}
                onChange={(e) => setText(e.target.value)}
                placeholder={`Enter text to analyze for ${analysisType === "ai-detection" ? "AI-generated content" : "plagiarism"}`}
                className="w-full p-4 border border-gray-300 rounded-lg h-64 focus:outline-none focus:ring-2 focus:ring-purple-500"
              ></textarea>
            ) : (
              <div className="border border-gray-300 rounded-lg p-4">
                <FileUpload 
                  onUpload={handleFileUpload} 
                  multiple={analysisType === "plagiarism"} 
                  acceptedFileTypes={analysisType === "ai-detection" ? 
                    ".txt,.doc,.docx,.pdf,.py,.java,.js,.jsx,.ts,.tsx,.c,.cpp,.cs,.html,.css" : 
                    undefined}
                />
                
                {files.length > 0 && (
                  <div className="mt-4">
                    <h4 className="font-medium text-gray-700 mb-2">Uploaded Files:</h4>
                    <ul className="max-h-40 overflow-y-auto">
                      {files.map((file, index) => (
                        <li key={index} className="flex justify-between items-center py-1 border-b">
                          <span className="text-sm">{file.name} ({(file.size / 1024).toFixed(1)} KB)</span>
                          <button 
                            onClick={() => {
                              const newFiles = [...files];
                              newFiles.splice(index, 1);
                              setFiles(newFiles);
                            }}
                            className="text-red-500 hover:text-red-700"
                          >
                            <i className="fas fa-times"></i>
                          </button>
                        </li>
                      ))}
                    </ul>
                    <button 
                      onClick={clearFiles}
                      className="mt-2 px-3 py-1 bg-red-100 text-red-700 rounded-md text-sm hover:bg-red-200"
                    >
                      Clear All
                    </button>
                  </div>
                )}
              </div>
            )}
          </div>

          <button
            onClick={handleAnalyze}
            disabled={loading || (inputMethod === "text" && !text.trim()) || (inputMethod === "file" && files.length === 0)}
            className="w-full bg-purple-700 text-white py-3 rounded-lg font-medium hover:bg-purple-800 transition-colors disabled:bg-purple-300 disabled:cursor-not-allowed"
          >
            {loading ? "Analyzing..." : `Analyze with Enhanced ${analysisType === "ai-detection" ? "AI Detection" : "Plagiarism Check"}`}
          </button>

          {result && (
            <div className="mt-8 border-t pt-6">
              <h3 className="text-xl font-semibold text-gray-800 mb-4">Analysis Results</h3>
              
              {/* Show document information */}
              <div className="mb-6">
                <h4 className="font-medium text-gray-800 mb-2">Document Information:</h4>
                <div className="bg-gray-50 p-3 rounded-lg">
                  {inputMethod === "text" ? (
                    <div className="flex items-center">
                      <i className="fas fa-file-alt text-gray-500 mr-2"></i>
                      <span>Text input - {text.length} characters</span>
                    </div>
                  ) : (
                    <div>
                      <div className="flex items-center mb-2">
                        <i className="fas fa-file-upload text-gray-500 mr-2"></i>
                        <span>{files.length} document{files.length !== 1 ? 's' : ''} analyzed:</span>
                      </div>
                      <ul className="ml-6 text-sm text-gray-600">
                        {files.map((file, index) => (
                          <li key={index} className="mb-1">
                            • {file.name} ({(file.size / 1024).toFixed(1)} KB)
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              </div>
              
              <div className="mb-6">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-gray-700 font-medium">
                    {analysisType === "ai-detection" ? "AI Content Probability" : "Plagiarism Probability"}:
                  </span>
                  <span className="font-bold text-lg">
                    {(result.probability * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-4">
                  <div
                    className={`h-4 rounded-full ${
                      result.probability < 0.3
                        ? "bg-green-500"
                        : result.probability < 0.7
                        ? "bg-yellow-500"
                        : "bg-red-500"
                    }`}
                    style={{ width: `${result.probability * 100}%` }}
                  ></div>
                </div>
                
                <div className="mt-2 text-center">
                  <span className={`font-medium text-sm px-2 py-1 rounded ${
                    result.probability < 0.3
                      ? "bg-green-100 text-green-800"
                      : result.probability < 0.7
                      ? "bg-yellow-100 text-yellow-800"
                      : "bg-red-100 text-red-800"
                  }`}>
                    {result.probability < 0.3
                      ? analysisType === "ai-detection" ? "Likely Human-Written" : "Minimal Plagiarism"
                      : result.probability < 0.7
                      ? analysisType === "ai-detection" ? "Possibly AI-Generated" : "Moderate Plagiarism"
                      : analysisType === "ai-detection" ? "Likely AI-Generated" : "Significant Plagiarism"
                    }
                  </span>
                </div>
              </div>

              {/* Show API source if using custom API */}
              {result.source && (
                <div className="mt-4 bg-purple-50 border-l-4 border-purple-400 p-3">
                  <div className="flex">
                    <div className="flex-shrink-0">
                      <svg className="h-5 w-5 text-purple-400" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                        <path fillRule="evenodd" d="M12.316 3.051a1 1 0 01.633 1.265l-4 12a1 1 0 11-1.898-.632l4-12a1 1 0 011.265-.633zM5.707 6.293a1 1 0 010 1.414L3.414 10l2.293 2.293a1 1 0 11-1.414 1.414l-3-3a1 1 0 010-1.414l3-3a1 1 0 011.414 0zm8.586 0a1 1 0 011.414 0l3 3a1 1 0 010 1.414l-3 3a1 1 0 11-1.414-1.414L16.586 10l-2.293-2.293a1 1 0 010-1.414z" clipRule="evenodd" />
                      </svg>
                    </div>
                    <div className="ml-3">
                      <p className="text-sm text-purple-700">
                        Analysis performed using: <span className="font-medium">{result.source}</span>
                      </p>
                    </div>
                  </div>
                </div>
              )}

              {/* Feature analysis section */}
              {result.detailed_results && (
                <div className="space-y-4 mt-6">
                  <h4 className="font-medium text-gray-800 border-b pb-2">Feature Analysis:</h4>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {Object.entries(result.detailed_results).map(([key, value]) => (
                      <div key={key} className="bg-gray-50 p-3 rounded-lg">
                        <div className="flex justify-between items-center">
                          <span className="text-gray-700 capitalize">{key.replace(/_/g, " ")}</span>
                          <span className="font-medium bg-white px-2 py-1 rounded border">
                            {typeof value === 'number' ? value.toFixed(3) : value.toString()}
                          </span>
                        </div>
                        <div className="mt-2 h-2 bg-gray-200 rounded-full overflow-hidden">
                          {typeof value === 'number' && (
                            <div 
                              className={`h-full rounded-full ${
                                key.includes('human') || key.includes('original') 
                                  ? "bg-green-500" : "bg-blue-500"
                              }`}
                              style={{ width: `${Math.max(Math.min(value * 100, 100), 0)}%` }}
                            ></div>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* File comparison results for plagiarism */}
              {analysisType === "plagiarism" && result.comparisons && (
                <div className="space-y-4 mt-6">
                  <h4 className="font-medium text-gray-800 border-b pb-2">File Comparisons:</h4>
                  {Object.entries(result.comparisons).map(([fileKey, compData], idx) => (
                    <div key={idx} className="bg-gray-50 p-3 rounded-lg">
                      <div className="font-medium mb-2">{fileKey}</div>
                      <div className="text-sm text-gray-600">
                        {typeof compData === 'object' ? (
                          <ul className="space-y-2">
                            {Object.entries(compData).map(([k, v], i) => (
                              <li key={i} className="flex justify-between">
                                <span>{k}:</span>
                                <span className="font-medium">{typeof v === 'number' ? v.toFixed(3) : v.toString()}</span>
                              </li>
                            ))}
                          </ul>
                        ) : (
                          <p>{compData}</p>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              )}

              <div className="mt-6 flex flex-wrap gap-3">
                <button
                  onClick={() => handleVisualize()}
                  className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
                >
                  Visualize Results
                </button>
                <button
                  onClick={() => handleExport()}
                  className="px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700"
                >
                  Export Results
                </button>
                <button
                  onClick={() => setActiveTab("visualization")}
                  className="px-4 py-2 bg-purple-600 text-white rounded-md hover:bg-purple-700"
                >
                  Advanced Analysis
                </button>
              </div>

              <div className="mt-8 border-t pt-4">
                <h4 className="font-medium text-gray-800 mb-2">Provide Feedback</h4>
                <div className="flex items-center mb-4">
                  <span className="mr-3 text-sm text-gray-600">Rating:</span>
                  {[1, 2, 3, 4, 5].map((rating) => (
                    <button
                      key={rating}
                      onClick={() => setFeedback({...feedback, rating})}
                      className={`mx-1 w-8 h-8 rounded-full ${
                        feedback.rating >= rating ? "bg-yellow-400" : "bg-gray-200"
                      } flex items-center justify-center text-gray-800`}
                    >
                      {rating}
                    </button>
                  ))}
                </div>
                <textarea
                  value={feedback.comments}
                  onChange={(e) => setFeedback({...feedback, comments: e.target.value})}
                  placeholder="Comments on result accuracy (optional)"
                  className="w-full p-3 border border-gray-300 rounded-lg h-24 text-sm focus:outline-none focus:ring-2 focus:ring-purple-500 mb-3"
                ></textarea>
                <button
                  onClick={handleSubmitFeedback}
                  disabled={feedback.rating === 0}
                  className="px-4 py-2 bg-gray-700 text-white rounded-md hover:bg-gray-800 disabled:bg-gray-400 disabled:cursor-not-allowed"
                >
                  Submit Feedback
                </button>
              </div>
            </div>
          )}
        </div>
      )}

      {activeTab === "history" && (
        <div className="bg-white rounded-xl shadow-md p-6">
          <div className="flex justify-between items-center mb-6">
            <h3 className="text-xl font-semibold text-gray-800">Analysis History</h3>
            {isAuthenticated ? (
              <span className="bg-green-100 text-green-800 text-xs px-3 py-1 rounded-full">
                Showing personal history for {user?.username}
              </span>
            ) : (
              <span className="bg-yellow-100 text-yellow-800 text-xs px-3 py-1 rounded-full">
                Showing session history only - login to access your saved history
              </span>
            )}
          </div>
          
          <div className="flex justify-end mb-4">
            <button 
              onClick={fetchHistory}
              className="px-3 py-1 bg-gray-200 hover:bg-gray-300 rounded text-gray-700 text-sm flex items-center"
            >
              <span className="mr-1">↻</span> Refresh History
            </button>
          </div>
          
          {history.length === 0 ? (
            <p className="text-gray-500 text-center py-8">No analysis history yet.</p>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-left">
                <thead>
                  <tr className="border-b">
                    <th className="py-3 px-4 font-medium">Date</th>
                    <th className="py-3 px-4 font-medium">Document</th>
                    <th className="py-3 px-4 font-medium">Type</th>
                    <th className="py-3 px-4 font-medium">Result</th>
                    <th className="py-3 px-4 font-medium">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {history.map((item) => (
                    <tr key={item.id} className="border-b hover:bg-gray-50">
                      <td className="py-3 px-4 whitespace-nowrap">{new Date(item.timestamp).toLocaleString()}</td>
                      <td className="py-3 px-4">
                        <div className="flex items-center">
                          <i className={`fas fa-${item.filename ? 'file-alt' : 'align-left'} text-gray-500 mr-2`}></i>
                          <span className="text-sm font-medium truncate max-w-[150px]" title={item.filename || "Text input"}>
                            {item.filename || "Text input"}
                          </span>
                        </div>
                      </td>
                      <td className="py-3 px-4">
                        <span className={`px-2 py-1 text-xs rounded-full ${
                          item.analysis_type === "ai-detection" 
                            ? "bg-blue-100 text-blue-800" 
                            : "bg-purple-100 text-purple-800"
                        }`}>
                          {item.analysis_type === "ai-detection" ? "AI Detection" : "Plagiarism"}
                        </span>
                      </td>
                      <td className="py-3 px-4">
                        <div className="flex items-center">
                          <div
                            className={`w-3 h-3 rounded-full mr-2 ${
                              item.probability < 0.3
                                ? "bg-green-500"
                                : item.probability < 0.7
                                ? "bg-yellow-500"
                                : "bg-red-500"
                            }`}
                          ></div>
                          {(item.probability * 100).toFixed(1)}%
                        </div>
                      </td>
                      <td className="py-3 px-4">
                        <div className="flex space-x-2">
                          <button
                            onClick={() => {
                              setResult(item);
                              setActiveTab("analysis");
                            }}
                            className="px-2 py-1 text-sm bg-blue-100 text-blue-700 rounded hover:bg-blue-200"
                          >
                            View
                          </button>
                          <button
                            onClick={() => {
                              handleVisualize(item.id);
                              setActiveTab("visualization");
                            }}
                            className="px-2 py-1 text-sm bg-purple-100 text-purple-700 rounded hover:bg-purple-200"
                          >
                            Visualize
                          </button>
                          <button
                            onClick={() => handleExport(item.id)}
                            className="px-2 py-1 text-sm bg-green-100 text-green-700 rounded hover:bg-green-200"
                          >
                            Export
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}

      {activeTab === "visualization" && (
        <div className="bg-white rounded-xl shadow-md p-6">
          <h3 className="text-xl font-semibold text-gray-800 mb-6">Visualizations</h3>
          
          <div className="mb-6">
            <label className="block text-gray-700 font-medium mb-2">Visualization Type</label>
            <div className="flex flex-wrap gap-2">
              <button
                onClick={() => setVisualizationType("features")}
                className={`px-3 py-2 rounded ${
                  visualizationType === "features"
                    ? "bg-purple-700 text-white"
                    : "bg-gray-100 text-gray-800"
                }`}
              >
                Feature Importance
              </button>
              <button
                onClick={() => setVisualizationType("distribution")}
                className={`px-3 py-2 rounded ${
                  visualizationType === "distribution"
                    ? "bg-purple-700 text-white"
                    : "bg-gray-100 text-gray-800"
                }`}
              >
                Score Distribution
              </button>
              <button
                onClick={() => setVisualizationType("comparison")}
                className={`px-3 py-2 rounded ${
                  visualizationType === "comparison"
                    ? "bg-purple-700 text-white"
                    : "bg-gray-100 text-gray-800"
                }`}
              >
                Comparison
              </button>
              <button
                onClick={() => setVisualizationType("time-series")}
                className={`px-3 py-2 rounded ${
                  visualizationType === "time-series"
                    ? "bg-purple-700 text-white"
                    : "bg-gray-100 text-gray-800"
                }`}
              >
                Time Series
              </button>
            </div>
          </div>

          {result ? (
            <div className="mb-4">
              <button
                onClick={() => handleVisualize()}
                disabled={visualizationLoading}
                className={`px-4 py-2 ${
                  visualizationLoading 
                    ? "bg-purple-500 cursor-not-allowed" 
                    : "bg-purple-700 hover:bg-purple-800"
                } text-white rounded-md flex items-center justify-center`}
              >
                {visualizationLoading && (
                  <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                )}
                {visualizationLoading ? "Generating..." : "Generate Visualization"}
              </button>
            </div>
          ) : (
            <p className="text-gray-500 mb-4">Please run an analysis or select from history first.</p>
          )}

          {/* Show error message if there was an error generating visualization */}
          {visualizationError && (
            <div className="mt-4 bg-red-50 border-l-4 border-red-400 p-4">
              <div className="flex">
                <div className="flex-shrink-0">
                  <svg className="h-5 w-5 text-red-400" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                  </svg>
                </div>
                <div className="ml-3">
                  <p className="text-sm text-red-700">{visualizationError}</p>
                </div>
              </div>
            </div>
          )}

          {visualizationLoading && (
            <div className="mt-6 flex justify-center items-center p-12 border rounded-lg">
              <div className="flex flex-col items-center">
                <svg className="animate-spin h-10 w-10 text-purple-500" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                <p className="mt-4 text-gray-600">Generating visualization...</p>
              </div>
            </div>
          )}

          {visualizationData && !visualizationLoading && !visualizationError && (
            <div className="mt-6 border rounded-lg p-4">
              {visualizationType === "features" && (
                <div>
                  <h4 className="text-lg font-medium mb-4">Feature Importance Visualization</h4>
                  <div className="aspect-w-16 aspect-h-9 bg-gray-100 flex items-center justify-center">
                    <img 
                      src={`data:image/png;base64,${visualizationData.image}`} 
                      alt="Feature Importance" 
                      className="max-w-full max-h-full"
                    />
                  </div>
                </div>
              )}
              
              {visualizationType === "distribution" && (
                <div>
                  <h4 className="text-lg font-medium mb-4">Score Distribution Visualization</h4>
                  <div className="aspect-w-16 aspect-h-9 bg-gray-100 flex items-center justify-center">
                    <img 
                      src={`data:image/png;base64,${visualizationData.image}`} 
                      alt="Score Distribution" 
                      className="max-w-full max-h-full"
                    />
                  </div>
                </div>
              )}
              
              {visualizationType === "comparison" && (
                <div>
                  <h4 className="text-lg font-medium mb-4">Comparison Visualization</h4>
                  <div className="aspect-w-16 aspect-h-9 bg-gray-100 flex items-center justify-center">
                    <img 
                      src={`data:image/png;base64,${visualizationData.image}`} 
                      alt="Comparison" 
                      className="max-w-full max-h-full"
                    />
                  </div>
                </div>
              )}
              
              {visualizationType === "time-series" && (
                <div>
                  <h4 className="text-lg font-medium mb-4">Time Series Visualization</h4>
                  <div className="aspect-w-16 aspect-h-9 bg-gray-100 flex items-center justify-center">
                    <img 
                      src={`data:image/png;base64,${visualizationData.image}`} 
                      alt="Time Series" 
                      className="max-w-full max-h-full"
                    />
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {activeTab === "settings" && (
        <div className="bg-white rounded-xl shadow-md p-6">
          <div className="flex justify-between items-center mb-6">
            <h3 className="text-xl font-semibold text-gray-800">Analysis Settings</h3>
            {isAuthenticated ? (
              <span className="bg-green-100 text-green-800 text-sm px-3 py-1 rounded-full">
                Settings will be saved to your account
              </span>
            ) : (
              <span className="bg-yellow-100 text-yellow-800 text-sm px-3 py-1 rounded-full">
                Settings are temporary for this session only
              </span>
            )}
          </div>
          
          {!isAuthenticated && (
            <div className="mb-6 p-4 bg-blue-50 border border-blue-200 rounded-lg">
              <div className="flex items-start">
                <div className="flex-shrink-0 text-blue-500">
                  <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg">
                    <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd"></path>
                  </svg>
                </div>
                <div className="ml-3">
                  <p className="text-sm text-blue-700">
                    <span className="font-medium">Your settings won't be saved across sessions.</span>
                    <span className="block mt-1">
                      <Link to="/login" className="font-medium underline hover:text-blue-800">
                        Login or create an account
                      </Link>{" "}
                      to save your settings permanently and access them from any device.
                    </span>
                  </p>
                </div>
              </div>
            </div>
          )}
          
          <div className="mb-6">
            <label className="block text-gray-700 font-medium mb-2">Default Analysis Type</label>
            <div className="flex flex-wrap gap-2">
              <button
                onClick={() => setAnalysisType("ai-detection")}
                className={`px-3 py-2 rounded ${
                  analysisType === "ai-detection"
                    ? "bg-purple-700 text-white"
                    : "bg-gray-100 text-gray-800"
                }`}
              >
                AI Detection
              </button>
              <button
                onClick={() => setAnalysisType("plagiarism")}
                className={`px-3 py-2 rounded ${
                  analysisType === "plagiarism"
                    ? "bg-purple-700 text-white"
                    : "bg-gray-100 text-gray-800"
                }`}
              >
                Plagiarism Check
              </button>
            </div>
          </div>
          
          <div className="mb-6">
            <label className="block text-gray-700 font-medium mb-2">Default Sensitivity</label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.1"
              value={sensitivity}
              onChange={(e) => setSensitivity(parseFloat(e.target.value))}
              className="w-full"
            />
            <div className="flex justify-between text-xs text-gray-500">
              <span>More Permissive ({(sensitivity * 100).toFixed(0)}%)</span>
              <span>More Strict</span>
            </div>
          </div>

          <div className="mb-6">
            <label className="block text-gray-700 font-medium mb-2">Default Visualization Type</label>
            <div className="flex flex-wrap gap-2">
              <button
                onClick={() => setVisualizationType("features")}
                className={`px-3 py-2 rounded ${
                  visualizationType === "features"
                    ? "bg-purple-700 text-white"
                    : "bg-gray-100 text-gray-800"
                }`}
              >
                Feature Importance
              </button>
              <button
                onClick={() => setVisualizationType("distribution")}
                className={`px-3 py-2 rounded ${
                  visualizationType === "distribution"
                    ? "bg-purple-700 text-white"
                    : "bg-gray-100 text-gray-800"
                }`}
              >
                Distribution
              </button>
            </div>
          </div>

          <div className="mb-6">
            <label className="block text-gray-700 font-medium mb-2">Default Export Format</label>
            <div className="flex flex-wrap gap-2">
              <button
                onClick={() => setExportFormat("pdf")}
                className={`px-3 py-2 rounded ${
                  exportFormat === "pdf"
                    ? "bg-purple-700 text-white"
                    : "bg-gray-100 text-gray-800"
                }`}
              >
                PDF
              </button>
              <button
                onClick={() => setExportFormat("csv")}
                className={`px-3 py-2 rounded ${
                  exportFormat === "csv"
                    ? "bg-purple-700 text-white"
                    : "bg-gray-100 text-gray-800"
                }`}
              >
                CSV
              </button>
              <button
                onClick={() => setExportFormat("json")}
                className={`px-3 py-2 rounded ${
                  exportFormat === "json"
                    ? "bg-purple-700 text-white"
                    : "bg-gray-100 text-gray-800"
                }`}
              >
                JSON
              </button>
            </div>
          </div>

          <button
            onClick={handleSaveSettings}
            className="px-4 py-2 bg-purple-700 text-white rounded-md hover:bg-purple-800"
          >
            Save Settings
          </button>
        </div>
      )}

      {activeTab === "api-connections" && (
        <div className="bg-white rounded-xl shadow-md p-6">
          <div className="flex justify-between items-center mb-6">
            <h3 className="text-xl font-semibold text-gray-800">API Connections</h3>
            <button 
              onClick={() => setShowApiModal(true)}
              className="px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 flex items-center"
            >
              <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-1" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M10 5a1 1 0 011 1v3h3a1 1 0 110 2h-3v3a1 1 0 11-2 0v-3H6a1 1 0 110-2h3V6a1 1 0 011-1z" clipRule="evenodd" />
              </svg>
              Add New Connection
            </button>
          </div>
          
          <div className="bg-blue-50 border-l-4 border-blue-400 p-4 mb-6">
            <div className="flex">
              <div className="flex-shrink-0">
                <svg className="h-5 w-5 text-blue-400" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2h-1V9a1 1 0 00-1-1z" clipRule="evenodd" />
                </svg>
              </div>
              <div className="ml-3">
                <p className="text-sm text-blue-700">
                  <span className="font-medium">Connect your own APIs and plugins</span>
                  <span className="block mt-1">
                    Add custom API endpoints for plagiarism checking or AI content detection. 
                    You can use any compatible service that matches our request/response format.
                  </span>
                </p>
              </div>
            </div>
          </div>

          {apiConnections.length === 0 ? (
            <div className="text-center py-12 bg-gray-50 rounded-lg border-2 border-dashed border-gray-300">
              <svg className="mx-auto h-12 w-12 text-gray-400" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M14 10l-2 1m0 0l-2-1m2 1v2.5M20 7l-2 1m2-1l-2-1m2 1v2.5M14 4l-2-1-2 1M4 7l2-1M4 7l2 1M4 7v2.5M12 21l-2-1m2 1l2-1m-2 1v-2.5M6 18l-2-1v-2.5M18 18l2-1v-2.5" />
              </svg>
              <h3 className="mt-2 text-sm font-medium text-gray-900">No API connections</h3>
              <p className="mt-1 text-sm text-gray-500">Get started by adding a new connection.</p>
              <div className="mt-6">
                <button
                  type="button"
                  onClick={() => setShowApiModal(true)}
                  className="inline-flex items-center px-4 py-2 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-purple-600 hover:bg-purple-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-purple-500"
                >
                  <svg className="-ml-1 mr-2 h-5 w-5" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M10 5a1 1 0 011 1v3h3a1 1 0 110 2h-3v3a1 1 0 11-2 0v-3H6a1 1 0 110-2h3V6a1 1 0 011-1z" clipRule="evenodd" />
                  </svg>
                  Add API Connection
                </button>
              </div>
            </div>
          ) : (
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
              {apiConnections.map(api => (
                <div key={api.id} className="border rounded-lg p-4 hover:shadow-md transition-shadow">
                  <div className="flex items-start justify-between">
                    <div>
                      <h4 className="font-medium text-gray-900">{api.name}</h4>
                      <p className="text-sm text-gray-500 mt-1">{api.endpoint}</p>
                      <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium mt-2 ${
                        api.type === "plagiarism" ? "bg-blue-100 text-blue-800" : "bg-green-100 text-green-800"
                      }`}>
                        {api.type === "plagiarism" ? "Plagiarism API" : "AI Detection API"}
                      </span>
                    </div>
                    <div className="flex">
                      <button 
                        onClick={() => handleUseApi(api)}
                        className={`text-sm px-2 py-1 rounded mr-1 ${
                          selectedApi && selectedApi.id === api.id
                            ? "bg-green-600 text-white"
                            : "bg-gray-100 text-gray-800 hover:bg-gray-200"
                        }`}
                      >
                        {selectedApi && selectedApi.id === api.id ? "Active" : "Use"}
                      </button>
                      <button 
                        onClick={() => handleDeleteApiConnection(api.id)}
                        className="text-sm px-2 py-1 bg-red-100 text-red-800 rounded hover:bg-red-200"
                      >
                        Delete
                      </button>
                    </div>
                  </div>
                  <div className="mt-4">
                    <button
                      className="text-sm text-purple-600 hover:text-purple-800 hover:underline"
                      onClick={() => {
                        // This would open a modal for API testing
                        alert(`Test functionality for ${api.name} would go here`);
                      }}
                    >
                      Test Connection
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
          
          {/* API Connection Modal */}
          {showApiModal && (
            <div className="fixed inset-0 bg-gray-500 bg-opacity-75 flex items-center justify-center z-50">
              <div className="bg-white rounded-lg p-6 max-w-md w-full">
                <div className="flex justify-between items-center mb-4">
                  <h3 className="text-xl font-medium text-gray-900">Add API Connection</h3>
                  <button onClick={() => setShowApiModal(false)} className="text-gray-500 hover:text-gray-700">
                    <svg className="h-6 w-6" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  </button>
                </div>
                
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700">Connection Name</label>
                    <input
                      type="text"
                      value={newApiConnection.name}
                      onChange={(e) => setNewApiConnection({...newApiConnection, name: e.target.value})}
                      className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm p-2"
                      placeholder="My Custom API"
                    />
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-700">Endpoint URL</label>
                    <input
                      type="text"
                      value={newApiConnection.endpoint}
                      onChange={(e) => setNewApiConnection({...newApiConnection, endpoint: e.target.value})}
                      className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm p-2"
                      placeholder="https://api.example.com/check"
                    />
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-700">API Key (optional)</label>
                    <input
                      type="password"
                      value={newApiConnection.apiKey}
                      onChange={(e) => setNewApiConnection({...newApiConnection, apiKey: e.target.value})}
                      className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm p-2"
                      placeholder="Your API key"
                    />
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-700">API Type</label>
                    <div className="mt-1 flex">
                      <button
                        onClick={() => setNewApiConnection({...newApiConnection, type: "plagiarism"})}
                        className={`flex-1 py-2 ${
                          newApiConnection.type === "plagiarism" 
                            ? "bg-purple-700 text-white" 
                            : "bg-gray-100 text-gray-800"
                        } rounded-l`}
                      >
                        Plagiarism
                      </button>
                      <button
                        onClick={() => setNewApiConnection({...newApiConnection, type: "ai-detection"})}
                        className={`flex-1 py-2 ${
                          newApiConnection.type === "ai-detection" 
                            ? "bg-purple-700 text-white" 
                            : "bg-gray-100 text-gray-800"
                        } rounded-r`}
                      >
                        AI Detection
                      </button>
                    </div>
                  </div>
                </div>
                
                <div className="mt-6 flex justify-end space-x-3">
                  <button
                    onClick={() => setShowApiModal(false)}
                    className="px-4 py-2 border border-gray-300 rounded-md text-gray-700 hover:bg-gray-50"
                  >
                    Cancel
                  </button>
                  <button
                    onClick={handleSaveApiConnection}
                    className="px-4 py-2 bg-purple-700 text-white rounded-md hover:bg-purple-800"
                  >
                    Save Connection
                  </button>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
