import React, { useState, useEffect } from "react";
import { 
  enhancedAIDetection, 
  enhancedPlagiarismCheck, 
  getAnalysisHistory, 
  getVisualization,
  exportResults,
  updateSettings,
  provideFeedback
} from "../utils/api";

export default function EnhancedAnalysis() {
  const [text, setText] = useState("");
  const [analysisType, setAnalysisType] = useState("ai-detection"); // "ai-detection" or "plagiarism"
  const [sensitivity, setSensitivity] = useState(0.5);
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [visualizationType, setVisualizationType] = useState("features");
  const [visualizationData, setVisualizationData] = useState(null);
  const [exportFormat, setExportFormat] = useState("pdf");
  const [feedback, setFeedback] = useState({ rating: 0, comments: "" });
  const [activeTab, setActiveTab] = useState("analysis");

  // Load history on component mount
  useEffect(() => {
    fetchHistory();
  }, []);

  const fetchHistory = async () => {
    try {
      const historyData = await getAnalysisHistory();
      setHistory(historyData.history || []);
    } catch (error) {
      console.error("Failed to fetch history:", error);
    }
  };

  const handleAnalyze = async () => {
    if (!text.trim()) return;

    setLoading(true);
    setResult(null);
    setVisualizationData(null);

    try {
      let response;
      if (analysisType === "ai-detection") {
        response = await enhancedAIDetection(text, sensitivity);
      } else {
        response = await enhancedPlagiarismCheck(text, sensitivity);
      }
      
      setResult(response);
      // Refresh history after analysis
      fetchHistory();
      setLoading(false);
    } catch (error) {
      console.error(`Error during ${analysisType}:`, error);
      setLoading(false);
    }
  };

  const handleVisualize = async (analysisId = result?.id) => {
    if (!analysisId) return;

    setLoading(true);
    try {
      const data = await getVisualization(analysisId, visualizationType);
      setVisualizationData(data);
      setLoading(false);
    } catch (error) {
      console.error("Error generating visualization:", error);
      setLoading(false);
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

  const handleSaveSettings = async () => {
    try {
      await updateSettings({ defaultSensitivity: sensitivity });
      alert("Settings saved successfully!");
    } catch (error) {
      console.error("Error saving settings:", error);
    }
  };

  return (
    <div className="max-w-6xl mx-auto px-4 py-8">
      <div className="mb-8 text-center">
        <h2 className="text-3xl font-bold text-purple-800 mb-2">Enhanced Content Analysis</h2>
        <p className="text-gray-600">
          Advanced AI detection and plagiarism checking with visualization tools and customizable settings
        </p>
      </div>

      <div className="mb-6 flex justify-center space-x-2">
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

            <textarea
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder={`Enter text to analyze for ${analysisType === "ai-detection" ? "AI-generated content" : "plagiarism"}`}
              className="w-full p-4 border border-gray-300 rounded-lg h-64 focus:outline-none focus:ring-2 focus:ring-purple-500"
            ></textarea>
          </div>

          <button
            onClick={handleAnalyze}
            disabled={loading || !text.trim()}
            className="w-full bg-purple-700 text-white py-3 rounded-lg font-medium hover:bg-purple-800 transition-colors disabled:bg-purple-300 disabled:cursor-not-allowed"
          >
            {loading ? "Analyzing..." : `Analyze with Enhanced ${analysisType === "ai-detection" ? "AI Detection" : "Plagiarism Check"}`}
          </button>

          {result && (
            <div className="mt-8 border-t pt-6">
              <h3 className="text-xl font-semibold text-gray-800 mb-4">Analysis Results</h3>
              
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
              </div>

              {result.detailed_results && (
                <div className="space-y-4">
                  <h4 className="font-medium text-gray-800">Feature Analysis:</h4>
                  {Object.entries(result.detailed_results).map(([key, value]) => (
                    <div key={key} className="flex justify-between">
                      <span className="text-gray-600">{key.replace(/_/g, " ")}:</span>
                      <span className="font-medium">{typeof value === 'number' ? value.toFixed(3) : value}</span>
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
          <h3 className="text-xl font-semibold text-gray-800 mb-6">Analysis History</h3>
          
          {history.length === 0 ? (
            <p className="text-gray-500 text-center py-8">No analysis history yet.</p>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-left">
                <thead>
                  <tr className="border-b">
                    <th className="py-3 px-4 font-medium">Date</th>
                    <th className="py-3 px-4 font-medium">Type</th>
                    <th className="py-3 px-4 font-medium">Result</th>
                    <th className="py-3 px-4 font-medium">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {history.map((item) => (
                    <tr key={item.id} className="border-b hover:bg-gray-50">
                      <td className="py-3 px-4">{new Date(item.timestamp).toLocaleString()}</td>
                      <td className="py-3 px-4">{item.analysis_type}</td>
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
                className="px-4 py-2 bg-purple-700 text-white rounded-md hover:bg-purple-800"
              >
                {loading ? "Generating..." : "Generate Visualization"}
              </button>
            </div>
          ) : (
            <p className="text-gray-500 mb-4">Please run an analysis or select from history first.</p>
          )}

          {visualizationData && (
            <div className="mt-6 border rounded-lg p-4">
              {visualizationType === "features" && (
                <div>
                  <h4 className="text-lg font-medium mb-4">Feature Importance Visualization</h4>
                  <div className="aspect-w-16 aspect-h-9 bg-gray-100 flex items-center justify-center">
                    {/* This would be replaced with actual visualization component */}
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
          <h3 className="text-xl font-semibold text-gray-800 mb-6">Analysis Settings</h3>
          
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
            <label className="block text-gray-700 font-medium mb-2">Export Format</label>
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
    </div>
  );
}
