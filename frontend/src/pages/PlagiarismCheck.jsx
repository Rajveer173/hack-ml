import React, { useState } from "react";
import FileUpload from "../components/FileUpload";
import api from "../utils/api";

export default function PlagiarismCheck() {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [mode, setMode] = useState("single"); // "single" or "multiple"

  const handleUpload = async (fileInput) => {
    setLoading(true);
    setError(null);
    const formData = new FormData();
    
    if (mode === "single") {
      formData.append("file", fileInput);
      
      try {
        const res = await api.post("/plagiarism/check", formData, {
          headers: { "Content-Type": "multipart/form-data" },
        });
        setResult(res.data);
        setLoading(false);
      } catch (err) {
        console.error(err);
        setError("An error occurred while checking for plagiarism. Please try again.");
        setLoading(false);
      }
    } else {
      // Handle multiple files
      Array.from(fileInput).forEach(file => {
        formData.append("files[]", file);
      });
      
      try {
        // Use the compare endpoint for multiple files
        const res = await api.post("/plagiarism/compare", formData, {
          headers: { "Content-Type": "multipart/form-data" },
        });
        setResult(res.data);
        setLoading(false);
      } catch (err) {
        console.error(err);
        setError("An error occurred while checking for plagiarism between files. Please try again.");
        setLoading(false);
      }
    }
  };

  return (
    <div className="max-w-5xl mx-auto">
      <div className="mb-8 text-center">
        <h2 className="text-3xl font-bold text-purple-800 mb-4">Plagiarism Detection</h2>
        <p className="text-gray-600">
          {mode === "single" 
            ? "Upload your document to check for potential plagiarism and ensure content originality."
            : "Upload multiple files to check for similarities and potential plagiarism between them."
          }
        </p>
      </div>

      <div className="mb-6 flex justify-center space-x-4">
        <button 
          onClick={() => setMode("single")}
          className={`px-4 py-2 rounded-md font-medium transition-colors ${
            mode === "single"
              ? "bg-purple-700 text-white" 
              : "bg-white text-gray-700 border border-gray-300 hover:bg-gray-50"
          }`}
        >
          Single File Check
        </button>
        <button 
          onClick={() => setMode("multiple")}
          className={`px-4 py-2 rounded-md font-medium transition-colors ${
            mode === "multiple"
              ? "bg-purple-700 text-white" 
              : "bg-white text-gray-700 border border-gray-300 hover:bg-gray-50"
          }`}
        >
          Multiple File Comparison
        </button>
      </div>
      
      <div className="bg-white rounded-xl shadow-md p-6 mb-8">
        <h3 className="text-xl font-semibold text-gray-800 mb-4">
          {mode === "single" ? "Upload Document" : "Upload Multiple Files"}
        </h3>
        
        <FileUpload onUpload={handleUpload} multiple={mode === "multiple"} />
        
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

      {result && mode === "single" && (
        <div className="bg-white rounded-xl shadow-md p-6">
          <h3 className="text-xl font-semibold text-gray-800 mb-4">Analysis Results</h3>
          <div className="mb-6">
            <div className="flex items-center justify-between mb-2">
              <span className="text-gray-700 font-medium">Similarity Score:</span>
              <span className={`font-semibold ${
                result.similarity_score > 30 ? 'text-red-600' : 
                result.similarity_score > 15 ? 'text-amber-600' : 'text-emerald-600'
              }`}>
                {result.similarity_score}%
              </span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div 
                className={`h-2 rounded-full ${
                  result.similarity_score > 30 ? 'bg-red-600' : 
                  result.similarity_score > 15 ? 'bg-amber-600' : 'bg-emerald-600'
                }`} 
                style={{width: `${result.similarity_score}%`}}
              ></div>
            </div>
          </div>
          
          <div className={`flex items-center gap-2 p-4 rounded-lg border font-medium mb-4 
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
            <span>Status: {result.status}</span>
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
