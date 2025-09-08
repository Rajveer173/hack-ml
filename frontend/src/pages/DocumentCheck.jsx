import { useState } from "react";
import FileUpload from "../components/FileUpload";
import api from "../utils/api";

export default function DocumentCheck() {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleUpload = async (file) => {
    setLoading(true);
    setError(null);
    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await api.post("/document/verify", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setResult(res.data);
      setLoading(false);
    } catch (err) {
      console.error(err);
      setError("An error occurred while analyzing the document. Please try again.");
      setLoading(false);
    }
  };

  return (
    <div className="max-w-4xl mx-auto">
      <div className="mb-8 text-center">
        <h2 className="text-3xl font-bold text-purple-800 mb-4">Document Analysis</h2>
        <p className="text-gray-600">Upload your document for comprehensive verification and analysis.</p>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
        <div className="bg-white rounded-lg p-5 shadow-md border-t-4 border-purple-500 flex flex-col items-center">
          <svg 
            className="w-10 h-10 text-purple-600 mb-3" 
            fill="none" 
            stroke="currentColor" 
            viewBox="0 0 24 24" 
            xmlns="http://www.w3.org/2000/svg"
          >
            <path 
              strokeLinecap="round" 
              strokeLinejoin="round" 
              strokeWidth={2} 
              d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" 
            />
          </svg>
          <h3 className="font-semibold text-lg mb-2">Authenticity</h3>
          <p className="text-center text-gray-600 text-sm">Verify document originality and authenticity</p>
        </div>
        <div className="bg-white rounded-lg p-5 shadow-md border-t-4 border-amber-500 flex flex-col items-center">
          <svg 
            className="w-10 h-10 text-amber-600 mb-3" 
            fill="none" 
            stroke="currentColor" 
            viewBox="0 0 24 24" 
            xmlns="http://www.w3.org/2000/svg"
          >
            <path 
              strokeLinecap="round" 
              strokeLinejoin="round" 
              strokeWidth={2} 
              d="M7 21a4 4 0 01-4-4V5a2 2 0 012-2h4a2 2 0 012 2v12a4 4 0 01-4 4zm0 0h12a2 2 0 002-2v-4a2 2 0 00-2-2h-2.343M11 7.343l1.657-1.657a2 2 0 012.828 0l2.829 2.829a2 2 0 010 2.828l-8.486 8.485M7 17h.01" 
            />
          </svg>
          <h3 className="font-semibold text-lg mb-2">Content Analysis</h3>
          <p className="text-center text-gray-600 text-sm">Evaluate structure, style, and coherence</p>
        </div>
        <div className="bg-white rounded-lg p-5 shadow-md border-t-4 border-emerald-500 flex flex-col items-center">
          <svg 
            className="w-10 h-10 text-emerald-600 mb-3" 
            fill="none" 
            stroke="currentColor" 
            viewBox="0 0 24 24" 
            xmlns="http://www.w3.org/2000/svg"
          >
            <path 
              strokeLinecap="round" 
              strokeLinejoin="round" 
              strokeWidth={2} 
              d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" 
            />
          </svg>
          <h3 className="font-semibold text-lg mb-2">Format Validation</h3>
          <p className="text-center text-gray-600 text-sm">Check for proper formatting and consistency</p>
        </div>
      </div>
      
      <div className="bg-white rounded-xl shadow-md p-6 mb-8">
        <h3 className="text-xl font-semibold text-gray-800 mb-4">Upload Document</h3>
        <FileUpload onUpload={handleUpload} />
        
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
          <h3 className="text-xl font-semibold text-gray-800 mb-4">Analysis Results</h3>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
            <div className="p-4 rounded-lg border">
              <div className="flex items-center mb-4">
                <div className={`p-2 rounded-md ${
                  result.authenticity === 'Authentic' ? 'bg-emerald-100' : 'bg-red-100'
                }`}>
                  <svg 
                    className={`w-6 h-6 ${
                      result.authenticity === 'Authentic' ? 'text-emerald-600' : 'text-red-600'
                    }`} 
                    fill="none" 
                    stroke="currentColor" 
                    viewBox="0 0 24 24" 
                    xmlns="http://www.w3.org/2000/svg"
                  >
                    {result.authenticity === 'Authentic' ? (
                      <path 
                        strokeLinecap="round" 
                        strokeLinejoin="round" 
                        strokeWidth={2} 
                        d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" 
                      />
                    ) : (
                      <path 
                        strokeLinecap="round" 
                        strokeLinejoin="round" 
                        strokeWidth={2} 
                        d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" 
                      />
                    )}
                  </svg>
                </div>
                <span className="ml-3 font-medium text-lg">Authenticity Check</span>
              </div>
              <p className={`font-semibold ${
                result.authenticity === 'Authentic' ? 'text-emerald-600' : 'text-red-600'
              }`}>
                {result.authenticity}
              </p>
            </div>
            
            <div className="p-4 rounded-lg border">
              <div className="flex items-center mb-4">
                <div className={`p-2 rounded-md ${
                  result.status === 'Valid' ? 'bg-emerald-100' : 
                  result.status === 'Warning' ? 'bg-amber-100' : 'bg-red-100'
                }`}>
                  <svg 
                    className={`w-6 h-6 ${
                      result.status === 'Valid' ? 'text-emerald-600' : 
                      result.status === 'Warning' ? 'text-amber-600' : 'text-red-600'
                    }`} 
                    fill="none" 
                    stroke="currentColor" 
                    viewBox="0 0 24 24" 
                    xmlns="http://www.w3.org/2000/svg"
                  >
                    {result.status === 'Valid' ? (
                      <path 
                        strokeLinecap="round" 
                        strokeLinejoin="round" 
                        strokeWidth={2} 
                        d="M5 13l4 4L19 7" 
                      />
                    ) : result.status === 'Warning' ? (
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
                        d="M6 18L18 6M6 6l12 12" 
                      />
                    )}
                  </svg>
                </div>
                <span className="ml-3 font-medium text-lg">Status</span>
              </div>
              <p className={`font-semibold ${
                result.status === 'Valid' ? 'text-emerald-600' : 
                result.status === 'Warning' ? 'text-amber-600' : 'text-red-600'
              }`}>
                {result.status}
              </p>
            </div>
          </div>
          
          <div className="p-4 bg-gray-50 rounded-lg border">
            <h4 className="font-semibold mb-2">Analysis Summary</h4>
            <p className="text-gray-700">
              Document verification complete. {
                result.authenticity === 'Authentic' && result.status === 'Valid' 
                  ? 'All checks have passed successfully. This document appears to be authentic and properly formatted.' 
                  : 'Some issues were detected with this document. Please review the details above for more information.'
              }
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
