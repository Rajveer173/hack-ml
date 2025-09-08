import { Link } from "react-router-dom";

export default function Home() {
  return (
    <div className="max-w-5xl mx-auto">
      <section className="text-center mb-12">
        <h1 className="text-4xl md:text-5xl font-bold text-purple-800 mb-6">
          Advanced ML Analysis Platform
        </h1>
        <p className="text-xl text-gray-600 max-w-3xl mx-auto">
          Harness the power of machine learning to analyze documents and detect plagiarism with high accuracy.
        </p>
      </section>

      <div className="mb-10 bg-gradient-to-r from-blue-500 to-purple-700 text-white rounded-xl shadow-lg p-6 overflow-hidden relative">
        <div className="absolute -right-10 -top-10 w-40 h-40 bg-white/10 rounded-full"></div>
        <div className="absolute right-10 bottom-5 w-20 h-20 bg-white/10 rounded-full"></div>
        <div className="relative z-10">
          <div className="flex items-center">
            <div className="p-2 bg-white/20 rounded-lg mr-4">
              <svg 
                className="w-8 h-8 text-white" 
                xmlns="http://www.w3.org/2000/svg" 
                fill="none" 
                viewBox="0 0 24 24" 
                stroke="currentColor"
              >
                <path 
                  strokeLinecap="round" 
                  strokeLinejoin="round" 
                  strokeWidth={2} 
                  d="M13 10V3L4 14h7v7l9-11h-7z" 
                />
              </svg>
            </div>
            <div>
              <span className="bg-amber-400 text-purple-900 text-xs font-bold px-2 py-1 rounded uppercase">New</span>
              <h2 className="text-2xl font-bold mt-1">Enhanced Analysis with ML</h2>
            </div>
          </div>
          <p className="mt-4 mb-6">
            Our new enhanced analysis feature leverages advanced machine learning for superior AI content detection and plagiarism checking with customizable sensitivity, visualizations, and detailed analytics.
          </p>
          <div className="flex space-x-3">
            <Link 
              to="/enhanced" 
              className="px-6 py-2 bg-white text-purple-700 hover:bg-amber-400 hover:text-purple-900 font-medium rounded-md transition-colors"
            >
              Try Enhanced Analysis
            </Link>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-12">
        <div className="bg-white rounded-xl shadow-md p-6 transition-all duration-200 hover:shadow-lg hover:-translate-y-1 border-t-4 border-purple-600">
          <div className="flex items-center mb-4">
            <div className="p-2 bg-purple-100 rounded-lg mr-4">
              <svg 
                className="w-8 h-8 text-purple-600" 
                xmlns="http://www.w3.org/2000/svg" 
                fill="none" 
                viewBox="0 0 24 24" 
                stroke="currentColor"
              >
                <path 
                  strokeLinecap="round" 
                  strokeLinejoin="round" 
                  strokeWidth={2} 
                  d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" 
                />
              </svg>
            </div>
            <h2 className="text-2xl font-bold text-gray-800">Document Analysis</h2>
          </div>
          <p className="text-gray-600 mb-4">
            Upload your documents to receive comprehensive analysis including sentiment, key topics, and content summary.
          </p>
          <Link 
            to="/document" 
            className="inline-flex items-center px-4 py-2 bg-purple-700 hover:bg-purple-800 text-white rounded-md font-medium transition-colors"
          >
            Analyze Documents
            <svg 
              className="ml-2 w-4 h-4" 
              fill="none" 
              stroke="currentColor" 
              viewBox="0 0 24 24" 
              xmlns="http://www.w3.org/2000/svg"
            >
              <path 
                strokeLinecap="round" 
                strokeLinejoin="round" 
                strokeWidth={2} 
                d="M14 5l7 7m0 0l-7 7m7-7H3" 
              />
            </svg>
          </Link>
        </div>

        <div className="bg-white rounded-xl shadow-md p-6 transition-all duration-200 hover:shadow-lg hover:-translate-y-1 border-t-4 border-emerald-600">
          <div className="flex items-center mb-4">
            <div className="p-2 bg-emerald-100 rounded-lg mr-4">
              <svg 
                className="w-8 h-8 text-emerald-600" 
                xmlns="http://www.w3.org/2000/svg" 
                fill="none" 
                viewBox="0 0 24 24" 
                stroke="currentColor"
              >
                <path 
                  strokeLinecap="round" 
                  strokeLinejoin="round" 
                  strokeWidth={2} 
                  d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" 
                />
              </svg>
            </div>
            <h2 className="text-2xl font-bold text-gray-800">Plagiarism Detection</h2>
          </div>
          <p className="text-gray-600 mb-4">
            Check your content against a vast database to identify potential plagiarism and ensure academic integrity.
          </p>
          <Link 
            to="/plagiarism" 
            className="inline-flex items-center px-4 py-2 bg-emerald-600 hover:bg-emerald-700 text-white rounded-md font-medium transition-colors"
          >
            Check Plagiarism
            <svg 
              className="ml-2 w-4 h-4" 
              fill="none" 
              stroke="currentColor" 
              viewBox="0 0 24 24" 
              xmlns="http://www.w3.org/2000/svg"
            >
              <path 
                strokeLinecap="round" 
                strokeLinejoin="round" 
                strokeWidth={2} 
                d="M14 5l7 7m0 0l-7 7m7-7H3" 
              />
            </svg>
          </Link>
        </div>
      </div>

      <section className="bg-gradient-to-r from-purple-800 to-purple-900 rounded-xl p-8 text-white mb-12">
        <div className="flex flex-col md:flex-row items-center">
          <div className="md:w-2/3 mb-6 md:mb-0">
            <h2 className="text-3xl font-bold mb-4">Powered by Advanced ML Algorithms</h2>
            <p className="text-purple-100">
              Our platform leverages state-of-the-art machine learning models to provide you with accurate, reliable, and insightful analysis of your documents.
            </p>
          </div>
          <div className="md:w-1/3 md:pl-6">
            <svg 
              className="w-full h-auto text-white opacity-80" 
              xmlns="http://www.w3.org/2000/svg" 
              fill="none" 
              viewBox="0 0 24 24" 
              stroke="currentColor"
            >
              <path 
                strokeLinecap="round" 
                strokeLinejoin="round" 
                strokeWidth={1.5} 
                d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z" 
              />
            </svg>
          </div>
        </div>
      </section>

      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-6">
        <div className="bg-white p-6 rounded-lg shadow-md border-l-4 border-amber-500">
          <h3 className="font-bold text-lg text-gray-800 mb-2">High Accuracy</h3>
          <p className="text-gray-600">Our models are trained on diverse datasets to ensure high precision results.</p>
        </div>
        <div className="bg-white p-6 rounded-lg shadow-md border-l-4 border-purple-500">
          <h3 className="font-bold text-lg text-gray-800 mb-2">Fast Processing</h3>
          <p className="text-gray-600">Get results quickly with our optimized algorithms and efficient processing.</p>
        </div>
        <div className="bg-white p-6 rounded-lg shadow-md border-l-4 border-emerald-500">
          <h3 className="font-bold text-lg text-gray-800 mb-2">Secure Analysis</h3>
          <p className="text-gray-600">Your documents are processed securely and never shared with third parties.</p>
        </div>
      </div>
    </div>
  );
}
