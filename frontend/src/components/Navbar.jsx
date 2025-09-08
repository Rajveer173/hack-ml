    import { Link, useLocation } from "react-router-dom";

export default function Navbar() {
  const location = useLocation();
  
  return (
    <nav className="bg-gradient-to-r from-purple-700 to-purple-900 shadow-lg p-4">
      <div className="container mx-auto flex flex-col sm:flex-row justify-between items-center">
        <Link to="/" className="flex items-center space-x-2 mb-3 sm:mb-0">
          <svg 
            className="w-8 h-8 text-amber-400" 
            xmlns="http://www.w3.org/2000/svg" 
            fill="none" 
            viewBox="0 0 24 24" 
            stroke="currentColor"
          >
            <path 
              strokeLinecap="round" 
              strokeLinejoin="round" 
              strokeWidth={2} 
              d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z" 
            />
          </svg>
          <h1 className="text-2xl font-bold tracking-tight text-white">ML <span className="text-amber-400">Analysis</span></h1>
        </Link>
        <div className="flex space-x-1">
          <Link 
            to="/plagiarism" 
            className={`px-4 py-2 rounded-lg font-medium transition-colors ${
              location.pathname === '/plagiarism' 
                ? 'bg-white text-purple-800' 
                : 'text-white hover:bg-purple-600'
            }`}
          >
            Plagiarism Detection
          </Link>
          <Link 
            to="/document" 
            className={`px-4 py-2 rounded-lg font-medium transition-colors ${
              location.pathname === '/document' 
                ? 'bg-white text-purple-800' 
                : 'text-white hover:bg-purple-600'
            }`}
          >
            Document Analysis
          </Link>
          <Link 
            to="/enhanced" 
            className={`px-4 py-2 rounded-lg font-medium transition-colors ${
              location.pathname === '/enhanced' 
                ? 'bg-white text-purple-800' 
                : 'text-white hover:bg-purple-600'
            }`}
          >
            <span className="flex items-center">
              Enhanced
              <span className="ml-1 text-xs bg-amber-400 text-purple-900 px-1 rounded">NEW</span>
            </span>
          </Link>
        </div>
      </div>
    </nav>
  );
}
