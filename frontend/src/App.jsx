import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Navbar from "./components/Navbar";
import PlagiarismCheck from "./pages/PlagiarismCheck";
import DocumentCheck from "./pages/DocumentCheck";
import EnhancedAnalysis from "./pages/EnhancedAnalysis";
import Authentication from "./pages/Authentication";
import Login from "./pages/Login";
import Home from "./pages/Home";
import { AuthProvider } from "./context/AuthContext";

export default function App() {
  return (
    <AuthProvider>
      <Router>
        <div className="min-h-screen bg-slate-50">
          <Navbar />
          <div className="container mx-auto px-4 py-8">
            <Routes>
              <Route path="/plagiarism" element={<PlagiarismCheck />} />
              <Route path="/document" element={<DocumentCheck />} />
              <Route path="/enhanced" element={<EnhancedAnalysis />} />
              <Route path="/enhanced-analysis" element={<EnhancedAnalysis />} />
              <Route path="/auth" element={<Authentication />} />
              <Route path="/login" element={<Login />} />
              <Route path="/" element={<Home />} />
            </Routes>
          </div>
          <footer className="bg-purple-900 text-white text-center py-6 mt-12">
            <div className="container mx-auto px-4">
              <p>© {new Date().getFullYear()} ML Analysis Platform</p>
              <p className="text-sm text-purple-300 mt-1">Empowering research with advanced machine learning</p>
            </div>
          </footer>
        </div>
      </Router>
    </AuthProvider>
  );
}
