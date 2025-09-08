import axios from "axios";

const api = axios.create({
  baseURL: "http://localhost:5000", // Flask backend
});

// Original API functions
export const checkPlagiarism = async (text) => {
  try {
    const response = await api.post("/plagiarism/check", { text });
    return response.data;
  } catch (error) {
    console.error("Error checking plagiarism:", error);
    throw error;
  }
};

export const checkDocument = async (formData) => {
  try {
    const response = await api.post("/document/check", formData, {
      headers: { "Content-Type": "multipart/form-data" },
    });
    return response.data;
  } catch (error) {
    console.error("Error checking document:", error);
    throw error;
  }
};

// Enhanced API functions
export const enhancedAIDetection = async (text, sensitivity = 0.5) => {
  try {
    const response = await api.post("/enhanced/ai-detection", { 
      text, 
      sensitivity 
    });
    return response.data;
  } catch (error) {
    console.error("Error in enhanced AI detection:", error);
    throw error;
  }
};

export const enhancedPlagiarismCheck = async (text, sensitivity = 0.5) => {
  try {
    const response = await api.post("/enhanced/plagiarism", { 
      text, 
      sensitivity 
    });
    return response.data;
  } catch (error) {
    console.error("Error in enhanced plagiarism check:", error);
    throw error;
  }
};

export const getAnalysisHistory = async () => {
  try {
    const response = await api.get("/enhanced/history");
    return response.data;
  } catch (error) {
    console.error("Error fetching analysis history:", error);
    throw error;
  }
};

export const getVisualization = async (analysisId, visualizationType) => {
  try {
    const response = await api.get(`/enhanced/visualize/${analysisId}`, {
      params: { type: visualizationType }
    });
    return response.data;
  } catch (error) {
    console.error("Error getting visualization:", error);
    throw error;
  }
};

export const exportResults = async (analysisId, format) => {
  try {
    const response = await api.get(`/enhanced/export/${analysisId}`, {
      params: { format },
      responseType: 'blob' // Important for file downloads
    });
    return response.data;
  } catch (error) {
    console.error("Error exporting results:", error);
    throw error;
  }
};

export const updateSettings = async (settings) => {
  try {
    const response = await api.post("/enhanced/settings", settings);
    return response.data;
  } catch (error) {
    console.error("Error updating settings:", error);
    throw error;
  }
};

export const provideFeedback = async (analysisId, feedbackData) => {
  try {
    const response = await api.post(`/enhanced/feedback/${analysisId}`, feedbackData);
    return response.data;
  } catch (error) {
    console.error("Error providing feedback:", error);
    throw error;
  }
};

export default api;
