import axios from "axios";

const api = axios.create({
  baseURL: "http://localhost:5000", // Flask backend
  withCredentials: true, // Important for session cookies
});

// Authentication API functions
export const registerUser = async (username, password) => {
  try {
    const response = await api.post("/user/register", { 
      username, 
      password 
    });
    return response.data;
  } catch (error) {
    console.error("Error registering user:", error);
    throw error;
  }
};

export const loginUser = async (username, password) => {
  try {
    const response = await api.post("/user/login", { 
      username, 
      password 
    });
    return response.data;
  } catch (error) {
    console.error("Error logging in:", error);
    throw error;
  }
};

export const logoutUser = async () => {
  try {
    const response = await api.post("/user/logout");
    return response.data;
  } catch (error) {
    console.error("Error logging out:", error);
    throw error;
  }
};

export const getUserSettings = async () => {
  try {
    const response = await api.get("/user/settings");
    return response.data;
  } catch (error) {
    console.error("Error getting user settings:", error);
    throw error;
  }
};

export const checkAuth = async () => {
  try {
    const response = await api.get("/user/check-auth");
    return response.data;
  } catch (error) {
    console.error("Error checking authentication:", error);
    throw error;
  }
};

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
    const response = await api.post("/enhanced/ai-detection/text", { 
      text, 
      sensitivity 
    });
    return response.data;
  } catch (error) {
    console.error("Error in enhanced AI detection:", error);
    throw error;
  }
};

export const enhancedAIDetectionFile = async (file, sensitivity = 0.5) => {
  try {
    // Validate file input
    if (!file) {
      throw new Error("No file provided for analysis");
    }
    
    const formData = new FormData();
    
    // Make sure we're using the correct parameter name expected by the backend
    formData.append('file', file);
    
    // Add sensitivity parameter as a string
    formData.append('sensitivity', String(sensitivity));
    
    console.log('Sending file:', file.name, 'with size:', file.size);
    
    const response = await api.post("/enhanced/ai-detection", formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    });
    
    return response.data;
  } catch (error) {
    console.error("Error in enhanced AI detection with file:", error);
    
    // Enhanced error handling with more specific error messages
    if (error.response) {
      const status = error.response.status;
      if (status === 400) {
        if (error.response.data && error.response.data.error) {
          throw new Error(`Bad request: ${error.response.data.error}`);
        } else {
          throw new Error("Bad request: Check file format and size. The backend might be expecting a different parameter name.");
        }
      } else if (status === 413) {
        throw new Error("File too large: The uploaded file exceeds the size limit.");
      } else if (status === 415) {
        throw new Error("Unsupported file type: The file format is not supported.");
      } else if (status === 500) {
        throw new Error("Server error processing file: The backend failed to analyze the file.");
      }
    }
    
    throw error;
  }
};

export const enhancedPlagiarismCheck = async (fileInput, sensitivity = 0.5, advancedSettings = {}) => {
  try {
    console.log("Processing input type:", typeof fileInput, fileInput instanceof File ? "File object" : "Not file object");
    
    // Handle if fileInput is a File object (from file upload)
    if (fileInput instanceof File) {
      console.log("Processing file input:", fileInput.name, fileInput.size);
      
      const formData = new FormData();
      formData.append("file", fileInput);
      formData.append("sensitivity", sensitivity.toString());
      
      // Add advanced settings to the form data
      Object.entries(advancedSettings).forEach(([key, value]) => {
        if (value !== null && value !== undefined) {
          formData.append(key, typeof value === 'boolean' ? value.toString() : value);
        }
      });
      
      // For FormData, we don't need to set Content-Type as the browser will set it with the boundary
      console.log("Sending file to /plagiarism/check");
      const response = await api.post("/plagiarism/check", formData);
      return response.data;
    } 
    // Handle if fileInput is a text string
    else if (typeof fileInput === 'string') {
      console.log("Processing text input, length:", fileInput.length);
      
      // Ensure text is not empty
      if (!fileInput.trim()) {
        throw new Error("Text input cannot be empty");
      }
      
      // Create payload with text content
      const payload = { 
        text: fileInput, 
        sensitivity: sensitivity,
        ...advancedSettings
      };
      
      console.log("Sending text to /plagiarism/check/text/enhanced", {
        textLength: fileInput.length,
        sensitivity,
        ...advancedSettings
      });
      
      const response = await api.post("/plagiarism/check/text/enhanced", payload, {
        headers: {
          'Content-Type': 'application/json'
        }
      });
      return response.data;
    } else {
      throw new Error("Invalid input: Please provide either a file or text content");
    }
  } catch (error) {
    console.error("Error in enhanced plagiarism check:", error);
    
    // Enhanced error handling
    if (error.response) {
      console.error("Response error data:", error.response.status, error.response.data);
      
      const status = error.response.status;
      if (status === 400) {
        if (error.response.data && error.response.data.error) {
          throw new Error(`Bad request: ${error.response.data.error}`);
        } else {
          throw new Error("Bad request: The server could not process your text. Please check your input.");
        }
      } else if (status === 413) {
        throw new Error("Text too large: The submitted text exceeds the size limit.");
      } else if (status === 500) {
        throw new Error("Server error: The plagiarism detection service is currently experiencing issues.");
      }
    }
    
    throw error;
  }
};

export const enhancedPlagiarismCheckFiles = async (files, sensitivity = 0.5, advancedSettings = {}) => {
  try {
    const formData = new FormData();
    
    // Add all files to the form data
    if (Array.isArray(files)) {
      files.forEach(file => {
        formData.append('files[]', file);
      });
    } else {
      // Handle single file case
      formData.append('file', files);
    }
    
    // Add sensitivity parameter
    formData.append('sensitivity', String(sensitivity));
    
    // Add advanced settings
    Object.entries(advancedSettings).forEach(([key, value]) => {
      // For boolean values, convert to string
      if (typeof value === 'boolean') {
        formData.append(key, value ? 'true' : 'false');
      } else {
        formData.append(key, value);
      }
    });
    
    // Use the correct plagiarism endpoint for file comparison
    console.log("Comparing files:", Array.from(files).map(f => f.name));
    
    // Add a parameter to ensure identical files get 100% match
    formData.append("ensureExactMatches", "true");
    
    const response = await api.post("/plagiarism/compare", formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    });
    
    // Log the response data for debugging
    console.log("Comparison results:", 
      response.data.comparisons ? 
      response.data.comparisons.map(c => `${c.file1} vs ${c.file2}: ${c.similarity_score}%`) : 
      "No comparisons found");
    
    return response.data;
  } catch (error) {
    console.error("Error in enhanced plagiarism check with files:", error);
    
    // Enhanced error handling
    if (error.response) {
      const status = error.response.status;
      if (status === 400) {
        if (error.response.data && error.response.data.error) {
          throw new Error(`Bad request: ${error.response.data.error}`);
        } else {
          throw new Error("Bad request: Check file format and size. The server could not process your files.");
        }
      } else if (status === 413) {
        throw new Error("Files too large: One or more uploaded files exceed the size limit.");
      } else if (status === 415) {
        throw new Error("Unsupported file type: One or more files are not in a supported format.");
      } else if (status === 500) {
        throw new Error("Server error: The plagiarism detection service is currently experiencing issues.");
      }
    }
    
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
    // Make sure we use the right visualization type parameter name that matches the backend
    // The backend expects 'type' as the query parameter
    const response = await api.get(`/enhanced/visualize/${analysisId}`, {
      params: { type: visualizationType }
    });
    
    // Check for empty or invalid response data
    if (!response.data || !response.data.image) {
      throw new Error("Visualization data is incomplete or invalid");
    }
    
    return response.data;
  } catch (error) {
    console.error("Error getting visualization:", error);
    
    // Check for specific error types and provide helpful messages
    if (error.response) {
      const status = error.response.status;
      if (status === 404) {
        throw new Error("Analysis not found or visualization endpoint not available. Check if the analysis ID is valid and the backend server is running correctly.");
      } else if (status === 400) {
        throw new Error("Invalid request to visualization endpoint. Check the visualization type parameter.");
      } else if (status === 500) {
        throw new Error("Server error generating visualization. The visualization module may not be properly configured.");
      }
    }
    
    // If it's a network error, provide a specific message
    if (error.message.includes('Network Error')) {
      throw new Error("Network error connecting to backend. Please check your internet connection or if the backend server is running.");
    }
    
    // For all other errors
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
    // Try to update user-specific settings first (requires authentication)
    const response = await api.post("/user/settings", settings);
    return response.data;
  } catch (error) {
    // If that fails (likely due to no authentication), use enhanced endpoint
    try {
      const response = await api.post("/enhanced/settings", settings);
      return response.data;
    } catch (innerError) {
      console.error("Error updating settings:", innerError);
      throw innerError;
    }
  }
};

// Get settings, either from user account or general enhanced settings
export const getSettings = async () => {
  try {
    // Try to get user-specific settings first (requires authentication)
    const response = await api.get("/user/settings");
    return response.data;
  } catch (error) {
    // If that fails (likely due to no authentication), use enhanced endpoint
    try {
      const response = await api.get("/enhanced/settings");
      return response.data;
    } catch (innerError) {
      console.error("Error getting settings:", innerError);
      throw innerError;
    }
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

// Custom API Connection Management
export const testApiConnection = async (apiConnection) => {
  try {
    // Test if the API is responsive
    const formData = new FormData();
    formData.append("test", "true");
    formData.append("sensitivity", 0.5);
    
    // Add API key to headers if provided
    const headers = {};
    if (apiConnection.apiKey) {
      headers["Authorization"] = `Bearer ${apiConnection.apiKey}`;
    }
    
    // Try a very small request to test connectivity
    const response = await fetch(apiConnection.endpoint, {
      method: "POST",
      headers,
      body: formData,
    });
    
    if (!response.ok) {
      throw new Error(`API responded with status: ${response.status}`);
    }
    
    const data = await response.json();
    return { success: true, message: "Connection successful", data };
  } catch (error) {
    console.error("Error testing API connection:", error);
    return { 
      success: false, 
      message: `Connection failed: ${error.message}` 
    };
  }
};

// Helper function to generate a custom API request for custom endpoints
export const callCustomApi = async (apiConnection, data, files = null) => {
  try {
    const formData = new FormData();
    
    // Add text data if provided
    if (data.text) {
      formData.append("text", data.text);
    }
    
    // Add files if provided
    if (files && files.length > 0) {
      for (let i = 0; i < files.length; i++) {
        formData.append("files", files[i]);
      }
    }
    
    // Add sensitivity setting
    formData.append("sensitivity", data.sensitivity || 0.5);
    
    // Add API key to headers if provided
    const headers = {};
    if (apiConnection.apiKey) {
      headers["Authorization"] = `Bearer ${apiConnection.apiKey}`;
    }
    
    // Make the API request
    const response = await fetch(apiConnection.endpoint, {
      method: "POST",
      headers,
      body: formData,
    });
    
    if (!response.ok) {
      throw new Error(`API responded with status: ${response.status}`);
    }
    
    const responseData = await response.json();
    
    // Add a source indicator to the response
    responseData.source = `External API: ${apiConnection.name}`;
    
    return responseData;
  } catch (error) {
    console.error("Error calling custom API:", error);
    throw error;
  }
};

export default api;
