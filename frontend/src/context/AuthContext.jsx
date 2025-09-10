import React, { createContext, useState, useEffect, useContext } from 'react';
import { loginUser, logoutUser, getUserSettings, checkAuth } from '../utils/api';

// Create the context
const AuthContext = createContext();

// Create a provider component
export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [userSettings, setUserSettings] = useState(null);

  // Check if user is already logged in on mount
  useEffect(() => {
    const checkLoggedIn = async () => {
      try {
        // First check authentication status
        const authData = await checkAuth();
        console.log('Auth check result:', authData);
        
        if (authData.authenticated) {
          setUser({ id: authData.user_id, username: authData.username });
          
          // Get user settings since we're authenticated
          try {
            const settings = await getUserSettings();
            setUserSettings(settings);
          } catch (settingsError) {
            console.error('Error loading settings:', settingsError);
          }
        } else {
          // Not logged in - clear any stale state
          setUser(null);
          setUserSettings(null);
        }
      } catch (error) {
        console.error('Error checking authentication:', error);
        setUser(null);
        setUserSettings(null);
      } finally {
        setLoading(false);
      }
    };

    checkLoggedIn();
  }, []);

  // Login function
  const login = async (username, password) => {
    try {
      setLoading(true);
      const response = await loginUser(username, password);
      setUser({ id: response.user_id, username });
      
      // Get user settings after login
      const settings = await getUserSettings();
      setUserSettings(settings);
      
      return { success: true };
    } catch (error) {
      return { 
        success: false, 
        error: error.response?.data?.error || 'Login failed'
      };
    } finally {
      setLoading(false);
    }
  };

  // Logout function
  const logout = async () => {
    try {
      setLoading(true);
      await logoutUser();
      setUser(null);
      setUserSettings(null);
      return { success: true };
    } catch (error) {
      return { 
        success: false, 
        error: error.response?.data?.error || 'Logout failed'
      };
    } finally {
      setLoading(false);
    }
  };

  // Update user settings in context
  const updateUserSettings = (settings) => {
    setUserSettings(settings);
  };

  // Value to be provided to consumers
  const value = {
    user,
    userSettings,
    updateUserSettings,
    loading,
    login,
    logout,
    isAuthenticated: !!user
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
}

// Custom hook for using the auth context
export function useAuth() {
  return useContext(AuthContext);
}
