import React, { createContext, useContext, useState, useEffect } from 'react';

const AuthContext = createContext();

export function useAuth() {
    return useContext(AuthContext);
}

export function AuthProvider({ children }) {
    const [user, setUser] = useState(null);
    const [loading, setLoading] = useState(true);

    // Load user from localStorage on mount
    useEffect(() => {
        try {
            const savedUser = localStorage.getItem('mediscan_user');
            if (savedUser) {
                setUser(JSON.parse(savedUser));
            }
        } catch (error) {
            console.error('Error loading user from localStorage:', error);
        } finally {
            setLoading(false);
        }
    }, []);

    const login = (userData) => {
        setUser(userData);
        try {
            localStorage.setItem('mediscan_user', JSON.stringify(userData));
        } catch (error) {
            console.error('Error saving user to localStorage:', error);
        }
    };

    const logout = () => {
        setUser(null);
        try {
            localStorage.removeItem('mediscan_user');
        } catch (error) {
            console.error('Error removing user from localStorage:', error);
        }
    };

    const updateUser = (newData) => {
        setUser(prev => {
            const updated = { ...prev, ...newData };
            try {
                localStorage.setItem('mediscan_user', JSON.stringify(updated));
            } catch (error) {
                console.error('Error updating user in localStorage:', error);
            }
            return updated;
        });
    };

    const value = {
        user,
        login,
        logout,
        updateUser,
        loading
    };

    // Don't render children until we've checked localStorage
    if (loading) {
        return null;
    }

    return (
        <AuthContext.Provider value={value}>
            {children}
        </AuthContext.Provider>
    );
}
