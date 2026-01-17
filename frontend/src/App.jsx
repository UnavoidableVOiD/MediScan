import React, { useEffect } from 'react';
import { Routes, Route, Navigate, useNavigate } from 'react-router-dom';
import { useSelector, useDispatch } from 'react-redux';
import { logout } from './store/slices/authSlice';
import { ToastContainer } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';

import Navbar from './components/Navbar';
import LandingPage from './pages/LandingPage';
import Signup from './pages/Signup';
import Login from './pages/Login';
import OTPVerification from './pages/OTPVerification';
import Dashboard from './pages/Dashboard';
import Profile from './pages/Profile';
import Chatbot from './components/Chatbot';
import './App.css';

// Protected Route Components
const PublicRoute = ({ children }) => {
  const { token } = useSelector((state) => state.auth);
  return token ? <Navigate to="/dashboard" /> : children;
};

const PrivateRoute = ({ children }) => {
  const { token } = useSelector((state) => state.auth);
  return token ? children : <Navigate to="/login" />;
};

function App() {
  const dispatch = useDispatch();
  const navigate = useNavigate();

  useEffect(() => {
    const handleUnauthorized = () => {
      dispatch(logout());
      navigate('/login');
    };

    window.addEventListener('mediscan-unauthorized', handleUnauthorized);
    return () => window.removeEventListener('mediscan-unauthorized', handleUnauthorized);
  }, [dispatch, navigate]);

  return (
    <div className="min-h-screen bg-white">
      <Navbar />
      <main>
        <Routes>
          <Route path="/" element={<PublicRoute><LandingPage /></PublicRoute>} />
          <Route path="/login" element={<PublicRoute><Login /></PublicRoute>} />
          <Route path="/signup" element={<PublicRoute><Signup /></PublicRoute>} />
          <Route path="/otp-verify" element={<PublicRoute><OTPVerification /></PublicRoute>} />
          <Route path="/dashboard" element={<PrivateRoute><Dashboard /></PrivateRoute>} />
          <Route path="/profile" element={<PrivateRoute><Profile /></PrivateRoute>} />
          <Route path="/demo" element={<div className="pt-20 text-center text-gray-900">Demo Page (Coming Soon)</div>} />
        </Routes>
      </main>
      <Chatbot />
      <ToastContainer position="top-right" autoClose={3000} />
    </div>
  );
}

export default App;
