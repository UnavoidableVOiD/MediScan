import React from 'react';
import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom';
import Navbar from './Components/Navbar.jsx';
import Chatbot from './Components/Chatbot.jsx';
import Home from './pages/Home.jsx';
import Login from './pages/Login.jsx';
import Signup from './pages/Signup.jsx';
import CheckReports from './pages/CheckReports.jsx';
import Membership from './pages/Membership.jsx';
import PatientDashboard from './pages/PatientDashboard.jsx';
import Analysis from './pages/Analysis.jsx';
import DoctorDashboard from './pages/DoctorDashboard.jsx';
import DoctorReview from './pages/DoctorReview.jsx';
import AdminDashboard from './pages/AdminDashboard.jsx';
import Profile from './pages/Profile.jsx';
import { AuthProvider } from './contexts/AuthContext.jsx';
export default function App() {
  return (
    <BrowserRouter>
      <AuthProvider>
        <div className="flex flex-col min-h-screen">
          <Navbar />
          <main className="flex-1 bg-slate-50">
            <Routes>
              <Route path="/" element={<Home />} />
              <Route path="/login" element={<Login />} />
              <Route path="/signup" element={<Signup />} />
              <Route path="/check-reports" element={<CheckReports />} />
              <Route path="/dashboard" element={<PatientDashboard />} />
              <Route path="/analysis" element={<Analysis />} />
              <Route path="/doctor-dashboard" element={<DoctorDashboard />} />
              <Route path="/doctor/review/:id" element={<DoctorReview />} />
              <Route path="/admin-dashboard" element={<AdminDashboard />} />
              <Route path="/profile" element={<Profile />} />
              <Route path="/membership" element={<Membership />} />
              <Route path="*" element={<Navigate to="/" />} />
            </Routes>
          </main>
          <Chatbot />
        </div>
      </AuthProvider>
    </BrowserRouter>
  );
}
