import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import { AuthProvider } from './contexts/AuthContext';
import { AdminProvider } from './contexts/AdminContext';
import Navbar from './Components/NavBar';
import Home from './pages/Home';
import About from './pages/About';
import CheckReports from './pages/CheckReports';
import Analysis from './pages/Analysis';
import Authentication from './pages/Authentication';
import Profile from './pages/Profile';
import PatientLogin from './pages/Patient/PatientLogin';
import PatientSignup from './pages/Patient/PatientSignup';
import PatientDashboard from './pages/Patient/Dashboard';
import PatientHealthDashboard from './pages/Patient/HealthDashboard';
import PatientUploadReport from './pages/Patient/UploadReport';
import PatientViewReport from './pages/Patient/ViewReport';
import PatientChat from './pages/Patient/Chat';
import DoctorLogin from './pages/Doctor/DoctorLogin';
import DoctorSignup from './pages/Doctor/DoctorSignup';
import DoctorDashboard from './pages/Doctor/Dashboard';
import AdminLogin from './pages/Admin/AdminLogin';
import AdminDashboard from './pages/Admin/Dashboard';
import AdminStatistics from './pages/Admin/Statistics';
import AdminUsers from './pages/Admin/Users';
import AdminVerifyDoctor from './pages/Admin/VerifyDoctor';
import './App.css';

function App() {
  return (
    <AuthProvider>
      <AdminProvider>
        <Router>
          <div className="min-h-screen bg-slate-50 flex flex-col">
            <Navbar />
            <div className="flex-1">
              <Routes>
                <Route path="/" element={<Home />} />
                <Route path="/login" element={<Authentication />} />
                <Route path="/signup" element={<Authentication />} />
                <Route path="/check-reports" element={<CheckReports />} />
                <Route path="/analysis" element={<Analysis />} />
                <Route path="/profile" element={<Profile />} />
                <Route path="/about" element={<About />} />
                <Route path="/patient/login" element={<PatientLogin />} />
                <Route path="/patient/signup" element={<PatientSignup />} />
                <Route path="/patient/dashboard" element={<PatientDashboard />} />
                <Route path="/patient/health-dashboard" element={<PatientHealthDashboard />} />
                <Route path="/patient/upload-report" element={<PatientUploadReport />} />
                <Route path="/patient/view-report" element={<PatientViewReport />} />
                <Route path="/patient/chat" element={<PatientChat />} />
                <Route path="/doctor/login" element={<DoctorLogin />} />
                <Route path="/doctor/signup" element={<DoctorSignup />} />
                <Route path="/doctor/dashboard" element={<DoctorDashboard />} />
                <Route path="/admin/login" element={<AdminLogin />} />
                <Route path="/admin/dashboard" element={<AdminDashboard />} />
                <Route path="/admin/statistics" element={<AdminStatistics />} />
                <Route path="/admin/users" element={<AdminUsers />} />
                <Route path="/admin/verify-doctor" element={<AdminVerifyDoctor />} />
                <Route path="*" element={
                  <div className="flex flex-col items-center justify-center min-h-[60vh]">
                    <h1 className="text-4xl font-bold text-slate-800">404 - Page Not Found</h1>
                    <p className="text-slate-600 mt-2">The page you are looking for doesn't exist.</p>
                    <Link to="/" className="mt-4 text-green-600 font-bold hover:underline">Go Home</Link>
                  </div>
                } />
              </Routes>
            </div>
          </div>
        </Router>
      </AdminProvider>
    </AuthProvider>
  );
}

export default App;
