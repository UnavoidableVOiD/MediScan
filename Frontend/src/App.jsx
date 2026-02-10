import React, { useEffect } from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import { useDispatch, useSelector } from 'react-redux';
import { checkAuth } from './store/slices/authSlice';
import MainLayout from './components/layout/MainLayout';
import LandingPage from './pages/LandingPage';
import AuthPage from './pages/AuthPage';
import OtpVerification from './pages/OtpVerification';
import PatientDashboard from './pages/PatientDashboard';
import CheckReports from './pages/CheckReports';
import PatientProfile from './pages/PatientProfile';
import ViewReportResult from './pages/ViewReportResult';

import About from './pages/About';
import Contact from './pages/Contact';
import { ProtectedRoute, PublicRoute, AdminRoute } from './components/auth/RouteGuards';

// Admin Pages
import AdminLayout from './layouts/AdminLayout';
import AdminLogin from './pages/admin/AdminLogin';
import AdminDoctors from './pages/admin/AdminDoctors';
import AdminPatients from './pages/admin/AdminPatients';
import CreateAdmin from './pages/admin/CreateAdmin';
import DoctorProfile from './pages/doctor/DoctorProfile';
import VerifyDoctor from './pages/doctor/VerifyDoctor';
import DoctorDashboard from './pages/doctor/DoctorDashboard';
import DoctorAppointments from './pages/doctor/DoctorAppointments';
import DoctorPatients from './pages/doctor/DoctorPatients';
import PatientDetailView from './pages/doctor/PatientDetailView';

// Placeholder pages
const Placeholder = ({ title }) => (
  <div className="max-w-7xl mx-auto px-6 py-20 text-center">
    <h1 className="text-4xl font-bold text-medic-dark mb-4">{title}</h1>
    <p className="text-gray-600">This is a placeholder page for the {title} route.</p>
  </div>
);

function App() {
  const dispatch = useDispatch();
  const { initialized } = useSelector(state => state.auth);

  useEffect(() => {
    dispatch(checkAuth());
  }, [dispatch]);

  if (!initialized) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-neutral-background">
        <div className="w-12 h-12 border-4 border-medic-dark/20 border-t-medic-dark rounded-full animate-spin" />
      </div>
    );
  }

  return (
    <Routes>
      <Route path="/" element={<MainLayout />}>
        <Route index element={
          <PublicRoute>
            <LandingPage />
          </PublicRoute>
        } />

        <Route path="login" element={
          <PublicRoute>
            <AuthPage />
          </PublicRoute>
        } />
        <Route path="signup" element={
          <PublicRoute>
            <AuthPage />
          </PublicRoute>
        } />
        <Route path="verify-otp" element={<OtpVerification />} />

        {/* Protected Routes */}
        <Route path="dashboard" element={
          <ProtectedRoute role="patient">
            <PatientDashboard />
          </ProtectedRoute>
        } />

        <Route path="check-reports" element={
          <ProtectedRoute role="patient">
            <CheckReports />
          </ProtectedRoute>
        } />
        <Route path="profile" element={
          <ProtectedRoute>
            <PatientProfile />
          </ProtectedRoute>
        } />

        <Route path="doctor-profile" element={
          <ProtectedRoute role="doctor">
            <DoctorProfile />
          </ProtectedRoute>
        } />
        <Route path="reports/:id/result" element={
          <ProtectedRoute role="patient">
            <ViewReportResult />
          </ProtectedRoute>
        } />


        <Route path="about" element={<About />} />
        <Route path="contact" element={<Contact />} />
        <Route path="services" element={<Placeholder title="Our Services" />} />
        <Route path="demo" element={<Placeholder title="Live Demo" />} />
        <Route path="privacy" element={<Placeholder title="Privacy Policy" />} />

        {/* Helper redirection */}
        {/* Doctor Routes */}
        <Route path="doctor-dashboard" element={
          <ProtectedRoute role="doctor">
            <DoctorDashboard />
          </ProtectedRoute>
        } />
        <Route path="appointments" element={
          <ProtectedRoute role="doctor">
            <DoctorAppointments />
          </ProtectedRoute>
        } />
        <Route path="patients" element={
          <ProtectedRoute role="doctor">
            <DoctorPatients />
          </ProtectedRoute>
        } />
        <Route path="patient/:id" element={
          <ProtectedRoute role="doctor">
            <PatientDetailView />
          </ProtectedRoute>
        } />
        <Route path="doctor-profile" element={
          <ProtectedRoute role="doctor">
            <DoctorProfile />
          </ProtectedRoute>
        } />
        <Route path="verify-doctor" element={
          <ProtectedRoute role="doctor">
            <VerifyDoctor />
          </ProtectedRoute>
        } />

        <Route path="*" element={<Placeholder title="404 Not Found" />} />
      </Route>

      {/* Admin Routes - Isolated from MainLayout */}
      <Route path="/admin/login" element={
        <PublicRoute>
          <AdminLogin />
        </PublicRoute>
      } />

      <Route path="/admin" element={
        <AdminRoute>
          <AdminLayout />
        </AdminRoute>
      }>
        <Route index element={<Navigate to="/admin/dashboard" replace />} />
        <Route path="dashboard" element={<Placeholder title="Admin Dashboard" />} />
        <Route path="doctors" element={<AdminDoctors />} />
        <Route path="patients" element={<AdminPatients />} />
        <Route path="create-admin" element={<CreateAdmin />} />
      </Route>
    </Routes>
  );
}

export default App;
