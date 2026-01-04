import React from 'react';
import { BrowserRouter, Routes, Route, Outlet } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import Home from './pages/Home';
import PatientLogin from './pages/patient/Login';
import AdminLogin from './pages/admin/Login';
import DoctorLogin from './pages/doctor/Login';



import Navbar from './components/layout/Navbar';
import PatientDashboard from './pages/patient/Dashboard';
import UploadReport from './pages/patient/UploadReport';
import Chat from './pages/patient/Chat';
import HealthDashboard from './pages/patient/HealthDashboard';
import ViewReport from './pages/patient/ViewReport';
import DashboardLayout from './components/layout/DashboardLayout';
import DoctorDashboard from './pages/doctor/Dashboard';
import UploadLicense from './pages/doctor/UploadLicense';
import PatientList from './pages/doctor/Patients';
import AdminDashboard from './pages/admin/Dashboard';
import VerifyDoctors from './pages/admin/VerifyDoctors';
import ManageUsers from './pages/admin/Users';
import SystemStatistics from './pages/admin/Statistics';




const queryClient = new QueryClient();

const PublicLayout = () => (
  <div className="min-h-screen flex flex-col">
    <Navbar />
    <main className="flex-1">
      <Outlet />
    </main>
    <footer className="py-6 text-center text-sm text-muted-foreground border-t">
      © 2024 MediScan System. All rights reserved.
    </footer>
  </div>
);

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Routes>
          <Route element={<PublicLayout />}>
            <Route path="/" element={<Home />} />
            <Route path="/patient/login" element={<PatientLogin />} />
            <Route path="/doctor/login" element={<DoctorLogin />} />
            <Route path="/admin/login" element={<AdminLogin />} />
          </Route>

          {/* Patient Routes */}
          <Route path="/patient" element={<DashboardLayout />}>
            <Route index element={<PatientDashboard />} />
            <Route path="upload-report" element={<UploadReport />} />
            <Route path="health-dashboard" element={<HealthDashboard />} />
            <Route path="chat" element={<Chat />} />
            <Route path="report/:id" element={<ViewReport />} />
          </Route>


          {/* Doctor Routes */}
          <Route path="/doctor" element={<DashboardLayout />}>
            <Route index element={<DoctorDashboard />} />
            <Route path="upload-license" element={<UploadLicense />} />
            <Route path="patients" element={<PatientList />} />
          </Route>

          {/* Admin Routes */}
          <Route path="/admin" element={<DashboardLayout />}>
            <Route index element={<AdminDashboard />} />
            <Route path="verify-doctors" element={<VerifyDoctors />} />
            <Route path="users" element={<ManageUsers />} />
            <Route path="statistics" element={<SystemStatistics />} />
          </Route>

        </Routes>

      </BrowserRouter>
    </QueryClientProvider>
  );
}

export default App;
