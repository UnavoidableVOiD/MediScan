import { lazy, Suspense } from "react";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { SmoothScroll } from "@/components/motion";
import { PublicLayout } from "@/components/layout/PublicLayout";
import { AppLayout } from "@/components/layout/AppLayout";
import { LogoMark } from "@/components/brand/Logo";

const Landing = lazy(() => import("@/pages/public/Landing"));
const M = {
  Services: lazy(() => import("@/pages/public/Marketing").then((m) => ({ default: m.Services }))),
  About: lazy(() => import("@/pages/public/Marketing").then((m) => ({ default: m.About }))),
  Contact: lazy(() => import("@/pages/public/Marketing").then((m) => ({ default: m.Contact }))),
  Privacy: lazy(() => import("@/pages/public/Marketing").then((m) => ({ default: m.Privacy }))),
  Doctors: lazy(() => import("@/pages/public/Marketing").then((m) => ({ default: m.Doctors }))),
  DoctorProfilePublic: lazy(() =>
    import("@/pages/public/Marketing").then((m) => ({ default: m.DoctorProfilePublic })),
  ),
};
const A = {
  Login: lazy(() => import("@/pages/auth/Auth").then((m) => ({ default: m.Login }))),
  Signup: lazy(() => import("@/pages/auth/Auth").then((m) => ({ default: m.Signup }))),
  VerifyOtp: lazy(() => import("@/pages/auth/Auth").then((m) => ({ default: m.VerifyOtp }))),
  AdminLogin: lazy(() => import("@/pages/auth/Auth").then((m) => ({ default: m.AdminLogin }))),
};
const P = {
  Dashboard: lazy(() => import("@/pages/patient/Dashboard")),
  ReportsList: lazy(() =>
    import("@/pages/patient/Reports").then((m) => ({ default: m.ReportsList })),
  ),
  Upload: lazy(() => import("@/pages/patient/Reports").then((m) => ({ default: m.UploadReport }))),
  Review: lazy(() => import("@/pages/patient/Reports").then((m) => ({ default: m.ReviewReport }))),
  Result: lazy(() => import("@/pages/patient/Reports").then((m) => ({ default: m.ReportResult }))),
  Appointments: lazy(() =>
    import("@/pages/patient/Account").then((m) => ({ default: m.PatientAppointments })),
  ),
  Profile: lazy(() =>
    import("@/pages/patient/Account").then((m) => ({ default: m.PatientProfile })),
  ),
  Book: lazy(() => import("@/pages/patient/Account").then((m) => ({ default: m.BookAppointment }))),
  PaySuccess: lazy(() =>
    import("@/pages/patient/Account").then((m) => ({ default: m.PaymentSuccess })),
  ),
};
const D = {
  Dashboard: lazy(() =>
    import("@/pages/doctor/Doctor").then((m) => ({ default: m.DoctorDashboard })),
  ),
  Appointments: lazy(() =>
    import("@/pages/doctor/Doctor").then((m) => ({ default: m.DoctorAppointments })),
  ),
  Patients: lazy(() =>
    import("@/pages/doctor/Doctor").then((m) => ({ default: m.DoctorPatients })),
  ),
  PatientDetail: lazy(() =>
    import("@/pages/doctor/Doctor").then((m) => ({ default: m.PatientDetail })),
  ),
  Availability: lazy(() =>
    import("@/pages/doctor/Doctor").then((m) => ({ default: m.DoctorAvailability })),
  ),
  Profile: lazy(() => import("@/pages/doctor/Doctor").then((m) => ({ default: m.DoctorProfile }))),
  Verify: lazy(() => import("@/pages/doctor/Doctor").then((m) => ({ default: m.DoctorVerify }))),
};
const Ad = {
  Dashboard: lazy(() => import("@/pages/admin/Admin").then((m) => ({ default: m.AdminDashboard }))),
  Doctors: lazy(() => import("@/pages/admin/Admin").then((m) => ({ default: m.AdminDoctors }))),
  Patients: lazy(() => import("@/pages/admin/Admin").then((m) => ({ default: m.AdminPatients }))),
  Create: lazy(() => import("@/pages/admin/Admin").then((m) => ({ default: m.CreateAdmin }))),
};
const Dev = {
  Pages: lazy(() => import("@/pages/dev/Dev").then((m) => ({ default: m.PagesIndex }))),
  Style: lazy(() => import("@/pages/dev/Dev").then((m) => ({ default: m.StyleGuide }))),
  NotFound: lazy(() => import("@/pages/dev/Dev").then((m) => ({ default: m.NotFound }))),
};

const qc = new QueryClient();

function Loader() {
  return (
    <div className="grid min-h-screen place-items-center">
      <div className="animate-float">
        <LogoMark size={56} />
      </div>
    </div>
  );
}

export default function App() {
  return (
    <QueryClientProvider client={qc}>
      <BrowserRouter>
        <SmoothScroll>
          <Suspense fallback={<Loader />}>
            <Routes>
              {/* public site */}
              <Route element={<PublicLayout />}>
                <Route path="/" element={<Landing />} />
                <Route path="/services" element={<M.Services />} />
                <Route path="/about" element={<M.About />} />
                <Route path="/contact" element={<M.Contact />} />
                <Route path="/privacy" element={<M.Privacy />} />
                <Route path="/doctors" element={<M.Doctors />} />
                <Route path="/doctors/:id" element={<M.DoctorProfilePublic />} />
                <Route path="/pages" element={<Dev.Pages />} />
                <Route path="/styleguide" element={<Dev.Style />} />
              </Route>

              {/* auth (no chrome) */}
              <Route path="/login" element={<A.Login />} />
              <Route path="/signup" element={<A.Signup />} />
              <Route path="/verify-otp" element={<A.VerifyOtp />} />
              <Route path="/admin/login" element={<A.AdminLogin />} />

              {/* patient portal — no guards during the building phase */}
              <Route element={<AppLayout role="patient" />}>
                <Route path="/dashboard" element={<P.Dashboard />} />
                <Route path="/reports" element={<P.ReportsList />} />
                <Route path="/reports/upload" element={<P.Upload />} />
                <Route path="/reports/:id/review" element={<P.Review />} />
                <Route path="/reports/:id/result" element={<P.Result />} />
                <Route path="/appointments" element={<P.Appointments />} />
                <Route path="/book-appointment/:id" element={<P.Book />} />
                <Route path="/payment/success" element={<P.PaySuccess />} />
                <Route path="/profile" element={<P.Profile />} />
              </Route>

              {/* doctor portal */}
              <Route element={<AppLayout role="doctor" />}>
                <Route path="/doctor/dashboard" element={<D.Dashboard />} />
                <Route path="/doctor/appointments" element={<D.Appointments />} />
                <Route path="/doctor/patients" element={<D.Patients />} />
                <Route path="/doctor/patients/:id" element={<D.PatientDetail />} />
                <Route path="/doctor/availability" element={<D.Availability />} />
                <Route path="/doctor/profile" element={<D.Profile />} />
                <Route path="/doctor/verify" element={<D.Verify />} />
              </Route>

              {/* admin portal */}
              <Route element={<AppLayout role="admin" />}>
                <Route path="/admin" element={<Ad.Dashboard />} />
                <Route path="/admin/dashboard" element={<Ad.Dashboard />} />
                <Route path="/admin/doctors" element={<Ad.Doctors />} />
                <Route path="/admin/patients" element={<Ad.Patients />} />
                <Route path="/admin/create-admin" element={<Ad.Create />} />
              </Route>

              <Route path="*" element={<Dev.NotFound />} />
            </Routes>
          </Suspense>
        </SmoothScroll>
      </BrowserRouter>
    </QueryClientProvider>
  );
}
