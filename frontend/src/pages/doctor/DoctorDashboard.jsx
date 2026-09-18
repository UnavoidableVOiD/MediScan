import React, { useState, useEffect } from "react";
import { motion } from "framer-motion";
import {
  User,
  Clock,
  CheckCircle2,
  Sparkles,
  Search,
  Calendar,
  ArrowRight,
  BadgeCheck,
  ArrowUpRight,
  ChevronRight,
  Activity,
  Users,
  Loader2,
  Wallet,
  DollarSign,
} from "lucide-react";
import { useSelector, useDispatch } from "react-redux";
import { useNavigate } from "react-router-dom";
import PatientReportSummary from "../../components/doctor/PatientReportSummary";
import {
  fetchDoctorStats,
  fetchMyPatients,
} from "../../store/slices/doctorSlice";
import { fetchAppointments } from "../../store/slices/appointmentSlice";

const StatCard = ({ title, count, icon: Icon, color, trend, trendValue }) => (
  <motion.div
    initial={{ opacity: 0, y: 20 }}
    animate={{ opacity: 1, y: 0 }}
    className="bg-white rounded-3xl p-6 shadow-sm border border-gray-100 flex flex-col justify-between gap-4 hover:shadow-xl hover:shadow-medic-dark/5 transition-all group relative overflow-hidden"
  >
    <div
      className={`absolute top-0 right-0 w-32 h-32 ${color} opacity-5 rounded-full -translate-y-1/2 translate-x-1/3 group-hover:scale-110 transition-transform duration-500`}
    />

    <div className="flex items-start justify-between relative z-10">
      <div
        className={`w-12 h-12 ${color.replace("bg-", "bg-opacity-10 text-")} rounded-2xl flex items-center justify-center shadow-sm`}
      >
        <Icon className={`w-6 h-6 ${color.replace("bg-", "text-")}`} />
      </div>
      {trend && (
        <span
          className={`text-[10px] font-bold px-2.5 py-1 rounded-full flex items-center gap-1 ${trend === "up" ? "bg-green-50 text-green-600" : "bg-gray-50 text-gray-400"}`}
        >
          {trend === "up" && <ArrowUpRight size={10} />}
          {trendValue}
        </span>
      )}
    </div>

    <div className="relative z-10">
      <p className="text-4xl font-black text-gray-900 tracking-tight">
        {count}
      </p>
      <p className="text-sm text-gray-500 font-medium mt-1 uppercase tracking-wide">
        {title}
      </p>
    </div>
  </motion.div>
);

const DoctorDashboard = () => {
  const { user } = useSelector((state) => state.auth);
  const { stats, patients, statsLoading, loading } = useSelector(
    (state) => state.doctor,
  );
  const { appointments, loading: appointmentsLoading } = useSelector(
    (state) => state.appointment,
  );
  const dispatch = useDispatch();
  const navigate = useNavigate();

  const [selectedPatientId, setSelectedPatientId] = useState(null);
  const [selectedPatientName, setSelectedPatientName] = useState("");

  useEffect(() => {
    dispatch(fetchDoctorStats());
    dispatch(fetchMyPatients());
    dispatch(fetchAppointments());
  }, [dispatch]);

  const handlePatientClick = (patientId, patientName) => {
    setSelectedPatientId(patientId);
    setSelectedPatientName(patientName);
  };

  const closePatientSummary = () => {
    setSelectedPatientId(null);
    setSelectedPatientName("");
  };

  const statsData = stats || {
    total_patients: 0,
    ongoing_patients: 0,
    completed_patients: 0,
    new_patients_7_days: 0,
  };

  const statCards = [
    {
      title: "Total Patients",
      count: statsData.total_patients,
      icon: Users,
      color: "bg-blue-500",
      trend: "up",
      trendValue: "All time",
    },
    {
      title: "Gross Earnings",
      count: `Rs. ${statsData.revenue?.total_gross?.toLocaleString() || 0}`,
      icon: DollarSign,
      color: "bg-emerald-500",
      trend: "up",
      trendValue: "Gross",
    },
    {
      title: "Net Revenue",
      count: `Rs. ${statsData.revenue?.total_net?.toLocaleString() || 0}`,
      icon: Wallet,
      color: "bg-medic-dark",
      trend: "up",
      trendValue: "75% Share",
    },
    {
      title: "Active Cases",
      count: statsData.ongoing_patients,
      icon: Activity,
      color: "bg-medic-primary",
      trend: "up",
      trendValue: "Current",
    },
  ];

  if (loading || statsLoading)
    return (
      <div className="min-h-screen flex items-center justify-center bg-neutral-background">
        <Loader2 className="w-10 h-10 text-medic-dark animate-spin" />
      </div>
    );

  return (
    <div className="min-h-[calc(100vh-80px)] bg-neutral-background pt-32 pb-8 px-6 space-y-8 relative font-sans">
      {selectedPatientId && (
        <PatientReportSummary
          patientId={selectedPatientId}
          patientName={selectedPatientName}
          onClose={closePatientSummary}
        />
      )}

      <div className="max-w-7xl mx-auto space-y-10">
        {/* Header Section */}
        <div className="flex flex-col md:flex-row md:items-end justify-between gap-6 pb-4 border-b border-gray-100/50">
          <div className="space-y-2">
            <h1 className="text-4xl font-black text-gray-900 tracking-tight">
              Dashboard
            </h1>
            <div className="flex items-center gap-3 text-gray-500 font-medium">
              <span>Welcome back, Dr. {user?.last_name}</span>
              <span className="w-1 h-1 bg-gray-300 rounded-full" />
              <div className="flex items-center gap-1.5 text-emerald-600 text-xs font-bold bg-emerald-50 px-2.5 py-0.5 rounded-full">
                <BadgeCheck size={12} />
                {user?.doctor_status}
              </div>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <div className="text-right hidden sm:block">
              <p className="text-xs font-bold text-gray-400 uppercase tracking-wider">
                Today's Date
              </p>
              <p className="text-lg font-bold text-gray-900">
                {new Date().toLocaleDateString(undefined, {
                  weekday: "long",
                  month: "long",
                  day: "numeric",
                })}
              </p>
            </div>
          </div>
        </div>

        {/* Stats Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
          {statCards.map((stat, idx) => (
            <StatCard key={idx} {...stat} />
          ))}
        </div>

        {/* Main Content Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Recent Patients Table */}
          <div className="lg:col-span-2 space-y-6">
            <div className="flex items-center justify-between">
              <h2 className="text-2xl font-black text-gray-900 tracking-tight flex items-center gap-2">
                <Users className="w-6 h-6 text-medic-dark" />
                Recent Patients
              </h2>
              <button
                onClick={() => navigate("/patients")}
                className="text-sm font-bold text-medic-dark hover:text-medic-primary flex items-center gap-1 transition-colors px-4 py-2 hover:bg-medic-light/10 rounded-xl"
              >
                View Directory
                <ChevronRight size={16} />
              </button>
            </div>

            <div className="bg-white rounded-[2.5rem] shadow-xl shadow-medic-dark/5 border border-medic-light/20 overflow-hidden">
              <div className="overflow-x-auto">
                <table className="w-full text-left">
                  <thead>
                    <tr className="bg-gray-50/50 border-b border-gray-100">
                      <th className="px-8 py-5 text-gray-400 text-[10px] font-black uppercase tracking-widest">
                        Patient Details
                      </th>
                      <th className="px-8 py-5 text-gray-400 text-[10px] font-black uppercase tracking-widest">
                        Status
                      </th>
                      <th className="px-8 py-5 text-gray-400 text-[10px] font-black uppercase tracking-widest text-right">
                        Last Visit
                      </th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-50">
                    {patients.length > 0 ? (
                      patients.slice(0, 5).map((patient) => (
                        <tr
                          key={patient.id}
                          onClick={() =>
                            handlePatientClick(
                              patient.id,
                              `${patient.first_name} ${patient.last_name}`,
                            )
                          }
                          className="hover:bg-blue-50/30 transition-all group cursor-pointer"
                        >
                          <td className="px-8 py-5">
                            <div className="flex items-center gap-4">
                              <div className="w-10 h-10 rounded-2xl bg-medic-light/20 flex items-center justify-center font-black text-medic-dark uppercase text-sm">
                                {patient.first_name[0]}
                              </div>
                              <div className="flex flex-col">
                                <span className="font-bold text-gray-900 group-hover:text-medic-dark transition-colors">
                                  {patient.first_name} {patient.last_name}
                                </span>
                                <span className="text-xs text-gray-400 font-medium">
                                  {patient.email}
                                </span>
                              </div>
                            </div>
                          </td>
                          <td className="px-8 py-5">
                            <span
                              className={`px-3 py-1 rounded-full text-[10px] font-black border uppercase tracking-wide ${
                                patient.status === "ONGOING"
                                  ? "bg-blue-50 text-blue-600 border-blue-100"
                                  : "bg-emerald-50 text-emerald-600 border-emerald-100"
                              }`}
                            >
                              {patient.status}
                            </span>
                          </td>
                          <td className="px-8 py-5 text-right">
                            <span className="text-sm text-gray-500 font-bold">
                              {patient.last_visit
                                ? new Date(
                                    patient.last_visit,
                                  ).toLocaleDateString()
                                : "No visits"}
                            </span>
                          </td>
                        </tr>
                      ))
                    ) : (
                      <tr>
                        <td
                          colSpan="3"
                          className="px-8 py-16 text-center text-gray-400 font-medium"
                        >
                          No patients found.
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Sidebar / Quick Actions */}
          <div className="space-y-8">
            <div className="space-y-6">
              <h2 className="text-2xl font-black text-gray-900 tracking-tight flex items-center gap-2">
                <Calendar className="w-6 h-6 text-medic-primary" />
                Upcoming
              </h2>

              <div className="space-y-4">
                {appointmentsLoading ? (
                  <div className="flex justify-center p-6 bg-white rounded-3xl">
                    <Loader2 className="animate-spin text-medic-dark" />
                  </div>
                ) : (
                  (() => {
                    const now = new Date();
                    const today = now.toISOString().split("T")[0];
                    const upcoming = appointments
                      .filter((a) => {
                        const isPaid = a.status === "PAID";
                        const isFutureDate = a.appointment_date >= today;
                        return isPaid && isFutureDate;
                      })
                      .sort(
                        (a, b) =>
                          a.appointment_date.localeCompare(
                            b.appointment_date,
                          ) || a.start_time.localeCompare(b.start_time),
                      )
                      .slice(0, 3); // Show only top 3

                    return upcoming.length > 0 ? (
                      upcoming.map((appt) => (
                        <div
                          key={appt.id}
                          className="p-5 bg-white border border-gray-100 rounded-[2rem] shadow-sm hover:shadow-lg hover:shadow-medic-dark/5 hover:-translate-y-1 transition-all group cursor-pointer"
                          onClick={() =>
                            handlePatientClick(
                              appt.patient,
                              appt.patient_first_name,
                            )
                          }
                        >
                          <div className="flex justify-between items-start mb-3">
                            <span className="px-3 py-1 bg-medic-light/10 text-medic-dark rounded-full text-[10px] font-black uppercase tracking-wider">
                              {appt.start_time.slice(0, 5)}
                            </span>
                            <div className="text-[10px] font-bold text-gray-400 uppercase">
                              {new Date(
                                appt.appointment_date,
                              ).toLocaleDateString(undefined, {
                                month: "short",
                                day: "numeric",
                              })}
                            </div>
                          </div>

                          <h4 className="font-bold text-gray-900 text-lg group-hover:text-medic-dark transition-colors truncate">
                            {appt.patient_first_name} {appt.patient_last_name}
                          </h4>
                          <p className="text-xs text-gray-500 font-medium mt-1">
                            General Consultation
                          </p>
                        </div>
                      ))
                    ) : (
                      <div className="p-8 bg-white rounded-[2rem] border border-dashed border-gray-200 text-center">
                        <p className="text-gray-400 font-bold text-sm">
                          No upcoming appointments.
                        </p>
                      </div>
                    );
                  })()
                )}
              </div>
            </div>

            <div className="p-1 rounded-[2rem] bg-gray-50/50 border border-gray-100">
              {[
                {
                  title: "Manage Schedule",
                  icon: Clock,
                  onClick: () => navigate("/doctor-profile"),
                },
                {
                  title: "View All Patients",
                  icon: Users,
                  onClick: () => navigate("/patients"),
                },
                {
                  title: "Full Calendar",
                  icon: Calendar,
                  onClick: () => navigate("/appointments"),
                },
              ].map((action, idx) => (
                <button
                  key={idx}
                  onClick={action.onClick}
                  className="w-full p-4 flex items-center gap-4 hover:bg-white hover:shadow-sm rounded-2xl transition-all group mb-1 last:mb-0"
                >
                  <div className="w-10 h-10 rounded-xl bg-white border border-gray-100 flex items-center justify-center text-gray-400 group-hover:bg-medic-dark group-hover:text-white group-hover:border-medic-dark transition-all shadow-sm">
                    <action.icon size={18} />
                  </div>
                  <span className="font-bold text-gray-600 group-hover:text-gray-900">
                    {action.title}
                  </span>
                  <ChevronRight
                    size={16}
                    className="ml-auto text-gray-300 group-hover:text-medic-dark"
                  />
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DoctorDashboard;
