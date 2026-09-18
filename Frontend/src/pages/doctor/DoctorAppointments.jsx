import React, { useEffect, useState, useMemo } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Calendar,
  Clock,
  User,
  Filter,
  Search,
  ChevronRight,
  MoreVertical,
  CheckCircle2,
  XCircle,
  Loader2,
  Mail,
  History,
  TrendingUp,
  DollarSign,
  Wallet,
  ArrowUpRight,
} from "lucide-react";
import { useDispatch, useSelector } from "react-redux";
import { fetchAppointments } from "../../store/slices/appointmentSlice";

const StatusBadge = ({ status }) => {
  const styles = {
    PAID: "bg-emerald-50 text-emerald-600 border-emerald-100",
    COMPLETED: "bg-blue-50 text-blue-600 border-blue-100",
    CANCELLED: "bg-red-50 text-red-600 border-red-100",
    PENDING: "bg-orange-50 text-orange-600 border-orange-100",
  };

  const icons = {
    PAID: CheckCircle2,
    COMPLETED: CheckCircle2,
    CANCELLED: XCircle,
    PENDING: Clock,
  };

  const Icon = icons[status] || Clock;

  return (
    <span
      className={`px-3 py-1 rounded-full text-[10px] font-black tracking-wider uppercase border flex items-center gap-1.5 w-fit ${styles[status] || styles.PENDING}`}
    >
      <Icon size={12} />
      {status}
    </span>
  );
};

const RevenueCard = ({ title, amount, icon: Icon, color, subtitle }) => (
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
      <span className="text-[10px] font-bold text-gray-400 uppercase tracking-widest">
        {subtitle}
      </span>
    </div>

    <div className="relative z-10">
      <p className="text-3xl font-black text-gray-900 tracking-tight">
        Rs. {amount.toLocaleString()}
      </p>
      <p className="text-xs text-gray-400 font-bold mt-1 uppercase tracking-widest">
        {title}
      </p>
    </div>
  </motion.div>
);

const DoctorAppointments = () => {
  const dispatch = useDispatch();
  const { appointments, loading } = useSelector((state) => state.appointment);
  const [searchTerm, setSearchTerm] = useState("");
  const [activeTab, setActiveTab] = useState("UPCOMING"); // UPCOMING or HISTORY

  useEffect(() => {
    dispatch(fetchAppointments());
  }, [dispatch]);

  const revenueStats = useMemo(() => {
    const paidAppts = appointments.filter(
      (a) => a.status !== "PENDING" && a.status !== "CANCELLED",
    );
    const cancelledAppts = appointments.filter(
      (a) => a.status === "CANCELLED" && a.amount_paid > 0,
    );

    const totalGross = appointments.reduce(
      (acc, a) => acc + (parseFloat(a.amount_paid) || 0),
      0,
    );
    const totalDoctorNet = appointments.reduce((acc, a) => {
      const stored = parseFloat(a.doctor_revenue) || 0;
      const paid = parseFloat(a.amount_paid) || 0;
      const refunded = parseFloat(a.refund_amount) || 0;
      const retained = paid - refunded;
      // If stored is 0 but there's retained amount, compute 75%
      const net = stored > 0 ? stored : retained * 0.75;
      return acc + net;
    }, 0);

    const totalAdminComm = appointments.reduce((acc, a) => {
      const stored = parseFloat(a.admin_revenue) || 0;
      const paid = parseFloat(a.amount_paid) || 0;
      const refunded = parseFloat(a.refund_amount) || 0;
      const retained = paid - refunded;
      // If stored is 0 but there's retained amount, compute 25%
      const comm = stored > 0 ? stored : retained * 0.25;
      return acc + comm;
    }, 0);

    return {
      totalGross,
      totalDoctorNet,
      totalAdminComm,
      confirmedCount: paidAppts.length,
      cancelledRetention: cancelledAppts.length,
    };
  }, [appointments]);

  const isUpcoming = (apptDate) => {
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    const date = new Date(apptDate);
    return date >= today;
  };

  const filteredAppointments = appointments
    .filter((appt) => {
      const name =
        `${appt.patient_first_name} ${appt.patient_last_name}`.toLowerCase();
      const matchesSearch =
        appt.patient_email.toLowerCase().includes(searchTerm.toLowerCase()) ||
        name.includes(searchTerm.toLowerCase());
      const matchesTab =
        activeTab === "UPCOMING"
          ? isUpcoming(appt.appointment_date) &&
            appt.status !== "CANCELLED" &&
            appt.status !== "COMPLETED"
          : !isUpcoming(appt.appointment_date) ||
            appt.status === "CANCELLED" ||
            appt.status === "COMPLETED";
      return matchesSearch && matchesTab;
    })
    .sort((a, b) => {
      if (activeTab === "UPCOMING") {
        return new Date(a.appointment_date) - new Date(b.appointment_date);
      }
      return new Date(b.appointment_date) - new Date(a.appointment_date);
    });

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-neutral-background">
        <Loader2 className="w-10 h-10 text-medic-dark animate-spin" />
      </div>
    );
  }

  return (
    <div className="min-h-[calc(100vh-80px)] bg-neutral-background pt-32 pb-8 px-6 space-y-8 font-sans">
      <div className="max-w-7xl mx-auto space-y-10">
        {/* Header */}
        <div className="flex flex-col md:flex-row md:items-end justify-between gap-6 pb-6 border-b border-gray-100/50">
          <div className="space-y-3">
            <h1 className="text-4xl font-black text-gray-900 tracking-tight">
              Clinical Schedule
            </h1>
            <p className="text-gray-500 font-medium max-w-md leading-relaxed">
              Track your appointments and financial summaries from clinical
              sessions.
            </p>
          </div>

          <div className="flex items-center gap-4 flex-wrap">
            <div className="relative group">
              <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400 group-focus-within:text-medic-dark transition-colors" />
              <input
                type="text"
                placeholder="Search patient or email..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="pl-11 pr-6 py-3 bg-white border border-gray-100 rounded-2xl outline-none focus:border-medic-dark focus:shadow-lg focus:shadow-medic-dark/5 transition-all w-72 text-sm font-medium"
              />
            </div>
          </div>
        </div>

        {/* Revenue Summary */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <RevenueCard
            title="Your Net Earnings"
            amount={revenueStats.totalDoctorNet}
            icon={Wallet}
            color="bg-medic-dark"
            subtitle="75% Split"
          />
          <RevenueCard
            title="Admin Commission"
            amount={revenueStats.totalAdminComm}
            icon={TrendingUp}
            color="bg-blue-500"
            subtitle="25% App Fee"
          />
          <RevenueCard
            title="Gross Transactions"
            amount={revenueStats.totalGross}
            icon={DollarSign}
            color="bg-emerald-500"
            subtitle="Total Inflow"
          />
        </div>

        {/* Filters and Table */}
        <div className="space-y-6">
          <div className="flex bg-white p-1.5 rounded-2xl border border-gray-100 shadow-sm w-fit">
            {[
              { id: "UPCOMING", label: "Upcoming Sessions", icon: Clock },
              { id: "HISTORY", label: "Clinical History", icon: History },
            ].map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center gap-2 px-6 py-3 rounded-xl text-xs font-black uppercase tracking-wider transition-all ${
                  activeTab === tab.id
                    ? "bg-medic-dark text-white shadow-lg shadow-medic-dark/20"
                    : "text-gray-400 hover:text-gray-600 hover:bg-gray-50"
                }`}
              >
                <tab.icon size={14} />
                {tab.label}
              </button>
            ))}
          </div>

          <div className="bg-white rounded-[2.5rem] shadow-xl shadow-medic-dark/5 border border-medic-light/20 overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full text-left border-collapse">
                <thead>
                  <tr className="bg-gray-50/50 border-b border-gray-100">
                    <th className="px-8 py-6 text-gray-400 text-[10px] font-black uppercase tracking-widest">
                      Patient Details
                    </th>
                    <th className="px-8 py-6 text-gray-400 text-[10px] font-black uppercase tracking-widest">
                      Schedule
                    </th>
                    <th className="px-8 py-6 text-gray-400 text-[10px] font-black uppercase tracking-widest">
                      Status
                    </th>
                    <th className="px-8 py-6 text-gray-400 text-[10px] font-black uppercase tracking-widest">
                      Earnings
                    </th>
                    <th className="px-8 py-6 text-gray-400 text-[10px] font-black uppercase tracking-widest text-right">
                      Actions
                    </th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-50">
                  <AnimatePresence mode="popLayout">
                    {filteredAppointments.length > 0 ? (
                      filteredAppointments.map((appt, idx) => (
                        <motion.tr
                          layout
                          initial={{ opacity: 0, y: 10 }}
                          animate={{ opacity: 1, y: 0 }}
                          exit={{ opacity: 0, scale: 0.98 }}
                          transition={{ duration: 0.2, delay: idx * 0.03 }}
                          key={appt.id}
                          className="hover:bg-neutral-soft/30 transition-colors group cursor-pointer"
                        >
                          <td className="px-8 py-6">
                            <div className="flex items-center gap-4">
                              <div className="w-12 h-12 rounded-2xl bg-medic-light/20 flex items-center justify-center font-black text-medic-dark uppercase text-lg shadow-sm group-hover:bg-medic-dark group-hover:text-white transition-all duration-300">
                                {appt.patient_first_name?.[0] ||
                                  appt.patient_email[0]}
                              </div>
                              <div className="flex flex-col">
                                <span className="font-bold text-gray-900 group-hover:text-medic-dark transition-colors">
                                  {appt.patient_first_name}{" "}
                                  {appt.patient_last_name}
                                </span>
                                <span className="text-xs text-gray-400 font-medium">
                                  {appt.patient_email}
                                </span>
                              </div>
                            </div>
                          </td>
                          <td className="px-8 py-6">
                            <div className="flex flex-col gap-1.5">
                              <div className="flex items-center gap-2 text-gray-900 font-bold text-sm">
                                <Calendar
                                  size={14}
                                  className="text-medic-primary"
                                />
                                {new Date(
                                  appt.appointment_date + "T00:00:00",
                                ).toLocaleDateString(undefined, {
                                  weekday: "short",
                                  month: "short",
                                  day: "numeric",
                                })}
                              </div>
                              <div className="flex items-center gap-2 text-gray-500 font-bold text-[9px] tracking-wider uppercase bg-gray-50 w-fit px-2.5 py-1 rounded-lg border border-gray-100">
                                <Clock size={10} />
                                {appt.start_time.slice(0, 5)} -{" "}
                                {appt.end_time.slice(0, 5)}
                              </div>
                            </div>
                          </td>
                          <td className="px-8 py-6">
                            <StatusBadge status={appt.status} />
                          </td>
                          <td className="px-8 py-6">
                            <div className="flex flex-col gap-1 w-[160px]">
                              {parseFloat(appt.refund_amount) > 0 ? (
                                <>
                                  <div className="flex items-center justify-between gap-4">
                                    <span className="text-[10px] font-black text-gray-400 uppercase tracking-widest">
                                      Initial Paid
                                    </span>
                                    <span className="text-xs font-bold text-gray-500">
                                      Rs.{" "}
                                      {parseFloat(
                                        appt.amount_paid,
                                      ).toLocaleString()}
                                    </span>
                                  </div>
                                  <div className="flex items-center justify-between gap-4">
                                    <span className="text-[10px] font-black text-orange-400 uppercase tracking-widest">
                                      Refunded (60%)
                                    </span>
                                    <span className="text-xs font-bold text-orange-400">
                                      - Rs.{" "}
                                      {parseFloat(
                                        appt.refund_amount,
                                      ).toLocaleString()}
                                    </span>
                                  </div>
                                  <div className="h-px bg-gray-100 my-1" />
                                </>
                              ) : null}
                              <div className="flex items-center justify-between gap-4">
                                <span className="text-[10px] font-black text-gray-400 uppercase tracking-widest">
                                  {parseFloat(appt.refund_amount) > 0
                                    ? "Retained Gross"
                                    : "Gross"}
                                </span>
                                <span className="text-xs font-bold text-gray-500">
                                  Rs.{" "}
                                  {(
                                    parseFloat(appt.amount_paid) -
                                    parseFloat(appt.refund_amount || 0)
                                  ).toLocaleString()}
                                </span>
                              </div>
                              <div className="flex items-center justify-between gap-4">
                                <span className="text-[10px] font-black text-rose-400 uppercase tracking-widest">
                                  Platform Cut
                                </span>
                                <span className="text-xs font-bold text-rose-400">
                                  - Rs.{" "}
                                  {(
                                    parseFloat(appt.admin_revenue) ||
                                    (parseFloat(appt.amount_paid) -
                                      parseFloat(appt.refund_amount || 0)) *
                                      0.25
                                  ).toLocaleString()}
                                </span>
                              </div>
                              <div className="h-px bg-gray-100 my-1" />
                              <div className="flex items-center justify-between gap-4">
                                <span className="text-[10px] font-black text-medic-dark uppercase tracking-widest">
                                  Net Profit
                                </span>
                                <span className="font-black text-sm text-gray-900">
                                  Rs.{" "}
                                  {(
                                    parseFloat(appt.doctor_revenue) ||
                                    (parseFloat(appt.amount_paid) -
                                      parseFloat(appt.refund_amount || 0)) *
                                      0.75
                                  ).toLocaleString()}
                                </span>
                              </div>
                            </div>
                          </td>
                          <td className="px-8 py-6 text-right">
                            <div className="flex items-center justify-end gap-2 group-hover:translate-x-[-4px] transition-transform">
                              <button className="p-2.5 bg-white border border-gray-100 rounded-xl text-gray-400 hover:border-medic-dark hover:text-medic-dark transition-all shadow-sm">
                                <Mail size={16} />
                              </button>
                              <div className="w-10 h-10 bg-gray-50 rounded-xl flex items-center justify-center text-gray-300 group-hover:bg-medic-dark group-hover:text-white transition-all">
                                <ArrowUpRight size={20} />
                              </div>
                            </div>
                          </td>
                        </motion.tr>
                      ))
                    ) : (
                      <motion.tr
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        key="empty"
                      >
                        <td colSpan="5" className="px-8 py-32 text-center">
                          <div className="flex flex-col items-center gap-4 opacity-40">
                            <div className="w-20 h-20 bg-gray-100 rounded-full flex items-center justify-center">
                              {activeTab === "UPCOMING" ? (
                                <Calendar size={32} className="text-gray-400" />
                              ) : (
                                <History size={32} className="text-gray-400" />
                              )}
                            </div>
                            <p className="font-black uppercase tracking-widest text-xs text-gray-400">
                              {activeTab === "UPCOMING"
                                ? "No scheduled sessions found"
                                : "No clinical history found"}
                            </p>
                          </div>
                        </td>
                      </motion.tr>
                    )}
                  </AnimatePresence>
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DoctorAppointments;
