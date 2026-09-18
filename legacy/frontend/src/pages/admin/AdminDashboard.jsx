import React, { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { useNavigate } from "react-router-dom";
import {
  Users,
  ShieldCheck,
  UserPlus,
  Settings,
  BookOpen,
  Bell,
  ArrowRight,
  ClipboardList,
  Activity,
  UserCheck,
  FileText,
  TrendingUp,
  ArrowUpRight,
  ArrowDownRight,
} from "lucide-react";
import { useSelector, useDispatch } from "react-redux";
import {
  fetchAdminDoctors,
  fetchAdminPatients,
  fetchAdminFinancialStats,
} from "../../store/slices/adminSlice";

const StatCard = ({
  title,
  value,
  icon: Icon,
  color,
  trend,
  trendValue,
  delay,
}) => (
  <motion.div
    initial={{ opacity: 0, y: 20 }}
    animate={{ opacity: 1, y: 0 }}
    transition={{ delay }}
    className="bg-white p-7 rounded-[2.5rem] shadow-sm border border-slate-100 hover:shadow-xl hover:shadow-slate-200/50 transition-all group"
  >
    <div className="flex items-start justify-between mb-6">
      <div
        className={`p-4 rounded-2xl ${color} bg-opacity-10 transition-transform group-hover:scale-110 duration-500`}
      >
        <Icon className={`w-7 h-7 ${color.replace("bg-", "text-")}`} />
      </div>
      <div
        className={`flex items-center gap-1 px-3 py-1.5 rounded-full text-[10px] font-black uppercase tracking-wider ${trend === "up" ? "bg-emerald-50 text-emerald-600" : "bg-rose-50 text-rose-600"}`}
      >
        {trend === "up" ? (
          <ArrowUpRight size={14} />
        ) : (
          <ArrowDownRight size={14} />
        )}
        {trendValue}%
      </div>
    </div>
    <div>
      <p className="text-[11px] font-black text-slate-400 uppercase tracking-[0.15em] mb-1">
        {title}
      </p>
      <h3 className="text-3xl font-black text-slate-900 tracking-tight">
        {value}
      </h3>
    </div>
  </motion.div>
);

const AdminDashboard = () => {
  const navigate = useNavigate();
  const dispatch = useDispatch();
  const { doctors, patients, financialStats, loading } = useSelector(
    (state) => state.admin,
  );

  useEffect(() => {
    dispatch(fetchAdminDoctors("all"));
    dispatch(fetchAdminPatients());
    dispatch(fetchAdminFinancialStats());
  }, [dispatch]);

  const stats = [
    {
      title: "Total Healthcare Providers",
      value: doctors.length,
      icon: UserCheck,
      color: "bg-[#6366F1]",
      trend: "up",
      trendValue: 12,
      delay: 0.1,
    },
    {
      title: "Total Platform Commission",
      value: `Rs. ${financialStats?.overall?.total_admin_revenue?.toLocaleString() || "0"}`,
      icon: TrendingUp,
      color: "bg-emerald-500",
      trend: "up",
      trendValue: 25,
      delay: 0.2,
    },
    {
      title: "Pending Verifications",
      value: doctors.filter(
        (d) =>
          d.doctor_status === "PENDING" || d.license_info?.status === "PENDING",
      ).length,
      icon: ClipboardList,
      color: "bg-[#EF4444]",
      trend: "down",
      trendValue: 5,
      delay: 0.3,
    },
    {
      title: "Gross Transaction Value",
      value: `Rs. ${financialStats?.overall?.total_gross?.toLocaleString() || "0"}`,
      icon: Activity,
      color: "bg-amber-500",
      trend: "up",
      trendValue: 15,
      delay: 0.4,
    },
  ];

  return (
    <div className="pt-32 pb-20 space-y-10 max-w-[1600px] mx-auto">
      {/* Welcome Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-6">
        <div>
          <h1 className="text-4xl font-black text-slate-900 tracking-tight">
            System Overview
          </h1>
          <p className="text-slate-500 font-medium mt-1 text-lg">
            Central control for MediScan network infrastructure.
          </p>
        </div>
        <div className="flex items-center gap-3">
          <button className="px-6 py-3 bg-white border border-slate-200 rounded-2xl font-bold text-sm text-slate-600 hover:bg-slate-50 transition-all flex items-center gap-2 shadow-sm">
            Generate Report <FileText size={18} />
          </button>
          <button
            onClick={() => navigate("/admin/create-admin")}
            className="px-6 py-3 bg-[#0F172A] text-white rounded-2xl font-bold text-sm hover:translate-y-[-2px] hover:shadow-lg hover:shadow-slate-900/20 transition-all flex items-center gap-2"
          >
            Add Administrator <UserPlus size={18} />
          </button>
        </div>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
        {stats.map((stat, idx) => (
          <StatCard key={idx} {...stat} />
        ))}
      </div>

      {/* Detailed Insights Section */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-10">
        {/* Revenue Breakdown */}
        <div className="lg:col-span-2 space-y-6">
          <div className="flex items-center justify-between px-2">
            <h2 className="text-2xl font-black text-slate-900 tracking-tight flex items-center gap-3">
              <TrendingUp className="text-emerald-500" /> Doctor Revenue
              Breakdown
            </h2>
            <div className="flex items-center gap-2">
              <span className="text-[10px] font-black text-slate-400 uppercase tracking-widest">
                Platform commission: 25%
              </span>
            </div>
          </div>

          <div className="bg-white rounded-[3rem] p-4 border border-slate-100 shadow-sm overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full text-left">
                <thead>
                  <tr className="border-b border-slate-50">
                    <th className="px-6 py-4 text-[10px] font-black text-slate-400 uppercase tracking-widest">
                      Doctor
                    </th>
                    <th className="px-6 py-4 text-[10px] font-black text-slate-400 uppercase tracking-widest text-right">
                      Initial Paid
                    </th>
                    <th className="px-6 py-4 text-[10px] font-black text-rose-400 uppercase tracking-widest text-right">
                      Refunds
                    </th>
                    <th className="px-6 py-4 text-[10px] font-black text-slate-400 uppercase tracking-widest text-right">
                      Retained
                    </th>
                    <th className="px-6 py-4 text-[10px] font-black text-emerald-600 uppercase tracking-widest text-right">
                      Commission
                    </th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-50">
                  {financialStats?.doctor_breakdown?.map((doc, idx) => (
                    <tr
                      key={idx}
                      className="hover:bg-slate-50 transition-colors"
                    >
                      <td className="px-6 py-4">
                        <div className="flex flex-col">
                          <span className="font-bold text-slate-900">
                            {doc.doctor__first_name} {doc.doctor__last_name}
                          </span>
                          <span className="text-[10px] text-slate-400 font-medium">
                            {doc.doctor__email} • {doc.appt_count} Appts
                          </span>
                        </div>
                      </td>
                      <td className="px-6 py-4 text-right font-medium text-slate-500">
                        Rs. {doc.paid?.toLocaleString() || 0}
                      </td>
                      <td className="px-6 py-4 text-right font-medium text-rose-400">
                        - {doc.refunded?.toLocaleString() || 0}
                      </td>
                      <td className="px-6 py-4 text-right font-black text-slate-900">
                        Rs. {doc.gross?.toLocaleString() || 0}
                      </td>
                      <td className="px-6 py-4 text-right">
                        <span className="font-black text-emerald-600">
                          Rs. {doc.admin_share?.toLocaleString() || 0}
                        </span>
                      </td>
                    </tr>
                  ))}
                  {(!financialStats?.doctor_breakdown ||
                    financialStats.doctor_breakdown.length === 0) && (
                    <tr>
                      <td
                        colSpan="4"
                        className="px-6 py-12 text-center text-slate-400 font-bold text-sm"
                      >
                        No transaction data available
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>
        </div>

        {/* System Health */}
        <div className="space-y-6">
          <h2 className="text-2xl font-black text-slate-900 tracking-tight">
            System Health
          </h2>
          <div className="bg-slate-900 rounded-[3rem] p-8 text-white relative overflow-hidden shadow-2xl space-y-8">
            <div className="absolute -right-20 -top-20 w-64 h-64 bg-indigo-500 rounded-full blur-[100px] opacity-20"></div>

            <div className="relative z-10 space-y-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-[10px] font-black text-indigo-400 uppercase tracking-widest">
                    Active Verification
                  </p>
                  <h4 className="text-xl font-bold mt-1">
                    {
                      doctors.filter((d) => d.doctor_status === "VERIFIED")
                        .length
                    }{" "}
                    Verified Providers
                  </h4>
                </div>
              </div>

              <div className="space-y-4">
                {[
                  {
                    label: "License verification rate",
                    value:
                      doctors.length > 0
                        ? Math.round(
                            (doctors.filter(
                              (d) => d.doctor_status === "VERIFIED",
                            ).length /
                              doctors.length) *
                              100,
                          )
                        : 0,
                    color: "bg-indigo-500",
                  },
                  {
                    label: "Patient Activation",
                    value: patients.length > 0 ? 100 : 0,
                    color: "bg-emerald-500",
                  },
                ].map((item, i) => (
                  <div key={i} className="space-y-1.5">
                    <div className="flex justify-between text-[10px] font-black uppercase tracking-wider text-white/50">
                      <span>{item.label}</span>
                      <span>{item.value}%</span>
                    </div>
                    <div className="h-1.5 w-full bg-white/10 rounded-full overflow-hidden">
                      <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: `${item.value}%` }}
                        className={`h-full ${item.color}`}
                      />
                    </div>
                  </div>
                ))}
              </div>

              <button
                onClick={() => navigate("/admin/doctors")}
                className="w-full py-4 rounded-2xl bg-white text-slate-900 transition-all font-black text-xs uppercase tracking-widest shadow-xl flex items-center justify-center gap-2"
              >
                Manage Providers <ArrowRight size={14} />
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AdminDashboard;
