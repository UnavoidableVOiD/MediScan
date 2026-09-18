import React, { useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Users, Search, Activity, Loader2, ArrowUpRight } from "lucide-react";
import { useDispatch, useSelector } from "react-redux";
import { useNavigate } from "react-router-dom";
import { fetchMyPatients } from "../../store/slices/doctorSlice";

const PatientCard = ({ patient, idx, navigate }) => (
  <motion.div
    initial={{ opacity: 0, scale: 0.95 }}
    animate={{ opacity: 1, scale: 1 }}
    transition={{ delay: idx * 0.05 }}
    onClick={() => navigate(`/patient/${patient.id}`)}
    className="bg-white rounded-[2.5rem] p-8 border border-medic-light/20 shadow-xl shadow-medic-dark/5 hover:shadow-2xl hover:shadow-medic-dark/10 hover:-translate-y-1 transition-all cursor-pointer group relative overflow-hidden"
  >
    <div className="absolute top-0 right-0 p-6 opacity-0 group-hover:opacity-100 transition-all translate-x-4 group-hover:translate-x-0 z-10">
      <div className="bg-medic-dark text-white p-2 rounded-full shadow-lg">
        <ArrowUpRight size={20} />
      </div>
    </div>

    <div className="space-y-6 relative z-0">
      <div className="flex items-center gap-5">
        <div className="w-16 h-16 rounded-3xl bg-medic-light/20 flex items-center justify-center font-black text-medic-dark text-2xl uppercase shadow-sm relative group-hover:bg-medic-dark group-hover:text-white transition-colors duration-300">
          {patient.first_name[0]}
          {patient.status === "ONGOING" && (
            <span className="absolute -top-1.5 -right-1.5 flex h-4 w-4">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-blue-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-4 w-4 bg-blue-500 border-2 border-white"></span>
            </span>
          )}
        </div>
        <div className="flex flex-col flex-1">
          <h3 className="font-black text-gray-900 text-xl leading-tight group-hover:text-medic-dark transition-colors">
            {patient.first_name} {patient.last_name}
          </h3>
          <span className="text-sm text-gray-400 font-bold truncate max-w-[180px]">
            {patient.email}
          </span>
          {patient.phone_number && (
            <span className="text-[10px] text-slate-400 font-black uppercase tracking-widest mt-0.5">
              {patient.phone_number}
            </span>
          )}
        </div>
      </div>

      <div className="h-px bg-gray-50" />

      <div className="space-y-4">
        <div className="flex items-center justify-between text-xs font-bold">
          <div className="space-y-1">
            <p className="text-gray-400 uppercase tracking-wider text-[10px]">
              Condition
            </p>
            <span
              className={`font-black uppercase tracking-tight ${
                patient.condition === "High"
                  ? "text-rose-500"
                  : patient.condition === "Medium"
                    ? "text-amber-500"
                    : "text-emerald-500"
              }`}
            >
              {patient.condition || "Stable"} Risk
            </span>
          </div>
          <div className="text-right space-y-1">
            <p className="text-gray-400 uppercase tracking-wider text-[10px]">
              Last Visit
            </p>
            <p className="text-gray-700">
              {patient.last_visit ? (
                new Date(patient.last_visit).toLocaleDateString(undefined, {
                  month: "short",
                  day: "numeric",
                })
              ) : (
                <span className="text-gray-300">N/A</span>
              )}
            </p>
          </div>
        </div>

        <div className="flex items-center justify-between pt-2">
          <span
            className={`px-4 py-1.5 rounded-full text-[10px] font-black uppercase tracking-wider border ${
              patient.status === "ONGOING"
                ? "bg-blue-50 text-blue-600 border-blue-100"
                : "bg-emerald-50 text-emerald-600 border-emerald-100"
            }`}
          >
            {patient.status}
          </span>
          <div className="flex -space-x-2">
            {[1, 2, 3].map((i) => (
              <div
                key={i}
                className="w-6 h-6 rounded-full border-2 border-white bg-slate-100 flex items-center justify-center"
              >
                <Activity size={10} className="text-slate-400" />
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  </motion.div>
);

const DoctorPatients = () => {
  const dispatch = useDispatch();
  const navigate = useNavigate();
  const { patients, loading } = useSelector((state) => state.doctor);
  const [searchTerm, setSearchTerm] = useState("");
  const [statusFilter, setStatusFilter] = useState("ALL"); // ALL, ONGOING

  useEffect(() => {
    dispatch(fetchMyPatients());
  }, [dispatch]);

  const filteredPatients = patients.filter((patient) => {
    const matchesSearch =
      patient.first_name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      patient.last_name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      patient.email.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesStatus =
      statusFilter === "ALL" || patient.status === statusFilter;
    return matchesSearch && matchesStatus;
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
        <div className="flex flex-col md:flex-row md:items-end justify-between gap-6 pb-4 border-b border-gray-100/50">
          <div className="space-y-2">
            <h1 className="text-4xl font-black text-gray-900 tracking-tight">
              Patient Directory
            </h1>
            <p className="text-gray-500 font-medium">
              Access records and track patient progress.
            </p>
          </div>

          <div className="flex items-center gap-4 flex-wrap">
            <div className="relative group">
              <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400 group-focus-within:text-medic-dark transition-colors" />
              <input
                type="text"
                placeholder="Search patient name..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="pl-11 pr-6 py-3 bg-white border border-gray-100 rounded-2xl outline-none focus:border-medic-dark focus:shadow-lg focus:shadow-medic-dark/5 transition-all w-64 text-sm font-medium"
              />
            </div>
            <div className="flex bg-white p-1 rounded-2xl border border-gray-100 shadow-sm">
              {[
                { id: "ALL", label: "All Patients" },
                { id: "ONGOING", label: "Active Treatment" },
              ].map((tab) => (
                <button
                  key={tab.id}
                  onClick={() => setStatusFilter(tab.id)}
                  className={`px-5 py-2.5 rounded-xl text-xs font-black uppercase tracking-wider transition-all ${
                    statusFilter === tab.id
                      ? "bg-medic-dark text-white shadow-md shadow-medic-dark/20"
                      : "text-gray-400 hover:text-gray-600 hover:bg-gray-50"
                  }`}
                >
                  {tab.label}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Patients Grid */}
        {filteredPatients.length > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
            <AnimatePresence>
              {filteredPatients.map((patient, idx) => (
                <PatientCard
                  key={patient.id}
                  patient={patient}
                  idx={idx}
                  navigate={navigate}
                />
              ))}
            </AnimatePresence>
          </div>
        ) : (
          <div className="flex flex-col items-center justify-center py-20 opacity-40">
            <div className="w-24 h-24 bg-gray-100 rounded-full flex items-center justify-center mb-6">
              <Users size={40} className="text-gray-400" />
            </div>
            <h3 className="text-xl font-black text-gray-400 uppercase tracking-widest text-center">
              No patients found
            </h3>
            <p className="text-gray-400 text-sm mt-2 text-center max-w-xs">
              Try adjusting your search or filters to find what you&apos;re
              looking for.
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

export default DoctorPatients;
