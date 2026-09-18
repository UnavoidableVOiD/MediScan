import React, { useEffect, useState, useMemo } from "react";
import {
  User,
  Phone,
  Mail,
  ShieldAlert,
  Trash2,
  Search,
  Filter,
  ChevronRight,
  Users,
  Activity,
} from "lucide-react";
import { useSelector, useDispatch } from "react-redux";
import { motion, AnimatePresence } from "framer-motion";
import {
  fetchAdminPatients,
  updatePatient,
  deletePatient,
} from "../../store/slices/adminSlice";

const AdminPatients = () => {
  const dispatch = useDispatch();
  const { user } = useSelector((state) => state.auth);
  const { patients, loading } = useSelector((state) => state.admin);
  const [searchQuery, setSearchQuery] = useState("");

  useEffect(() => {
    dispatch(fetchAdminPatients());
  }, [dispatch]);

  const filteredPatients = useMemo(() => {
    return patients.filter((p) => {
      const fullName = `${p.first_name} ${p.last_name}`.toLowerCase();
      const email = p.email.toLowerCase();
      const query = searchQuery.toLowerCase();
      return fullName.includes(query) || email.includes(query);
    });
  }, [patients, searchQuery]);

  const handleUpdatePatient = async (id, data) => {
    try {
      await dispatch(updatePatient({ id, data })).unwrap();
    } catch (error) {}
  };

  const handleDeletePatient = async (id) => {
    if (
      !window.confirm(
        "CRITICAL: Permanent deletion of patient record. Proceed?",
      )
    )
      return;
    try {
      await dispatch(deletePatient(id)).unwrap();
    } catch (error) {}
  };

  return (
    <div className="space-y-10 max-w-[1600px] mx-auto pb-20">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-6 px-2">
        <div className="flex items-center gap-5">
          <div className="w-14 h-14 bg-amber-50 text-amber-600 rounded-[1.25rem] flex items-center justify-center">
            <Users size={30} />
          </div>
          <div>
            <h1 className="text-3xl font-black text-slate-900 tracking-tight">
              Patient Registry
            </h1>
            <p className="text-slate-500 font-medium">
              Monitor patient activity and manage system access.
            </p>
          </div>
        </div>

        <div className="relative group min-w-[320px]">
          <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400 group-focus-within:text-amber-600 transition-colors" />
          <input
            type="text"
            placeholder="Search patients..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-11 pr-5 py-3.5 bg-white border border-slate-100 shadow-sm rounded-2xl outline-none focus:ring-4 focus:ring-amber-500/5 transition-all font-medium text-sm"
          />
        </div>
      </div>

      {loading ? (
        <div className="flex flex-col items-center justify-center py-24 gap-4">
          <div className="w-12 h-12 border-4 border-slate-100 border-t-amber-600 rounded-full animate-spin" />
          <p className="text-slate-400 font-bold text-xs uppercase tracking-widest text-center">
            Synchronizing Registry...
          </p>
        </div>
      ) : filteredPatients.length === 0 ? (
        <div className="text-center py-24 text-slate-400 font-medium bg-white rounded-[2.5rem] border border-dashed border-slate-200">
          No matching patients discovered.
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {filteredPatients.map((patient) => (
            <motion.div
              layout
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              key={patient.id}
              className="bg-white p-8 rounded-[2.5rem] shadow-sm border border-slate-100 hover:shadow-xl hover:shadow-slate-200/50 transition-all group relative overflow-hidden"
            >
              <div className="flex items-center gap-5 mb-8">
                <div className="w-16 h-16 bg-slate-50 text-slate-400 group-hover:bg-amber-50 group-hover:text-amber-600 rounded-2xl flex items-center justify-center font-black text-xl transition-all duration-500">
                  {(patient.first_name?.[0] || "") +
                    (patient.last_name?.[0] || "")}
                </div>
                <div className="flex-1 min-w-0">
                  <h3 className="font-black text-slate-900 text-lg truncate leading-tight mb-1">
                    {patient.first_name} {patient.last_name}
                  </h3>
                  <div className="flex gap-2 items-center">
                    <span className="text-[10px] bg-slate-100 text-slate-500 px-3 py-1 rounded-full font-black uppercase tracking-widest">
                      PATIENT
                    </span>
                    {!patient.is_active && (
                      <span className="text-[10px] bg-rose-50 text-rose-600 px-3 py-1 rounded-full font-black uppercase tracking-widest flex items-center gap-1">
                        <ShieldAlert size={12} /> Blocked
                      </span>
                    )}
                  </div>
                </div>
              </div>

              <div className="space-y-4 text-xs font-bold text-slate-500 border-t border-slate-50 pt-6 mb-8">
                <div className="flex items-center gap-3">
                  <div className="w-8 h-8 rounded-lg bg-slate-50 flex items-center justify-center">
                    <Mail size={14} className="text-slate-400" />
                  </div>
                  <span className="truncate">{patient.email}</span>
                </div>
                <div className="flex items-center gap-3">
                  <div className="w-8 h-8 rounded-lg bg-slate-50 flex items-center justify-center">
                    <Phone size={14} className="text-slate-400" />
                  </div>
                  <span>{patient.phone_number || "No Phone Data"}</span>
                </div>
                <div className="flex items-center gap-3">
                  <div className="w-8 h-8 rounded-lg bg-slate-50 flex items-center justify-center">
                    <Activity size={14} className="text-slate-400" />
                  </div>
                  <span>
                    Joined{" "}
                    {new Date(
                      patient.date_joined || Date.now(),
                    ).toLocaleDateString()}
                  </span>
                </div>
              </div>

              <div className="flex gap-3">
                {(user?.is_staff || user?.is_superuser) && (
                  <>
                    <button
                      onClick={() =>
                        handleUpdatePatient(patient.id, {
                          is_active: !patient.is_active,
                        })
                      }
                      className={`flex-1 py-4 rounded-2xl font-black text-[10px] uppercase tracking-[0.15em] flex items-center justify-center gap-2 transition-all ${
                        patient.is_active
                          ? "bg-slate-50 text-slate-600 hover:bg-rose-50 hover:text-rose-600 hover:border-rose-100 border border-slate-100"
                          : "bg-rose-600 text-white hover:bg-rose-700 shadow-lg shadow-rose-200"
                      }`}
                    >
                      <ShieldAlert size={14} />
                      {patient.is_active ? "Restrict" : "Enable"}
                    </button>
                    <button
                      onClick={() => handleDeletePatient(patient.id)}
                      className="w-12 h-12 flex items-center justify-center rounded-2xl bg-white border border-slate-100 text-slate-300 hover:text-rose-600 hover:border-rose-200 transition-all shadow-sm"
                    >
                      <Trash2 size={16} />
                    </button>
                  </>
                )}
              </div>
            </motion.div>
          ))}
        </div>
      )}
    </div>
  );
};

export default AdminPatients;
