import React, { useState, useEffect, useMemo } from "react";
import {
  CheckCircle,
  XCircle,
  Clock,
  FileText,
  ChevronRight,
  X,
  ShieldAlert,
  Trash2,
  Search,
  ExternalLink,
  ShieldCheck,
  Stethoscope,
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { useSelector, useDispatch } from "react-redux";
import {
  fetchAdminDoctors,
  verifyDoctor,
  updateDoctor,
  deleteDoctor,
  unverifyDoctor,
} from "../../store/slices/adminSlice";

import AvailabilityManager from "../../components/doctor/AvailabilityManager";
import { appointmentApi } from "../../services/api";

const AdminDoctors = () => {
  const [activeTab, setActiveTab] = useState("pending");
  const [selectedDoctor, setSelectedDoctor] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [showAvailability, setShowAvailability] = useState(false);
  const [availabilityData, setAvailabilityData] = useState([]);
  const [isSavingAvailability, setIsSavingAvailability] = useState(false);

  const dispatch = useDispatch();
  const { user } = useSelector((state) => state.auth);
  const { doctors, loading } = useSelector((state) => state.admin);

  useEffect(() => {
    dispatch(fetchAdminDoctors(activeTab));
  }, [activeTab, dispatch]);

  // Reset availability view when closing modal or changing doctor
  useEffect(() => {
    if (!selectedDoctor) {
      setShowAvailability(false);
      setAvailabilityData([]);
    }
  }, [selectedDoctor]);

  const loadAvailability = async (doctorId) => {
    try {
      const response = await appointmentApi.getAvailability(doctorId);
      setAvailabilityData(Array.isArray(response.data) ? response.data : []);
    } catch (error) {
      console.error("Failed to load availability", error);
    }
  };

  const handleOpenSchedule = async () => {
    if (selectedDoctor) {
      setShowAvailability(true);
      await loadAvailability(selectedDoctor.id);
    }
  };

  const handleSaveAvailability = async () => {
    if (!selectedDoctor) return;
    setIsSavingAvailability(true);
    try {
      // Ensure times are HH:MM:SS for backend
      const cleanData = availabilityData.map(({ tempId, id, ...rest }) => ({
        ...rest,
        start_time:
          rest.start_time.length === 5
            ? `${rest.start_time}:00`
            : rest.start_time,
        end_time:
          rest.end_time.length === 5 ? `${rest.end_time}:00` : rest.end_time,
      }));

      const response = await appointmentApi.syncAvailability(
        cleanData,
        selectedDoctor.id,
      );
      if (response.data) {
        setAvailabilityData(Array.isArray(response.data) ? response.data : []);
      }
      // Success toast or feedback could be added here
    } catch (error) {
      console.error("Failed to save availability", error);
    } finally {
      setIsSavingAvailability(false);
    }
  };

  const filteredDoctors = useMemo(() => {
    return doctors.filter((doctor) => {
      const fullName = `${doctor.first_name} ${doctor.last_name}`.toLowerCase();
      const email = doctor.email.toLowerCase();
      const specialty = (doctor.specialization || "").toLowerCase();
      const query = searchQuery.toLowerCase();
      return (
        fullName.includes(query) ||
        email.includes(query) ||
        specialty.includes(query)
      );
    });
  }, [doctors, searchQuery]);

  const handleCloseModal = () => {
    setSelectedDoctor(null);
    setPreviewUrl(null);
  };

  const handleVerify = async (status, rejectionReason = "") => {
    if (!selectedDoctor?.license_info?.id) return;
    try {
      await dispatch(
        verifyDoctor({
          id: selectedDoctor.license_info.id,
          data: { status, rejection_reason: rejectionReason },
        }),
      ).unwrap();
      handleCloseModal();
      dispatch(fetchAdminDoctors(activeTab));
    } catch (error) {
      console.error(error);
    }
  };

  const handleUpdateDoctor = async (id, data) => {
    try {
      await dispatch(updateDoctor({ id, data })).unwrap();
      dispatch(fetchAdminDoctors(activeTab));
      if (selectedDoctor && selectedDoctor.id === id)
        setSelectedDoctor({ ...selectedDoctor, ...data });
    } catch (error) {
      console.error(error);
    }
  };

  const handleDeleteDoctor = async (id) => {
    if (!window.confirm("CRITICAL: Permanent record deletion. Proceed?"))
      return;
    try {
      await dispatch(deleteDoctor(id)).unwrap();
      handleCloseModal();
    } catch (error) {
      console.error(error);
    }
  };

  const handleUnverify = async (id) => {
    if (!window.confirm("DANGER: Rescinding verification. Proceed?")) return;
    try {
      await dispatch(unverifyDoctor(id)).unwrap();
      handleCloseModal();
      dispatch(fetchAdminDoctors(activeTab));
    } catch (error) {
      console.error(error);
    }
  };

  return (
    <div className="space-y-10 max-w-[1600px] mx-auto pb-20">
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-6">
        <div className="flex items-center gap-5">
          <div className="w-14 h-14 bg-indigo-50 text-indigo-600 rounded-[1.25rem] flex items-center justify-center">
            <Stethoscope size={30} />
          </div>
          <div>
            <h1 className="text-3xl font-black text-slate-900 tracking-tight">
              Staff Management
            </h1>
            <p className="text-slate-500 font-medium">
              Verify credentials and manage provider accounts.
            </p>
          </div>
        </div>
        <div className="relative group min-w-[320px]">
          <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400 group-focus-within:text-indigo-600 transition-colors" />
          <input
            type="text"
            placeholder="Search practitioners..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-11 pr-5 py-3.5 bg-white border border-slate-100 shadow-sm rounded-2xl outline-none focus:ring-4 focus:ring-indigo-500/5 transition-all font-medium text-sm"
          />
        </div>
      </div>

      <div className="bg-white rounded-[2.5rem] shadow-sm border border-slate-100 overflow-hidden">
        <div className="flex items-center gap-2 p-3 bg-slate-50 border-b border-slate-100">
          {[
            {
              id: "pending",
              label: "Pending Review",
              icon: Clock,
              color: "text-amber-500",
            },
            {
              id: "verified",
              label: "Verified Staff",
              icon: ShieldCheck,
              color: "text-emerald-500",
            },
            {
              id: "rejected",
              label: "Rejected",
              icon: XCircle,
              color: "text-slate-400",
            },
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex items-center gap-2 px-6 py-3.5 rounded-2xl font-black text-xs uppercase tracking-widest transition-all ${
                activeTab === tab.id
                  ? "bg-white text-slate-900 shadow-lg"
                  : "text-slate-400 hover:text-slate-600"
              }`}
            >
              <tab.icon
                className={`w-4 h-4 ${activeTab === tab.id ? tab.color : "text-slate-400"}`}
              />
              {tab.label}
            </button>
          ))}
        </div>

        <div className="overflow-x-auto">
          {loading ? (
            <div className="flex flex-col items-center justify-center py-24 gap-4">
              <div className="w-12 h-12 border-4 border-slate-100 border-t-indigo-600 rounded-full animate-spin" />
              <p className="text-slate-400 font-bold text-xs uppercase tracking-widest">
                Loading Records...
              </p>
            </div>
          ) : filteredDoctors.length === 0 ? (
            <div className="text-center py-24 text-slate-400 font-medium">
              No results found for your search.
            </div>
          ) : (
            <table className="w-full text-left">
              <thead>
                <tr className="text-[11px] font-black text-slate-400 uppercase tracking-[0.2em] bg-slate-50/30">
                  <th className="pl-10 pr-6 py-6 border-b border-slate-100">
                    Practitioner
                  </th>
                  <th className="px-6 py-6 border-b border-slate-100">
                    Specialization
                  </th>
                  <th className="px-6 py-6 border-b border-slate-100">
                    Status
                  </th>
                  <th className="pr-10 pl-6 py-6 border-b border-slate-100 text-right">
                    Action
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-50">
                {filteredDoctors.map((doctor) => (
                  <tr
                    key={doctor.id}
                    onClick={() => setSelectedDoctor(doctor)}
                    className="hover:bg-slate-50 transition-colors group cursor-pointer"
                  >
                    <td className="pl-10 pr-6 py-6">
                      <div className="flex items-center gap-4">
                        <div className="w-12 h-12 rounded-xl bg-indigo-50 text-indigo-600 flex items-center justify-center font-black">
                          {(doctor.first_name?.[0] || "") +
                            (doctor.last_name?.[0] || "")}
                        </div>
                        <div>
                          <p className="font-black text-slate-900 leading-none mb-1 text-sm">
                            {doctor.first_name} {doctor.last_name}
                          </p>
                          <p className="text-xs text-slate-500">
                            {doctor.email}
                          </p>
                        </div>
                      </div>
                    </td>
                    <td className="px-6 py-6 text-xs font-black uppercase tracking-widest text-slate-600">
                      {doctor.specialization?.replace(/_/g, " ") || "N/A"}
                    </td>
                    <td className="px-6 py-6">
                      <div className="flex items-center gap-2">
                        <div
                          className={`w-2 h-2 rounded-full ${activeTab === "pending" ? "bg-amber-500" : activeTab === "verified" ? "bg-emerald-500" : "bg-slate-300"}`}
                        />
                        <span className="text-[11px] font-black uppercase tracking-widest text-slate-600">
                          {activeTab}
                        </span>
                      </div>
                    </td>
                    <td className="pr-10 pl-6 py-6 text-right">
                      <ChevronRight className="w-5 h-5 text-slate-300 ml-auto" />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>

      <AnimatePresence>
        {selectedDoctor && (
          <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={handleCloseModal}
              className="absolute inset-0 bg-slate-900/60 backdrop-blur-sm"
            />
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.95 }}
              className="bg-white w-full max-w-4xl rounded-[3rem] shadow-2xl relative z-10 max-h-[90vh] overflow-auto"
            >
              <div className="p-10 space-y-12">
                <div className="flex items-center justify-between border-b pb-8">
                  <div className="flex items-center gap-6">
                    <div className="w-20 h-20 bg-[#0F172A] text-white rounded-[1.5rem] flex items-center justify-center text-3xl font-black">
                      {(selectedDoctor.first_name?.[0] || "") +
                        (selectedDoctor.last_name?.[0] || "")}
                    </div>
                    <div>
                      <h2 className="text-3xl font-black text-slate-900 tracking-tight">
                        Dr. {selectedDoctor.first_name}{" "}
                        {selectedDoctor.last_name}
                      </h2>
                      <p className="text-slate-500 font-bold">
                        {selectedDoctor.specialization?.replace(/_/g, " ")}
                      </p>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    {showAvailability ? (
                      <button
                        onClick={() => setShowAvailability(false)}
                        className="px-4 py-2 text-slate-500 font-bold text-xs hover:bg-slate-50 rounded-xl"
                      >
                        Back to Details
                      </button>
                    ) : null}
                    <button
                      onClick={handleCloseModal}
                      className="p-4 hover:bg-slate-100 rounded-2xl"
                    >
                      <X size={24} />
                    </button>
                  </div>
                </div>

                {showAvailability ? (
                  <div className="space-y-6">
                    <div className="flex items-center justify-between">
                      <h3 className="text-lg font-bold text-slate-900">
                        Manage Schedule
                      </h3>
                    </div>
                    <AvailabilityManager
                      availabilityData={availabilityData}
                      onChange={setAvailabilityData}
                      onSave={handleSaveAvailability}
                      isSaving={isSavingAvailability}
                    />
                  </div>
                ) : (
                  <div className="grid md:grid-cols-2 gap-10">
                    <div className="space-y-6">
                      <h4 className="font-black uppercase tracking-widest text-slate-400 text-xs">
                        Credential Review
                      </h4>
                      {selectedDoctor.license_info ? (
                        <div
                          onClick={() =>
                            window.open(
                              selectedDoctor.license_info.license_file,
                              "_blank",
                            )
                          }
                          className="p-8 bg-slate-50 rounded-[2rem] border-2 border-dashed border-slate-200 cursor-pointer hover:border-indigo-400 group transition-all"
                        >
                          <FileText className="w-12 h-12 text-indigo-500 mb-4 group-hover:scale-110 transition-transform" />
                          <p className="font-black text-slate-900 underline">
                            Medical License Doc
                          </p>
                          <p className="text-xs text-slate-500 mt-1">
                            License ID:{" "}
                            {selectedDoctor.license_info.license_number}
                          </p>
                        </div>
                      ) : (
                        <div className="p-8 bg-slate-50 rounded-3xl text-slate-400 font-bold text-center">
                          No docs submitted
                        </div>
                      )}
                    </div>
                    <div className="space-y-6">
                      <h4 className="font-black uppercase tracking-widest text-slate-400 text-xs">
                        Governance Actions
                      </h4>
                      <div className="space-y-4">
                        <button
                          onClick={handleOpenSchedule}
                          className="w-full py-4 bg-indigo-500 text-white rounded-2xl font-black text-sm hover:bg-indigo-600 hover:shadow-xl transition-all flex items-center justify-center gap-2"
                        >
                          <Clock size={16} />
                          Manage Schedule
                        </button>

                        {activeTab === "pending" && (
                          <button
                            onClick={() => handleVerify("APPROVED")}
                            className="w-full py-4 bg-[#0F172A] text-white rounded-2xl font-black text-sm hover:shadow-xl transition-all"
                          >
                            Verify & Approve
                          </button>
                        )}
                        <button
                          onClick={() =>
                            handleUpdateDoctor(selectedDoctor.id, {
                              is_active: !selectedDoctor.is_active,
                            })
                          }
                          className="w-full py-4 border-2 border-slate-100 rounded-2xl font-black text-sm hover:bg-rose-50 hover:text-rose-600 transition-all"
                        >
                          {selectedDoctor.is_active
                            ? "Restrict Account"
                            : "Enable Account"}
                        </button>
                        <button
                          onClick={() => handleDeleteDoctor(selectedDoctor.id)}
                          className="w-full py-4 text-rose-500 font-black text-xs uppercase tracking-widest hover:underline"
                        >
                          Purge Record
                        </button>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </motion.div>
          </div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default AdminDoctors;
