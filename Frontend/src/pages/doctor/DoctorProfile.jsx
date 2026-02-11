import React from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  ShieldAlert,
  Mail,
  User,
  CheckCircle2,
  AlertCircle,
  ArrowRight,
  Clock,
  FileSearch,
  ShieldCheck,
  BadgeCheck,
  Loader2,
  Plus,
  Trash2,
  Calendar,
} from "lucide-react";
import { useNavigate } from "react-router-dom";
import { useSelector, useDispatch } from "react-redux";
import { appointmentApi } from "../../services/api";
import { updateProfile } from "../../store/slices/authSlice";
import { toast } from "react-toastify";

const days = [
  "Sunday",
  "Monday",
  "Tuesday",
  "Wednesday",
  "Thursday",
  "Friday",
  "Saturday",
];

const DoctorProfile = () => {
  const navigate = useNavigate();
  const dispatch = useDispatch();
  const { user } = useSelector((state) => state.auth);

  const doctorName = user ? `${user.first_name} ${user.last_name}` : "Doctor";
  const doctorSpecialization = user?.specialization
    ? user.specialization.replace(/_/g, " ")
    : "Physician";
  const doctorEmail = user?.email || "";

  const isPending = user?.doctor_status === "PENDING";
  const isVerified = user?.doctor_status === "VERIFIED";

  const [availabilityData, setAvailabilityData] = React.useState([]);
  const [fee, setFee] = React.useState(user?.consultation_fee || 0);
  const [isSaving, setIsSaving] = React.useState(false);
  const [activeDay, setActiveDay] = React.useState(new Date().getDay());

  const canManageAvailability =
    isVerified || isPending || user?.doctor_status === "UNVERIFIED";

  // Fetch availability on load
  React.useEffect(() => {
    const fetchCurrentAvailability = async () => {
      if (user?.id && canManageAvailability) {
        try {
          const response = await appointmentApi.getAvailability(user.id);
          setAvailabilityData(
            Array.isArray(response.data) ? response.data : [],
          );
        } catch (err) {
          console.error("Failed to fetch availability", err);
        }
      }
    };
    fetchCurrentAvailability();
  }, [user?.id, canManageAvailability]);

  // Sync state with loaded user data (fixes reload issues)
  React.useEffect(() => {
    if (user?.consultation_fee) {
      setFee(user.consultation_fee);
    }
  }, [user?.consultation_fee]);

  const addSlot = (dayIdx) => {
    // Find existing slots for this day to suggest a better time
    const daySlots = availabilityData.filter(
      (slot) => Number(slot.day_of_week) === dayIdx,
    );
    let lastEndTime = "09:00";

    if (daySlots.length > 0) {
      // Sort by start time and get the latest end time
      const lastSlot = [...daySlots]
        .sort((a, b) => a.start_time.localeCompare(b.start_time))
        .pop();
      lastEndTime = lastSlot.end_time;
    }

    const [hours, minutes] = lastEndTime.split(":").map(Number);
    const nextStartTime = `${String(hours).padStart(2, "0")}:${String(minutes).padStart(2, "0")}`;
    const nextEndTime = `${String(hours + 1 > 23 ? 23 : hours + 1).padStart(2, "0")}:${String(minutes).padStart(2, "0")}`;

    const newSlot = {
      day_of_week: dayIdx,
      start_time: nextStartTime,
      end_time: nextEndTime,
      is_active: true,
      tempId: Date.now() + Math.random(),
    };
    setAvailabilityData((prev) => [...prev, newSlot]);
  };

  const validateSlots = () => {
    const slotsByDay = {};
    for (const slot of availabilityData) {
      const day = slot.day_of_week;
      if (!slotsByDay[day]) slotsByDay[day] = [];
      slotsByDay[day].push(slot);
    }

    for (const [day, slots] of Object.entries(slotsByDay)) {
      const sorted = [...slots].sort((a, b) =>
        a.start_time.localeCompare(b.start_time),
      );
      for (let i = 0; i < sorted.length - 1; i++) {
        if (sorted[i].end_time > sorted[i + 1].start_time) {
          toast.error(
            `Overlap on ${days[day]}: ${sorted[i].start_time}-${sorted[i].end_time} and ${sorted[i + 1].start_time}-${sorted[i + 1].end_time}`,
          );
          return false;
        }
      }
    }
    return true;
  };

  const removeSlot = (slotToRemove) => {
    setAvailabilityData((prev) => prev.filter((s) => s !== slotToRemove));
  };

  const updateSlotValue = (targetSlot, field, value) => {
    setAvailabilityData((prev) =>
      prev.map((s) => (s === targetSlot ? { ...s, [field]: value } : s)),
    );
  };

  const handleSaveSettings = async () => {
    if (!validateSlots()) return;

    setIsSaving(true);
    try {
      // 1. Update Profile (Consultation Fee)
      await dispatch(updateProfile({ consultation_fee: fee })).unwrap();

      // 2. Sync Availability
      const cleanData = availabilityData.map(({ tempId, ...rest }) => rest);
      const syncResponse = await appointmentApi.syncAvailability(cleanData);

      if (syncResponse.data) {
        setAvailabilityData(
          Array.isArray(syncResponse.data) ? syncResponse.data : [],
        );
      }

      toast.success("Settings saved successfully!");
    } catch (err) {
      console.error(err);
      const errorMsg =
        err.response?.data?.error || "Failed to save some settings";
      toast.error(errorMsg);
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <div className="min-h-[calc(100vh-80px)] bg-neutral-background py-8 px-6">
      <div className="max-w-4xl mx-auto space-y-6">
        {/* Status Banners */}
        <AnimatePresence>
          {isPending && (
            <motion.div
              initial={{ opacity: 0, y: -20 }}
              animate={{ opacity: 1, y: 0 }}
              className="bg-amber-50 border border-amber-200 rounded-2xl p-4 flex items-center justify-center gap-3 text-amber-800 shadow-sm"
            >
              <Clock className="w-5 h-5 animate-pulse" />
              <p className="font-bold text-sm">
                Your clinical status is currently under review
              </p>
            </motion.div>
          )}
          {isVerified && (
            <motion.div
              initial={{ opacity: 0, y: -20 }}
              animate={{ opacity: 1, y: 0 }}
              className="bg-green-50 border border-green-200 rounded-2xl p-4 flex items-center justify-center gap-3 text-green-800 shadow-sm"
            >
              <BadgeCheck className="w-5 h-5" />
              <p className="font-bold text-sm">
                Your professional profile is verified
              </p>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Header Section */}
        <div className="text-center space-y-2">
          <h1 className="text-3xl font-bold text-gray-900 tracking-tight">
            Doctor Profile
          </h1>
          <p className="text-gray-500 max-w-lg mx-auto">
            Manage your professional details and monitor your clinical
            verification status.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          {/* Profile Card */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="md:col-span-1 bg-white rounded-3xl shadow-xl shadow-medic-dark/5 p-8 border border-medic-light/20 flex flex-col items-center text-center space-y-6"
          >
            <div className="w-32 h-32 rounded-3xl bg-medic-light/30 flex items-center justify-center overflow-hidden border-4 border-white shadow-inner">
              <User className="w-16 h-16 text-medic-dark/50" />
            </div>
            <div className="space-y-1">
              <h2 className="text-xl font-bold text-gray-900">{doctorName}</h2>
              <p className="text-medic-dark font-semibold text-sm px-3 py-1 bg-medic-light/30 rounded-full inline-block">
                {doctorSpecialization}
              </p>
            </div>
            <div className="w-full pt-4 border-t border-gray-100 flex items-center justify-center gap-2 text-gray-500 text-sm">
              <Mail className="w-4 h-4" />
              <span className="truncate">{doctorEmail}</span>
            </div>
          </motion.div>

          {/* Verification Status Card */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="md:col-span-2 bg-white rounded-3xl shadow-xl shadow-medic-dark/5 p-8 border border-medic-light/20 space-y-8 flex flex-col justify-between"
          >
            <div className="space-y-6">
              <div className="flex items-center justify-between">
                <h3 className="text-lg font-bold text-gray-900">
                  Verification Status
                </h3>
                {isVerified ? (
                  <div className="flex items-center gap-2 px-4 py-2 bg-green-50 text-green-600 rounded-2xl text-sm font-bold border border-green-100">
                    <BadgeCheck size={18} />
                    Verified
                  </div>
                ) : isPending ? (
                  <div className="flex items-center gap-2 px-4 py-2 bg-orange-50 text-orange-600 rounded-2xl text-sm font-bold border border-orange-100">
                    <Clock size={18} />
                    In Progress
                  </div>
                ) : (
                  <div className="flex items-center gap-2 px-4 py-2 bg-yellow-50 text-yellow-600 rounded-2xl text-sm font-bold border border-yellow-100">
                    <AlertCircle size={18} />
                    Verification Required
                  </div>
                )}
              </div>

              <div
                className={`${isVerified ? "bg-green-50 border-green-100" : isPending ? "bg-blue-50 border-blue-100" : "bg-orange-50 border-orange-100"} rounded-2xl p-6 border`}
              >
                <div className="flex gap-4">
                  <div className="w-12 h-12 bg-white rounded-xl shadow-sm flex items-center justify-center flex-shrink-0">
                    {isVerified ? (
                      <ShieldCheck className="w-6 h-6 text-green-500" />
                    ) : (
                      <FileSearch className="w-6 h-6 text-blue-500" />
                    )}
                  </div>
                  <div className="space-y-1">
                    <h4
                      className={`font-bold ${isVerified ? "text-green-900" : isPending ? "text-blue-900" : "text-yellow-900"}`}
                    >
                      {isVerified
                        ? "Approval Confirmed"
                        : isPending
                          ? "Under Review"
                          : "Missing Credentials"}
                    </h4>
                    <p className="text-sm text-gray-600 leading-relaxed">
                      {isVerified
                        ? "Your credentials have been verified. You now have full clinical access."
                        : isPending
                          ? "We are currently validating your medical license. This usually takes 24-48 hours."
                          : "Please upload your medical license and certificates to gain clinical access."}
                    </p>
                  </div>
                </div>
              </div>

              {isVerified ? (
                <button
                  onClick={() => navigate("/doctor-dashboard")}
                  className="w-full bg-medic-dark text-white py-4 rounded-2xl font-bold shadow-lg shadow-medic-dark/20 hover:bg-medic-primary transition-all flex items-center justify-center gap-3 group"
                >
                  <span>Go to Dashboard</span>
                  <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
                </button>
              ) : (
                !isPending && (
                  <button
                    onClick={() => navigate("/verify-doctor")}
                    className="w-full bg-medic-dark text-white py-4 rounded-2xl font-bold shadow-lg shadow-medic-dark/20 hover:bg-medic-primary transition-all flex items-center justify-center gap-3 group"
                  >
                    <span>Submit Credentials</span>
                    <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
                  </button>
                )
              )}
            </div>
          </motion.div>
        </div>

        {/* Consultation & Availability */}
        {canManageAvailability && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-white rounded-[2.5rem] p-8 border border-gray-100 shadow-xl shadow-medic-dark/5 space-y-8"
          >
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-6 border-b border-gray-100">
              <div className="space-y-1">
                <h3 className="text-xl font-bold text-gray-900">
                  Consultation Settings
                </h3>
                <p className="text-sm text-gray-500 font-medium">
                  Manage your clinical fees and patient time slots.
                </p>
              </div>
              <div className="flex items-center gap-3 bg-neutral-soft p-1.5 rounded-2xl border border-gray-100">
                <span className="px-3 text-[10px] font-black text-gray-400 uppercase tracking-widest">
                  Fee (Rs.)
                </span>
                <input
                  type="number"
                  value={fee}
                  onChange={(e) => setFee(e.target.value)}
                  className="w-24 py-2 px-4 bg-white rounded-xl border-none outline-none font-bold text-medic-dark shadow-sm focus:ring-2 focus:ring-medic-dark/20 transition-all"
                />
              </div>
            </div>

            <div className="space-y-8">
              {/* Day Selection Tabs */}
              <div className="flex gap-2 p-1 bg-neutral-soft rounded-2xl overflow-x-auto no-scrollbar">
                {days.map((dayName, dayIdx) => (
                  <button
                    key={dayName}
                    onClick={() => setActiveDay(dayIdx)}
                    className={`px-6 py-2.5 rounded-xl text-xs font-bold transition-all whitespace-nowrap ${
                      activeDay === dayIdx
                        ? "bg-medic-dark text-white shadow-lg shadow-medic-dark/20"
                        : "text-gray-500 hover:text-medic-dark hover:bg-white"
                    }`}
                  >
                    {dayName}
                  </button>
                ))}
              </div>

              {/* Active Day Section */}
              <div className="space-y-6">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 bg-medic-light/20 rounded-xl flex items-center justify-center text-medic-dark">
                      <Calendar size={20} />
                    </div>
                    <div>
                      <h4 className="font-bold text-gray-900 leading-none">
                        {days[activeDay]} Schedule
                      </h4>
                      <p className="text-[10px] text-gray-400 font-bold uppercase tracking-widest mt-1">
                        {
                          availabilityData.filter(
                            (s) => Number(s.day_of_week) === activeDay,
                          ).length
                        }{" "}
                        Slots Configured
                      </p>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    {availabilityData.filter(
                      (s) => Number(s.day_of_week) === activeDay,
                    ).length > 0 && (
                      <button
                        onClick={() => {
                          if (
                            window.confirm(
                              `Clear all slots for ${days[activeDay]}?`,
                            )
                          ) {
                            setAvailabilityData((prev) =>
                              prev.filter(
                                (s) => Number(s.day_of_week) !== activeDay,
                              ),
                            );
                          }
                        }}
                        className="p-2 text-red-400 hover:text-red-500 hover:bg-red-50 rounded-xl transition-all"
                        title="Clear all for this day"
                      >
                        <Trash2 size={18} />
                      </button>
                    )}
                    <button
                      onClick={() => addSlot(activeDay)}
                      className="flex items-center gap-2 px-5 py-2.5 bg-medic-dark text-white text-xs font-bold rounded-xl hover:bg-medic-primary transition-all shadow-lg shadow-medic-dark/10"
                    >
                      <Plus size={16} />
                      Add New Slot
                    </button>
                  </div>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                  {availabilityData.filter(
                    (s) => Number(s.day_of_week) === activeDay,
                  ).length > 0 ? (
                    availabilityData
                      .filter((s) => Number(s.day_of_week) === activeDay)
                      .map((slot, idx) => (
                        <div
                          key={slot.id || slot.tempId || idx}
                          className="p-5 bg-white rounded-3xl border border-gray-100 shadow-sm hover:shadow-md transition-all group relative flex flex-col gap-4"
                        >
                          <div className="flex items-center justify-between">
                            {slot.id ? (
                              <span className="flex items-center gap-1.5 text-[10px] font-black text-green-600 bg-green-50 px-3 py-1 rounded-full uppercase tracking-widest border border-green-100">
                                <BadgeCheck size={12} />
                                Active Slot
                              </span>
                            ) : (
                              <span className="flex items-center gap-1.5 text-[10px] font-black text-amber-600 bg-amber-50 px-3 py-1 rounded-full uppercase tracking-widest border border-amber-100 italic">
                                <Clock size={12} />
                                Pending Sync
                              </span>
                            )}
                            <button
                              onClick={() => removeSlot(slot)}
                              className="p-1.5 text-gray-300 hover:text-red-500 hover:bg-red-50 rounded-lg transition-all"
                            >
                              <Trash2 size={14} />
                            </button>
                          </div>

                          <div className="flex items-center gap-3">
                            <div className="flex-1 space-y-1.5">
                              <label className="text-[10px] font-bold text-gray-400 uppercase tracking-widest px-1">
                                From
                              </label>
                              <input
                                type="time"
                                value={slot.start_time.slice(0, 5)}
                                onChange={(e) =>
                                  updateSlotValue(
                                    slot,
                                    "start_time",
                                    e.target.value,
                                  )
                                }
                                className="w-full p-2.5 bg-neutral-soft rounded-xl text-xs font-bold border border-transparent focus:border-medic-dark/20 text-medic-dark outline-none transition-all"
                              />
                            </div>
                            <div className="pt-6">
                              <ArrowRight size={14} className="text-gray-300" />
                            </div>
                            <div className="flex-1 space-y-1.5">
                              <label className="text-[10px] font-bold text-gray-400 uppercase tracking-widest px-1">
                                Until
                              </label>
                              <input
                                type="time"
                                value={slot.end_time.slice(0, 5)}
                                onChange={(e) =>
                                  updateSlotValue(
                                    slot,
                                    "end_time",
                                    e.target.value,
                                  )
                                }
                                className="w-full p-2.5 bg-neutral-soft rounded-xl text-xs font-bold border border-transparent focus:border-medic-dark/20 text-medic-dark outline-none transition-all"
                              />
                            </div>
                          </div>
                        </div>
                      ))
                  ) : (
                    <div className="col-span-full py-12 flex flex-col items-center justify-center gap-3 bg-neutral-soft/50 rounded-[2rem] border-2 border-dashed border-gray-200">
                      <div className="w-12 h-12 bg-white rounded-2xl flex items-center justify-center text-gray-300">
                        <Clock size={24} />
                      </div>
                      <div className="text-center">
                        <p className="text-sm font-bold text-gray-400 uppercase tracking-widest leading-none">
                          No Slots Configured
                        </p>
                        <p className="text-xs text-gray-400 mt-2">
                          Add availability to appear in patient searches for{" "}
                          {days[activeDay]}.
                        </p>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>

            <div className="pt-4">
              <button
                disabled={isSaving}
                className="w-full py-5 bg-medic-dark text-white rounded-3xl font-bold text-lg hover:bg-medic-primary transition-all shadow-xl shadow-medic-dark/10 flex items-center justify-center gap-3 group disabled:opacity-50"
                onClick={handleSaveSettings}
              >
                {isSaving ? (
                  <>
                    <Loader2 className="w-6 h-6 animate-spin" />
                    <span>Synchronizing Schedule...</span>
                  </>
                ) : (
                  <>
                    <span>Confirm & Sync Schedule</span>
                    <ShieldCheck
                      size={22}
                      className="group-hover:scale-110 transition-transform"
                    />
                  </>
                )}
              </button>
              <p className="text-[10px] text-gray-400 text-center mt-4 font-bold uppercase tracking-widest">
                Updates will be reflected instantly across the patient search
                results
              </p>
            </div>
          </motion.div>
        )}
        <div className="flex items-center justify-center gap-8 py-4 opacity-50 grayscale">
          <div className="flex items-center gap-2">
            <ShieldAlert size={20} />
            <span className="text-sm font-medium">HIPAA Compliant</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DoctorProfile;
