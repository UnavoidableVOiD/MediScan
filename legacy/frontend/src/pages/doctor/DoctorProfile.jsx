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
import AvailabilityManager from "../../components/doctor/AvailabilityManager";

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
  const [experience, setExperience] = React.useState(user?.experience || 0);
  const [bio, setBio] = React.useState(user?.bio || "");
  const [isSaving, setIsSaving] = React.useState(false);

  const canManageAvailability =
    isVerified || isPending || user?.doctor_status === "UNVERIFIED";

  const [upcomingAppointments, setUpcomingAppointments] = React.useState([]);
  const [apptLoading, setApptLoading] = React.useState(false);

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

  // Fetch upcoming appointments
  React.useEffect(() => {
    const fetchAppointments = async () => {
      setApptLoading(true);
      try {
        const res = await appointmentApi.getAppointments();
        const now = new Date();
        const today = now.toISOString().split("T")[0];
        const currentTime = now.toTimeString().split(" ")[0];

        const upcoming = (res.data || []).filter((a) => {
          const isPaid = a.status === "PAID";
          const isFutureDate = a.appointment_date > today;
          const isToday = a.appointment_date === today;
          const isFutureTime = a.start_time >= currentTime;
          return isPaid && (isFutureDate || (isToday && isFutureTime));
        });
        upcoming.sort(
          (a, b) =>
            a.appointment_date.localeCompare(b.appointment_date) ||
            a.start_time.localeCompare(b.start_time),
        );
        setUpcomingAppointments(upcoming);
      } catch (err) {
        console.error("Failed to fetch appointments", err);
      } finally {
        setApptLoading(false);
      }
    };
    fetchAppointments();
  }, []);

  // Sync state with loaded user data (fixes reload issues)
  React.useEffect(() => {
    if (user) {
      if (user.consultation_fee !== undefined) setFee(user.consultation_fee);
      if (user.experience !== undefined) setExperience(user.experience);
      if (user.bio !== undefined) setBio(user.bio || "");
    }
  }, [user]);

  // Validate slots before saving
  const validateSlots = () => {
    const slotsByGroup = {};
    for (const slot of availabilityData) {
      if (!slot.date) {
        toast.error("Every slot must have a date assigned.");
        return false;
      }
      const groupKey = slot.date;
      if (!slotsByGroup[groupKey]) slotsByGroup[groupKey] = [];
      slotsByGroup[groupKey].push(slot);
    }

    for (const [date, slots] of Object.entries(slotsByGroup)) {
      const sorted = [...slots].sort((a, b) =>
        a.start_time.localeCompare(b.start_time),
      );
      for (let i = 0; i < sorted.length - 1; i++) {
        if (sorted[i].end_time > sorted[i + 1].start_time) {
          toast.error(
            `Overlap on ${date}: ${sorted[i].start_time}-${sorted[i].end_time} and ${sorted[i + 1].start_time}-${sorted[i + 1].end_time}`,
          );
          return false;
        }
      }
    }
    return true;
  };

  const handleSaveSettings = async () => {
    if (!validateSlots()) return;

    setIsSaving(true);
    try {
      // 1. Update Profile (Consultation Fee, Experience, Bio)
      await dispatch(
        updateProfile({
          consultation_fee: fee,
          experience: experience,
          bio: bio,
        }),
      ).unwrap();

      // 2. Sync Availability
      const cleanData = availabilityData.map(({ tempId, id, ...rest }) => ({
        ...rest,
        day_of_week: null, // Ensure weekly recursion is null
        // Ensure times are HH:MM:SS for backend
        start_time:
          rest.start_time.length === 5
            ? `${rest.start_time}:00`
            : rest.start_time,
        end_time:
          rest.end_time.length === 5 ? `${rest.end_time}:00` : rest.end_time,
      }));
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
    <div className="min-h-[calc(100vh-80px)] bg-neutral-background pt-32 pb-8 px-6">
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
            <div className="space-y-6 pb-8 border-b border-gray-100">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                <div className="space-y-1">
                  <h3 className="text-xl font-bold text-gray-900">
                    Professional Information
                  </h3>
                  <p className="text-sm text-gray-500 font-medium">
                    Update your consultation fee, years of experience, and bio.
                  </p>
                </div>
                <div className="flex flex-wrap items-center gap-4">
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
                  <div className="flex items-center gap-3 bg-neutral-soft p-1.5 rounded-2xl border border-gray-100">
                    <span className="px-3 text-[10px] font-black text-gray-400 uppercase tracking-widest">
                      Exp (Yrs)
                    </span>
                    <input
                      type="number"
                      value={experience}
                      onChange={(e) => setExperience(e.target.value)}
                      className="w-20 py-2 px-4 bg-white rounded-xl border-none outline-none font-bold text-medic-dark shadow-sm focus:ring-2 focus:ring-medic-dark/20 transition-all"
                    />
                  </div>
                </div>
              </div>

              <div className="space-y-2">
                <label className="text-[10px] font-black text-gray-400 uppercase tracking-widest px-2">
                  Professional Bio & Expertise
                </label>
                <textarea
                  value={bio}
                  onChange={(e) => setBio(e.target.value)}
                  placeholder="Share your expertise, work experience, and what patients can expect..."
                  className="w-full p-6 bg-neutral-soft border-transparent focus:bg-white focus:border-medic-dark/20 rounded-3xl text-sm outline-none transition-all min-h-[120px] shadow-inner"
                />
              </div>
            </div>

            <AvailabilityManager
              availabilityData={availabilityData}
              onChange={setAvailabilityData}
              onSave={handleSaveSettings}
              isSaving={isSaving}
            />
          </motion.div>
        )}

        {/* Upcoming Appointments */}
        {canManageAvailability && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-white rounded-[2.5rem] p-8 border border-gray-100 shadow-xl shadow-medic-dark/5 space-y-6"
          >
            <div className="flex items-center gap-3 pb-4 border-b border-gray-100">
              <div className="w-10 h-10 bg-blue-50 rounded-xl flex items-center justify-center text-blue-600">
                <Calendar size={20} />
              </div>
              <div>
                <h3 className="text-xl font-bold text-gray-900 leading-none">
                  Upcoming Appointments
                </h3>
                <p className="text-sm text-gray-500 font-medium mt-1">
                  {upcomingAppointments.length} confirmed appointment
                  {upcomingAppointments.length !== 1 ? "s" : ""}
                </p>
              </div>
            </div>

            {apptLoading ? (
              <div className="flex justify-center py-8">
                <Loader2 className="w-8 h-8 text-medic-dark animate-spin" />
              </div>
            ) : upcomingAppointments.length > 0 ? (
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                {upcomingAppointments.map((appt) => (
                  <div
                    key={appt.id}
                    className="p-5 bg-neutral-soft/50 rounded-3xl border border-gray-100 space-y-3"
                  >
                    <div className="flex items-center gap-3">
                      <div className="w-10 h-10 rounded-2xl bg-medic-light/30 flex items-center justify-center font-bold text-medic-dark text-sm uppercase">
                        {appt.patient_first_name?.[0] || "P"}
                      </div>
                      <div>
                        <h4 className="font-bold text-gray-900 text-sm leading-tight">
                          {appt.patient_first_name} {appt.patient_last_name}
                        </h4>
                        <p className="text-[10px] text-gray-400 font-bold">
                          {appt.patient_email}
                        </p>
                      </div>
                    </div>
                    <div className="flex items-center gap-4 text-xs">
                      <div className="flex items-center gap-1.5 text-medic-dark font-bold">
                        <Calendar size={14} />
                        {new Date(
                          appt.appointment_date + "T00:00:00",
                        ).toLocaleDateString(undefined, {
                          weekday: "short",
                          month: "short",
                          day: "numeric",
                        })}
                      </div>
                      <div className="flex items-center gap-1.5 text-gray-500 font-bold">
                        <Clock size={14} />
                        {appt.start_time?.slice(0, 5)} -{" "}
                        {appt.end_time?.slice(0, 5)}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="py-10 flex flex-col items-center justify-center gap-3 opacity-40">
                <Calendar size={36} />
                <p className="text-sm font-bold text-gray-400 uppercase tracking-widest">
                  No upcoming appointments
                </p>
              </div>
            )}
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
