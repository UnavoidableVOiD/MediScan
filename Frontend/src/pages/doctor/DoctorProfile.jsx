import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
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
    Calendar
} from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { useSelector, useDispatch } from 'react-redux';
import { appointmentApi } from '../../services/api';
import { updateProfile } from '../../store/slices/authSlice';
import { toast } from 'react-toastify';

const days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'];

const DoctorProfile = () => {
    const navigate = useNavigate();
    const dispatch = useDispatch();
    const { user } = useSelector(state => state.auth);

    const doctorName = user ? `${user.first_name} ${user.last_name}` : "Doctor";
    const doctorSpecialization = user?.specialization ? user.specialization.replace(/_/g, ' ') : "Physician";
    const doctorEmail = user?.email || "";

    const isPending = user?.doctor_status === 'PENDING';
    const isVerified = user?.doctor_status === 'VERIFIED';

    const [availabilityData, setAvailabilityData] = React.useState([]);
    const [fee, setFee] = React.useState(user?.consultation_fee || 0);
    const [isSaving, setIsSaving] = React.useState(false);

    const canManageAvailability = isVerified || isPending;

    // Fetch availability on load
    React.useEffect(() => {
        const fetchCurrentAvailability = async () => {
            if (user?.id && canManageAvailability) {
                try {
                    const response = await appointmentApi.getAvailability(user.id);
                    setAvailabilityData(Array.isArray(response.data) ? response.data : []);
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
        const daySlots = availabilityData.filter(slot => Number(slot.day_of_week) === dayIdx);
        let lastEndTime = "09:00";

        if (daySlots.length > 0) {
            // Sort by start time and get the latest end time
            const lastSlot = [...daySlots].sort((a, b) => a.start_time.localeCompare(b.start_time)).pop();
            lastEndTime = lastSlot.end_time;
        }

        const [hours, minutes] = lastEndTime.split(':').map(Number);
        const nextStartTime = `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}`;
        const nextEndTime = `${String(hours + 1 > 23 ? 23 : hours + 1).padStart(2, '0')}:${String(minutes).padStart(2, '0')}`;

        const newSlot = {
            day_of_week: dayIdx,
            start_time: nextStartTime,
            end_time: nextEndTime,
            is_active: true,
            tempId: Date.now() + Math.random()
        };
        setAvailabilityData(prev => [...prev, newSlot]);
    };

    const validateSlots = () => {
        const slotsByDay = {};
        for (const slot of availabilityData) {
            const day = slot.day_of_week;
            if (!slotsByDay[day]) slotsByDay[day] = [];
            slotsByDay[day].push(slot);
        }

        for (const [day, slots] of Object.entries(slotsByDay)) {
            const sorted = [...slots].sort((a, b) => a.start_time.localeCompare(b.start_time));
            for (let i = 0; i < sorted.length - 1; i++) {
                if (sorted[i].end_time > sorted[i + 1].start_time) {
                    toast.error(`Overlap on ${days[day]}: ${sorted[i].start_time}-${sorted[i].end_time} and ${sorted[i + 1].start_time}-${sorted[i + 1].end_time}`);
                    return false;
                }
            }
        }
        return true;
    };

    const removeSlot = (slotToRemove) => {
        setAvailabilityData(prev => prev.filter(s => s !== slotToRemove));
    };

    const updateSlotValue = (targetSlot, field, value) => {
        setAvailabilityData(prev => prev.map(s =>
            s === targetSlot ? { ...s, [field]: value } : s
        ));
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
                setAvailabilityData(Array.isArray(syncResponse.data) ? syncResponse.data : []);
            }

            toast.success("Settings saved successfully!");
        } catch (err) {
            console.error(err);
            const errorMsg = err.response?.data?.error || "Failed to save some settings";
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
                            <p className="font-bold text-sm">Your clinical status is currently under review</p>
                        </motion.div>
                    )}
                    {isVerified && (
                        <motion.div
                            initial={{ opacity: 0, y: -20 }}
                            animate={{ opacity: 1, y: 0 }}
                            className="bg-green-50 border border-green-200 rounded-2xl p-4 flex items-center justify-center gap-3 text-green-800 shadow-sm"
                        >
                            <BadgeCheck className="w-5 h-5" />
                            <p className="font-bold text-sm">Your professional profile is verified</p>
                        </motion.div>
                    )}
                </AnimatePresence>

                {/* Header Section */}
                <div className="text-center space-y-2">
                    <h1 className="text-3xl font-bold text-gray-900 tracking-tight">
                        Doctor Profile
                    </h1>
                    <p className="text-gray-500 max-w-lg mx-auto">
                        Manage your professional details and monitor your clinical verification status.
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
                                <h3 className="text-lg font-bold text-gray-900">Verification Status</h3>
                                {isVerified ? (
                                    <div className="flex items-center gap-2 px-4 py-2 bg-green-50 text-green-600 rounded-2xl text-sm font-bold border border-green-100">
                                        <BadgeCheck size={18} />
                                        Verified
                                    </div>
                                ) : (
                                    isPending ? (
                                        <div className="flex items-center gap-2 px-4 py-2 bg-orange-50 text-orange-600 rounded-2xl text-sm font-bold border border-orange-100">
                                            <Clock size={18} />
                                            In Progress
                                        </div>
                                    ) : (
                                        <div className="flex items-center gap-2 px-4 py-2 bg-red-50 text-red-600 rounded-2xl text-sm font-bold border border-red-100">
                                            <AlertCircle size={18} />
                                            Unverified
                                        </div>
                                    )
                                )}
                            </div>

                            <div className={`${isVerified ? 'bg-green-50 border-green-100' : (isPending ? 'bg-blue-50 border-blue-100' : 'bg-orange-50 border-orange-100')} rounded-2xl p-6 border`}>
                                <div className="flex gap-4">
                                    <div className="w-12 h-12 bg-white rounded-xl shadow-sm flex items-center justify-center flex-shrink-0">
                                        {isVerified ? <ShieldCheck className="w-6 h-6 text-green-500" /> : <FileSearch className="w-6 h-6 text-blue-500" />}
                                    </div>
                                    <div className="space-y-1">
                                        <h4 className={`font-bold ${isVerified ? 'text-green-900' : 'text-blue-900'}`}>
                                            {isVerified ? "Approval Confirmed" : "Under Review"}
                                        </h4>
                                        <p className="text-sm text-gray-600 leading-relaxed">
                                            {isVerified
                                                ? "Your credentials have been verified. You now have full clinical access."
                                                : "We are currently validating your medical license. This usually takes 24-48 hours."}
                                        </p>
                                    </div>
                                </div>
                            </div>

                            {isVerified && (
                                <button
                                    onClick={() => navigate('/doctor-dashboard')}
                                    className="w-full bg-medic-dark text-white py-4 rounded-2xl font-bold shadow-lg shadow-medic-dark/20 hover:bg-medic-primary transition-all flex items-center justify-center gap-3 group"
                                >
                                    <span>Go to Dashboard</span>
                                    <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
                                </button>
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
                        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                            <div className="space-y-1">
                                <h3 className="text-xl font-bold text-gray-900">Consultation Settings</h3>
                                <p className="text-sm text-gray-500 font-medium">Manage your clinical fees and patient time slots.</p>
                            </div>
                            <div className="flex items-center gap-3 bg-neutral-soft p-1 rounded-2xl border border-gray-100">
                                <span className="px-4 text-xs font-bold text-gray-400 uppercase tracking-widest">Fee (Rs.)</span>
                                <input
                                    type="number"
                                    value={fee}
                                    onChange={(e) => setFee(e.target.value)}
                                    className="w-24 py-2 px-4 bg-white rounded-xl border-none outline-none font-bold text-medic-dark shadow-sm"
                                />
                            </div>
                        </div>

                        <div className="space-y-8">
                            {days.map((dayName, dayIdx) => {
                                const daySlots = availabilityData.filter(slot => Number(slot.day_of_week) === dayIdx);
                                return (
                                    <div key={dayName} className="space-y-4">
                                        <div className="flex items-center justify-between">
                                            <h4 className="font-bold text-gray-700 flex items-center gap-2">
                                                <div className="w-2 h-2 rounded-full bg-medic-dark"></div>
                                                {dayName}
                                            </h4>
                                            <button
                                                onClick={() => addSlot(dayIdx)}
                                                className="flex items-center gap-1.5 text-xs font-black text-medic-dark hover:text-medic-primary transition-colors bg-medic-light/20 px-3 py-1.5 rounded-xl uppercase tracking-wider"
                                            >
                                                <Plus size={14} />
                                                Add Slot
                                            </button>
                                        </div>

                                        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                                            {daySlots.length > 0 ? (
                                                daySlots.map((slot, idx) => (
                                                    <div key={slot.id || slot.tempId || idx} className="p-4 bg-neutral-soft/30 rounded-2xl border border-gray-100 flex flex-col gap-3 group relative">
                                                        <div className="flex items-center justify-between">
                                                            {slot.id ? (
                                                                <span className="flex items-center gap-1 text-[9px] font-black text-green-600 bg-green-50 px-2 py-0.5 rounded-full uppercase tracking-tighter">
                                                                    <CheckCircle2 size={10} />
                                                                    Persisted
                                                                </span>
                                                            ) : (
                                                                <span className="flex items-center gap-1 text-[9px] font-black text-amber-600 bg-amber-50 px-2 py-0.5 rounded-full uppercase tracking-tighter">
                                                                    <Clock size={10} />
                                                                    Unsaved
                                                                </span>
                                                            )}
                                                        </div>
                                                        <div className="flex items-center gap-2 text-gray-600">
                                                            <div className="flex-1 flex flex-col gap-1">
                                                                <span className="text-[9px] font-bold text-gray-400 uppercase">Start</span>
                                                                <input
                                                                    type="time"
                                                                    value={slot.start_time.slice(0, 5)}
                                                                    onChange={(e) => updateSlotValue(slot, 'start_time', e.target.value)}
                                                                    className="p-2 bg-white rounded-lg text-xs font-bold outline-none border border-transparent focus:border-medic-dark transition-all"
                                                                />
                                                            </div>
                                                            <div className="pt-4"><ArrowRight size={10} className="text-gray-300" /></div>
                                                            <div className="flex-1 flex flex-col gap-1">
                                                                <span className="text-[9px] font-bold text-gray-400 uppercase">End</span>
                                                                <input
                                                                    type="time"
                                                                    value={slot.end_time.slice(0, 5)}
                                                                    onChange={(e) => updateSlotValue(slot, 'end_time', e.target.value)}
                                                                    className="p-2 bg-white rounded-lg text-xs font-bold outline-none border border-transparent focus:border-medic-dark transition-all"
                                                                />
                                                            </div>
                                                        </div>
                                                        <button
                                                            onClick={() => removeSlot(slot)}
                                                            className="absolute -top-2 -right-2 w-7 h-7 bg-white text-red-500 rounded-xl shadow-md flex items-center justify-center hover:bg-red-500 hover:text-white transition-all opacity-0 group-hover:opacity-100 z-10"
                                                        >
                                                            <Trash2 size={14} />
                                                        </button>
                                                    </div>
                                                ))
                                            ) : (
                                                <div className="col-span-full py-4 text-center text-[10px] text-gray-400 font-bold uppercase tracking-widest bg-gray-50/50 rounded-2xl border border-dashed border-gray-200">
                                                    No slots configured for {dayName}
                                                </div>
                                            )}
                                        </div>
                                    </div>
                                );
                            })}
                        </div>

                        <button
                            disabled={isSaving}
                            className="w-full py-4 bg-medic-dark text-white rounded-2xl font-bold hover:bg-medic-primary transition-all shadow-lg shadow-medic-dark/10 flex items-center justify-center gap-2 group disabled:opacity-50"
                            onClick={handleSaveSettings}
                        >
                            {isSaving ? <Loader2 className="w-5 h-5 animate-spin" /> : <span>Sync Clinical Schedule</span>}
                        </button>
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
