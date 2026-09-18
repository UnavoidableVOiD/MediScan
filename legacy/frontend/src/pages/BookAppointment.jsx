import React, { useEffect, useState, useMemo } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Calendar,
  Clock,
  ArrowRight,
  ShieldCheck,
  Loader2,
  ChevronLeft,
  AlertCircle,
  BadgeCheck,
  Stethoscope,
} from "lucide-react";
import { useParams, useNavigate, Link } from "react-router-dom";
import { useDispatch, useSelector } from "react-redux";
import { appointmentApi } from "../services/api";
import {
  bookAppointment,
  initiateKhaltiPayment,
  fetchAvailability,
  fetchAppointments,
} from "../store/slices/appointmentSlice";
import { toast } from "react-toastify";

const BookAppointment = () => {
  const { id } = useParams();
  const navigate = useNavigate();
  const dispatch = useDispatch();

  const {
    recommendedDoctors,
    currentDoctor,
    availability,
    availabilityLoading,
    bookingLoading,
    appointments,
  } = useSelector((state) => state.appointment);
  const { user } = useSelector((state) => state.auth);

  const [selectedDoctor, setSelectedDoctor] = useState(null);
  const [selectedDate, setSelectedDate] = useState("");
  const [selectedSlot, setSelectedSlot] = useState(null);
  const [appointmentNote, setAppointmentNote] = useState("");
  const [bookedSlots, setBookedSlots] = useState([]);

  // Find the doctor from recommendedDoctors or fetch if not available
  useEffect(() => {
    const doc = recommendedDoctors.find((d) => d.id === parseInt(id));
    if (doc) {
      setSelectedDoctor(doc);
    } else if (currentDoctor && currentDoctor.id === parseInt(id)) {
      setSelectedDoctor(currentDoctor);
    } else {
      import("../store/slices/appointmentSlice").then((module) => {
        dispatch(module.fetchDoctorById(id));
      });
    }
  }, [id, recommendedDoctors, currentDoctor, dispatch]);

  useEffect(() => {
    if (id) {
      dispatch(fetchAvailability(id));
      dispatch(fetchAppointments());
    }
  }, [dispatch, id]);

  useEffect(() => {
    if (selectedDate && id) {
      setSelectedSlot(null);
      appointmentApi
        .getBookedSlots(id, selectedDate)
        .then((res) => setBookedSlots(Array.isArray(res.data) ? res.data : []))
        .catch(() => setBookedSlots([]));
    }
  }, [id, selectedDate]);

  const hasActiveAppointment = useMemo(() => {
    return appointments.some((appt) =>
      ["PAID", "PENDING"].includes(appt.status),
    );
  }, [appointments]);

  const dayOfWeekMap = { 0: 6, 1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5 };
  const selectedDayOfWeek = selectedDate
    ? dayOfWeekMap[new Date(selectedDate + "T00:00:00").getDay()]
    : null;

  const filteredSlots = useMemo(() => {
    if (!selectedDate || availability.length === 0) return [];

    const dateSpecificSlots = availability.filter(
      (s) => s.date === selectedDate,
    );
    if (dateSpecificSlots.length > 0) return dateSpecificSlots;

    return availability.filter(
      (s) => Number(s.day_of_week) === selectedDayOfWeek && !s.date,
    );
  }, [selectedDate, availability, selectedDayOfWeek]);

  const isSlotBooked = (slot) =>
    bookedSlots.some(
      (b) =>
        b.start_time.slice(0, 5) === slot.start_time.slice(0, 5) &&
        b.end_time.slice(0, 5) === slot.end_time.slice(0, 5),
    );

  const todayStr = new Date().toISOString().split("T")[0];

  const handleBooking = async () => {
    if (!selectedDate) return toast.warning("Please select a date");
    if (!selectedSlot) return toast.warning("Please select a time slot");

    try {
      const apptData = {
        doctor: id,
        appointment_date: selectedDate,
        start_time: selectedSlot.start_time,
        end_time: selectedSlot.end_time,
        notes: appointmentNote,
      };

      const appt = await dispatch(bookAppointment(apptData)).unwrap();

      const paymentData = {
        appointmentId: appt.id,
        returnUrl: `${window.location.origin}/payment/success`,
        websiteUrl: window.location.origin,
      };

      const khaltiResponse = await dispatch(
        initiateKhaltiPayment(paymentData),
      ).unwrap();

      if (khaltiResponse.payment_url) {
        window.location.href = khaltiResponse.payment_url;
      } else {
        toast.error("Failed to get payment URL from Khalti");
      }
    } catch (error) {
      console.error("Booking error:", error);
    }
  };

  if (!selectedDoctor && !availabilityLoading) {
    // If doctor not found in recommended list, we should fetch it.
    // However, if we don't have a fetchDoctor endpoint, we might have issues.
    // Let's at least show a message or try to fetch availability which confirms doctor existence.
  }

  return (
    <div className="max-w-4xl mx-auto px-6 pt-32 pb-10 space-y-8">
      <Link
        to="/doctors"
        className="inline-flex items-center gap-2 text-[10px] font-black text-gray-400 uppercase tracking-widest hover:text-medic-dark transition-colors"
      >
        <ChevronLeft size={14} /> Back to Search
      </Link>

      <div className="bg-white rounded-[2.5rem] shadow-xl border border-gray-100 overflow-hidden">
        <div className="p-8 md:p-12 space-y-10">
          {/* Doctor Header */}
          <div className="flex flex-col md:flex-row gap-8 items-center md:items-start text-center md:text-left">
            <div className="w-24 h-24 bg-medic-dark rounded-[2rem] flex items-center justify-center text-white text-3xl font-bold shadow-lg shadow-medic-dark/20">
              {selectedDoctor?.first_name?.[0]}
              {selectedDoctor?.last_name?.[0]}
            </div>
            <div className="space-y-3 flex-1">
              <div className="flex flex-col md:flex-row md:items-center gap-3">
                <h1 className="text-3xl font-bold text-gray-900">
                  Dr. {selectedDoctor?.first_name} {selectedDoctor?.last_name}
                </h1>
                <span className="inline-flex items-center gap-1.5 px-3 py-1 bg-green-50 text-green-600 rounded-full text-[10px] font-black uppercase tracking-widest border border-green-100">
                  <BadgeCheck size={12} /> Verified Specialist
                </span>
              </div>
              <div className="flex flex-wrap justify-center md:justify-start items-center gap-x-6 gap-y-2">
                <div className="flex items-center gap-2 text-medic-dark font-bold text-xs uppercase tracking-widest">
                  <Stethoscope size={16} />
                  {selectedDoctor?.specialization?.replace("_", " ")}
                </div>
                <div className="flex items-center gap-2 text-gray-500 font-bold text-xs uppercase tracking-widest">
                  <Clock size={16} />
                  {selectedDoctor?.experience || 0}+ Years Experience
                </div>
              </div>
            </div>
            <div className="text-right">
              <span className="block text-[10px] font-black text-gray-400 uppercase tracking-widest mb-1">
                Consultation Fee
              </span>
              <span className="text-2xl font-black text-medic-dark">
                Rs. {selectedDoctor?.consultation_fee}
              </span>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 pt-8 border-t border-gray-100">
            {/* Left: Scheduling */}
            <div className="space-y-8">
              <div className="space-y-4">
                <div className="flex items-center gap-3 mb-2">
                  <Calendar size={20} className="text-medic-dark" />
                  <h3 className="text-lg font-bold text-gray-900">
                    Select Date
                  </h3>
                </div>
                <input
                  type="date"
                  value={selectedDate}
                  min={todayStr}
                  onChange={(e) => setSelectedDate(e.target.value)}
                  className="w-full p-4 bg-neutral-soft rounded-2xl border border-transparent focus:bg-white focus:border-medic-dark/20 text-sm font-bold text-medic-dark outline-none transition-all shadow-inner"
                />
              </div>

              <div className="space-y-4">
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-3">
                    <Clock size={20} className="text-medic-dark" />
                    <h3 className="text-lg font-bold text-gray-900">
                      Available Slots
                    </h3>
                  </div>
                  {selectedDate && (
                    <span className="text-[10px] font-black text-gray-400 uppercase tracking-widest">
                      {
                        [
                          "Monday",
                          "Tuesday",
                          "Wednesday",
                          "Thursday",
                          "Friday",
                          "Saturday",
                          "Sunday",
                        ][selectedDayOfWeek]
                      }
                    </span>
                  )}
                </div>

                <div className="space-y-6">
                  {!selectedDate ? (
                    <div className="py-8 text-center bg-neutral-soft/50 rounded-2xl border-2 border-dashed border-gray-100">
                      <p className="text-xs text-gray-400 font-bold uppercase tracking-widest italic">
                        Please select a date first
                      </p>
                    </div>
                  ) : availabilityLoading ? (
                    <div className="py-8 flex justify-center">
                      <Loader2 className="w-6 h-6 animate-spin text-medic-dark" />
                    </div>
                  ) : filteredSlots.length > 0 ? (
                    ["Morning", "Afternoon", "Evening"].map((period) => {
                      const periodSlots = filteredSlots
                        .filter((slot) => {
                          const hour = parseInt(slot.start_time.split(":")[0]);
                          if (period === "Morning") return hour < 12;
                          if (period === "Afternoon")
                            return hour >= 12 && hour < 17;
                          return hour >= 17;
                        })
                        .sort((a, b) =>
                          a.start_time.localeCompare(b.start_time),
                        );

                      if (periodSlots.length === 0) return null;

                      return (
                        <div key={period} className="space-y-3">
                          <h5 className="text-[10px] font-black text-gray-400 uppercase tracking-widest pl-1">
                            {period}
                          </h5>
                          <div className="flex flex-wrap gap-3">
                            {periodSlots.map((slot, i) => {
                              const booked = isSlotBooked(slot);
                              const isSelected = selectedSlot === slot;
                              return (
                                <button
                                  key={i}
                                  disabled={booked}
                                  onClick={() => setSelectedSlot(slot)}
                                  className={`px-4 py-3 rounded-2xl text-xs font-bold transition-all border flex items-center gap-2 ${
                                    booked
                                      ? "bg-gray-50 text-gray-300 border-gray-100 cursor-not-allowed decoration-slice"
                                      : isSelected
                                        ? "bg-medic-dark text-white border-medic-dark shadow-lg shadow-medic-dark/20 scale-105"
                                        : "bg-white text-gray-600 border-gray-200 hover:border-medic-dark hover:text-medic-dark"
                                  }`}
                                >
                                  {booked ? (
                                    <BadgeCheck
                                      size={14}
                                      className="opacity-50"
                                    />
                                  ) : (
                                    <Clock
                                      size={14}
                                      className={
                                        isSelected
                                          ? "text-white"
                                          : "text-medic-primary"
                                      }
                                    />
                                  )}
                                  <span>
                                    {slot.start_time.slice(0, 5)} -{" "}
                                    {slot.end_time.slice(0, 5)}
                                  </span>
                                </button>
                              );
                            })}
                          </div>
                        </div>
                      );
                    })
                  ) : (
                    <div className="py-8 text-center bg-red-50/50 rounded-2xl border-2 border-dashed border-red-100">
                      <p className="text-xs text-red-400 font-bold uppercase tracking-widest italic">
                        No slots available
                      </p>
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Right: Notes & Confirm */}
            <div className="space-y-8">
              <div className="space-y-4">
                <div className="flex items-center gap-3 mb-2">
                  <AlertCircle size={20} className="text-medic-dark" />
                  <h3 className="text-lg font-bold text-gray-900">
                    Appointment Notes
                  </h3>
                </div>
                <textarea
                  value={appointmentNote}
                  onChange={(e) => setAppointmentNote(e.target.value)}
                  placeholder="Tell us about your symptoms or medical history related to this visit..."
                  className="w-full p-6 bg-neutral-soft border-transparent focus:bg-white focus:border-medic-dark/20 rounded-3xl text-sm outline-none transition-all min-h-[160px] shadow-inner"
                />
              </div>

              {hasActiveAppointment && (
                <div className="p-4 bg-amber-50 border border-amber-200 rounded-2xl flex items-start gap-3">
                  <AlertCircle className="w-5 h-5 text-amber-500 shrink-0 mt-0.5" />
                  <p className="text-xs text-amber-800 font-bold leading-relaxed">
                    You already have an active appointment. Please visit your
                    dashboard to manage existing bookings.
                  </p>
                </div>
              )}

              <button
                onClick={handleBooking}
                disabled={bookingLoading || !selectedSlot}
                className="w-full py-5 bg-medic-dark text-white rounded-3xl font-black text-lg hover:bg-medic-primary transition-all shadow-xl shadow-medic-dark/20 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-4 group"
              >
                {bookingLoading ? (
                  <>
                    <Loader2 className="w-6 h-6 animate-spin" />
                    <span>Processing...</span>
                  </>
                ) : (
                  <>
                    <span>Confirm & Pay with Khalti</span>
                    <ArrowRight className="group-hover:translate-x-1 transition-transform" />
                  </>
                )}
              </button>

              <div className="flex items-center justify-center gap-3 py-2 grayscale opacity-50">
                <ShieldCheck size={20} />
                <span className="text-[10px] font-black uppercase tracking-widest">
                  Secured Health Data
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default BookAppointment;
