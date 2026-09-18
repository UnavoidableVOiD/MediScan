import React, { useState, useMemo } from "react";
import { Clock, Trash2, Plus, Calendar, Save, Tag } from "lucide-react";
import { toast } from "react-toastify";

const AvailabilityManager = ({
  availabilityData,
  onChange,
  onSave,
  isSaving,
}) => {
  const [selectedDate, setSelectedDate] = useState(
    new Date().toISOString().split("T")[0],
  );

  const activeDates = useMemo(() => {
    const dates = availabilityData.filter((s) => s.date).map((s) => s.date);
    return [...new Set(dates)].sort();
  }, [availabilityData]);

  const getTimeSlotsSummary = (date) => {
    const slots = availabilityData.filter((s) => s.date === date);
    if (slots.length === 0) return "No slots";
    return slots
      .sort((a, b) => a.start_time.localeCompare(b.start_time))
      .map((s) => `${s.start_time.slice(0, 5)}-${s.end_time.slice(0, 5)}`)
      .join(", ");
  };

  const addSlot = (date) => {
    const daySlots = availabilityData.filter((slot) => slot.date === date);
    let lastEndTime = "09:00";

    if (daySlots.length > 0) {
      const lastSlot = [...daySlots]
        .sort((a, b) => a.start_time.localeCompare(b.start_time))
        .pop();
      lastEndTime = lastSlot.end_time;
    }

    const [hours, minutes] = lastEndTime.split(":").map(Number);
    const nextStartTime = `${String(hours).padStart(2, "0")}:${String(minutes).padStart(2, "0")}`;
    const nextEndTime = `${String(hours + 1 > 23 ? 23 : hours + 1).padStart(2, "0")}:${String(minutes).padStart(2, "0")}`;

    const newSlot = {
      date: date,
      start_time: nextStartTime,
      end_time: nextEndTime,
      label: "",
      is_active: true,
      tempId: Date.now() + Math.random(),
    };
    onChange([...availabilityData, newSlot]);
  };

  const removeSlot = (slotToRemove) => {
    onChange(availabilityData.filter((s) => s !== slotToRemove));
  };

  const updateSlotValue = (targetSlot, field, value) => {
    onChange(
      availabilityData.map((s) =>
        s === targetSlot ? { ...s, [field]: value } : s,
      ),
    );
  };

  const currentSlots = availabilityData.filter((s) => s.date === selectedDate);

  return (
    <div className="space-y-8">
      {/* Date-Only Schedule Overview */}
      <div className="bg-neutral-soft/30 rounded-3xl p-6 border border-gray-50 space-y-4">
        <div className="flex items-center gap-2 mb-2">
          <div className="w-1.5 h-6 bg-medic-dark rounded-full" />
          <h4 className="text-sm font-bold text-gray-900 uppercase tracking-wider">
            Clinical Availability by Date
          </h4>
        </div>

        <div className="flex flex-col gap-4">
          <div className="flex items-center gap-3">
            <span className="text-[10px] font-black text-gray-400 uppercase tracking-widest">
              Manage Date:
            </span>
            <input
              type="date"
              value={selectedDate}
              min={new Date().toISOString().split("T")[0]}
              onChange={(e) => setSelectedDate(e.target.value)}
              className="p-2.5 bg-white rounded-xl text-sm font-bold border border-gray-100 outline-none text-medic-dark shadow-sm focus:border-medic-dark/30 transition-all"
            />
          </div>

          <div className="space-y-3">
            <span className="text-[10px] font-black text-gray-400 uppercase tracking-widest block px-1">
              Active Availability Dates
            </span>
            {activeDates.length > 0 ? (
              <div className="flex flex-wrap gap-2">
                {activeDates.map((date) => {
                  const times = getTimeSlotsSummary(date);
                  const isPast = date < new Date().toISOString().split("T")[0];
                  return (
                    <button
                      key={date}
                      onClick={() => setSelectedDate(date)}
                      className={`px-4 py-2 border rounded-xl transition-all flex flex-col gap-1 ${
                        selectedDate === date
                          ? "bg-medic-dark text-white border-medic-dark shadow-md"
                          : `bg-white ${isPast ? "opacity-50 grayscale" : "text-medic-dark"} border-gray-100 hover:border-medic-dark/30`
                      }`}
                    >
                      <div className="flex items-center gap-1.5">
                        <Calendar
                          size={12}
                          className={
                            selectedDate === date
                              ? "text-white"
                              : "text-medic-dark"
                          }
                        />
                        <span className="text-[10px] font-black uppercase tracking-widest">
                          {new Date(date + "T00:00:00").toLocaleDateString(
                            undefined,
                            { month: "short", day: "numeric" },
                          )}
                        </span>
                      </div>
                      <span
                        className={`text-[9px] font-bold ${selectedDate === date ? "text-white/80" : "text-gray-400"}`}
                      >
                        {times}
                      </span>
                    </button>
                  );
                })}
              </div>
            ) : (
              <p className="text-[10px] font-medium text-gray-400 italic px-1">
                No availability configured yet. Select a date above to start.
              </p>
            )}
          </div>
        </div>
      </div>

      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-medic-light/20 rounded-xl flex items-center justify-center text-medic-dark">
              <Calendar size={20} />
            </div>
            <div>
              <h4 className="font-bold text-gray-900 leading-none">
                Schedule for{" "}
                {new Date(selectedDate + "T00:00:00").toLocaleDateString(
                  undefined,
                  { weekday: "long", month: "long", day: "numeric" },
                )}
              </h4>
              <p className="text-[10px] text-gray-400 font-bold uppercase tracking-widest mt-1">
                {currentSlots.length} Slots Configured
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            {currentSlots.length > 0 && (
              <button
                onClick={() => {
                  if (window.confirm(`Clear all slots for ${selectedDate}?`)) {
                    onChange(
                      availabilityData.filter((s) => s.date !== selectedDate),
                    );
                  }
                }}
                className="p-2 text-red-400 hover:text-red-500 hover:bg-red-50 rounded-xl transition-all"
                title="Clear all for this date"
              >
                <Trash2 size={18} />
              </button>
            )}
            <button
              onClick={() => addSlot(selectedDate)}
              className="flex items-center gap-2 px-5 py-2.5 bg-medic-dark text-white text-xs font-bold rounded-xl hover:bg-medic-primary transition-all shadow-lg shadow-medic-dark/10"
            >
              <Plus size={16} />
              Add New Slot
            </button>
          </div>
        </div>

        {/* Slot Grid with Labels */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          {currentSlots
            .sort((a, b) => a.start_time.localeCompare(b.start_time))
            .map((slot, idx) => (
              <div
                key={slot.id || slot.tempId || idx}
                className="group relative bg-white border border-gray-100 rounded-[2rem] p-5 hover:border-medic-dark hover:shadow-xl hover:shadow-medic-dark/5 transition-all space-y-4"
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2 text-medic-dark">
                    <Clock size={16} />
                    <span className="text-[10px] font-black uppercase tracking-widest">
                      Time Slot
                    </span>
                  </div>
                  <button
                    onClick={() => removeSlot(slot)}
                    className="p-1.5 text-gray-300 hover:text-red-500 hover:bg-red-50 rounded-lg transition-all"
                  >
                    <Trash2 size={16} />
                  </button>
                </div>

                <div className="grid grid-cols-2 gap-3">
                  <div className="space-y-1">
                    <label className="text-[9px] font-black text-gray-400 uppercase tracking-widest px-1">
                      Start
                    </label>
                    <input
                      type="time"
                      value={slot.start_time.slice(0, 5)}
                      onChange={(e) =>
                        updateSlotValue(slot, "start_time", e.target.value)
                      }
                      className="w-full p-2 bg-neutral-soft rounded-xl text-sm font-bold text-gray-900 border-none outline-none focus:ring-1 focus:ring-medic-dark/20"
                    />
                  </div>
                  <div className="space-y-1">
                    <label className="text-[9px] font-black text-gray-400 uppercase tracking-widest px-1">
                      End
                    </label>
                    <input
                      type="time"
                      value={slot.end_time.slice(0, 5)}
                      onChange={(e) =>
                        updateSlotValue(slot, "end_time", e.target.value)
                      }
                      className="w-full p-2 bg-neutral-soft rounded-xl text-sm font-bold text-gray-900 border-none outline-none focus:ring-1 focus:ring-medic-dark/20"
                    />
                  </div>
                </div>

                <div className="space-y-1">
                  <label className="text-[9px] font-black text-gray-400 uppercase tracking-widest px-1">
                    Label / Exception Name
                  </label>
                  <div className="relative">
                    <Tag
                      className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-300 pointer-events-none"
                      size={14}
                    />
                    <input
                      type="text"
                      placeholder="e.g., Morning Clinic, ER Duty..."
                      value={slot.label || ""}
                      onChange={(e) =>
                        updateSlotValue(slot, "label", e.target.value)
                      }
                      className="w-full pl-9 pr-4 py-2 bg-neutral-soft rounded-xl text-xs font-bold text-gray-700 placeholder:text-gray-300 border-none outline-none focus:ring-1 focus:ring-medic-dark/20"
                    />
                  </div>
                </div>
              </div>
            ))}

          {currentSlots.length === 0 && (
            <div className="sm:col-span-2 py-12 flex flex-col items-center justify-center gap-3 bg-neutral-soft/50 rounded-[2.5rem] border border-dashed border-gray-200">
              <Clock className="text-gray-300" size={32} />
              <p className="text-xs font-bold text-gray-400 uppercase tracking-widest">
                No slots defined for this date
              </p>
              <button
                onClick={() => addSlot(selectedDate)}
                className="mt-2 text-medic-dark text-[10px] font-black uppercase tracking-widest hover:underline"
              >
                + Add First Slot
              </button>
            </div>
          )}
        </div>
      </div>

      <div className="flex justify-end pt-8 border-t border-gray-100">
        <button
          onClick={onSave}
          disabled={isSaving}
          className="px-8 py-4 bg-medic-dark text-white rounded-2xl font-bold hover:bg-medic-primary transition-all shadow-xl shadow-medic-dark/20 flex items-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {isSaving ? (
            "Saving..."
          ) : (
            <>
              <Save size={20} /> Save Availability
            </>
          )}
        </button>
      </div>
    </div>
  );
};

export default AvailabilityManager;
