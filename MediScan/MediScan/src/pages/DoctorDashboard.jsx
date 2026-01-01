import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { MOCK_DATA } from '../Data/mockData.js';
export default function DoctorDashboard() {
  const patients = MOCK_DATA.patients;
  const navigate = useNavigate();
  const [appointments, setAppointments] = useState([]);
  useEffect(() => {
    const saved = localStorage.getItem('mediScanAppointments');
    if (saved) {
      setAppointments(JSON.parse(saved));
    } else {
      setAppointments([
        { id: 99, patientName: 'Alice Brown', date: 'Feb 24, 2025', time: '09:00 AM', condition: 'Thyroid Checkup', status: 'Confirmed' }
      ]);
    }
  }, []);
  return (
    <div className="max-w-7xl mx-auto px-4 py-8 fade-in">
      <div className="flex justify-between items-center mb-8">
        <div>
          <h2 className="text-2xl font-bold text-slate-900">Doctor Dashboard</h2>
          <p className="text-slate-500">Overview of patient reports and appointments.</p>
        </div>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
        <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm">
          <div className="text-slate-500 text-sm font-medium uppercase mb-2">Upcoming Appointments</div>
          <div className="text-3xl font-bold text-blue-600">{appointments.length}</div>
        </div>
        <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm">
          <div className="text-slate-500 text-sm font-medium uppercase mb-2">Pending Reviews</div>
          <div className="text-3xl font-bold text-red-600">3</div>
        </div>
        <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm">
          <div className="text-slate-500 text-sm font-medium uppercase mb-2">Total Patients</div>
          <div className="text-3xl font-bold text-slate-900">142</div>
        </div>
      </div>
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        <div className="lg:col-span-2 bg-white rounded-2xl shadow-sm border border-slate-200 overflow-hidden h-fit">
          <div className="p-6 border-b border-slate-100">
            <h3 className="font-bold text-lg text-slate-900">Recent Reports</h3>
          </div>
          <table className="w-full text-sm text-left">
            <thead className="bg-slate-50 text-slate-500 uppercase font-medium">
              <tr>
                <th className="px-6 py-3">Patient</th>
                <th className="px-6 py-3">Report</th>
                <th className="px-6 py-3">Status</th>
                <th className="px-6 py-3">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {patients.map((p) => (
                <tr key={p.id} className="hover:bg-slate-50">
                  <td className="px-6 py-4 font-medium">{p.name}</td>
                  <td className="px-6 py-4">{p.type}</td>
                  <td className="px-6 py-4">
                    <span
                      className={`px-2 py-1 rounded-full text-xs font-bold ${
                        p.status === 'Normal' ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'
                      }`}
                    >
                      {p.status}
                    </span>
                  </td>
                  <td className="px-6 py-4">
                    <button
                      onClick={() => navigate(`/doctor/review/${p.id}`)}
                      className="text-blue-600 font-medium hover:underline"
                    >
                      Review
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-6 h-fit">
          <h3 className="font-bold text-lg text-slate-900 mb-4 flex items-center gap-2">
            <i className="fa-regular fa-calendar-check text-blue-600"></i> Appointments
          </h3>
          <div className="space-y-4">
            {appointments.length === 0 && <p className="text-slate-500 text-sm">No appointments scheduled.</p>}
            {appointments.map((apt) => (
              <div key={apt.id} className="border-l-4 border-blue-500 bg-blue-50 p-3 rounded-r-lg hover:shadow-md transition">
                <div className="flex justify-between items-start">
                  <div className="font-bold text-slate-900">{apt.patientName}</div>
                  <span className="text-xs bg-green-200 text-green-800 px-2 py-0.5 rounded-full">{apt.status}</span>
                </div>
                <div className="text-sm text-slate-600 mt-1">{apt.condition}</div>
                <div className="text-xs text-slate-500 mt-2 flex items-center gap-2">
                  <i className="fa-regular fa-clock"></i> {apt.date} • {apt.time}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
