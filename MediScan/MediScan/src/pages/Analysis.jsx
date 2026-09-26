import React, { useMemo, useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext.jsx';
import { MOCK_DATA } from '../Data/mockData.js';
export default function Analysis() {
  const { user } = useAuth();
  const data = MOCK_DATA.analysis;
  const [summary, setSummary] = useState(data.summary);
  const [summaryMode, setSummaryMode] = useState('patient'); // 'patient' or 'doctor'

  useEffect(() => {
    setSummary(summaryMode === 'patient' ? data.summary : data.doctorSummary);
  }, [summaryMode, data]);
  const isPremium = true; // All features are now free
  const highRiskConditions = useMemo(
    () => data.riskAssessment.filter((r) => r.probability > 80),
    [data.riskAssessment]
  );
  const isCritical = highRiskConditions.length > 0;
  const suggestedDoctors = [
    {
      id: 1,
      name: 'Dr. Sarah Wilson',
      specialty: 'Hematologist',
      rating: 4.8,
      date: 'Feb 25, 2025',
      time: '10:00 AM',
      hospital: 'City General Hospital'
    },
    {
      id: 2,
      name: 'Dr. James Carter',
      specialty: 'General Physician',
      rating: 4.9,
      date: 'Feb 26, 2025',
      time: '02:30 PM',
      hospital: 'MediCare Clinic'
    }
  ];
  const speak = () => {
    const u = new SpeechSynthesisUtterance(summary);
    window.speechSynthesis.speak(u);
  };
  const translate = () => {
    setSummary(
      'यो रिपोर्टले हल्का रक्तअल्पता (कम हेमोग्लोबिन) र बढेको ग्लुकोज स्तर देखाउँछ जसले पूर्व-मधुमेहको जोखिम संकेत गर्दछ।'
    );
  };
  const bookAppointment = (doctor) => {
    const newAppointment = {
      id: crypto.randomUUID(),
      patientName: data.patientName,
      doctorName: doctor.name,
      date: doctor.date,
      time: doctor.time,
      condition: highRiskConditions[0]?.condition || 'General Checkup',
      status: 'Confirmed'
    };
    const existingAppointments = JSON.parse(localStorage.getItem('mediScanAppointments') || '[]');
    localStorage.setItem('mediScanAppointments', JSON.stringify([newAppointment, ...existingAppointments]));
    alert(
      `Appointment Confirmed!\n\nDoctor: ${doctor.name}\nDate: ${doctor.date}\nTime: ${doctor.time}\n\nThe doctor has been notified.`
    );
  };
  return (
    <div className="max-w-7xl mx-auto px-4 py-8 fade-in">
      <div className="flex justify-between items-center mb-6">
        <Link to="/dashboard" className="text-slate-500 hover:text-slate-800 flex items-center gap-2">
          <i className="fa-solid fa-arrow-left"></i> Back
        </Link>
        <div className="flex gap-2">
          <button className="bg-white border border-slate-300 text-slate-700 px-4 py-2 rounded-lg text-sm hover:bg-slate-50">
            <i className="fa-solid fa-share-nodes mr-2"></i> Share
          </button>
          <button className="bg-blue-600 text-white px-4 py-2 rounded-lg text-sm hover:bg-blue-700">
            <i className="fa-solid fa-download mr-2"></i> Download
          </button>
        </div>
      </div>
      {isCritical && (
        <div className="bg-red-50 border-l-4 border-red-500 p-4 mb-6 rounded-r-lg shadow-sm flex flex-col sm:flex-row justify-between items-center gap-4 animate-pulse">
          <div className="flex items-center gap-3">
            <div className="bg-red-100 p-3 rounded-full text-red-600">
              <i className="fa-solid fa-triangle-exclamation text-xl"></i>
            </div>
            <div>
              <h3 className="text-red-800 font-bold text-lg">Critical Attention Required</h3>
              <p className="text-red-600 text-sm">
                Your report indicates high-risk values for{' '}
                <span className="font-bold underline">{highRiskConditions.map((c) => c.condition).join(', ')}</span>. Please
                consult a doctor immediately.
              </p>
            </div>
          </div>
        </div>
      )}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        <div className="lg:col-span-2 space-y-6">
          <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-6">
            <div className="flex items-center gap-3 mb-4">
              <div className="bg-purple-100 text-purple-600 p-2 rounded-lg">
                <i className="fa-solid fa-wand-magic-sparkles"></i>
              </div>
              <h3 className="font-bold text-lg text-slate-900">AI Report Summary</h3>
            </div>

            <div className="flex bg-slate-100 p-1 rounded-lg mb-4 w-fit">
              <button
                onClick={() => setSummaryMode('patient')}
                className={`px-3 py-1.5 text-xs font-bold rounded-md transition ${summaryMode === 'patient' ? 'bg-white text-green-700 shadow-sm' : 'text-slate-500 hover:text-slate-700'}`}
              >
                Patient View
              </button>
              <button
                onClick={() => setSummaryMode('doctor')}
                className={`px-3 py-1.5 text-xs font-bold rounded-md transition ${summaryMode === 'doctor' ? 'bg-white text-blue-700 shadow-sm' : 'text-slate-500 hover:text-slate-700'}`}
              >
                Doctor View
              </button>
            </div>

            <p className="text-slate-700 leading-relaxed bg-slate-50 p-4 rounded-xl border border-slate-100 min-h-[80px]">{summary}</p>
            <div className="mt-4 flex gap-4">
              <button onClick={speak} className="text-sm text-blue-600 font-medium hover:underline">
                <i className="fa-solid fa-volume-high mr-1"></i> Listen
              </button>
              <button onClick={translate} className="text-sm text-blue-600 font-medium hover:underline">
                <i className="fa-solid fa-language mr-1"></i> Translate (Nepali)
              </button>
            </div>
          </div>
          <div className="bg-white rounded-2xl shadow-sm border border-slate-200 overflow-hidden">
            <div className="p-6 border-b border-slate-100">
              <h3 className="font-bold text-lg text-slate-900">Extracted Values</h3>
            </div>
            <table className="w-full text-sm text-left">
              <thead className="bg-slate-50 text-slate-500 uppercase font-medium">
                <tr>
                  <th className="px-6 py-3">Test</th>
                  <th className="px-6 py-3">Result</th>
                  <th className="px-6 py-3">Range</th>
                  <th className="px-6 py-3">Status</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {data.extractedData.map((row, i) => (
                  <tr key={i} className="hover:bg-slate-50">
                    <td className="px-6 py-4 font-medium">{row.test}</td>
                    <td className="px-6 py-4 font-bold">
                      {row.value} <span className="text-xs font-normal text-slate-400">{row.unit}</span>
                    </td>
                    <td className="px-6 py-4 text-slate-500">{row.refRange}</td>
                    <td className="px-6 py-4">
                      <span
                        className={`px-2 py-1 rounded-full text-xs font-bold uppercase ${row.status === 'normal' ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'
                          }`}
                      >
                        {row.status}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
        <div className="space-y-6">
          <div className="relative bg-white rounded-2xl shadow-sm border border-slate-200 p-6 overflow-hidden">
            <div className={!isPremium ? 'blur-lock' : ''}>
              <div className="flex items-center gap-3 mb-4">
                <div className="bg-red-100 text-red-600 p-2 rounded-lg">
                  <i className="fa-solid fa-notes-medical"></i>
                </div>
                <h3 className="font-bold text-lg text-slate-900">
                  Risk Analysis{' '}

                </h3>
              </div>
              <div className="space-y-4">
                {data.riskAssessment.map((risk, i) => (
                  <div key={i}>
                    <div className="flex justify-between text-sm mb-1">
                      <span className="font-medium text-slate-700">{risk.condition}</span>
                      <span className="font-bold text-slate-500">{risk.probability}%</span>
                    </div>
                    <div className="w-full bg-slate-100 rounded-full h-2">
                      <div
                        className={`h-2 rounded-full ${risk.probability > 60 ? 'bg-red-500' : 'bg-green-500'}`}
                        style={{ width: `${risk.probability}%` }}
                      ></div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
            {!isPremium && (
              <div className="absolute inset-0 z-10 flex flex-col items-center justify-center bg-white/60 backdrop-blur-[2px]">
                <div className="bg-white p-6 rounded-xl shadow-xl text-center border border-slate-200 max-w-xs">
                  <div className="bg-yellow-100 w-12 h-12 rounded-full flex items-center justify-center mx-auto mb-3 text-yellow-600">
                    <i className="fa-solid fa-lock text-xl"></i>
                  </div>
                  <h4 className="font-bold text-slate-900 mb-2">Premium Feature</h4>
                  <p className="text-xs text-slate-500 mb-4">Unlock detailed disease risk analysis and probability graphs.</p>
                  <Link to="/membership" className="block bg-blue-600 text-white py-2 rounded-lg text-sm font-bold hover:bg-blue-700">
                    Upgrade to View
                  </Link>
                </div>
              </div>
            )}
          </div>
          {isCritical && (
            <div className="relative bg-white rounded-2xl shadow-lg border border-red-100 overflow-hidden fade-in">
              <div className={!isPremium ? 'blur-lock' : ''}>
                <div className="bg-gradient-to-r from-red-500 to-rose-600 p-4 text-white">
                  <h3 className="font-bold text-lg flex items-center gap-2">
                    <i className="fa-solid fa-user-doctor"></i> Recommended Specialists
                  </h3>
                  <p className="text-red-100 text-xs mt-1">Based on your condition: {highRiskConditions[0].condition}</p>
                </div>
                <div className="p-4 space-y-4">
                  {suggestedDoctors.map((doc) => (
                    <div
                      key={doc.id}
                      className="border border-slate-100 rounded-xl p-3 hover:shadow-md transition bg-slate-50"
                    >
                      <div className="flex justify-between items-start mb-2">
                        <div>
                          <div className="font-bold text-slate-900">{doc.name}</div>
                          <div className="text-xs text-slate-500">{doc.specialty}</div>
                          <div className="text-xs text-slate-400">{doc.hospital}</div>
                        </div>
                        <div className="flex items-center text-yellow-500 text-xs font-bold bg-yellow-50 px-2 py-1 rounded">
                          <i className="fa-solid fa-star mr-1"></i> {doc.rating}
                        </div>
                      </div>
                      <div className="flex items-center gap-2 text-xs text-slate-600 mb-3 bg-white p-2 rounded border border-slate-100">
                        <i className="fa-regular fa-calendar text-blue-500"></i> Available:{' '}
                        <span className="font-bold text-slate-800">{doc.date} @ {doc.time}</span>
                      </div>
                      <button
                        onClick={() => bookAppointment(doc)}
                        className="w-full bg-blue-600 text-white py-2 rounded-lg text-sm font-bold hover:bg-blue-700 transition flex items-center justify-center gap-2"
                      >
                        Book Now
                      </button>
                    </div>
                  ))}
                </div>
              </div>
              {!isPremium && (
                <div className="absolute inset-0 z-10 flex flex-col items-center justify-center bg-white/60 backdrop-blur-[2px]">
                  <Link
                    to="/membership"
                    className="bg-blue-600 text-white px-6 py-3 rounded-xl text-sm font-bold hover:bg-blue-700 shadow-lg flex items-center gap-2"
                  >
                    <i className="fa-solid fa-lock"></i> Unlock Doctor Booking
                  </Link>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
