import React, { useState } from 'react';
import { MOCK_DATA } from '../Data/mockData.js';
export default function AdminDashboard() {
  const [pending, setPending] = useState(MOCK_DATA.admin.pendingVerifications);
  const stats = MOCK_DATA.admin.stats;
  const models = MOCK_DATA.admin.modelHealth;
  const handleApprove = (id) => {
    if (confirm('Approve this doctor registration?')) {
      setPending((prev) => prev.filter((d) => d.id !== id));
    }
  };
  const handleReject = (id) => {
    if (confirm('Reject this application?')) {
      setPending((prev) => prev.filter((d) => d.id !== id));
    }
  };
  return (
    <div className="max-w-7xl mx-auto px-4 py-8 fade-in">
      <div className="flex justify-between items-center mb-8">
        <div>
          <h2 className="text-2xl font-bold text-slate-900">Admin Panel</h2>
          <p className="text-slate-500">System overview and backend monitoring.</p>
        </div>
        <div className="flex gap-3">
          <div className="bg-green-100 text-green-700 px-4 py-2 rounded-lg text-sm font-bold shadow-sm flex items-center">
            <i className="fa-solid fa-server mr-2"></i> System: Online
          </div>
        </div>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
        <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm relative overflow-hidden group">
          <div className="absolute right-0 top-0 h-full w-1 bg-blue-500 group-hover:w-2 transition-all"></div>
          <div className="text-slate-500 text-xs font-bold uppercase mb-2 tracking-wider">Total Users</div>
          <div className="text-3xl font-extrabold text-slate-900">{stats.totalUsers}</div>
          <div className="text-xs text-green-500 mt-1 font-bold">
            <i className="fa-solid fa-arrow-up"></i> 12% vs last month
          </div>
        </div>
        <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm relative overflow-hidden group">
          <div className="absolute right-0 top-0 h-full w-1 bg-green-500 group-hover:w-2 transition-all"></div>
          <div className="text-slate-500 text-xs font-bold uppercase mb-2 tracking-wider">Verified Doctors</div>
          <div className="text-3xl font-extrabold text-slate-900">{stats.totalDoctors}</div>
          <div className="text-xs text-slate-400 mt-1">across 12 specialties</div>
        </div>
        <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm relative overflow-hidden group">
          <div className="absolute right-0 top-0 h-full w-1 bg-purple-500 group-hover:w-2 transition-all"></div>
          <div className="text-slate-500 text-xs font-bold uppercase mb-2 tracking-wider">Reports Processed</div>
          <div className="text-3xl font-extrabold text-slate-900">{stats.reportsAnalyzed}</div>
          <div className="text-xs text-purple-600 mt-1 font-bold">1.2s avg processing</div>
        </div>
        <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm relative overflow-hidden group">
          <div className="absolute right-0 top-0 h-full w-1 bg-orange-500 group-hover:w-2 transition-all"></div>
          <div className="text-slate-500 text-xs font-bold uppercase mb-2 tracking-wider">Pending Approvals</div>
          <div className="text-3xl font-extrabold text-slate-900">{pending.length}</div>
          <div className="text-xs text-orange-500 mt-1 font-bold">Action Required</div>
        </div>
      </div>
      <div className="bg-white rounded-2xl shadow-sm border border-slate-200 overflow-hidden mb-8">
        <div className="p-6 border-b border-slate-100 flex justify-between items-center bg-slate-50">
          <div className="flex items-center gap-3">
            <div className="bg-yellow-100 text-yellow-600 p-2 rounded-lg">
              <i className="fa-solid fa-crown"></i>
            </div>
            <div>
              <h3 className="font-bold text-lg text-slate-900">User Subscriptions & Billing</h3>
              <p className="text-xs text-slate-500">Monitor active premiums, renewals, and payment methods.</p>
            </div>
          </div>
          <button className="text-blue-600 text-sm font-bold hover:underline">View All Transactions</button>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm text-left">
            <thead className="bg-slate-50 text-slate-500 uppercase font-medium">
              <tr>
                <th className="px-6 py-3">User</th>
                <th className="px-6 py-3">Plan Type</th>
                <th className="px-6 py-3">Status</th>
                <th className="px-6 py-3">Renewal</th>
                <th className="px-6 py-3">Billing</th>
                <th className="px-6 py-3">Payment</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              <tr className="hover:bg-slate-50">
                <td className="px-6 py-4 font-bold text-slate-900">
                  John Doe <br />
                  <span className="text-xs font-normal text-slate-500">
                    Nepal <i className="fa-solid fa-location-dot ml-1"></i>
                  </span>
                </td>
                <td className="px-6 py-4">
                  <span className="bg-blue-100 text-blue-700 text-xs font-bold px-2 py-1 rounded">Premium</span>
                </td>
                <td className="px-6 py-4 text-green-600 font-bold">
                  <i className="fa-solid fa-circle-check"></i> Active
                </td>
                <td className="px-6 py-4 text-slate-600">
                  Mar 24, 2025 <br />
                  <span className="text-xs text-slate-400">Started: Feb 24</span>
                </td>
                <td className="px-6 py-4 font-bold">$9.99 <span className="text-xs text-slate-400 font-normal">/mo</span></td>
                <td className="px-6 py-4 flex items-center gap-2">
                  <i className="fa-brands fa-cc-visa text-xl text-slate-600"></i> Card
                </td>
              </tr>
              <tr className="hover:bg-slate-50">
                <td className="px-6 py-4 font-bold text-slate-900">
                  Sarah Smith <br />
                  <span className="text-xs font-normal text-slate-500">
                    USA <i className="fa-solid fa-location-dot ml-1"></i>
                  </span>
                </td>
                <td className="px-6 py-4">
                  <span className="bg-purple-100 text-purple-700 text-xs font-bold px-2 py-1 rounded">Family Plan</span>
                </td>
                <td className="px-6 py-4 text-green-600 font-bold">
                  <i className="fa-solid fa-circle-check"></i> Active
                </td>
                <td className="px-6 py-4 text-slate-600">
                  Mar 01, 2025 <br />
                  <span className="text-xs text-slate-400">Started: Feb 01</span>
                </td>
                <td className="px-6 py-4 font-bold">$14.99 <span className="text-xs text-slate-400 font-normal">/mo</span></td>
                <td className="px-6 py-4 flex items-center gap-2">
                  <i className="fa-brands fa-paypal text-xl text-blue-600"></i> PayPal
                </td>
              </tr>
              <tr className="hover:bg-slate-50">
                <td className="px-6 py-4 font-bold text-slate-900">
                  Ram Sharma <br />
                  <span className="text-xs font-normal text-slate-500">
                    Nepal <i className="fa-solid fa-location-dot ml-1"></i>
                  </span>
                </td>
                <td className="px-6 py-4">
                  <span className="bg-orange-100 text-orange-700 text-xs font-bold px-2 py-1 rounded">Student</span>
                </td>
                <td className="px-6 py-4 text-green-600 font-bold">
                  <i className="fa-solid fa-circle-check"></i> Active
                </td>
                <td className="px-6 py-4 text-slate-600">
                  Mar 15, 2025 <br />
                  <span className="text-xs text-slate-400">Started: Feb 15</span>
                </td>
                <td className="px-6 py-4 font-bold">NPR 499 <span className="text-xs text-slate-400 font-normal">/mo</span></td>
                <td className="px-6 py-4 flex items-center gap-2">
                  <i className="fa-solid fa-wallet text-xl text-green-600"></i> eSewa
                </td>
              </tr>
              <tr className="hover:bg-slate-50 opacity-60">
                <td className="px-6 py-4 font-bold text-slate-900">
                  Emily Blunt <br />
                  <span className="text-xs font-normal text-slate-500">
                    UK <i className="fa-solid fa-location-dot ml-1"></i>
                  </span>
                </td>
                <td className="px-6 py-4">
                  <span className="bg-blue-100 text-blue-700 text-xs font-bold px-2 py-1 rounded">Premium</span>
                </td>
                <td className="px-6 py-4 text-red-500 font-bold">
                  <i className="fa-solid fa-circle-xmark"></i> Expired
                </td>
                <td className="px-6 py-4 text-slate-600">
                  Jan 01, 2025 <br />
                  <span className="text-xs text-slate-400">Ended</span>
                </td>
                <td className="px-6 py-4 font-bold">$9.99 <span className="text-xs text-slate-400 font-normal">/mo</span></td>
                <td className="px-6 py-4 flex items-center gap-2">
                  <i className="fa-brands fa-google-play text-xl text-slate-600"></i> Google Play
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-8">
        <div className="bg-white rounded-2xl shadow-sm border border-slate-200 overflow-hidden h-fit">
          <div className="p-6 border-b border-slate-100 flex justify-between items-center bg-slate-50">
            <div>
              <h3 className="font-bold text-lg text-slate-900">Pending Doctor Verifications</h3>
              <p className="text-xs text-slate-500">Review license details before approving.</p>
            </div>
            <span className="text-xs bg-orange-100 text-orange-800 font-bold px-3 py-1 rounded-full border border-orange-200">
              {pending.length} Requests
            </span>
          </div>
          {pending.length === 0 ? (
            <div className="p-16 text-center text-slate-500">
              <div className="w-16 h-16 bg-green-100 text-green-500 rounded-full flex items-center justify-center mx-auto mb-4 text-2xl">
                <i className="fa-solid fa-check"></i>
              </div>
              <h4 className="font-bold text-slate-900">All Caught Up!</h4>
              <p className="text-sm">No pending doctor verifications at the moment.</p>
            </div>
          ) : (
            <div className="divide-y divide-slate-100">
              {pending.map((doctor) => (
                <div
                  key={doctor.id}
                  className="p-6 hover:bg-slate-50 transition flex flex-col sm:flex-row gap-4 justify-between items-start sm:items-center"
                >
                  <div className="flex items-start gap-4">
                    <div className="w-12 h-12 bg-blue-100 text-blue-600 rounded-full flex items-center justify-center text-xl shrink-0">
                      <i className="fa-solid fa-user-doctor"></i>
                    </div>
                    <div>
                      <div className="font-bold text-slate-900 text-lg">{doctor.name}</div>
                      <div className="text-sm text-slate-600 mb-1">{doctor.email}</div>
                      <div className="flex flex-wrap gap-2 text-xs">
                        <span className="bg-slate-100 px-2 py-1 rounded border border-slate-200 font-mono text-slate-600">
                          <i className="fa-solid fa-id-card mr-1"></i> {doctor.license}
                        </span>
                        <span className="bg-blue-50 px-2 py-1 rounded border border-blue-100 text-blue-700 font-bold">
                          {doctor.specialty}
                        </span>
                        <span className="text-slate-400 py-1">
                          <i className="fa-regular fa-clock mr-1"></i> {doctor.applied}
                        </span>
                      </div>
                    </div>
                  </div>
                  <div className="flex gap-3 w-full sm:w-auto">
                    <button
                      onClick={() => handleApprove(doctor.id)}
                      className="flex-1 sm:flex-none bg-green-600 hover:bg-green-700 text-white px-4 py-2 rounded-lg text-sm font-bold transition shadow-sm flex items-center justify-center gap-2"
                    >
                      <i className="fa-solid fa-check"></i> Approve
                    </button>
                    <button
                      onClick={() => handleReject(doctor.id)}
                      className="flex-1 sm:flex-none bg-white hover:bg-red-50 text-red-600 border border-slate-200 hover:border-red-200 px-4 py-2 rounded-lg text-sm font-bold transition flex items-center justify-center gap-2"
                    >
                      <i className="fa-solid fa-xmark"></i> Reject
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
        <div className="space-y-6">
          <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-6">
            <div className="flex items-center gap-3 mb-6">
              <div className="bg-purple-100 text-purple-600 p-2 rounded-lg">
                <i className="fa-solid fa-brain"></i>
              </div>
              <div>
                <h3 className="font-bold text-lg text-slate-900">AI Model Health</h3>
                <p className="text-xs text-slate-500">Real-time performance metrics of the ML Core.</p>
              </div>
            </div>
            <div className="space-y-4">
              {models.map((m, i) => (
                <div
                  key={i}
                  className="flex flex-col p-4 bg-slate-50 rounded-xl border border-slate-100 hover:border-blue-200 transition"
                >
                  <div className="flex justify-between items-start mb-3">
                    <div className="flex items-center gap-2">
                      <div className="font-bold text-slate-900">{m.name}</div>
                      <span
                        className={`text-[10px] px-2 py-0.5 rounded-full font-bold uppercase ${
                          m.status === 'Healthy' ? 'bg-green-100 text-green-700' : 'bg-yellow-100 text-yellow-700'
                        }`}
                      >
                        {m.status}
                      </span>
                    </div>
                    <div className="text-right">
                      <div className="font-bold text-slate-900 text-lg">{m.accuracy}</div>
                      <div className="text-[10px] text-slate-500 uppercase font-bold tracking-wider">Accuracy</div>
                    </div>
                  </div>
                  <div className="w-full bg-slate-200 rounded-full h-1.5 mb-3">
                    <div
                      className={`h-1.5 rounded-full ${parseFloat(m.accuracy) > 95 ? 'bg-green-500' : 'bg-blue-500'}`}
                      style={{ width: m.accuracy }}
                    ></div>
                  </div>
                  <div className="flex justify-between text-xs text-slate-500 font-mono">
                    <span>
                      <i className="fa-solid fa-stopwatch mr-1"></i> Latency: {m.latency}
                    </span>
                    <span>
                      <i className="fa-solid fa-check-double mr-1"></i> 99.9% Uptime
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
          <div className="bg-slate-900 rounded-2xl shadow-lg border border-slate-800 p-6 text-white">
            <div className="flex items-center justify-between mb-4">
              <h3 className="font-bold text-lg">
                <i className="fa-solid fa-terminal mr-2 text-blue-400"></i> System Logs
              </h3>
              <span className="text-xs text-slate-400 bg-slate-800 px-2 py-1 rounded">Live</span>
            </div>
            <div className="font-mono text-xs space-y-2 text-slate-300 h-32 overflow-y-auto">
              <div className="flex gap-2">
                <span className="text-blue-400">10:42:15</span> <span>[INFO] New report uploaded by User #482</span>
              </div>
              <div className="flex gap-2">
                <span className="text-green-400">10:42:18</span> <span>[SUCCESS] OCR Extraction completed (1.2s)</span>
              </div>
              <div className="flex gap-2">
                <span className="text-blue-400">10:45:00</span> <span>[INFO] Doctor #101 verified successfully</span>
              </div>
              <div className="flex gap-2">
                <span className="text-yellow-400">10:46:12</span> <span>[WARN] High latency detected in Chatbot RAG (1.5s)</span>
              </div>
              <div className="flex gap-2">
                <span className="text-blue-400">10:48:30</span> <span>[INFO] User #33 upgraded to Premium</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}