import React, { useState, useEffect } from 'react';

function Statistics() {
  const [stats, setStats] = useState({
    totalUsers: 0,
    totalDoctors: 0,
    totalPatients: 0,
    verifiedDoctors: 0,
    pendingDoctors: 0,
    totalReports: 0
  });

  useEffect(() => {
    // Fetch statistics from API
    // setStats(data);
  }, []);

  return (
    <div className="flex-1 min-h-screen bg-slate-50 px-4 py-8">
      <div className="max-w-7xl mx-auto">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-slate-900 mb-2">Statistics</h1>
          <p className="text-slate-600">Platform overview and analytics</p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          <div className="bg-white p-6 rounded-lg shadow-md border border-slate-200">
            <h3 className="text-lg font-semibold text-slate-700 mb-2">Total Users</h3>
            <p className="text-3xl font-bold text-purple-600">{stats.totalUsers}</p>
          </div>

          <div className="bg-white p-6 rounded-lg shadow-md border border-slate-200">
            <h3 className="text-lg font-semibold text-slate-700 mb-2">Total Patients</h3>
            <p className="text-3xl font-bold text-green-600">{stats.totalPatients}</p>
          </div>

          <div className="bg-white p-6 rounded-lg shadow-md border border-slate-200">
            <h3 className="text-lg font-semibold text-slate-700 mb-2">Total Doctors</h3>
            <p className="text-3xl font-bold text-blue-600">{stats.totalDoctors}</p>
          </div>

          <div className="bg-white p-6 rounded-lg shadow-md border border-slate-200">
            <h3 className="text-lg font-semibold text-slate-700 mb-2">Verified Doctors</h3>
            <p className="text-3xl font-bold text-green-600">{stats.verifiedDoctors}</p>
          </div>

          <div className="bg-white p-6 rounded-lg shadow-md border border-slate-200">
            <h3 className="text-lg font-semibold text-slate-700 mb-2">Pending Doctors</h3>
            <p className="text-3xl font-bold text-yellow-600">{stats.pendingDoctors}</p>
          </div>

          <div className="bg-white p-6 rounded-lg shadow-md border border-slate-200">
            <h3 className="text-lg font-semibold text-slate-700 mb-2">Total Reports</h3>
            <p className="text-3xl font-bold text-purple-600">{stats.totalReports}</p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Statistics;
