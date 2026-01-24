import React from 'react';
import { Link } from 'react-router-dom';

function Dashboard() {
  return (
    <div className="flex-1 min-h-screen bg-slate-50 px-4 py-8">
      <div className="max-w-7xl mx-auto">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-slate-900 mb-2">Admin Dashboard</h1>
          <p className="text-slate-600">Manage your healthcare platform</p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          <Link
            to="/admin/statistics"
            className="bg-white p-6 rounded-lg shadow-md hover:shadow-lg transition cursor-pointer border border-slate-200"
          >
            <h2 className="text-xl font-bold text-slate-900 mb-2">Statistics</h2>
            <p className="text-slate-600">View platform statistics and analytics</p>
          </Link>

          <Link
            to="/admin/users"
            className="bg-white p-6 rounded-lg shadow-md hover:shadow-lg transition cursor-pointer border border-slate-200"
          >
            <h2 className="text-xl font-bold text-slate-900 mb-2">Users</h2>
            <p className="text-slate-600">Manage patients and doctors</p>
          </Link>

          <Link
            to="/admin/verify-doctor"
            className="bg-white p-6 rounded-lg shadow-md hover:shadow-lg transition cursor-pointer border border-slate-200"
          >
            <h2 className="text-xl font-bold text-slate-900 mb-2">Verify Doctor</h2>
            <p className="text-slate-600">Review and verify doctor registrations</p>
          </Link>
        </div>
      </div>
    </div>
  );
}

export default Dashboard;
