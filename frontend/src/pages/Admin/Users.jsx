import React, { useState, useEffect } from 'react';

function Users() {
  const [users, setUsers] = useState([]);
  const [filter, setFilter] = useState('all'); // 'all', 'patients', 'doctors'

  useEffect(() => {
    // Fetch users from API
    // setUsers(data);
  }, [filter]);

  return (
    <div className="flex-1 min-h-screen bg-slate-50 px-4 py-8">
      <div className="max-w-7xl mx-auto">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-slate-900 mb-2">Users Management</h1>
          <p className="text-slate-600">Manage all platform users</p>
        </div>

        <div className="mb-6">
          <div className="flex gap-4">
            <button
              onClick={() => setFilter('all')}
              className={`px-4 py-2 rounded-lg font-semibold transition ${
                filter === 'all'
                  ? 'bg-purple-600 text-white'
                  : 'bg-white text-slate-700 border border-slate-300'
              }`}
            >
              All Users
            </button>
            <button
              onClick={() => setFilter('patients')}
              className={`px-4 py-2 rounded-lg font-semibold transition ${
                filter === 'patients'
                  ? 'bg-purple-600 text-white'
                  : 'bg-white text-slate-700 border border-slate-300'
              }`}
            >
              Patients
            </button>
            <button
              onClick={() => setFilter('doctors')}
              className={`px-4 py-2 rounded-lg font-semibold transition ${
                filter === 'doctors'
                  ? 'bg-purple-600 text-white'
                  : 'bg-white text-slate-700 border border-slate-300'
              }`}
            >
              Doctors
            </button>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-md border border-slate-200 overflow-hidden">
          <table className="w-full">
            <thead className="bg-slate-100">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-semibold text-slate-700 uppercase">Name</th>
                <th className="px-6 py-3 text-left text-xs font-semibold text-slate-700 uppercase">Email</th>
                <th className="px-6 py-3 text-left text-xs font-semibold text-slate-700 uppercase">Type</th>
                <th className="px-6 py-3 text-left text-xs font-semibold text-slate-700 uppercase">Status</th>
                <th className="px-6 py-3 text-left text-xs font-semibold text-slate-700 uppercase">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-200">
              {users.length === 0 ? (
                <tr>
                  <td colSpan="5" className="px-6 py-4 text-center text-slate-500">
                    No users found
                  </td>
                </tr>
              ) : (
                users.map((user) => (
                  <tr key={user.id} className="hover:bg-slate-50">
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-900">{user.name}</td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-600">{user.email}</td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-600">{user.type}</td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span
                        className={`px-2 py-1 text-xs font-semibold rounded-full ${
                          user.status === 'active'
                            ? 'bg-green-100 text-green-800'
                            : 'bg-gray-100 text-gray-800'
                        }`}
                      >
                        {user.status}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm">
                      <button className="text-purple-600 hover:text-purple-800 font-semibold">
                        View
                      </button>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

export default Users;
