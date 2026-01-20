import React, { useState, useEffect } from 'react';

function VerifyDoctor() {
  const [doctors] = useState([]);
  const [filter, setFilter] = useState('pending'); // 'pending', 'verified', 'rejected'

  useEffect(() => {
    // Fetch doctors from API
    // setDoctors(data);
  }, [filter]);

  const handleVerify = (doctorId) => {
    // Handle doctor verification
    console.log('Verify doctor:', doctorId);
  };

  const handleReject = (doctorId) => {
    // Handle doctor rejection
    console.log('Reject doctor:', doctorId);
  };

  return (
    <div className="flex-1 min-h-screen bg-slate-50 px-4 py-8">
      <div className="max-w-7xl mx-auto">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-slate-900 mb-2">Verify Doctors</h1>
          <p className="text-slate-600">Review and verify doctor registrations</p>
        </div>

        <div className="mb-6">
          <div className="flex gap-4">
            <button
              onClick={() => setFilter('pending')}
              className={`px-4 py-2 rounded-lg font-semibold transition ${filter === 'pending'
                  ? 'bg-purple-600 text-white'
                  : 'bg-white text-slate-700 border border-slate-300'
                }`}
            >
              Pending
            </button>
            <button
              onClick={() => setFilter('verified')}
              className={`px-4 py-2 rounded-lg font-semibold transition ${filter === 'verified'
                  ? 'bg-purple-600 text-white'
                  : 'bg-white text-slate-700 border border-slate-300'
                }`}
            >
              Verified
            </button>
            <button
              onClick={() => setFilter('rejected')}
              className={`px-4 py-2 rounded-lg font-semibold transition ${filter === 'rejected'
                  ? 'bg-purple-600 text-white'
                  : 'bg-white text-slate-700 border border-slate-300'
                }`}
            >
              Rejected
            </button>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {doctors.length === 0 ? (
            <div className="col-span-full text-center text-slate-500 py-8">
              No doctors found
            </div>
          ) : (
            doctors.map((doctor) => (
              <div key={doctor.id} className="bg-white p-6 rounded-lg shadow-md border border-slate-200">
                <h3 className="text-xl font-bold text-slate-900 mb-2">{doctor.name}</h3>
                <p className="text-slate-600 mb-1">Email: {doctor.email}</p>
                <p className="text-slate-600 mb-1">Specialization: {doctor.specialization}</p>
                <p className="text-slate-600 mb-1">License: {doctor.licenseNumber}</p>
                <p className="text-slate-600 mb-4">Hospital: {doctor.hospital}</p>

                {filter === 'pending' && (
                  <div className="flex gap-2">
                    <button
                      onClick={() => handleVerify(doctor.id)}
                      className="flex-1 bg-green-600 text-white font-semibold py-2 rounded-lg hover:bg-green-700 transition"
                    >
                      Verify
                    </button>
                    <button
                      onClick={() => handleReject(doctor.id)}
                      className="flex-1 bg-red-600 text-white font-semibold py-2 rounded-lg hover:bg-red-700 transition"
                    >
                      Reject
                    </button>
                  </div>
                )}

                <div className="mt-2">
                  <span
                    className={`px-2 py-1 text-xs font-semibold rounded-full ${doctor.status === 'verified'
                        ? 'bg-green-100 text-green-800'
                        : doctor.status === 'rejected'
                          ? 'bg-red-100 text-red-800'
                          : 'bg-yellow-100 text-yellow-800'
                      }`}
                  >
                    {doctor.status}
                  </span>
                </div>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}

export default VerifyDoctor;
