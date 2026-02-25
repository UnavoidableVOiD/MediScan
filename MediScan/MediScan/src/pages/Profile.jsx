import React, { useEffect, useState } from 'react';
import { Navigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext.jsx';
export default function Profile() {
  const { user, updateUser } = useAuth();
  const [isEditing, setIsEditing] = useState(false);
  const [formData, setFormData] = useState({});
  useEffect(() => {
    if (user) setFormData(user);
  }, [user]);
  const handleChange = (e) => {
    setFormData((prev) => ({ ...prev, [e.target.name]: e.target.value }));
  };
  const handleSave = () => {
    updateUser(formData);
    setIsEditing(false);
    alert('Profile updated successfully!');
  };
  const handleImageUpload = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onloadend = () => {
      setFormData((prev) => ({ ...prev, profileImage: reader.result }));
    };
    reader.readAsDataURL(file);
  };
  if (!user) return <Navigate to="/login" />;
  return (
    <div className="max-w-4xl mx-auto px-4 py-8 fade-in">
      <div className="flex justify-between items-center mb-8">
        <h2 className="text-2xl font-bold text-slate-900">My Profile</h2>
        <button
          onClick={() => (isEditing ? handleSave() : setIsEditing(true))}
          className={`px-6 py-2 rounded-lg text-sm font-bold shadow-sm transition ${
            isEditing ? 'bg-green-600 text-white hover:bg-green-700' : 'bg-blue-600 text-white hover:bg-blue-700'
          }`}
        >
          {isEditing ? (
            <>
              <i className="fa-solid fa-save mr-2"></i> Save Changes
            </>
          ) : (
            <>
              <i className="fa-solid fa-pen-to-square mr-2"></i> Edit Profile
            </>
          )}
        </button>
      </div>
      <div className="bg-white rounded-2xl shadow-xl border border-slate-100 overflow-hidden">
        <div className="bg-slate-100 p-8 flex items-center gap-6 border-b border-slate-200">
          <div className="relative group">
            {formData.profileImage ? (
              <img
                src={formData.profileImage}
                alt="Profile"
                className="w-24 h-24 rounded-full object-cover shadow-md border-4 border-white"
              />
            ) : (
              <div className="w-24 h-24 bg-white rounded-full flex items-center justify-center text-4xl text-blue-600 shadow-md border-4 border-white">
                <i
                  className={`fa-solid ${
                    user.role === 'doctor' ? 'fa-user-doctor' : user.role === 'admin' ? 'fa-shield-halved' : 'fa-user'
                  }`}
                ></i>
              </div>
            )}
            {isEditing && (
              <label className="absolute bottom-0 right-0 bg-blue-600 text-white p-2 rounded-full cursor-pointer hover:bg-blue-700 shadow-sm transition transform hover:scale-110">
                <i className="fa-solid fa-camera text-xs"></i>
                <input type="file" accept="image/*" className="hidden" onChange={handleImageUpload} />
              </label>
            )}
          </div>
          <div>
            <h3 className="text-2xl font-bold text-slate-900">{formData.name || user.name}</h3>
            <p className="text-slate-500 capitalize">{user.role} Account</p>
            <p className="text-slate-400 text-sm mt-1">{formData.email}</p>
          </div>
        </div>
        <div className="p-8 grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <label className="block text-xs font-bold text-slate-500 uppercase mb-2">Full Name</label>
            <input
              type="text"
              name="name"
              value={formData.name || ''}
              onChange={handleChange}
              disabled={!isEditing}
              className="w-full border border-slate-300 rounded-lg px-4 py-3 focus:outline-none focus:border-blue-500 disabled:bg-slate-50 disabled:text-slate-500"
            />
          </div>
          <div>
            <label className="block text-xs font-bold text-slate-500 uppercase mb-2">Email Address</label>
            <input
              type="email"
              name="email"
              value={formData.email || ''}
              onChange={handleChange}
              disabled={!isEditing}
              className="w-full border border-slate-300 rounded-lg px-4 py-3 focus:outline-none focus:border-blue-500 disabled:bg-slate-50 disabled:text-slate-500"
            />
          </div>
          <div>
            <label className="block text-xs font-bold text-slate-500 uppercase mb-2">Phone Number</label>
            <input
              type="tel"
              name="phone"
              value={formData.phone || ''}
              onChange={handleChange}
              disabled={!isEditing}
              placeholder="+1 234 567 890"
              className="w-full border border-slate-300 rounded-lg px-4 py-3 focus:outline-none focus:border-blue-500 disabled:bg-slate-50 disabled:text-slate-500"
            />
          </div>
          {user.role === 'patient' && (
            <>
              <div>
                <label className="block text-xs font-bold text-slate-500 uppercase mb-2">Age</label>
                <input
                  type="number"
                  name="age"
                  value={formData.age || ''}
                  onChange={handleChange}
                  disabled={!isEditing}
                  className="w-full border border-slate-300 rounded-lg px-4 py-3 focus:outline-none focus:border-blue-500 disabled:bg-slate-50 disabled:text-slate-500"
                />
              </div>
              <div>
                <label className="block text-xs font-bold text-slate-500 uppercase mb-2">Blood Group</label>
                <input
                  type="text"
                  name="bloodGroup"
                  value={formData.bloodGroup || ''}
                  onChange={handleChange}
                  disabled={!isEditing}
                  className="w-full border border-slate-300 rounded-lg px-4 py-3 focus:outline-none focus:border-blue-500 disabled:bg-slate-50 disabled:text-slate-500"
                />
              </div>
              <div className="md:col-span-2">
                <label className="block text-xs font-bold text-slate-500 uppercase mb-2">Address</label>
                <input
                  type="text"
                  name="address"
                  value={formData.address || ''}
                  onChange={handleChange}
                  disabled={!isEditing}
                  className="w-full border border-slate-300 rounded-lg px-4 py-3 focus:outline-none focus:border-blue-500 disabled:bg-slate-50 disabled:text-slate-500"
                />
              </div>
            </>
          )}
          {user.role === 'doctor' && (
            <>
              <div>
                <label className="block text-xs font-bold text-slate-500 uppercase mb-2">Specialty</label>
                <input
                  type="text"
                  name="specialty"
                  value={formData.specialty || ''}
                  onChange={handleChange}
                  disabled={!isEditing}
                  className="w-full border border-slate-300 rounded-lg px-4 py-3 focus:outline-none focus:border-blue-500 disabled:bg-slate-50 disabled:text-slate-500"
                />
              </div>
              <div>
                <label className="block text-xs font-bold text-slate-500 uppercase mb-2">License Number</label>
                <input
                  type="text"
                  name="license"
                  value={formData.license || ''}
                  onChange={handleChange}
                  disabled={!isEditing}
                  className="w-full border border-slate-300 rounded-lg px-4 py-3 focus:outline-none focus:border-blue-500 disabled:bg-slate-50 disabled:text-slate-500"
                />
              </div>
              <div className="md:col-span-2">
                <label className="block text-xs font-bold text-slate-500 uppercase mb-2">Hospital / Clinic</label>
                <input
                  type="text"
                  name="hospital"
                  value={formData.hospital || ''}
                  onChange={handleChange}
                  disabled={!isEditing}
                  className="w-full border border-slate-300 rounded-lg px-4 py-3 focus:outline-none focus:border-blue-500 disabled:bg-slate-50 disabled:text-slate-500"
                />
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
