import React from 'react';
import { Navigate, useNavigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext.jsx';
export default function Membership() {
  const { user, updateUser } = useAuth();
  const navigate = useNavigate();
  if (!user || user.role !== 'patient') return <Navigate to="/" />;
  const handleUpgrade = () => {
    if (confirm('Confirm purchase of Premium Plan for $9.99/mo?')) {
      updateUser({ plan: 'premium' });
      alert('Welcome to MediScan Premium! You now have unlimited access.');
      navigate('/dashboard');
    }
  };
  return (
    <div className="max-w-6xl mx-auto px-4 py-16 fade-in text-center">
      <h2 className="text-3xl font-bold text-slate-900 mb-4">Choose the Right Plan for You</h2>
      <p className="text-slate-500 mb-12 max-w-2xl mx-auto">
        Unlock the full power of AI health analysis with our premium features. Get detailed insights,
        risk graphs, and unlimited doctor support.
      </p>
      <div className="grid md:grid-cols-2 gap-8 max-w-4xl mx-auto">
        <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-8 flex flex-col relative">
          <div className="text-lg font-bold text-slate-900 mb-2">Basic</div>
          <div className="text-4xl font-extrabold text-slate-900 mb-6">Free</div>
          <ul className="space-y-4 text-left mb-8 flex-1">
            <li className="flex items-center gap-3">
              <i className="fa-solid fa-check text-green-500"></i> Basic Report Summary
            </li>
            <li className="flex items-center gap-3">
              <i className="fa-solid fa-check text-green-500"></i> Critical Alert Notifications
            </li>
            <li className="flex items-center gap-3">
              <i className="fa-solid fa-check text-green-500"></i> 5 Chatbot Messages / Day
            </li>
            <li className="flex items-center gap-3 text-slate-400">
              <i className="fa-solid fa-xmark"></i> Detailed Risk Graphs
            </li>
            <li className="flex items-center gap-3 text-slate-400">
              <i className="fa-solid fa-xmark"></i> Priority Doctor Booking
            </li>
          </ul>
          {user.plan !== 'premium' ? (
            <button
              disabled
              className="w-full bg-slate-100 text-slate-400 font-bold py-3 rounded-xl cursor-default"
            >
              Current Plan
            </button>
          ) : (
            <button
              onClick={() => updateUser({ plan: 'free' })}
              className="w-full bg-slate-100 text-slate-600 font-bold py-3 rounded-xl hover:bg-slate-200"
            >
              Downgrade
            </button>
          )}
        </div>
        <div className="bg-slate-900 text-white rounded-2xl shadow-xl p-8 flex flex-col relative transform scale-105 border-4 border-blue-500">
          <div className="absolute top-0 right-0 bg-blue-500 text-white text-xs font-bold px-3 py-1 rounded-bl-xl rounded-tr-lg">
            RECOMMENDED
          </div>
          <div className="text-lg font-bold text-blue-400 mb-2">Premium</div>
          <div className="text-4xl font-extrabold mb-6">
            $9.99 <span className="text-lg text-slate-400 font-medium">/mo</span>
          </div>
          <ul className="space-y-4 text-left mb-8 flex-1">
            <li className="flex items-center gap-3">
              <i className="fa-solid fa-check text-blue-400"></i> Everything in Free
            </li>
            <li className="flex items-center gap-3">
              <i className="fa-solid fa-check text-blue-400"></i> Detailed Disease Risk Graphs
            </li>
            <li className="flex items-center gap-3">
              <i className="fa-solid fa-check text-blue-400"></i> Unlimited AI Chatbot
            </li>
            <li className="flex items-center gap-3">
              <i className="fa-solid fa-check text-blue-400"></i> Instant Doctor Booking
            </li>
            <li className="flex items-center gap-3">
              <i className="fa-solid fa-check text-blue-400"></i> Family Profile Support
            </li>
          </ul>
          {user.plan === 'premium' ? (
            <button
              disabled
              className="w-full bg-green-500 text-white font-bold py-3 rounded-xl cursor-default"
            >
              Active Plan
            </button>
          ) : (
            <button
              onClick={handleUpgrade}
              className="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 rounded-xl transition shadow-lg shadow-blue-900/50"
            >
              Upgrade Now
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
