import React from 'react';
import { Link } from 'react-router-dom';
export default function Home() {
  return (
    <div className="flex-1 relative pt-16 pb-32 flex content-center items-center justify-center min-h-[calc(100vh-64px)]">
      <div
        className="absolute top-0 w-full h-full bg-center bg-cover"
        style={{
          backgroundImage:
            "url('https://images.unsplash.com/photo-1576091160399-112ba8d25d1d?ixlib=rb-1.2.1&auto=format&fit=crop&w=1950&q=80')",
          opacity: 0.1
        }}
      ></div>

      <div className="container relative mx-auto px-4 text-center fade-in">
        <div className="inline-block p-2 rounded-full bg-green-100 text-green-600 mb-4 text-xs font-bold tracking-wide uppercase">
          AI-Powered Healthcare
        </div>
        <h1 className="text-slate-900 font-semibold text-5xl leading-tight mb-4">
          Understand Your Medical Reports in <span className="text-green-600">Seconds</span>
        </h1>
        <p className="text-lg text-slate-600 max-w-2xl mx-auto mb-8">
          MediScan uses advanced AI to analyze blood reports, X-rays, and prescriptions. Get instant
          explanations, risk predictions, and connect with doctors.
        </p>
        <div className="flex justify-center gap-4 flex-wrap">
          <button
            onClick={() => window.location.href = '/check-reports?demo=true'}
            className="bg-green-600 text-white font-bold px-8 py-4 rounded-xl shadow-lg hover:bg-green-700 transition transform hover:-translate-y-1"
          >
            Try Demo (Instant Report)
          </button>
          <Link
            to="/login"
            className="bg-white text-slate-800 font-bold px-8 py-4 rounded-xl shadow-md hover:bg-slate-50 transition border border-slate-200"
          >
            For Doctors
          </Link>
        </div>
      </div>
    </div>
  );
}