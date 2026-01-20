import React from 'react';
import { Link } from 'react-router-dom';

export default function Home() {
  return (
    <div className="flex-1 min-h-screen flex items-center justify-center bg-slate-50 px-4">
      <div className="w-full max-w-2xl">
        <div className="text-center mb-8">
          <img src="/logo.jpg" alt="MediScan Logo" className="h-20 w-auto mx-auto mb-6 mix-blend-multiply" />
          <div className="inline-block p-2 rounded-full bg-green-100 text-green-600 mb-4 text-[10px] md:text-xs font-bold tracking-wide uppercase">
            AI-Powered Healthcare
          </div>
          <h1 className="text-slate-900 font-semibold text-3xl md:text-5xl leading-tight mb-4">
            Understand Your Medical Reports in <span className="text-green-600">Seconds</span>
          </h1>
          <p className="text-base md:text-lg text-slate-600 max-w-2xl mx-auto mb-8 px-2">
            MediScan uses advanced AI to analyze blood reports, X-rays, and prescriptions. Get instant
            explanations, risk predictions, and connect with doctors.
          </p>
        </div>

        <div className="flex justify-center gap-4 flex-col sm:flex-row">
          <button
            onClick={() => window.location.href = '/check-reports?demo=true'}
            className="bg-green-600 text-white font-bold px-8 py-4 rounded-xl shadow-lg hover:bg-green-700 transition transform hover:-translate-y-1 hover:shadow-xl w-full sm:w-auto"
          >
            Try Demo (Instant Report)
          </button>
          <Link
            to="/login"
            className="bg-white text-slate-800 font-bold px-8 py-4 rounded-xl shadow-md hover:bg-slate-50 transition border border-slate-200 block text-center w-full sm:w-auto transform hover:-translate-y-1 hover:shadow-lg"
          >
            Login / Sign Up
          </Link>
        </div>
      </div>
    </div >
  );
}