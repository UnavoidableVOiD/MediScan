import React from 'react';

export default function About() {
  return (
    <div className="flex-1 min-h-screen flex items-center justify-center bg-slate-50 px-4">
      <div className="w-full max-w-2xl">
        <div className="text-center mb-12">
          <div className="inline-block p-2 rounded-full bg-green-100 text-green-600 mb-4 text-[10px] md:text-xs font-bold tracking-wide uppercase">
            About MediScan
          </div>
          <h1 className="text-slate-900 font-semibold text-3xl md:text-5xl leading-tight mb-4">
            Making Healthcare <span className="text-green-600">Accessible</span>
          </h1>
        </div>

        <div className="space-y-8">
          <div className="bg-white rounded-xl p-6 md:p-8 shadow-md border border-slate-200">
            <h2 className="text-2xl font-bold text-slate-900 mb-3">Our Mission</h2>
            <p className="text-slate-600 leading-relaxed">
              MediScan is revolutionizing healthcare by using advanced AI to make medical reports understandable to everyone. 
              We believe that patients should have instant access to explanations of their medical tests, helping them make informed decisions about their health.
            </p>
          </div>

          <div className="bg-white rounded-xl p-6 md:p-8 shadow-md border border-slate-200">
            <h2 className="text-2xl font-bold text-slate-900 mb-3">What We Do</h2>
            <p className="text-slate-600 leading-relaxed mb-4">
              We provide an AI-powered platform that analyzes:
            </p>
            <ul className="space-y-2 text-slate-600">
              <li className="flex items-center gap-3">
                <span className="w-2 h-2 bg-green-600 rounded-full"></span>
                Blood reports and lab tests
              </li>
              <li className="flex items-center gap-3">
                <span className="w-2 h-2 bg-green-600 rounded-full"></span>
                X-rays and medical imaging
              </li>
              <li className="flex items-center gap-3">
                <span className="w-2 h-2 bg-green-600 rounded-full"></span>
                Prescriptions and medications
              </li>
              <li className="flex items-center gap-3">
                <span className="w-2 h-2 bg-green-600 rounded-full"></span>
                Risk predictions and health insights
              </li>
            </ul>
          </div>

          <div className="bg-white rounded-xl p-6 md:p-8 shadow-md border border-slate-200">
            <h2 className="text-2xl font-bold text-slate-900 mb-3">Why Choose Us?</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <h3 className="font-semibold text-slate-900 mb-2 flex items-center gap-2">
                  <i className="fa-solid fa-bolt text-green-600"></i> Instant Results
                </h3>
                <p className="text-slate-600 text-sm">Get explanations in seconds, not hours</p>
              </div>
              <div>
                <h3 className="font-semibold text-slate-900 mb-2 flex items-center gap-2">
                  <i className="fa-solid fa-lock text-green-600"></i> Secure & Private
                </h3>
                <p className="text-slate-600 text-sm">Your health data is encrypted and secure</p>
              </div>
              <div>
                <h3 className="font-semibold text-slate-900 mb-2 flex items-center gap-2">
                  <i className="fa-solid fa-user-doctor text-green-600"></i> Doctor Support
                </h3>
                <p className="text-slate-600 text-sm">Connect with qualified healthcare professionals</p>
              </div>
              <div>
                <h3 className="font-semibold text-slate-900 mb-2 flex items-center gap-2">
                  <i className="fa-solid fa-brain text-green-600"></i> AI-Powered
                </h3>
                <p className="text-slate-600 text-sm">Advanced technology for accurate analysis</p>
              </div>
            </div>
          </div>

          <div className="bg-green-50 rounded-xl p-6 md:p-8 border border-green-200">
            <p className="text-slate-600 leading-relaxed mb-4">
              MediScan is your trusted companion in understanding your health. Whether you're a patient looking to understand your medical reports 
              or a healthcare professional seeking better tools, we're here to help.
            </p>
            <p className="text-slate-600 leading-relaxed">
              Join thousands of users who are taking control of their health with MediScan.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
