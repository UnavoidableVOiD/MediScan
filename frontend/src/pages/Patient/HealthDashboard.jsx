import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';

function HealthDashboard() {
  const [healthMetrics] = useState({
    bloodPressure: { systolic: 120, diastolic: 80 },
    heartRate: 72,
    temperature: 98.6,
    weight: 70,
    bloodSugar: 90
  });
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Fetch health metrics from API
    // setHealthMetrics(data);
    setTimeout(() => setLoading(false), 500);
  }, []);

  const getStatusColor = (type, value) => {
    // Status color logic based on health metrics ranges
    if (type === 'bloodPressure') {
      if (value.systolic > 140 || value.diastolic > 90) return 'text-red-600';
      if (value.systolic > 120 || value.diastolic > 80) return 'text-yellow-600';
      return 'text-green-600';
    }
    return 'text-blue-600';
  };

  const healthCards = [
    {
      title: 'Blood Pressure',
      value: `${healthMetrics.bloodPressure.systolic}/${healthMetrics.bloodPressure.diastolic}`,
      unit: 'mmHg',
      icon: 'fa-heartbeat',
      color: 'from-red-500 to-red-600',
      bgColor: 'bg-red-50',
      status: 'Normal'
    },
    {
      title: 'Heart Rate',
      value: healthMetrics.heartRate,
      unit: 'bpm',
      icon: 'fa-heart',
      color: 'from-green-500 to-green-600',
      bgColor: 'bg-green-50',
      status: 'Normal'
    },
    {
      title: 'Temperature',
      value: healthMetrics.temperature,
      unit: '°F',
      icon: 'fa-temperature-high',
      color: 'from-orange-500 to-orange-600',
      bgColor: 'bg-orange-50',
      status: 'Normal'
    },
    {
      title: 'Weight',
      value: healthMetrics.weight,
      unit: 'kg',
      icon: 'fa-weight-scale',
      color: 'from-blue-500 to-blue-600',
      bgColor: 'bg-blue-50',
      status: 'Normal'
    },
    {
      title: 'Blood Sugar',
      value: healthMetrics.bloodSugar,
      unit: 'mg/dL',
      icon: 'fa-vial',
      color: 'from-purple-500 to-purple-600',
      bgColor: 'bg-purple-50',
      status: 'Normal'
    }
  ];

  if (loading) {
    return (
      <div className="flex-1 min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 flex items-center justify-center">
        <div className="text-center">
          <i className="fa-solid fa-spinner fa-spin text-4xl text-green-600 mb-4"></i>
          <p className="text-slate-600">Loading health metrics...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 px-4 py-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8 flex items-center justify-between">
          <div>
            <h1 className="text-4xl font-bold text-slate-900 mb-2">Health Dashboard</h1>
            <p className="text-slate-600 text-lg">Monitor your health metrics and trends</p>
          </div>
          <Link
            to="/patient/upload-report"
            className="bg-green-600 text-white px-6 py-3 rounded-lg font-semibold hover:bg-green-700 transition shadow-lg hover:shadow-xl"
          >
            <i className="fa-solid fa-plus mr-2"></i>Add Record
          </Link>
        </div>

        {/* Health Metrics Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
          {healthCards.map((card, index) => (
            <div
              key={index}
              className="bg-white p-6 rounded-xl shadow-lg border border-slate-200 hover:shadow-xl transition-all hover:-translate-y-1"
            >
              <div className="flex items-start justify-between mb-4">
                <div className={`p-3 rounded-xl bg-gradient-to-br ${card.color}`}>
                  <i className={`fa-solid ${card.icon} text-white text-xl`}></i>
                </div>
                <span className="px-3 py-1 text-xs font-semibold rounded-full bg-green-100 text-green-700">
                  {card.status}
                </span>
              </div>
              <h3 className="text-lg font-semibold text-slate-700 mb-2">{card.title}</h3>
              <div className="flex items-baseline gap-2">
                <p className={`text-4xl font-bold ${getStatusColor('bloodPressure', card.value)}`}>
                  {card.value}
                </p>
                <p className="text-slate-500 text-lg">{card.unit}</p>
              </div>
            </div>
          ))}
        </div>

        {/* Health Trends Chart Placeholder */}
        <div className="bg-white p-6 rounded-xl shadow-lg border border-slate-200 mb-8">
          <h2 className="text-2xl font-bold text-slate-900 mb-4">Health Trends</h2>
          <div className="h-64 flex items-center justify-center bg-slate-50 rounded-lg border-2 border-dashed border-slate-300">
            <div className="text-center">
              <i className="fa-solid fa-chart-line text-4xl text-slate-400 mb-3"></i>
              <p className="text-slate-500">Chart visualization will appear here</p>
              <p className="text-sm text-slate-400 mt-1">Track your health metrics over time</p>
            </div>
          </div>
        </div>

        {/* Recent Health Records */}
        <div className="bg-white p-6 rounded-xl shadow-lg border border-slate-200">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-2xl font-bold text-slate-900">Recent Health Records</h2>
            <Link
              to="/patient/view-report"
              className="text-green-600 hover:text-green-700 font-semibold text-sm"
            >
              View All <i className="fa-solid fa-arrow-right ml-1"></i>
            </Link>
          </div>
          <div className="text-center text-slate-500 py-12">
            <i className="fa-solid fa-clipboard-list text-5xl mb-4 opacity-50"></i>
            <p className="mb-2">No recent health records found</p>
            <p className="text-sm text-slate-400">Upload reports to track your health metrics</p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default HealthDashboard;
