import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { useAuth } from '../../contexts/AuthContext';

function Dashboard() {
  const { user } = useAuth();
  const [stats, setStats] = useState({
    totalReports: 0,
    recentReports: 0,
    activeChats: 0,
    pendingAnalysis: 0
  });

  useEffect(() => {
    // Fetch dashboard statistics from API
    // setStats(data);
  }, []);

  const quickActions = [
    {
      title: 'Health Dashboard',
      description: 'Monitor your health metrics and trends',
      icon: 'fa-heart-pulse',
      link: '/patient/health-dashboard',
      color: 'from-red-500 to-red-600'
    },
    {
      title: 'Upload Report',
      description: 'Upload medical reports for AI analysis',
      icon: 'fa-file-upload',
      link: '/patient/upload-report',
      color: 'from-blue-500 to-blue-600'
    },
    {
      title: 'View Reports',
      description: 'Access and manage your medical reports',
      icon: 'fa-folder-open',
      link: '/patient/view-report',
      color: 'from-green-500 to-green-600'
    }
  ];

  return (
    <div className="flex-1 min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 px-4 py-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-slate-900 mb-2">
            Welcome back{user?.name ? `, ${user.name.split(' ')[0]}` : ''}!
          </h1>
          <p className="text-slate-600 text-lg">Manage your health records and consultations</p>
        </div>

        {/* Statistics Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <div className="bg-white p-6 rounded-xl shadow-lg border border-slate-200 hover:shadow-xl transition-shadow">
            <div className="flex items-center justify-between mb-4">
              <div className="p-3 bg-blue-100 rounded-lg">
                <i className="fa-solid fa-file-medical text-blue-600 text-xl"></i>
              </div>
            </div>
            <h3 className="text-2xl font-bold text-slate-900 mb-1">{stats.totalReports}</h3>
            <p className="text-slate-600 text-sm">Total Reports</p>
          </div>

          <div className="bg-white p-6 rounded-xl shadow-lg border border-slate-200 hover:shadow-xl transition-shadow">
            <div className="flex items-center justify-between mb-4">
              <div className="p-3 bg-green-100 rounded-lg">
                <i className="fa-solid fa-clock-rotate-left text-green-600 text-xl"></i>
              </div>
            </div>
            <h3 className="text-2xl font-bold text-slate-900 mb-1">{stats.recentReports}</h3>
            <p className="text-slate-600 text-sm">Recent Reports</p>
          </div>

          <div className="bg-white p-6 rounded-xl shadow-lg border border-slate-200 hover:shadow-xl transition-shadow">
            <div className="flex items-center justify-between mb-4">
              <div className="p-3 bg-purple-100 rounded-lg">
                <i className="fa-solid fa-message text-purple-600 text-xl"></i>
              </div>
            </div>
            <h3 className="text-2xl font-bold text-slate-900 mb-1">{stats.activeChats}</h3>
            <p className="text-slate-600 text-sm">Active Chats</p>
          </div>

          <div className="bg-white p-6 rounded-xl shadow-lg border border-slate-200 hover:shadow-xl transition-shadow">
            <div className="flex items-center justify-between mb-4">
              <div className="p-3 bg-orange-100 rounded-lg">
                <i className="fa-solid fa-hourglass-half text-orange-600 text-xl"></i>
              </div>
            </div>
            <h3 className="text-2xl font-bold text-slate-900 mb-1">{stats.pendingAnalysis}</h3>
            <p className="text-slate-600 text-sm">Pending Analysis</p>
          </div>
        </div>

        {/* Quick Actions */}
        <div className="mb-8">
          <h2 className="text-2xl font-bold text-slate-900 mb-6">Quick Actions</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {quickActions.map((action, index) => (
              <Link
                key={index}
                to={action.link}
                className="group bg-white p-6 rounded-xl shadow-lg hover:shadow-2xl transition-all duration-300 border border-slate-200 hover:-translate-y-1"
              >
                <div className={`w-14 h-14 rounded-xl bg-gradient-to-br ${action.color} flex items-center justify-center mb-4 group-hover:scale-110 transition-transform`}>
                  <i className={`fa-solid ${action.icon} text-white text-2xl`}></i>
                </div>
                <h3 className="text-xl font-bold text-slate-900 mb-2 group-hover:text-green-600 transition-colors">
                  {action.title}
                </h3>
                <p className="text-slate-600 text-sm">{action.description}</p>
              </Link>
            ))}
          </div>
        </div>

        {/* Recent Activity Placeholder */}
        <div className="bg-white p-6 rounded-xl shadow-lg border border-slate-200">
          <h2 className="text-2xl font-bold text-slate-900 mb-4">Recent Activity</h2>
          <div className="text-center text-slate-500 py-8">
            <i className="fa-solid fa-inbox text-4xl mb-4 opacity-50"></i>
            <p>No recent activity to display</p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Dashboard;
