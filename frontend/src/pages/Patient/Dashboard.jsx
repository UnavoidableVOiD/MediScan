import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { useAuth } from '../../contexts/AuthContext';
import healthLogo from '../../assets/health_logo.png';
import uploadLogo from '../../assets/upload_logo.png';
import reportsLogo from '../../assets/reports_logo.png';
import totalReportsLogo from '../../assets/total_reports_logo.png';
import recentReportsLogo from '../../assets/recent_reports_logo.png';
import chatsLogo from '../../assets/chats_logo.svg';
import analysisLogo from '../../assets/analysis_logo.svg';
import defaultMale from '../../assets/default_male.svg';
import defaultFemale from '../../assets/default_female.svg';

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
      logo: healthLogo,
      link: '/patient/health-dashboard',
      color: 'bg-red-50'
    },
    {
      title: 'Upload Report',
      description: 'Upload medical reports for AI analysis',
      logo: uploadLogo,
      link: '/patient/upload-report',
      color: 'bg-blue-50'
    },
    {
      title: 'View Reports',
      description: 'Access and manage your medical reports',
      logo: reportsLogo,
      link: '/patient/view-report',
      color: 'bg-green-50'
    }
  ];

  return (
    <div className="flex-1 min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 px-4 py-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8 flex flex-col md:flex-row md:items-center gap-6">
          <div className="w-20 h-20 rounded-full border-4 border-white shadow-lg overflow-hidden shrink-0">
            <img
              src={user?.profileImage || (user?.gender === 'female' ? defaultFemale : defaultMale)}
              alt=""
              className="w-full h-full object-cover"
            />
          </div>
          <div>
            <h1 className="text-4xl font-bold text-slate-900 mb-2">
              Welcome back{user?.name ? `, ${user.name.split(' ')[0]}` : ''}!
            </h1>
            <p className="text-slate-600 text-lg">Manage your health records and consultations</p>
          </div>
        </div>

        {/* Statistics Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <div className="bg-white p-6 rounded-xl shadow-lg border border-slate-200 hover:shadow-xl transition-shadow">
            <div className="flex items-center justify-between mb-4">
              <div className="w-12 h-12 bg-blue-50 rounded-lg overflow-hidden p-2">
                <img src={totalReportsLogo} alt="" className="w-full h-full object-contain" />
              </div>
            </div>
            <h3 className="text-2xl font-bold text-slate-900 mb-1">{stats.totalReports}</h3>
            <p className="text-slate-600 text-sm">Total Reports</p>
          </div>

          <div className="bg-white p-6 rounded-xl shadow-lg border border-slate-200 hover:shadow-xl transition-shadow">
            <div className="flex items-center justify-between mb-4">
              <div className="w-12 h-12 bg-green-50 rounded-lg overflow-hidden p-2">
                <img src={recentReportsLogo} alt="" className="w-full h-full object-contain" />
              </div>
            </div>
            <h3 className="text-2xl font-bold text-slate-900 mb-1">{stats.recentReports}</h3>
            <p className="text-slate-600 text-sm">Recent Reports</p>
          </div>

          <div className="bg-white p-6 rounded-xl shadow-lg border border-slate-200 hover:shadow-xl transition-shadow">
            <div className="flex items-center justify-between mb-4">
              <div className="w-12 h-12 rounded-lg overflow-hidden shadow-sm">
                <img src={chatsLogo} alt="" className="w-full h-full object-cover" />
              </div>
            </div>
            <h3 className="text-2xl font-bold text-slate-900 mb-1">{stats.activeChats}</h3>
            <p className="text-slate-600 text-sm">Active Chats</p>
          </div>

          <div className="bg-white p-6 rounded-xl shadow-lg border border-slate-200 hover:shadow-xl transition-shadow">
            <div className="flex items-center justify-between mb-4">
              <div className="w-12 h-12 rounded-lg overflow-hidden shadow-sm">
                <img src={analysisLogo} alt="" className="w-full h-full object-cover" />
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
                <div className={`w-16 h-16 rounded-2xl ${action.color} flex items-center justify-center mb-4 group-hover:scale-110 transition-transform shadow-sm overflow-hidden p-3`}>
                  <img src={action.logo} alt="" className="w-full h-full object-contain" />
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
