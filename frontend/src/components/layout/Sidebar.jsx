import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { cn } from '@/lib/utils';
import {
    LayoutDashboard,
    FileText,
    MessageSquare,
    Activity,
    Upload,
    Users,
    ShieldCheck,
    Stethoscope,
    Pill,
    LogOut
} from 'lucide-react';
import { useAuthStore } from '@/store/authStore';
import { Button } from '@/components/ui/button';

const Sidebar = () => {
    const { user, logout } = useAuthStore();
    const location = useLocation();
    const pathname = location.pathname;

    if (!user) return null;

    const links = {
        patient: [
            { href: '/patient', label: 'Dashboard', icon: LayoutDashboard },
            { href: '/patient/upload-report', label: 'Upload Report', icon: Upload },
            { href: '/patient/health-dashboard', label: 'Health Status', icon: Activity },
            { href: '/patient/chat', label: 'MediBot AI', icon: MessageSquare },
        ],
        doctor: [
            { href: '/doctor', label: 'Dashboard', icon: LayoutDashboard },
            { href: '/doctor/patients', label: 'My Patients', icon: Users },
            { href: '/doctor/upload-license', label: 'License Setup', icon: ShieldCheck },
        ],
        admin: [
            { href: '/admin', label: 'Dashboard', icon: LayoutDashboard },
            { href: '/admin/verify-doctors', label: 'Verify Doctors', icon: ShieldCheck },
            { href: '/admin/statistics', label: 'Statistics', icon: Activity },
            { href: '/admin/users', label: 'Manage Users', icon: Users },
        ]
    };

    const navItems = links[user.role] || [];

    return (
        <div className="flex flex-col h-screen w-64 bg-card border-r fixed left-0 top-0">
            <div className="p-6 border-b">
                <h2 className="text-2xl font-bold text-primary flex items-center gap-2">
                    <img src="/logo-v2.png" alt="MediScan" className="h-14 w-auto object-contain" />
                </h2>


                <span className="text-xs text-muted-foreground uppercase tracking-wider mt-1 block">
                    {user.role} Portal
                </span>
            </div>

            <nav className="flex-1 p-4 space-y-2 overflow-y-auto">
                {navItems.map((item) => {
                    const Icon = item.icon;
                    const isActive = pathname === item.href || (pathname.startsWith(item.href) && item.href !== `/${user.role}`);

                    return (
                        <Link
                            key={item.href}
                            to={item.href}
                            className={cn(
                                "flex items-center gap-3 px-3 py-2 rounded-md transition-colors text-sm font-medium",
                                isActive
                                    ? "bg-primary/10 text-primary"
                                    : "text-muted-foreground hover:bg-muted hover:text-foreground"
                            )}
                        >
                            <Icon className="w-4 h-4" />
                            {item.label}
                        </Link>
                    );
                })}
            </nav>

            <div className="p-4 border-t">
                <div className="mb-4 flex items-center gap-3 px-2">
                    <div className="w-8 h-8 rounded-full bg-primary/20 flex items-center justify-center text-primary font-bold">
                        {user.name.charAt(0)}
                    </div>
                    <div className="overflow-hidden">
                        <p className="text-sm font-medium truncate">{user.name}</p>
                        <p className="text-xs text-muted-foreground truncate">{user.email}</p>
                    </div>
                </div>
                <Button
                    variant="outline"
                    className="w-full justify-start gap-2 text-destructive hover:text-destructive hover:bg-destructive/10 border-destructive/20"
                    onClick={logout}
                >
                    <LogOut className="w-4 h-4" />
                    Sign Out
                </Button>
            </div>
        </div>
    );
};

export default Sidebar;
