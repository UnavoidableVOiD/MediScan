import React from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { mockReports, mockHealthStats } from '@/lib/mockData';
import { Link } from 'react-router-dom';
import { Activity, FileText, AlertTriangle, ArrowRight } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const PatientDashboard = () => {
    return (
        <div className="space-y-6">
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-3xl font-bold tracking-tight text-primary">Dashboard</h1>
                    <p className="text-muted-foreground">Welcome back, here's your health overview.</p>
                </div>
                <Button asChild>
                    <Link to="/patient/upload-report">Upload New Report</Link>
                </Button>
            </div>

            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
                <Card>
                    <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                        <CardTitle className="text-sm font-medium">Total Reports</CardTitle>
                        <FileText className="h-4 w-4 text-muted-foreground" />
                    </CardHeader>
                    <CardContent>
                        <div className="text-2xl font-bold">{mockReports.length}</div>
                        <p className="text-xs text-muted-foreground">+1 from last month</p>
                    </CardContent>
                </Card>
                <Card>
                    <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                        <CardTitle className="text-sm font-medium">Health Status</CardTitle>
                        <Activity className="h-4 w-4 text-secondary" />
                    </CardHeader>
                    <CardContent>
                        <div className="text-2xl font-bold text-secondary">Normal</div>
                        <p className="text-xs text-muted-foreground">Last checkup: 5 days ago</p>
                    </CardContent>
                </Card>
                <Card>
                    <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                        <CardTitle className="text-sm font-medium">Pending Actions</CardTitle>
                        <AlertTriangle className="h-4 w-4 text-accent" />
                    </CardHeader>
                    <CardContent>
                        <div className="text-2xl font-bold">1</div>
                        <p className="text-xs text-muted-foreground">Doctor review pending</p>
                    </CardContent>
                </Card>
            </div>

            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-7">
                <Card className="col-span-4">
                    <CardHeader>
                        <CardTitle>Health Trends</CardTitle>
                    </CardHeader>
                    <CardContent className="pl-2">
                        <ResponsiveContainer width="100%" height={300}>
                            <LineChart data={mockHealthStats}>
                                <CartesianGrid strokeDasharray="3 3" />
                                <XAxis dataKey="name" stroke="#888888" fontSize={12} tickLine={false} axisLine={false} />
                                <YAxis stroke="#888888" fontSize={12} tickLine={false} axisLine={false} tickFormatter={(value) => `${value}`} />
                                <Tooltip />
                                <Line type="monotone" dataKey="bpm" stroke="#2563EB" strokeWidth={2} dot={{ r: 4 }} activeDot={{ r: 8 }} />
                                <Line type="monotone" dataKey="sugar" stroke="#F97316" strokeWidth={2} />
                            </LineChart>
                        </ResponsiveContainer>
                    </CardContent>
                </Card>
                <Card className="col-span-3">
                    <CardHeader>
                        <CardTitle>Recent Reports</CardTitle>
                        <CardDescription>
                            You have {mockReports.length} reports this year.
                        </CardDescription>
                    </CardHeader>
                    <CardContent>
                        <div className="space-y-4">
                            {mockReports.map((report) => (
                                <div key={report.id} className="flex items-center justify-between border-b pb-2 last:border-0 last:pb-0">
                                    <div className="space-y-1">
                                        <p className="text-sm font-medium leading-none">{report.type}</p>
                                        <p className="text-xs text-muted-foreground">{report.date} • {report.doctor}</p>
                                    </div>
                                    <div className={`text-xs font-bold px-2 py-1 rounded-full ${report.risk === 'Low' ? 'bg-secondary/20 text-secondary' :
                                            report.risk === 'Medium' ? 'bg-accent/20 text-accent' : 'bg-destructive/20 text-destructive'
                                        }`}>
                                        {report.risk}
                                    </div>
                                </div>
                            ))}
                            <Button variant="ghost" className="w-full text-xs" asChild>
                                <Link to="/patient/health-dashboard">
                                    View All <ArrowRight className="ml-2 h-3 w-3" />
                                </Link>
                            </Button>
                        </div>
                    </CardContent>
                </Card>
            </div>
        </div>
    );
};

export default PatientDashboard;
