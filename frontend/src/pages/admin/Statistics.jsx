import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';

const SystemStatistics = () => {
    const data = [
        { name: 'Jan', reports: 400 },
        { name: 'Feb', reports: 300 },
        { name: 'Mar', reports: 550 },
        { name: 'Apr', reports: 450 },
        { name: 'May', reports: 600 },
        { name: 'Jun', reports: 700 },
    ];

    return (
        <div className="space-y-6">
            <h1 className="text-3xl font-bold tracking-tight text-primary">System Statistics</h1>
            <Card>
                <CardHeader>
                    <CardTitle>Reports Processed (Monthly)</CardTitle>
                </CardHeader>
                <CardContent className="pl-2">
                    <ResponsiveContainer width="100%" height={350}>
                        <BarChart data={data}>
                            <XAxis dataKey="name" stroke="#888888" fontSize={12} tickLine={false} axisLine={false} />
                            <YAxis stroke="#888888" fontSize={12} tickLine={false} axisLine={false} />
                            <Tooltip cursor={{ fill: 'transparent' }} />
                            <Bar dataKey="reports" fill="#2563EB" radius={[4, 4, 0, 0]} />
                        </BarChart>
                    </ResponsiveContainer>
                </CardContent>
            </Card>
        </div>
    );
};

export default SystemStatistics;
