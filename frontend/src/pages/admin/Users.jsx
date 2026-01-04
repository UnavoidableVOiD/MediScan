import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Trash2, Ban } from 'lucide-react';

const ManageUsers = () => {
    // Dummy user list
    const users = [
        { id: 1, name: 'Alex Carter', role: 'Patient', status: 'Active' },
        { id: 2, name: 'Dr. Sarah Smith', role: 'Doctor', status: 'Active' },
        { id: 3, name: 'John Doe', role: 'Patient', status: 'Blocked' },
    ];

    return (
        <div className="space-y-6">
            <h1 className="text-3xl font-bold tracking-tight text-primary">Manage Users</h1>
            <Card>
                <CardHeader>
                    <CardTitle>All System Users</CardTitle>
                </CardHeader>
                <CardContent>
                    <div className="space-y-4">
                        {users.map((u) => (
                            <div key={u.id} className="flex items-center justify-between border-b pb-4 last:border-0 last:pb-0">
                                <div>
                                    <p className="font-medium">{u.name}</p>
                                    <p className="text-sm text-muted-foreground">{u.role} • {u.status}</p>
                                </div>
                                <div className="flex gap-2">
                                    <Button size="sm" variant="outline" className="text-orange-600 border-orange-200 hover:bg-orange-50">
                                        <Ban className="w-4 h-4 mr-1" /> {u.status === 'Active' ? 'Block' : 'Unblock'}
                                    </Button>
                                    <Button size="sm" variant="outline" className="text-destructive border-destructive hover:bg-destructive/10">
                                        <Trash2 className="w-4 h-4 mr-1" /> Remove
                                    </Button>
                                </div>
                            </div>
                        ))}
                    </div>
                </CardContent>
            </Card>
        </div>
    );
};

export default ManageUsers;
