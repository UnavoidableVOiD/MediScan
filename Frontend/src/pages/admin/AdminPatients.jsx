import React, { useState, useEffect } from 'react';
import { adminApi } from '../../services/api';
import { toast } from 'react-toastify';
import { User, Phone, Mail } from 'lucide-react';

const AdminPatients = () => {
    const [patients, setPatients] = useState([]);
    const [loading, setLoading] = useState(false);

    useEffect(() => {
        fetchPatients();
    }, []);

    const fetchPatients = async () => {
        setLoading(true);
        try {
            const response = await adminApi.getPatients();
            setPatients(response.data);
        } catch (error) {
            toast.error("Failed to fetch patients");
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="space-y-6">
            <h1 className="text-2xl font-bold text-gray-900">Manage Patients</h1>

            {loading ? (
                <div className="flex justify-center py-12">
                    <div className="w-8 h-8 border-2 border-medic-dark/30 border-t-medic-dark rounded-full animate-spin" />
                </div>
            ) : patients.length === 0 ? (
                <div className="text-center py-12 text-gray-500 bg-white rounded-2xl border border-dashed border-gray-200">
                    No patients found.
                </div>
            ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                    {patients.map((patient) => (
                        <div key={patient.id} className="bg-white p-6 rounded-2xl shadow-sm border border-gray-100 hover:shadow-md transition-all">
                            <div className="flex items-center gap-4 mb-4">
                                <div className="w-12 h-12 bg-medic-accent/10 text-medic-accent rounded-full flex items-center justify-center font-bold text-lg">
                                    {patient.first_name[0]}{patient.last_name[0]}
                                </div>
                                <div>
                                    <h3 className="font-bold text-gray-900">{patient.first_name} {patient.last_name}</h3>
                                    <span className="text-xs bg-green-100 text-green-700 px-2 py-0.5 rounded-full font-bold">Patient</span>
                                </div>
                            </div>

                            <div className="space-y-3 text-sm text-gray-600">
                                <div className="flex items-center gap-3">
                                    <Mail className="w-4 h-4 text-gray-400" />
                                    <span className="truncate">{patient.email}</span>
                                </div>
                                <div className="flex items-center gap-3">
                                    <Phone className="w-4 h-4 text-gray-400" />
                                    <span>{patient.phone_number || 'N/A'}</span>
                                </div>
                            </div>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
};

export default AdminPatients;
