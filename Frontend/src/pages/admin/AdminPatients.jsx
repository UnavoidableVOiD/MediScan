import React, { useEffect } from 'react';
import { User, Phone, Mail, ShieldAlert, Trash2 } from 'lucide-react';
import { useSelector, useDispatch } from 'react-redux';
import { fetchAdminPatients, updatePatient, deletePatient } from '../../store/slices/adminSlice';

const AdminPatients = () => {
    const dispatch = useDispatch();
    const { user } = useSelector(state => state.auth);
    const { patients, loading } = useSelector(state => state.admin);

    useEffect(() => {
        dispatch(fetchAdminPatients());
    }, [dispatch]);

    const handleUpdatePatient = async (id, data) => {
        try {
            await dispatch(updatePatient({ id, data })).unwrap();
        } catch (error) {
            // toast handled by slice
        }
    };

    const handleDeletePatient = async (id) => {
        if (!window.confirm("Are you sure you want to delete this patient? This action cannot be undone.")) return;
        try {
            await dispatch(deletePatient(id)).unwrap();
        } catch (error) {
            // toast handled by slice
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
                                    {(patient.first_name?.[0] || '') + (patient.last_name?.[0] || '')}
                                </div>
                                <div>
                                    <h3 className="font-bold text-gray-900">{patient.first_name} {patient.last_name}</h3>
                                    <div className="flex gap-2 items-center">
                                        <span className="text-xs bg-green-100 text-green-700 px-2 py-0.5 rounded-full font-bold">Patient</span>
                                        {!patient.is_active && (
                                            <span className="text-xs bg-red-100 text-red-700 px-2 py-0.5 rounded-full font-bold flex items-center gap-1">
                                                <ShieldAlert className="w-3 h-3" />
                                                Blocked
                                            </span>
                                        )}
                                    </div>
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

                            <div className="mt-6 pt-4 border-t flex gap-2">
                                {(user?.is_staff || user?.is_superuser) && (
                                    <>
                                        <button
                                            onClick={() => handleUpdatePatient(patient.id, { is_active: !patient.is_active })}
                                            title={patient.is_active ? "Block Patient" : "Unblock Patient"}
                                            className={`flex-1 py-2 rounded-lg font-bold text-xs flex items-center justify-center gap-2 transition-all ${patient.is_active
                                                ? 'bg-orange-50 text-orange-600 hover:bg-orange-100'
                                                : 'bg-orange-600 text-white hover:bg-orange-700'
                                                }`}
                                        >
                                            <ShieldAlert className="w-3.5 h-3.5" />
                                            {patient.is_active ? 'Block' : 'Unblock'}
                                        </button>
                                        <button
                                            onClick={() => handleDeletePatient(patient.id)}
                                            title="Delete Patient"
                                            className="flex-1 py-2 rounded-lg font-bold text-xs bg-red-50 text-red-600 hover:bg-red-100 flex items-center justify-center gap-2 transition-all"
                                        >
                                            <Trash2 className="w-3.5 h-3.5" />
                                            Delete
                                        </button>
                                    </>
                                )}
                            </div>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
};

export default AdminPatients;
