import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api';
export const BASE_URL = API_URL.replace('/api', '');

const api = axios.create({
    baseURL: API_URL,
    withCredentials: true,
    headers: {
        'Content-Type': 'application/json',
    },
});

export const authApi = {
    // Corrected to match doctor/urls.py
    submitDoctorVerification: (formData) => api.put('/doctor/verify/', formData, {
        headers: {
            'Content-Type': 'multipart/form-data',
        },
    }),
    updateProfile: (data) => api.patch('/auth/profile/', data),
};

export const doctorApi = {
    getStats: () => api.get('/doctor/my-patients/stats/'),
    getPatients: () => api.get('/doctor/my-patients/'),
    getPatientReports: (id) => api.get(`/doctor/my-patients/${id}/reports/`),
    updatePatientNotes: (id, notes) => api.post(`/doctor/my-patients/${id}/update_notes/`, { notes }),
    updateClinicalObservations: (id, observations) => api.post(`/doctor/my-patients/${id}/update_observations/`, { clinical_observations: observations }),
    markPatientCompleted: (id) => api.post(`/doctor/my-patients/${id}/complete/`),
    addComment: (data) => api.post('/doctor/comments/', data),
    getPatientTrends: (patientId) => api.get(`/reports/trends/?patient_id=${patientId}`),
};

export const reportApi = {
    getTrends: () => api.get('/reports/trends/'),
};

export const appointmentApi = {
    getAvailability: (doctorId) => api.get(`/doctor/availability/?doctor=${doctorId}`),
    manageAvailability: (data) => api.post('/doctor/availability/', data),
    syncAvailability: (data, doctorId) => {
        const url = doctorId ? `/doctor/availability/sync/?doctor=${doctorId}` : '/doctor/availability/sync/';
        return api.post(url, data);
    },
    deleteAvailability: (id) => api.delete(`/doctor/availability/${id}/`),
    getAppointments: () => api.get('/doctor/appointments/'),
    bookAppointment: (data) => api.post('/doctor/appointments/', data),
    verifyPayment: (appointmentId, data) => api.post(`/doctor/appointments/${appointmentId}/verify_payment/`, data),
    initiateKhaltiPayment: (data) => api.post('/doctor/payment/khalti/init/', data),
    verifyKhaltiPayment: (data) => api.post('/doctor/payment/khalti/verify/', data),
    getRecommendedDoctors: (params) => {
        const queryParams = new URLSearchParams(params).toString();
        return api.get(`/doctor/list/?${queryParams}`);
    },
    getBookedSlots: (doctorId, date) => api.get(`/doctor/appointments/booked_slots/?doctor=${doctorId}&date=${date}`),
    getDoctorById: (id) => api.get(`/doctor/list/${id}/`),
    cancelAppointment: (id) => api.post(`/doctor/appointments/${id}/cancel/`),
};

// Response interceptor to handle token refresh
api.interceptors.response.use(
    (response) => response,
    async (error) => {
        const originalRequest = error.config;

        // If error is 401 and we haven't tried to refresh yet
        if (error.response?.status === 401 && !originalRequest._retry) {
            // Skip refresh for checkAuth and logout
            if (originalRequest.url.includes('/auth/profile/') || originalRequest.url.includes('/auth/logout/')) {
                return Promise.reject(error);
            }

            originalRequest._retry = true;

            try {
                // Attempt to refresh the token
                // The refresh cookie is HttpOnly, so the backend will pick it up
                await axios.post(`${API_URL}/auth/token/refresh/`, {}, { withCredentials: true });

                // If refresh succeeds, retry the original request
                return api(originalRequest);
            } catch (refreshError) {
                // If refresh fails, redirect to login or handle as needed
                // For now, let the error bubble up to be handled by the slice
                return Promise.reject(refreshError);
            }
        }

        return Promise.reject(error);
    }
);


export const adminApi = {
    login: (credentials) => api.post('/admin/login/', credentials),
    getDoctors: (status) => api.get(`/admin/doctors/${status ? `?status=${status}` : ''}`),
    verifyDoctor: (id, data) => api.patch(`/admin/verify-doctor/${id}/`, data),
    getPatients: () => api.get('/admin/patients/'),
    createAdmin: (data) => api.post('/admin/create-admin/', data),
    unverifyDoctor: (id) => api.post(`/admin/doctors/${id}/unverify/`),
    // Flexible methods for user management
    updateDoctor: (id, data) => api.patch(`/admin/doctors/${id}/`, data),
    deleteDoctor: (id) => api.delete(`/admin/doctors/${id}/`),
    updatePatient: (id, data) => api.patch(`/admin/patients/${id}/`, data),
    deletePatient: (id) => api.delete(`/admin/patients/${id}/`),
    getFinancialStats: () => api.get('/doctor/appointments/admin_financial_stats/'),
};

export default api;

