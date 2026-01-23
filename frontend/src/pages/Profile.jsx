import React, { useState, useRef } from 'react';
import { useAuth } from '../contexts/AuthContext';
import defaultMale from '../assets/default_male.svg';
import defaultFemale from '../assets/default_female.svg';

function Profile() {
    const { user, updateUser } = useAuth();
    const fileInputRef = useRef(null);
    const videoRef = useRef(null);
    const [isEditing, setIsEditing] = useState(false);
    const [showOptions, setShowOptions] = useState(false);
    const [showCamera, setShowCamera] = useState(false);
    const [hasUnsavedChanges, setHasUnsavedChanges] = useState(false);
    const [profileImg, setProfileImg] = useState(user?.profileImage || (user?.gender === 'female' ? defaultFemale : defaultMale));
    const [formData, setFormData] = useState({
        name: user?.name || 'Guest User',
        email: user?.email || 'guest@example.com',
        phone: user?.phone || '+977 98XXXXXXXX',
        address: user?.address || 'Kathmandu, Nepal',
        gender: user?.gender || 'male'
    });

    const handleChange = (e) => {
        setFormData({ ...formData, [e.target.name]: e.target.value });
    };

    const handleSave = () => {
        setIsEditing(false);
        updateUser({
            ...formData,
            profileImage: profileImg
        });
        setHasUnsavedChanges(false);
        alert('Profile updated successfully!');
    };

    const handleImageClick = () => {
        setShowOptions(true);
    };

    const handleUploadClick = () => {
        setShowOptions(false);
        fileInputRef.current.click();
    };

    const handleCameraClick = async () => {
        setShowOptions(false);
        setShowCamera(true);
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: true });
            if (videoRef.current) {
                videoRef.current.srcObject = stream;
            }
        } catch (err) {
            console.error("Error accessing camera:", err);
            alert("Could not access camera. Please check permissions.");
            setShowCamera(false);
        }
    };

    const capturePhoto = () => {
        const video = videoRef.current;
        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        const dataUrl = canvas.toDataURL('image/jpeg');
        setProfileImg(dataUrl);
        setHasUnsavedChanges(true);
        stopCamera();
    };

    const stopCamera = () => {
        const stream = videoRef.current?.srcObject;
        const tracks = stream?.getTracks();
        tracks?.forEach(track => track.stop());
        setShowCamera(false);
    };

    const handleFileChange = (e) => {
        const file = e.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onloadend = () => {
                setProfileImg(reader.result);
                setHasUnsavedChanges(true);
            };
            reader.readAsDataURL(file);
        }
    };

    const handleGenderChange = (newGender) => {
        setFormData({ ...formData, gender: newGender });
        // Only set default if user hasn't uploaded a custom photo
        if (!user?.profileImage || user.profileImage === defaultMale || user.profileImage === defaultFemale) {
            setProfileImg(newGender === 'male' ? defaultMale : defaultFemale);
        }
    };

    return (
        <div className="flex-1 min-h-screen bg-slate-50 px-4 py-8">
            <div className="max-w-3xl mx-auto">
                <h1 className="text-3xl font-bold text-slate-900 mb-8">My Profile</h1>

                <div className="bg-white rounded-2xl shadow-sm border border-slate-100 overflow-hidden relative">
                    <div className="bg-gradient-to-r from-green-600 to-green-500 px-8 py-10 text-white flex items-center gap-6">
                        <div className="relative group">
                            <div className="w-28 h-28 rounded-full bg-white p-1 shadow-xl overflow-hidden cursor-pointer relative" onClick={handleImageClick}>
                                <img src={profileImg} alt="Profile" className="w-full h-full rounded-full object-cover" />
                                <div className="absolute inset-0 bg-black/20 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity">
                                    <i className="fa-solid fa-camera text-2xl text-white"></i>
                                </div>
                            </div>
                            <input
                                type="file"
                                ref={fileInputRef}
                                onChange={handleFileChange}
                                className="hidden"
                                accept="image/*"
                            />
                            {hasUnsavedChanges && (
                                <div className="absolute -top-4 -right-20 bg-yellow-400 text-yellow-950 text-[10px] font-black px-2 py-1 rounded-md shadow-lg animate-bounce border border-yellow-500 whitespace-nowrap z-20">
                                    <i className="fa-solid fa-arrow-down mr-1"></i> Save Changes!
                                </div>
                            )}
                        </div>
                        <div>
                            <h2 className="text-3xl font-bold">{formData.name}</h2>
                            <p className="text-green-100 opacity-90 font-medium">{user?.role ? user.role.charAt(0).toUpperCase() + user.role.slice(1) : 'Patient'}</p>
                        </div>
                    </div>

                    <div className="p-8">
                        <div className="flex justify-between items-center mb-6">
                            <h3 className="text-lg font-bold text-slate-800">Personal Information</h3>
                            <button
                                onClick={() => (isEditing || hasUnsavedChanges) ? handleSave() : setIsEditing(true)}
                                className={`px-4 py-2 rounded-lg font-bold text-sm transition ${(isEditing || hasUnsavedChanges) ? 'bg-green-600 text-white hover:bg-green-700 ring-4 ring-green-500/10' : 'bg-slate-100 text-slate-600 hover:bg-slate-200'}`}
                            >
                                {(isEditing || hasUnsavedChanges) ? 'Save Changes' : 'Edit Profile'}
                            </button>
                        </div>

                        <div className="space-y-6">
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                <div>
                                    <label className="block text-sm font-semibold text-slate-500 mb-2">Full Name</label>
                                    <input
                                        name="name"
                                        value={formData.name}
                                        onChange={handleChange}
                                        disabled={!isEditing}
                                        className="w-full px-4 py-2 bg-slate-50 border border-slate-200 rounded-lg focus:outline-none focus:border-green-500 disabled:opacity-60 disabled:cursor-not-allowed"
                                    />
                                </div>
                                <div>
                                    <label className="block text-sm font-semibold text-slate-500 mb-2">Email Address</label>
                                    <input
                                        name="email"
                                        value={formData.email}
                                        onChange={handleChange}
                                        disabled={!isEditing} // Usually email is not editable directly
                                        className="w-full px-4 py-2 bg-slate-50 border border-slate-200 rounded-lg focus:outline-none focus:border-green-500 disabled:opacity-60 disabled:cursor-not-allowed"
                                    />
                                </div>
                                <div>
                                    <label className="block text-sm font-semibold text-slate-500 mb-2">Gender</label>
                                    <div className="flex gap-4">
                                        <button
                                            onClick={() => isEditing && handleGenderChange('male')}
                                            disabled={!isEditing}
                                            className={`flex-1 py-2 rounded-lg border font-bold transition ${formData.gender === 'male' ? 'bg-green-50 border-green-500 text-green-600' : 'bg-slate-50 border-slate-200 text-slate-500 opacity-60'}`}
                                        >
                                            Male
                                        </button>
                                        <button
                                            onClick={() => isEditing && handleGenderChange('female')}
                                            disabled={!isEditing}
                                            className={`flex-1 py-2 rounded-lg border font-bold transition ${formData.gender === 'female' ? 'bg-green-50 border-green-500 text-green-600' : 'bg-slate-50 border-slate-200 text-slate-500 opacity-60'}`}
                                        >
                                            Female
                                        </button>
                                    </div>
                                </div>
                                <div className="md:col-span-2 border-t border-slate-100 pt-6">
                                    <h4 className="text-sm font-bold text-slate-800 mb-4 uppercase tracking-wider">Contact Details</h4>
                                </div>
                                <div>
                                    <label className="block text-sm font-semibold text-slate-500 mb-2">Phone Number</label>
                                    <input
                                        name="phone"
                                        value={formData.phone}
                                        onChange={handleChange}
                                        disabled={!isEditing}
                                        className="w-full px-4 py-2 bg-slate-50 border border-slate-200 rounded-lg focus:outline-none focus:border-green-500 disabled:opacity-60 disabled:cursor-not-allowed"
                                    />
                                </div>
                                <div>
                                    <label className="block text-sm font-semibold text-slate-500 mb-2">Address</label>
                                    <input
                                        name="address"
                                        value={formData.address}
                                        onChange={handleChange}
                                        disabled={!isEditing}
                                        className="w-full px-4 py-2 bg-slate-50 border border-slate-200 rounded-lg focus:outline-none focus:border-green-500 disabled:opacity-60 disabled:cursor-not-allowed"
                                    />
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            {/* Photo Source Selection Modal */}
            {showOptions && (
                <div className="fixed inset-0 bg-black/60 backdrop-blur-sm z-[100] flex items-center justify-center p-4">
                    <div className="bg-white rounded-2xl p-6 w-full max-w-sm shadow-2xl transform transition-all">
                        <h3 className="text-xl font-bold text-slate-900 mb-6 flex items-center gap-3">
                            <i className="fa-solid fa-camera text-green-600"></i> Change Photo
                        </h3>
                        <div className="grid gap-4">
                            <button
                                onClick={handleUploadClick}
                                className="flex items-center gap-4 p-4 rounded-xl border-2 border-slate-100 hover:border-green-500 hover:bg-green-50 transition group"
                            >
                                <div className="w-12 h-12 rounded-lg bg-blue-100 flex items-center justify-center text-blue-600 group-hover:scale-110 transition">
                                    <i className="fa-solid fa-image text-xl"></i>
                                </div>
                                <div className="text-left">
                                    <div className="font-bold text-slate-800">Upload Photo</div>
                                    <div className="text-sm text-slate-500">Pick from gallery</div>
                                </div>
                            </button>
                            <button
                                onClick={handleCameraClick}
                                className="flex items-center gap-4 p-4 rounded-xl border-2 border-slate-100 hover:border-green-500 hover:bg-green-50 transition group"
                            >
                                <div className="w-12 h-12 rounded-lg bg-purple-100 flex items-center justify-center text-purple-600 group-hover:scale-110 transition">
                                    <i className="fa-solid fa-camera-retro text-xl"></i>
                                </div>
                                <div className="text-left">
                                    <div className="font-bold text-slate-800">Take Photo</div>
                                    <div className="text-sm text-slate-500">Click from camera</div>
                                </div>
                            </button>
                        </div>
                        <button
                            onClick={() => setShowOptions(false)}
                            className="mt-6 w-full py-3 text-slate-500 font-bold hover:text-slate-700 transition"
                        >
                            Cancel
                        </button>
                    </div>
                </div>
            )}

            {/* Camera Overlay */}
            {showCamera && (
                <div className="fixed inset-0 bg-black z-[110] flex flex-col">
                    <div className="p-6 flex justify-between items-center text-white">
                        <h3 className="text-xl font-bold">Capture Photo</h3>
                        <button onClick={stopCamera} className="p-2 hover:bg-white/10 rounded-full transition">
                            <i className="fa-solid fa-xmark text-2xl"></i>
                        </button>
                    </div>

                    <div className="flex-1 relative flex items-center justify-center overflow-hidden">
                        <video
                            ref={videoRef}
                            autoPlay
                            playsInline
                            className="h-full w-full object-cover md:rounded-3xl shadow-2xl max-w-4xl"
                        />
                        <div className="absolute inset-0 border-[40px] border-black/40 pointer-events-none md:rounded-3xl"></div>
                        <div className="absolute inset-x-0 bottom-12 flex justify-center gap-8 items-center">
                            <button
                                onClick={stopCamera}
                                className="w-14 h-14 rounded-full bg-white/20 text-white flex items-center justify-center hover:bg-white/30 transition shadow-lg"
                            >
                                <i className="fa-solid fa-xmark"></i>
                            </button>
                            <button
                                onClick={capturePhoto}
                                className="w-20 h-20 rounded-full bg-white flex items-center justify-center shadow-2xl hover:scale-110 active:scale-95 transition"
                            >
                                <div className="w-16 h-16 rounded-full border-4 border-slate-900"></div>
                            </button>
                            <div className="w-14 h-14"></div> {/* Spacer */}
                        </div>
                    </div>

                    <div className="p-12 bg-black text-white/50 text-center text-sm">
                        Align your face within the frame and click the shutter
                    </div>
                </div>
            )}
        </div>
    );
}

export default Profile;
