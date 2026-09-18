import React, { useEffect, useState } from "react";
import { useSearchParams, useNavigate } from "react-router-dom";
import { useDispatch } from "react-redux";
import { CheckCircle2, AlertCircle, Loader2, ArrowRight } from "lucide-react";
import { verifyKhaltiPayment } from "../store/slices/appointmentSlice";

const PaymentSuccess = () => {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const dispatch = useDispatch();
  const pidx = searchParams.get("pidx");
  const [status, setStatus] = useState("verifying"); // verifying, success, failed
  const [message, setMessage] = useState("Verifying your payment...");
  const [bookingDetails, setBookingDetails] = useState(null);

  useEffect(() => {
    if (!pidx) {
      setStatus("failed");
      setMessage("Invalid payment reference.");
      return;
    }

    const verify = async () => {
      try {
        const resultAction = await dispatch(verifyKhaltiPayment({ pidx }));

        if (verifyKhaltiPayment.fulfilled.match(resultAction)) {
          setStatus("success");
          setMessage("Payment successful! Your appointment is confirmed.");
          setBookingDetails(resultAction.payload);
        } else {
          setStatus("failed");
          setMessage(
            resultAction.payload?.error || "Payment verification failed.",
          );
        }
      } catch (error) {
        console.error("Verification error", error);
        setStatus("failed");
        setMessage("An error occurred while verifying payment.");
      }
    };

    verify();
  }, [pidx, dispatch]);

  return (
    <div className="min-h-screen bg-neutral-background flex items-center justify-center p-6">
      <div className="bg-white rounded-[2rem] shadow-xl p-8 max-w-md w-full text-center space-y-6">
        {status === "verifying" && (
          <div className="flex flex-col items-center gap-4">
            <Loader2 className="w-16 h-16 text-medic-dark animate-spin" />
            <h2 className="text-2xl font-bold text-gray-900">
              Verifying Payment
            </h2>
            <p className="text-gray-500">{message}</p>
          </div>
        )}

        {status === "success" && (
          <div className="flex flex-col items-center gap-4">
            <div className="w-20 h-20 bg-green-100 text-green-600 rounded-full flex items-center justify-center">
              <CheckCircle2 className="w-10 h-10" />
            </div>
            <h2 className="text-2xl font-bold text-gray-900">
              Payment Successful!
            </h2>
            <p className="text-gray-600">
              Transaction ID:{" "}
              <span className="font-mono font-bold">{pidx}</span>
            </p>
            <p className="text-sm text-gray-500">
              Your appointment has been confirmed. You will receive a
              confirmation email shortly.
            </p>

            <button
              onClick={() => navigate("/dashboard")}
              className="w-full py-4 bg-medic-dark text-white rounded-2xl font-bold hover:bg-medic-primary transition-all flex items-center justify-center gap-2 mt-4"
            >
              Go to Dashboard <ArrowRight className="w-5 h-5" />
            </button>
          </div>
        )}

        {status === "failed" && (
          <div className="flex flex-col items-center gap-4">
            <div className="w-20 h-20 bg-red-100 text-red-600 rounded-full flex items-center justify-center">
              <AlertCircle className="w-10 h-10" />
            </div>
            <h2 className="text-2xl font-bold text-gray-900">Payment Failed</h2>
            <p className="text-red-500 font-medium">{message}</p>
            <p className="text-sm text-gray-400">
              If you were charged, please contact support with reference ID:{" "}
              {pidx}
            </p>
            <button
              onClick={() => navigate("/dashboard")}
              className="w-full py-4 border-2 border-medic-dark text-medic-dark rounded-2xl font-bold hover:bg-medic-light/10 transition-all mt-4"
            >
              Return to Dashboard
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

export default PaymentSuccess;
