import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";

import { authMe } from "../../api/client";
export default function AuthGuard({ children }) {
  const [status, setStatus] = useState("checking");
  const navigate = useNavigate();

  useEffect(() => {
    authMe()
      .then((user) => {
        if (!user.authenticated) {
          navigate("/login", { replace: true });
          return;
        }
        setStatus("ok");
      })
      .catch(() => navigate("/login", { replace: true }));
  }, [navigate]);

  if (status !== "ok") {
    return (
      <div className="flex min-h-screen items-center justify-center bg-slate-100">
        <div className="h-10 w-10 animate-spin rounded-full border-4 border-blue-600 border-t-transparent" />
      </div>
    );
  }

  return children;
}
