import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { authLogin, authRegister } from "../api/client";

const DEMO_USER = "";
const DEMO_PASS = "";

export default function Login() {
  const [isSignup, setIsSignup] = useState(false);
  const [username, setUsername] = useState(DEMO_USER);
  const [password, setPassword] = useState(DEMO_PASS);
  const [error, setError]       = useState("");
  const [loading, setLoading]   = useState(false);
  const navigate = useNavigate();

  async function handleSubmit(e) {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      const form = new FormData();
      form.set("username", username.trim());
      form.set("password", password);
      if (isSignup) await authRegister(form);
      else await authLogin(form);
      navigate("/", { replace: true });
    } catch (err) {
      setError(err.message || "Login failed.");
      setLoading(false);
    }
  }

  // function switchMode(signup) {
  //   setIsSignup(signup);
  //   setError("");
  //   setUsername(DEMO_USER);
  //   setPassword(DEMO_PASS);
  // }

  return (
    <main className="min-h-screen bg-slate-950 text-white">
      <div className="grid min-h-screen lg:grid-cols-2">

        {/* Left hero panel */}
        <section className="relative hidden overflow-hidden bg-gradient-to-br from-indigo-600 via-violet-700 to-slate-950 p-12 lg:flex lg:flex-col lg:justify-between">
          <div className="absolute -right-24 -top-24 h-80 w-80 rounded-full bg-fuchsia-400/20 blur-3xl" />
          <div className="absolute -bottom-24 -left-20 h-96 w-96 rounded-full bg-cyan-400/20 blur-3xl" />
          <span className="relative text-xl font-bold tracking-wide">
            Source<span className="text-cyan-300">Diff</span>
          </span>
          <div className="relative max-w-lg">
            <div className="mb-6 inline-flex rounded-full border border-white/20 bg-white/10 px-4 py-2 text-sm text-indigo-100">
              AI-powered media verification
            </div>
            <h1 className="text-5xl font-bold leading-tight">
              Know what&apos;s real.<br />Spot what&apos;s changed.
            </h1>
            <p className="mt-6 max-w-md text-lg leading-8 text-indigo-100">
              Compare files, review your analysis history, and keep your
              verified results organised in one place.
            </p>
          </div>
          <p className="relative text-sm text-indigo-200">© {new Date().getFullYear()} SourceDiff</p>
        </section>

        {/* Form panel */}
        <section className="flex items-center justify-center px-5 py-12 sm:px-8">
          <div className="w-full max-w-md">
            <span className="mb-10 inline-block text-xl font-bold tracking-wide lg:hidden">
              Source<span className="text-cyan-400">Diff</span>
            </span>

            <div className="rounded-3xl border border-white/10 bg-white/[0.04] p-6 shadow-2xl shadow-black/30 sm:p-9">
              <div className="mb-8">
                <p className="text-sm font-semibold uppercase tracking-[0.2em] text-cyan-400">
                  Welcome {isSignup ? "aboard" : "back"}
                </p>
                <h2 className="mt-3 text-3xl font-bold">
                  {isSignup ? "Create your account" : "Sign in to SourceDiff"}
                </h2>
                <p className="mt-2 text-sm text-slate-400">
                  {isSignup ? "Create an account to get started." : "Enter your details to continue."}
                </p>
              </div>

              {/* Toggle */}
              {/* <div className="mb-6 grid grid-cols-2 rounded-xl bg-slate-900 p-1">
                {[false, true].map((s) => (
                  <button key={String(s)} type="button" onClick={() => switchMode(s)}
                    className={`rounded-lg py-2.5 text-sm font-semibold transition ${
                      isSignup === s ? "bg-indigo-500 text-white shadow" : "text-slate-400 hover:text-white"
                    }`}>
                    {s ? "Sign up" : "Login"}
                  </button>
                ))}
              </div> */}

              <form onSubmit={handleSubmit} className="space-y-5">
                <div>
                  <label className="mb-2 block text-sm font-medium text-slate-300">Username</label>
                  <input type="text" value={username} onChange={(e) => setUsername(e.target.value)}
                    placeholder="your_username" required
                    className="w-full rounded-xl border border-slate-700 bg-slate-900/80 px-4 py-3.5 text-white outline-none transition placeholder:text-slate-500 focus:border-indigo-400 focus:ring-4 focus:ring-indigo-500/10" />
                </div>
                <div>
                  <label className="mb-2 block text-sm font-medium text-slate-300">Password</label>
                  <input type="password" value={password} onChange={(e) => setPassword(e.target.value)}
                    placeholder="Enter your password" required
                    autoComplete={isSignup ? "new-password" : "current-password"}
                    className="w-full rounded-xl border border-slate-700 bg-slate-900/80 px-4 py-3.5 text-white outline-none transition placeholder:text-slate-500 focus:border-indigo-400 focus:ring-4 focus:ring-indigo-500/10" />
                </div>

                {error && (
                  <p role="alert" className="rounded-lg border border-red-500/20 bg-red-500/10 px-4 py-3 text-sm text-red-300">
                    {error}
                  </p>
                )}

                <button type="submit" disabled={loading}
                  className="w-full rounded-xl bg-gradient-to-r from-indigo-500 to-violet-500 px-4 py-3.5 font-semibold text-white shadow-lg shadow-indigo-500/20 transition hover:from-indigo-400 hover:to-violet-400 disabled:cursor-not-allowed disabled:opacity-60">
                  {loading ? "Please wait…" : isSignup ? "Create account" : "Login"}
                </button>
              </form>

              <p className="mt-6 text-center text-sm text-slate-400">
                {isSignup ? "Already have an account?" : "New to SourceDiff?"}{" "}
                <button type="button" onClick={() => switchMode(!isSignup)}
                  className="font-semibold text-cyan-400 hover:text-cyan-300">
                  {isSignup ? "Login" : "Create an account"}
                </button>
              </p>
            </div>

          </div>
        </section>
      </div>
    </main>
  );
}