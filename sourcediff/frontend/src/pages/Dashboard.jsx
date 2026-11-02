import { useState, useRef, useEffect } from "react";
import {
  ShieldCheck, LayoutDashboard, History, Folder,
  Settings, LogOut, Bell, UploadCloud, Search, Menu, User,
} from "lucide-react";
import { analyzeFile, authLogout, authMe } from "../api/client";
import ResultCard from "../components/analyze/ResultCard";
import HistoryTable from "../components/history/HistoryTable";
import LibraryList from "../components/library/LibraryList";

const TABS = ["Check a File", "History", "Library"];

export default function Dashboard() {
  const [activeTab, setActiveTab]   = useState("Check a File");
  const [file, setFile]             = useState(null);
  const [result, setResult]         = useState(null);   // { card, case_id }
  const [loading, setLoading]       = useState(false);
  const [error, setError]           = useState("");
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [username, setUsername]     = useState("");
  const fileInputRef                = useRef();

  // Fetch current user for display + history
  useEffect(() => {
    authMe().then((d) => setUsername(d.username || "")).catch(() => {});
  }, []);

  async function checkFile(e) {
    e.preventDefault();
    if (!file) return;
    const fd = new FormData();
    fd.append("file", file);
    setLoading(true); setError(""); setResult(null);
    try {
      const data = await analyzeFile(fd);
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  async function logout() {
    await authLogout().catch(() => {});
    window.location.href = "/login";
  }

  function navClick(label) {
    setActiveTab(label === "Dashboard" ? "Check a File" : label);
    setSidebarOpen(false);
  }

  return (
    <div className="min-h-screen bg-slate-100 text-slate-900">

      {/* ── Sidebar ── */}
      <aside className={`fixed inset-y-0 left-0 z-40 flex w-64 flex-col bg-slate-950 text-white transition-transform lg:translate-x-0 ${sidebarOpen ? "translate-x-0" : "-translate-x-full"}`}>
        <div className="flex h-20 items-center gap-3 border-b border-white/10 px-6">
          <div className="rounded-xl bg-blue-600 p-2.5"><ShieldCheck size={24} /></div>
          <span className="text-xl font-bold">SourceDiff</span>
        </div>

        <nav className="flex-1 space-y-1 px-3 py-6">
          {[[LayoutDashboard, "Dashboard"], [History, "History"], [Folder, "Library"], [Settings, "Settings"]].map(([Icon, label]) => {
            const active = activeTab === label || (label === "Dashboard" && activeTab === "Check a File");
            return (
              <button
                key={label}
                onClick={() => navClick(label)}
                className={`flex w-full items-center gap-3 rounded-xl px-4 py-3 text-left font-medium transition ${active ? "bg-blue-600 text-white" : "text-slate-400 hover:bg-white/10 hover:text-white"}`}
              >
                <Icon size={19} /> {label}
              </button>
            );
          })}
        </nav>

        <div className="m-4 rounded-2xl bg-gradient-to-br from-blue-600 to-violet-600 p-5">
          <ShieldCheck size={26} />
          <p className="mt-3 font-bold">Trusted Media</p>
          <p className="mt-1 text-sm text-blue-100">Verify, compare and protect your media.</p>
        </div>

        <button onClick={logout} className="flex items-center gap-3 border-t border-white/10 px-8 py-5 text-slate-400 hover:text-white transition">
          <LogOut size={19} /> Sign Out
        </button>
      </aside>

      {/* Overlay for mobile sidebar */}
      {sidebarOpen && (
        <div onClick={() => setSidebarOpen(false)} className="fixed inset-0 z-30 bg-black/40 lg:hidden" />
      )}

      {/* ── Main ── */}
      <main className="min-h-screen lg:ml-64">

        {/* Topbar */}
        <header className="flex h-16 items-center justify-between border-b border-slate-200 bg-white px-5 sm:px-8">
          <div className="flex items-center gap-3">
            <button onClick={() => setSidebarOpen(!sidebarOpen)} className="rounded-lg p-2 hover:bg-slate-100 lg:hidden">
              <Menu size={22} />
            </button>
            <div>
              <p className="font-bold text-sm">SourceDiff</p>
              <p className="text-xs text-slate-500">Media Verification Platform</p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <button className="relative rounded-full p-2 hover:bg-slate-100">
              <Bell size={20} />
              <span className="absolute right-2 top-2 h-2 w-2 rounded-full bg-red-500" />
            </button>
            <div className="flex h-9 w-9 items-center justify-center rounded-full bg-blue-600 font-bold text-white text-sm" title={username}>
              {username ? username[0].toUpperCase() : <User size={16} />}
            </div>
          </div>
        </header>

        <div className="mx-auto max-w-5xl space-y-6 p-5 sm:p-8">

          {/* Welcome banner */}
          <section className="relative overflow-hidden rounded-3xl bg-gradient-to-r from-blue-700 via-blue-600 to-violet-600 p-7 text-white sm:p-10">
            <div className="relative z-10 max-w-xl">
              <p className="mb-1 text-sm font-medium text-blue-100 uppercase tracking-wide">YOUR MEDIA, YOUR TRUST</p>
              <h1 className="text-3xl font-bold sm:text-4xl">
                {username ? `Welcome back, ${username}!` : "Welcome back!"}
              </h1>
              <p className="mt-3 leading-7 text-blue-100">
                Verify media authenticity and manage your trusted originals with SourceDiff.
              </p>
            </div>
            <ShieldCheck size={140} className="absolute -bottom-6 right-8 hidden text-white/10 sm:block" />
          </section>

          {/* Tabs */}
          <div className="flex flex-wrap gap-2">
            {TABS.map((tab) => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`rounded-xl px-5 py-2.5 font-semibold text-sm transition ${
                  activeTab === tab
                    ? "bg-blue-600 text-white shadow-lg shadow-blue-200"
                    : "border border-slate-200 bg-white text-slate-600 hover:bg-slate-50"
                }`}
              >
                {tab}
              </button>
            ))}
          </div>

          {/* ── CHECK A FILE TAB ── */}
          {activeTab === "Check a File" && (
            <>
              <form
                onSubmit={checkFile}
                className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm sm:p-8"
              >
                <h2 className="text-xl font-bold mb-1">Verify your media</h2>
                <p className="text-sm text-slate-500 mb-6">Upload an image or video to analyse.</p>

                {/* Drop zone */}
                <label className="flex cursor-pointer flex-col items-center justify-center rounded-2xl border-2 border-dashed border-blue-200 bg-blue-50/50 px-5 py-12 text-center transition hover:border-blue-500 hover:bg-blue-50">
                  <div className="mb-4 rounded-2xl bg-blue-100 p-4 text-blue-600">
                    <UploadCloud size={36} />
                  </div>
                  <span className="font-bold text-slate-800">
                    {file ? file.name : "Upload image or video"}
                  </span>
                  <span className="mt-2 text-sm text-slate-500">Click to browse your files</span>
                  <span className="mt-3 text-xs text-slate-400">JPG, PNG, WEBP, MP4, MOV, MKV, WEBM, AVI</span>
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept=".jpg,.jpeg,.png,.webp,.bmp,.mp4,.mov,.mkv,.webm,.avi"
                    required
                    className="hidden"
                    onChange={(e) => { setFile(e.target.files[0]); setResult(null); }}
                  />
                </label>

                {/* Final Path 2 configuration */}
                <div className="mt-6 rounded-xl border border-violet-200 bg-violet-50 px-4 py-3">
                  <p className="text-xs font-semibold uppercase tracking-wide text-violet-600">Path 2 fallback</p>
                  <p className="mt-1 text-sm font-bold text-slate-800">GenD PE-L</p>
                  <p className="mt-1 text-xs text-slate-500">Face + whole-frame analysis · automatic when no trusted original matches</p>
                </div>

                {error && (
                  <p className="mt-5 rounded-xl bg-red-50 border border-red-200 p-4 text-sm text-red-700">{error}</p>
                )}

                <button
                  disabled={loading || !file}
                  className="mt-6 flex items-center gap-2 rounded-xl bg-blue-600 px-7 py-4 font-bold text-white transition hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-50"
                >
                  <Search size={20} />
                  {loading ? "Analysing…" : "Check file"}
                </button>
              </form>

              {/* Result */}
              {result && <ResultCard card={result.card} caseId={result.case_id} />}
            </>
          )}

          {/* ── HISTORY TAB ── */}
          {activeTab === "History" && <HistoryTable username={username} />}

          {/* ── LIBRARY TAB ── */}
          {activeTab === "Library" && <LibraryList />}

          {/* ── SETTINGS TAB ── */}
          {activeTab === "Settings" && (
            <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
              <h2 className="text-xl font-bold mb-4">Settings</h2>
              <div className="text-slate-500 text-sm space-y-2">
                <p>Signed in as: <span className="font-semibold text-slate-700">{username}</span></p>
                <button
                  onClick={logout}
                  className="mt-4 flex items-center gap-2 rounded-xl bg-red-50 border border-red-200 px-5 py-3 text-sm font-semibold text-red-600 hover:bg-red-100 transition"
                >
                  <LogOut size={16} /> Sign out
                </button>
              </div>
            </section>
          )}

        </div>
      </main>
    </div>
  );
}