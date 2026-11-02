import { useState } from "react";
import { UploadCloud, CheckCircle2, AlertCircle, X } from "lucide-react";
import { addToLibrary } from "../../api/client";

export default function LibraryAddForm({ onSuccess, onCancel, adminKeyRequired }) {
  const [file, setFile]           = useState(null);
  const [door, setDoor]           = useState("image");
  const [title, setTitle]         = useState("");
  const [publisher, setPublisher] = useState("");
  const [license, setLicense]     = useState("");
  const [url, setUrl]             = useState("");
  const [notes, setNotes]         = useState("");
  const [adminKey, setAdminKey]   = useState("");
  const [rights, setRights]       = useState(false);
  const [loading, setLoading]     = useState(false);
  const [error, setError]         = useState("");
  const [success, setSuccess]     = useState("");

  async function handleSubmit(e) {
    e.preventDefault();
    if (!rights) { setError("Tick the rights checkbox."); return; }
    setLoading(true); setError(""); setSuccess("");
    try {
      const fd = new FormData();
      fd.append("file", file);
      fd.append("media", door);
      fd.append("title", title);
      fd.append("publisher", publisher);
      fd.append("license", license);
      fd.append("url", url);
      fd.append("notes", notes);
      fd.append("rights_confirmed", rights ? "true" : "false");
      if (adminKeyRequired) fd.append("admin_key", adminKey);
      const res = await addToLibrary(fd);
      setSuccess(res.message || "Added successfully.");
      setTimeout(onSuccess, 1500);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="border-b border-slate-100 bg-slate-50 p-6">
      <div className="flex items-center justify-between mb-5">
        <h3 className="font-bold text-slate-800">Add Trusted Original</h3>
        <button onClick={onCancel} className="rounded-full p-1.5 hover:bg-slate-200 transition">
          <X size={16} />
        </button>
      </div>

      <form onSubmit={handleSubmit} className="grid gap-4 sm:grid-cols-2">

        {/* File */}
        <div className="sm:col-span-2">
          <label className="mb-2 block text-sm font-medium text-slate-700">File *</label>
          <label className="flex cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed border-blue-200 bg-white px-5 py-8 text-center hover:border-blue-400 transition">
            <UploadCloud size={28} className="mb-2 text-blue-400" />
            <span className="font-medium text-slate-700">
              {file ? file.name : "Browse image or video"}
            </span>
            <span className="text-xs text-slate-400 mt-1">JPG, PNG, WEBP, MP4, MOV, MKV…</span>
            <input
              type="file"
              accept=".jpg,.jpeg,.png,.webp,.bmp,.mp4,.mov,.mkv,.webm,.avi"
              required
              className="hidden"
              onChange={(e) => setFile(e.target.files[0])}
            />
          </label>
        </div>

        {/* Media type */}
        <div>
          <label className="mb-2 block text-sm font-medium text-slate-700">Media type *</label>
          <select
            value={door}
            onChange={(e) => setDoor(e.target.value)}
            className="w-full rounded-xl border border-slate-200 bg-white px-4 py-3 text-sm outline-none focus:border-blue-500"
          >
            <option value="image">Image</option>
            <option value="video">Video</option>
          </select>
        </div>

        {/* Title */}
        <div>
          <label className="mb-2 block text-sm font-medium text-slate-700">Title *</label>
          <input
            type="text"
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            required
            placeholder="Descriptive title"
            className="w-full rounded-xl border border-slate-200 bg-white px-4 py-3 text-sm outline-none focus:border-blue-500"
          />
        </div>

        {/* Publisher */}
        <div>
          <label className="mb-2 block text-sm font-medium text-slate-700">Publisher *</label>
          <input
            type="text"
            value={publisher}
            onChange={(e) => setPublisher(e.target.value)}
            required
            placeholder="Source / publisher name"
            className="w-full rounded-xl border border-slate-200 bg-white px-4 py-3 text-sm outline-none focus:border-blue-500"
          />
        </div>

        {/* License */}
        <div>
          <label className="mb-2 block text-sm font-medium text-slate-700">License *</label>
          <input
            type="text"
            value={license}
            onChange={(e) => setLicense(e.target.value)}
            required
            placeholder="CC-BY 4.0, Public Domain…"
            className="w-full rounded-xl border border-slate-200 bg-white px-4 py-3 text-sm outline-none focus:border-blue-500"
          />
        </div>

        {/* URL */}
        <div>
          <label className="mb-2 block text-sm font-medium text-slate-700">Source URL</label>
          <input
            type="url"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            placeholder="https://..."
            className="w-full rounded-xl border border-slate-200 bg-white px-4 py-3 text-sm outline-none focus:border-blue-500"
          />
        </div>

        {/* Notes */}
        <div className="sm:col-span-2">
          <label className="mb-2 block text-sm font-medium text-slate-700">Notes</label>
          <textarea
            value={notes}
            onChange={(e) => setNotes(e.target.value)}
            rows={2}
            placeholder="Optional context"
            className="w-full rounded-xl border border-slate-200 bg-white px-4 py-3 text-sm outline-none focus:border-blue-500 resize-none"
          />
        </div>

        {/* Admin key */}
        {adminKeyRequired && (
          <div className="sm:col-span-2">
            <label className="mb-2 block text-sm font-medium text-slate-700">Admin key *</label>
            <input
              type="password"
              value={adminKey}
              onChange={(e) => setAdminKey(e.target.value)}
              required
              placeholder="SOURCEDIFF_ADMIN_KEY"
              className="w-full rounded-xl border border-slate-200 bg-white px-4 py-3 text-sm outline-none focus:border-blue-500"
            />
          </div>
        )}

        {/* Rights */}
        <div className="sm:col-span-2 flex items-start gap-3">
          <input
            id="rights"
            type="checkbox"
            checked={rights}
            onChange={(e) => setRights(e.target.checked)}
            className="mt-1 h-4 w-4 accent-blue-600"
          />
          <label htmlFor="rights" className="text-sm text-slate-600">
            I confirm I have the rights to add this file and that it is the <strong>genuine, unedited original</strong>.
          </label>
        </div>

        {/* Feedback */}
        {error && (
          <div className="sm:col-span-2 flex items-start gap-2 rounded-xl bg-red-50 border border-red-200 px-4 py-3 text-sm text-red-700">
            <AlertCircle size={15} className="mt-0.5 shrink-0" /> {error}
          </div>
        )}
        {success && (
          <div className="sm:col-span-2 flex items-center gap-2 rounded-xl bg-emerald-50 border border-emerald-200 px-4 py-3 text-sm text-emerald-700">
            <CheckCircle2 size={15} /> {success}
          </div>
        )}

        {/* Submit */}
        <div className="sm:col-span-2">
          <button
            type="submit"
            disabled={loading || !file}
            className="rounded-xl bg-blue-600 px-6 py-3 text-sm font-bold text-white hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition"
          >
            {loading ? "Uploading…" : "Add to library"}
          </button>
        </div>
      </form>
    </div>
  );
}
