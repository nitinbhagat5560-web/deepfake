import { useEffect, useState, useCallback } from "react";
import { Folder, RefreshCw, Image, Video, AlertCircle, Plus } from "lucide-react";
import LibraryAddForm from "./LibraryAddForm";
import { getLibrary } from "../../api/client";

export default function LibraryList() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [showForm, setShowForm] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const res = await getLibrary();
      setData(res);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { load(); }, [load]);

  const allItems = data
    ? [
        ...(data.items?.image || []).map((i) => ({ ...i, door: "image" })),
        ...(data.items?.video || []).map((i) => ({ ...i, door: "video" })),
      ].sort((a, b) => b.added_at?.localeCompare(a.added_at))
    : [];

  return (
    <section className="rounded-2xl border border-slate-200 bg-white shadow-sm overflow-hidden">

      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-3 px-6 py-5 border-b border-slate-100">
        <div className="flex items-center gap-3">
          <div className="rounded-xl bg-violet-100 p-2.5 text-violet-600"><Folder size={20} /></div>
          <div>
            <h2 className="font-bold text-slate-900">Trusted Library</h2>
            <p className="text-xs text-slate-500 mt-0.5">{allItems.length} original{allItems.length !== 1 ? "s" : ""}</p>
          </div>
        </div>
        <div className="flex gap-2">
          <button
            onClick={load}
            className="flex items-center gap-2 rounded-xl border border-slate-200 px-4 py-2 text-sm font-medium text-slate-600 hover:bg-slate-50 transition"
          >
            <RefreshCw size={15} /> Refresh
          </button>
          {data?.can_add && (
            <button
              onClick={() => setShowForm((v) => !v)}
              className="flex items-center gap-2 rounded-xl bg-blue-600 px-4 py-2 text-sm font-semibold text-white hover:bg-blue-700 transition"
            >
              <Plus size={15} /> Add Original
            </button>
          )}
        </div>
      </div>

      {/* Add form */}
      {showForm && data?.can_add && (
        <LibraryAddForm
          onSuccess={() => { setShowForm(false); load(); }}
          onCancel={() => setShowForm(false)}
          adminKeyRequired={data.admin_key_required}
        />
      )}

      {/* Content */}
      {loading ? (
        <div className="flex justify-center py-16">
          <div className="h-8 w-8 animate-spin rounded-full border-4 border-violet-600 border-t-transparent" />
        </div>
      ) : error ? (
        <div className="flex items-center gap-3 px-6 py-10 text-red-600">
          <AlertCircle size={20} /> {error}
        </div>
      ) : allItems.length === 0 ? (
        <div className="px-6 py-16 text-center text-slate-400">
          <Folder size={36} className="mx-auto mb-3 opacity-30" />
          <p className="font-medium">Library is empty</p>
          <p className="text-sm mt-1">Add trusted originals to enable verification.</p>
        </div>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="bg-slate-50 text-slate-500 text-xs uppercase">
              <tr>
                <th className="px-5 py-3 text-left">Type</th>
                <th className="px-5 py-3 text-left">Title</th>
                <th className="px-5 py-3 text-left">Publisher</th>
                <th className="px-5 py-3 text-left">License</th>
                <th className="px-5 py-3 text-left">Added</th>
                <th className="px-5 py-3 text-left">SHA-256</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {allItems.map((item) => (
                <tr key={`${item.door}-${item.source_id}`} className="hover:bg-slate-50 transition">
                  <td className="px-5 py-3.5">
                    <span className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-semibold ${
                      item.door === "image"
                        ? "bg-blue-100 text-blue-700"
                        : "bg-orange-100 text-orange-700"
                    }`}>
                      {item.door === "image" ? <Image size={12} /> : <Video size={12} />}
                      {item.door}
                    </span>
                  </td>
                  <td className="px-5 py-3.5 font-medium text-slate-800 max-w-xs">
                    <div className="truncate">{item.title}</div>
                    {item.url && (
                      <a href={item.url} target="_blank" rel="noreferrer"
                         className="text-xs text-blue-500 hover:underline truncate block">
                        {item.url}
                      </a>
                    )}
                  </td>
                  <td className="px-5 py-3.5 text-slate-500">{item.publisher || "—"}</td>
                  <td className="px-5 py-3.5 text-slate-500">{item.license || "—"}</td>
                  <td className="px-5 py-3.5 text-slate-500 whitespace-nowrap text-xs">
                    {item.added_at ? new Date(item.added_at).toLocaleDateString() : "—"}
                    {item.added_by && <div className="text-slate-400">{item.added_by}</div>}
                  </td>
                  <td className="px-5 py-3.5 font-mono text-xs text-slate-400">
                    {item.sha256?.slice(0, 14)}…
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
