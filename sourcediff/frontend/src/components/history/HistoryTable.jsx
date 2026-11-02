import { useEffect, useState, useCallback } from "react";
import { Clock, RefreshCw, ChevronRight, AlertCircle } from "lucide-react";
import VerdictBadge from "../shared/VerdictBadge";
import CaseDetailModal from "./CaseDetailModal";
import { deleteHistory, getHistory } from "../../api/client";
import { Trash2 } from "lucide-react";

export default function HistoryTable({ username }) {
  const [rows, setRows] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [selectedCase, setSelectedCase] = useState(null);
  const [deleteLoading, setDeleteLoading] = useState(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const data = await getHistory(username, 100);
      setRows(data.rows || []);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [username]);

  useEffect(() => { load(); }, [load]);

  async function handleDelete(row) {
    if (!window.confirm(`Delete history row "${row.filename}" and clear its cache?`)) return;
    setDeleteLoading(row.id);
    setError("");
    try {
      await deleteHistory(row.id);
      await load();
    } catch (e) {
      setError(e.message);
    } finally {
      setDeleteLoading(null);
    }
  }

  return (
    <section className="rounded-2xl border border-slate-200 bg-white shadow-sm overflow-hidden">

      {/* Header */}
      <div className="flex items-center justify-between px-6 py-5 border-b border-slate-100">
        <div className="flex items-center gap-3">
          <div className="rounded-xl bg-blue-100 p-2.5 text-blue-600"><Clock size={20} /></div>
          <div>
            <h2 className="font-bold text-slate-900">Verification History</h2>
            <p className="text-xs text-slate-500 mt-0.5">{rows.length} records</p>
          </div>
        </div>
        <button
          onClick={load}
          className="flex items-center gap-2 rounded-xl border border-slate-200 px-4 py-2 text-sm font-medium text-slate-600 hover:bg-slate-50 transition"
        >
          <RefreshCw size={15} /> Refresh
        </button>
      </div>

      {/* Body */}
      {loading ? (
        <div className="flex justify-center py-16">
          <div className="h-8 w-8 animate-spin rounded-full border-4 border-blue-600 border-t-transparent" />
        </div>
      ) : error ? (
        <div className="flex items-center gap-3 px-6 py-10 text-red-600">
          <AlertCircle size={20} /> {error}
        </div>
      ) : rows.length === 0 ? (
        <div className="px-6 py-16 text-center text-slate-400">
          <Clock size={36} className="mx-auto mb-3 opacity-30" />
          <p className="font-medium">No submissions yet</p>
          <p className="text-sm mt-1">Upload a file in the <em>Check a File</em> tab to get started.</p>
        </div>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="bg-slate-50 text-slate-500 text-xs uppercase">
              <tr>
                <th className="px-5 py-3 text-left">File</th>
                <th className="px-5 py-3 text-left">Type</th>
                <th className="px-5 py-3 text-left">Verdict</th>
                <th className="px-5 py-3 text-left">Date</th>
                <th className="px-5 py-3" />
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {rows.map((row) => (
                <tr key={row.id} className="hover:bg-slate-50 transition">
                  <td className="px-5 py-3.5 font-medium text-slate-800 max-w-xs truncate">
                    {row.filename === "[deleted]" ? <em className="text-slate-400">Deleted</em> : row.filename}
                  </td>
                  <td className="px-5 py-3.5 text-slate-500 capitalize">{row.door}</td>
                  <td className="px-5 py-3.5">
                    <VerdictBadge verdict={row.verdict} label={row.label} size="sm" />
                  </td>
                  <td className="px-5 py-3.5 text-slate-500 whitespace-nowrap">
                    {new Date(row.submitted_at).toLocaleString()}
                  </td>
                  <td className="px-5 py-3.5 text-right">
                    <div className="flex items-center justify-end gap-3">
                      <button
                        onClick={() => setSelectedCase(row.case_id)}
                        className="flex items-center gap-1 text-blue-600 hover:text-blue-800 font-medium text-xs"
                      >
                        View <ChevronRight size={14} />
                      </button>
                      <button
                        type="button"
                        disabled={deleteLoading === row.id}
                        onClick={() => handleDelete(row)}
                        className="flex items-center gap-1 text-red-600 hover:text-red-800 font-medium text-xs disabled:opacity-50"
                      >
                        <Trash2 size={14} />
                        {deleteLoading === row.id ? "Deleting…" : "Delete"}
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Case detail modal */}
      {selectedCase && (
        <CaseDetailModal caseId={selectedCase} onClose={() => setSelectedCase(null)} />
      )}
    </section>
  );
}
