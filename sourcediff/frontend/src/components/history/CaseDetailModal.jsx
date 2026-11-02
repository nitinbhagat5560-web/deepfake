import { useEffect, useState } from "react";
import { X, Loader2 } from "lucide-react";
import ResultCard from "../analyze/ResultCard";
import { getCaseCard } from "../../api/client";

export default function CaseDetailModal({ caseId, onClose }) {
  const [card, setCard] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    getCaseCard(caseId)
      .then(setCard)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [caseId]);

  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center overflow-y-auto bg-black/50 p-4 pt-12">
      <div className="relative w-full max-w-3xl">
        {/* Close button */}
        <button
          onClick={onClose}
          className="absolute -top-4 right-0 flex h-9 w-9 items-center justify-center rounded-full bg-white shadow-md hover:bg-slate-100 z-10"
        >
          <X size={18} />
        </button>

        {loading ? (
          <div className="flex items-center justify-center rounded-2xl bg-white p-16">
            <Loader2 size={28} className="animate-spin text-blue-600" />
          </div>
        ) : error ? (
          <div className="rounded-2xl bg-white p-10 text-center text-red-600">
            <p className="font-semibold">Could not load case</p>
            <p className="text-sm mt-1">{error}</p>
          </div>
        ) : (
          <ResultCard card={card} caseId={caseId} />
        )}
      </div>
    </div>
  );
}
