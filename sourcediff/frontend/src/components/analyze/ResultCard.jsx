import { CheckCircle2, AlertTriangle, HelpCircle, Image } from "lucide-react";
import VerdictBadge from "../shared/VerdictBadge";
import { getCaseFileUrl } from "../../api/client";

const CHANNEL_ICONS = {
  agrees:  <CheckCircle2 size={15} className="text-emerald-500" />,
  differs: <AlertTriangle size={15} className="text-red-500" />,
  not_run: <HelpCircle size={15} className="text-slate-400" />,
  na:      <HelpCircle size={15} className="text-slate-300" />,
};

export default function ResultCard({ card, caseId }) {
  if (!card) return null;
  const {
    title, verdict, summary, summary_hi,
    reasons = [], notes = [], warnings = [],
    channels = [], evidence = [], segments = [],
    images = {}, input = {}, path2, stub_used,
    exact_identity, engine_version, generated_at,
    card_sha256, change_explanation,
  } = card;

  const usedEvidence = evidence.filter((e) => e.counted);

  return (
    <section className="rounded-2xl border border-slate-200 bg-white shadow-sm overflow-hidden">

      {/* Header */}
      <div className={`px-6 py-5 border-b border-slate-100 flex flex-wrap items-start justify-between gap-3 ${
        verdict === "authentic"
          ? "bg-emerald-50"
          : verdict.startsWith("edited") || verdict === "video_edited"
          ? "bg-red-50"
          : "bg-slate-50"
      }`}>
        <div>
          <p className="text-xs font-semibold uppercase tracking-widest text-slate-500 mb-1">
            {input.door?.toUpperCase()} · {input.filename}
          </p>
          <h2 className="text-xl font-bold text-slate-900">{title}</h2>
          <p className="mt-1 text-sm text-slate-600 max-w-2xl">{summary}</p>
        </div>
        <VerdictBadge verdict={verdict} label={title} />
      </div>

      <div className="p-6 space-y-6">

        {/* Warnings */}
        {warnings.length > 0 && (
          <div className="rounded-xl bg-amber-50 border border-amber-200 p-4 space-y-1">
            {warnings.map((w, i) => (
              <p key={i} className="text-sm text-amber-700 flex items-start gap-2">
                <AlertTriangle size={15} className="mt-0.5 shrink-0" /> {w}
              </p>
            ))}
          </div>
        )}

        {/* Channels */}
        {channels.length > 0 && (
          <div>
            <h3 className="font-semibold text-slate-800 mb-3">Checks against the original</h3>
            <div className="overflow-x-auto rounded-xl border border-slate-100">
              <table className="w-full text-sm">
                <thead className="bg-slate-50 text-slate-500 text-xs uppercase">
                  <tr>
                    <th className="px-4 py-3 text-left">Channel</th>
                    <th className="px-4 py-3 text-left">Result</th>
                    <th className="px-4 py-3 text-left">Detail</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {channels.map((ch) => (
                    <tr key={ch.channel}>
                      <td className="px-4 py-3 font-medium text-slate-700 capitalize">{ch.channel}</td>
                      <td className="px-4 py-3">
                        <span className="flex items-center gap-1.5">
                          {CHANNEL_ICONS[ch.state] || CHANNEL_ICONS.na}
                          <span className="capitalize">{ch.state}</span>
                        </span>
                      </td>
                      <td className="px-4 py-3 text-slate-500">{ch.detail}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Changed parts and locations */}
        {change_explanation?.has_changes && (
          <div className="rounded-xl border border-red-200 bg-red-50 p-5">
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div>
                <h3 className="font-semibold text-red-900">What and where changed</h3>
                <p className="mt-1 text-sm text-red-800">{change_explanation.headline}</p>
              </div>
              {change_explanation.affected_area_pct !== null &&
               change_explanation.affected_area_pct !== undefined && (
                <span className="rounded-full bg-red-100 px-3 py-1 text-xs font-bold text-red-700">
                  Approximately {change_explanation.affected_area_pct}% affected
                </span>
              )}
            </div>

            {change_explanation.location_summary && (
              <div className="mt-4 grid gap-3 sm:grid-cols-2">
                <div className="rounded-lg bg-white/80 px-4 py-3">
                  <p className="text-xs font-semibold uppercase text-red-500">Detected location</p>
                  <p className="mt-1 font-semibold text-slate-800">{change_explanation.location_summary}</p>
                </div>
                {change_explanation.bounding_box && (
                  <div className="rounded-lg bg-white/80 px-4 py-3">
                    <p className="text-xs font-semibold uppercase text-red-500">Image region</p>
                    <p className="mt-1 font-mono text-sm text-slate-800">
                      x1={change_explanation.bounding_box[0]}, y1={change_explanation.bounding_box[1]},
                      {" "}x2={change_explanation.bounding_box[2]}, y2={change_explanation.bounding_box[3]}
                    </p>
                  </div>
                )}
              </div>
            )}

            <p className="mt-4 text-sm leading-6 text-red-900">{change_explanation.narrative}</p>

            {change_explanation.edit_types?.length > 0 && (
              <div className="mt-4 flex flex-wrap gap-2">
                {change_explanation.edit_types.map((type, i) => (
                  <span key={i} className="rounded-lg border border-red-200 bg-white px-3 py-1.5 text-xs font-semibold text-red-700">
                    {type}
                  </span>
                ))}
              </div>
            )}

            {change_explanation.items?.length > 0 && (
              <div className="mt-4 overflow-x-auto rounded-xl border border-red-200 bg-white">
                <table className="w-full text-sm">
                  <thead className="bg-red-100/60 text-left text-xs uppercase text-red-700">
                    <tr>
                      <th className="px-4 py-3">Where</th>
                      <th className="px-4 py-3">What changed</th>
                      <th className="px-4 py-3">Details / evidence</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-red-100">
                    {change_explanation.items.map((item, i) => (
                      <tr key={i}>
                        <td className="px-4 py-3 font-medium text-slate-800">{item.where}</td>
                        <td className="px-4 py-3 text-slate-700">{item.what}</td>
                        <td className="px-4 py-3 text-slate-600">{item.details}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}

        {/* Images */}
        {Object.keys(images).length > 0 && caseId && (
          <div>
            <h3 className="font-semibold text-slate-800 mb-3 flex items-center gap-2">
              <Image size={16} /> Visual Artifacts
            </h3>
            <div className="grid gap-4 sm:grid-cols-2">
              {Object.entries(images).map(([label, filename]) => (
                <div key={label} className="rounded-xl border border-slate-200 overflow-hidden">
                  <p className="px-3 py-2 text-xs font-medium text-slate-500 bg-slate-50 capitalize">
                    {label.replace(/_/g, " ")}
                  </p>
                  {/\.mp4|\.avi|\.mov|\.mkv|\.webm$/i.test(filename) ? (
                    <video
                      src={getCaseFileUrl(caseId, filename)}
                      controls
                      className="w-full max-h-64 bg-black"
                    />
                  ) : (
                    <img
                      src={getCaseFileUrl(caseId, filename)}
                      alt={label}
                      className="w-full object-contain max-h-64"
                    />
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Path 2 Section */}
        {path2 && (
          <div className="rounded-xl border border-violet-200 bg-violet-50 p-5">
            <h3 className="font-semibold text-violet-800 mb-2">Path 2 — AI Model Evidence</h3>
            <p className="text-sm text-violet-700 mb-1">
              <span className="font-medium">Final model:</span> GenD PE-L
            </p>
            <p className="text-sm text-violet-700 mb-3">
              <span className="font-medium">Analysis:</span> Face crops + whole frame
            </p>
            {path2.artifact && caseId && (
              <div className="mb-4 overflow-hidden rounded-xl border border-violet-200 bg-white">
                <p className="px-3 py-2 text-xs font-semibold uppercase tracking-wide text-violet-600 bg-violet-100">
                  Face / frame analysis
                </p>
                {/\.mp4|\.avi|\.mov|\.mkv|\.webm$/i.test(path2.artifact) ? (
                  <video
                    src={getCaseFileUrl(caseId, path2.artifact)}
                    controls
                    className="w-full max-h-[420px] bg-black"
                  />
                ) : (
                  <img
                    src={getCaseFileUrl(caseId, path2.artifact)}
                    alt="Path 2 face or frame analysis"
                    className="w-full max-h-[420px] object-contain"
                  />
                )}
              </div>
            )}
            {path2.explanation && (
              <>
                <div className="mb-3 rounded-lg border border-violet-200 bg-white px-4 py-3">
                  <p className="text-xs font-semibold uppercase tracking-wide text-violet-500">Final verdict</p>
                  <p className="mt-1 text-lg font-bold text-slate-900">
                    {path2.explanation.final_verdict || path2.explanation.headline}
                  </p>
                  <p className="mt-1 text-sm text-slate-700">
                    {path2.explanation.reason || path2.explanation.summary}
                  </p>
                </div>
                <p className="text-sm text-violet-700 mb-1">
                  <span className="font-medium">Status:</span> {path2.explanation.calibration_status}
                </p>
                <p className="text-sm text-violet-700 mb-3">{path2.explanation.headline}</p>
                {path2.explanation.observations?.length > 0 && (
                  <ul className="text-sm text-violet-700 list-disc ml-4 space-y-1 mb-2">
                    {path2.explanation.observations.map((obs, i) => <li key={i}>{obs}</li>)}
                  </ul>
                )}
                <p className="text-sm text-violet-600 italic">{path2.explanation.summary}</p>
                {path2.explanation.generative_report?.narrative && (
                  <div className="mt-3 rounded-lg bg-violet-100 p-3 text-sm text-violet-800">
                    <p className="font-medium mb-1">Generated Report</p>
                    <p>{path2.explanation.generative_report.narrative}</p>
                  </div>
                )}
              </>
            )}
          </div>
        )}

        {/* Evidence Table */}
        {usedEvidence.length > 0 && (
          <div>
            <h3 className="font-semibold text-slate-800 mb-3">Evidence that counted</h3>
            <div className="overflow-x-auto rounded-xl border border-slate-100">
              <table className="w-full text-sm">
                <thead className="bg-slate-50 text-slate-500 text-xs uppercase">
                  <tr>
                    <th className="px-4 py-3 text-left">Module</th>
                    <th className="px-4 py-3 text-left">Metric</th>
                    <th className="px-4 py-3 text-left">Value</th>
                    <th className="px-4 py-3 text-left">Genuine range</th>
                    <th className="px-4 py-3 text-left">Outside?</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {usedEvidence.map((e, i) => (
                    <tr key={i}>
                      <td className="px-4 py-3 font-mono text-xs text-slate-600">{e.module}</td>
                      <td className="px-4 py-3 text-slate-700">{e.metric}</td>
                      <td className="px-4 py-3 font-medium">{e.value} {e.unit}</td>
                      <td className="px-4 py-3 text-slate-500">{e.genuine_range ?? "—"}</td>
                      <td className="px-4 py-3">
                        {e.outside === null
                          ? <span className="text-slate-400">—</span>
                          : e.outside
                          ? <span className="text-red-600 font-semibold">Yes</span>
                          : <span className="text-emerald-600">No</span>
                        }
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Reasons */}
        {reasons.length > 0 && (
          <div>
            <h3 className="font-semibold text-slate-800 mb-2">Why</h3>
            <ul className="space-y-1">
              {reasons.map((r, i) => (
                <li key={i} className="text-sm text-slate-600 flex gap-2">
                  <span className="text-blue-400 mt-0.5">›</span> {r}
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* Notes */}
        {notes.length > 0 && (
          <div className="rounded-xl bg-slate-50 border border-slate-200 p-4">
            <h3 className="text-xs font-semibold uppercase text-slate-500 mb-2">Notes</h3>
            {notes.map((n, i) => (
              <p key={i} className="text-sm text-slate-600">{n}</p>
            ))}
          </div>
        )}

        {/* Hindi summary */}
        {summary_hi && (
          <div className="rounded-xl bg-indigo-50 border border-indigo-100 p-4">
            <p className="text-xs font-semibold uppercase text-indigo-400 mb-1">Hindi (Draft)</p>
            <p className="text-sm text-indigo-800">{summary_hi}</p>
          </div>
        )}

        {/* Footer metadata */}
        <div className="pt-2 border-t border-slate-100 text-xs text-slate-400 flex flex-wrap gap-x-4 gap-y-1">
          <span>SHA-256: <span className="font-mono">{input.sha256?.slice(0, 16)}…</span></span>
          <span>Card: <span className="font-mono">{card_sha256?.slice(0, 12)}…</span></span>
          <span>Engine: {engine_version}</span>
          {generated_at && <span>{new Date(generated_at).toLocaleString()}</span>}
          {stub_used && <span className="text-amber-500 font-semibold">⚠ Stub/Rehearsal result</span>}
        </div>
      </div>
    </section>
  );
}
