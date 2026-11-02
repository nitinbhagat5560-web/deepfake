/**
 * Colored badge for each verdict type returned by the backend.
 */
const STYLES = {
  authentic:              "bg-emerald-100 text-emerald-700 border-emerald-200",
  edited_picture:         "bg-red-100 text-red-700 border-red-200",
  edited_video:           "bg-red-100 text-red-700 border-red-200",
  edited_spliced:         "bg-red-100 text-red-700 border-red-200",
  edited_audio:           "bg-orange-100 text-orange-700 border-orange-200",
  lips_resynthesized:     "bg-orange-100 text-orange-700 border-orange-200",
  not_authentic_partial:  "bg-yellow-100 text-yellow-700 border-yellow-200",
  cant_verify:            "bg-slate-100 text-slate-600 border-slate-200",
  video_edited:           "bg-red-100 text-red-700 border-red-200",
  video_matched_no_edit:  "bg-blue-100 text-blue-700 border-blue-200",
  video_incomplete:       "bg-yellow-100 text-yellow-700 border-yellow-200",
  path2_analysis:         "bg-violet-100 text-violet-700 border-violet-200",
  path2_inconclusive:     "bg-slate-100 text-slate-600 border-slate-200",
  path2_error:            "bg-red-100 text-red-700 border-red-200",
};

const DOTS = {
  authentic:             "bg-emerald-500",
  edited_picture:        "bg-red-500",
  edited_video:          "bg-red-500",
  edited_spliced:        "bg-red-500",
  edited_audio:          "bg-orange-500",
  lips_resynthesized:    "bg-orange-500",
  not_authentic_partial: "bg-yellow-500",
  cant_verify:           "bg-slate-400",
  video_edited:          "bg-red-500",
  video_matched_no_edit: "bg-blue-500",
  video_incomplete:      "bg-yellow-500",
  path2_analysis:        "bg-violet-500",
  path2_inconclusive:    "bg-slate-400",
  path2_error:           "bg-red-500",
};

export default function VerdictBadge({ verdict, label, size = "md" }) {
  const cls = STYLES[verdict] || "bg-slate-100 text-slate-600 border-slate-200";
  const dot = DOTS[verdict] || "bg-slate-400";
  const px  = size === "sm" ? "px-2.5 py-1 text-xs" : "px-4 py-2 text-sm";
  return (
    <span className={`inline-flex items-center gap-2 rounded-full border font-semibold ${px} ${cls}`}>
      <span className={`h-2 w-2 rounded-full ${dot}`} />
      {label || verdict}
    </span>
  );
}
