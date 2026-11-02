/**
 * API client for SourceDiff.
 *
 * DEMO MODE (frontend-only):
 *   Set DEMO = true  → all calls return mock data, no backend needed.
 *   Set DEMO = false → real backend calls (needs uvicorn running on port 8000).
 */

const DEMO = false;

// ─────────────────────────────────────────────
// Mock data
// ─────────────────────────────────────────────
const MOCK_HISTORY = [
    { id: 1, filename: "press_photo.jpg", door: "image", verdict: "authentic", label: "Authentic", submitted_at: "2026-11-01T10:22:00Z", case_id: "mock-001", user_id: "demo", stub_used: false, reason_code: "full_coverage_all_agree" },
    { id: 2, filename: "viral_clip.mp4", door: "video", verdict: "video_edited", label: "Edited", submitted_at: "2026-11-01T08:05:00Z", case_id: "mock-002", user_id: "demo", stub_used: false, reason_code: "video_channel_differs" },
    { id: 3, filename: "thumbnail.png", door: "image", verdict: "edited_picture", label: "Picture edited", submitted_at: "2026-10-31T17:45:00Z", case_id: "mock-003", user_id: "demo", stub_used: false, reason_code: "channel_differs" },
    { id: 4, filename: "speech.mp4", door: "video", verdict: "cant_verify", label: "Can't Verify", submitted_at: "2026-10-30T12:00:00Z", case_id: "mock-004", user_id: "demo", stub_used: false, reason_code: "no_original" },
    { id: 5, filename: "documentary.mp4", door: "video", verdict: "video_matched_no_edit", label: "Matched — no edit found", submitted_at: "2026-10-29T09:10:00Z", case_id: "mock-005", user_id: "demo", stub_used: false, reason_code: "video_checks_inside_ranges" },
];

const MOCK_LIBRARY = {
    image: [
        { source_id: "reuters-img-001", title: "Reuters Official Press Photo", publisher: "Reuters", license: "Reuters License", url: "https://reuters.com", notes: "", door: "image", sha256: "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2", added_at: "2026-10-20T10:00:00", added_by: "admin" },
        { source_id: "ap-photo-042", title: "AP News Agency Photograph", publisher: "AP", license: "AP License", url: "https://ap.org", notes: "", door: "image", sha256: "b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6b2c3", added_at: "2026-10-18T08:30:00", added_by: "admin" },
    ],
    video: [
        { source_id: "bbc-clip-news-7", title: "BBC News Broadcast Clip", publisher: "BBC", license: "BBC License", url: "https://bbc.com", notes: "", door: "video", sha256: "c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6c3d4", added_at: "2026-10-15T14:00:00", added_by: "admin" },
    ],
};

const MOCK_CARD = (caseId) => ({
    title: "Authentic",
    verdict: "authentic",
    reason_code: "full_coverage_all_agree",
    summary: "Every part of this file matches a trusted source, and every check agrees.",
    summary_hi: "यह फ़ाइल विश्वसनीय स्रोत से पूरी तरह मेल खाती है।",
    reasons: ["Every part of the upload matches source reuters-img-001 on one alignment, and every channel ran and agrees: picture, alignment."],
    notes: [],
    warnings: [],
    channels: [
        { channel: "picture", state: "agrees", detail: "Inside the genuine range" },
        { channel: "alignment", state: "agrees", detail: "One consistent alignment" },
    ],
    evidence: [
        { module: "picture_diff", metric: "psnr", value: 48.2, unit: "dB", genuine_range: "[30, ∞)", outside: false, counted: true, role: "channel:picture", calibrated: true, timestamps: [], note: "" },
        { module: "picture_diff", metric: "ssim", value: 0.991, unit: "", genuine_range: "[0.95, 1]", outside: false, counted: true, role: "channel:picture", calibrated: true, timestamps: [], note: "" },
    ],
    segments: [],
    images: {},
    input: { filename: "sample.jpg", sha256: "abc123def456abc123def456abc123de", door: "image", duration_s: null },
    source: "reuters-img-001",
    exact_identity: false,
    stub_used: false,
    engine_version: "0.2-path1",
    generated_at: new Date().toISOString(),
    card_sha256: "deadbeef1234deadbeef1234",
});

const MOCK_ANALYZE_RESULT = {
    case_id: "mock-live-001",
    cache_hit: false,
    card: MOCK_CARD("mock-live-001"),
    history: { id: 99, submitted_at: new Date().toISOString() },
};

// ─────────────────────────────────────────────
// Real fetch helper
// ─────────────────────────────────────────────
async function request(method, path, body = null, isForm = false) {
    const opts = { method, credentials: "include" };
    if (body) {
        if (isForm) { opts.body = body; }
        else { opts.headers = { "Content-Type": "application/json" }; opts.body = JSON.stringify(body); }
    }
    const res = await fetch(path, opts);
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || data.message || "Request failed");
    return data;
}

// ─────────────────────────────────────────────
// Auth
// ─────────────────────────────────────────────
export const authMe = () => DEMO
    ? Promise.resolve({ authenticated: true, username: JSON.parse(localStorage.getItem("sd_user") || "{}").username || "demo" })
    : request("GET", "/api/auth/me");

export const authLogin = (form) => DEMO
    ? Promise.resolve({ ok: true })
    : request("POST", "/api/auth/login", form, true);

export const authRegister = (form) => DEMO
    ? Promise.resolve({ ok: true })
    : request("POST", "/api/auth/register", form, true);

export const authLogout = () => DEMO
    ? (localStorage.removeItem("sd_user"), Promise.resolve({ ok: true }))
    : request("POST", "/api/auth/logout");

// ─────────────────────────────────────────────
// Analyze
// ─────────────────────────────────────────────
export const analyzeFile = (form) => DEMO
    ? new Promise((res) => setTimeout(() => res(MOCK_ANALYZE_RESULT), 1800))
    : request("POST", "/api/analyze", form, true);

// ─────────────────────────────────────────────
// History
// ─────────────────────────────────────────────
export const getHistory = (_username, limit = 100) => DEMO
    ? Promise.resolve({ rows: MOCK_HISTORY })
    : (() => {
        const p = new URLSearchParams({ limit });
        if (_username) p.set("user_id", _username);
        return request("GET", `/api/history?${p}`);
    })();

export const deleteHistory = (rowId) =>
    request("POST", `/api/history/${rowId}/delete`);

export const getCaseCard = (caseId) => DEMO
    ? Promise.resolve(MOCK_CARD(caseId))
    : request("GET", `/api/case/${caseId}/card`);

export const getCaseFileUrl = (caseId, name) => `/api/case/${caseId}/file/${name}`;
export const forgetUser = (form) => request("POST", "/api/forget", form, true);

// ─────────────────────────────────────────────
// Library
// ─────────────────────────────────────────────
export const getLibrary = () => DEMO
    ? Promise.resolve({ items: MOCK_LIBRARY, can_add: true, admin_key_required: false, accept: { image: [".jpg", ".png", ".webp"], video: [".mp4", ".mov"] } })
    : request("GET", "/api/library");

export const addToLibrary = (form) => DEMO
    ? new Promise((res) => setTimeout(() => res({ added: {}, message: "Demo mode: file not actually saved." }), 1000))
    : request("POST", "/api/library/add", form, true);

// ─────────────────────────────────────────────
// Health
// ─────────────────────────────────────────────
export const getHealth = () => DEMO
    ? Promise.resolve({ ok: true, engine: "0.2-path1", library_trusted: { image: 2, video: 1 } })
    : request("GET", "/api/health");
