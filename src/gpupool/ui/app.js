/* gpupool management UI: one Alpine.js component, no build step. */
"use strict";

var KEY_STORE = "gpupool.adminKey";
var THEME_STORE = "gpupool.theme";

// Theme: "system" follows prefers-color-scheme (no attribute), "dark"/"light" force it via data-theme on <html>.
// Applied here, at script load in <head>, so the first paint already has the right colours.
function applyTheme(t) {
  var root = document.documentElement;
  if (t === "dark" || t === "light") root.setAttribute("data-theme", t);
  else root.removeAttribute("data-theme");
}
function savedTheme() {
  var t = "";
  try { t = localStorage.getItem(THEME_STORE) || ""; } catch (e) { t = ""; }
  return t === "dark" || t === "light" ? t : "system";
}
applyTheme(savedTheme());

// Inline SVG icons (stroke style). Keyed by name; rendered through icon().
var ICONS = {
  grid: '<rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" height="7" rx="1.5"/><rect x="3" y="14" width="7" height="7" rx="1.5"/><rect x="14" y="14" width="7" height="7" rx="1.5"/>',
  server: '<rect x="3" y="4" width="18" height="7" rx="2"/><rect x="3" y="13" width="18" height="7" rx="2"/><path d="M7 7.5h.01M7 16.5h.01"/>',
  chip: '<rect x="6" y="6" width="12" height="12" rx="2"/><path d="M9 2v4M15 2v4M9 18v4M15 18v4M2 9h4M2 15h4M18 9h4M18 15h4"/>',
  cube: '<path d="M12 2l9 5v10l-9 5-9-5V7z"/><path d="M12 12l9-5M12 12v10M12 12L3 7"/>',
  gear: '<circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.7 1.7 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-1.8-.3 1.7 1.7 0 0 0-1 1.5V21a2 2 0 1 1-4 0v-.1a1.7 1.7 0 0 0-1.1-1.5 1.7 1.7 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.7 1.7 0 0 0 .3-1.8 1.7 1.7 0 0 0-1.5-1H3a2 2 0 1 1 0-4h.1a1.7 1.7 0 0 0 1.5-1.1 1.7 1.7 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.7 1.7 0 0 0 1.8.3h0a1.7 1.7 0 0 0 1-1.5V3a2 2 0 1 1 4 0v.1a1.7 1.7 0 0 0 1 1.5 1.7 1.7 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.7 1.7 0 0 0-.3 1.8v0a1.7 1.7 0 0 0 1.5 1H21a2 2 0 1 1 0 4h-.1a1.7 1.7 0 0 0-1.5 1z"/>',
  play: '<path d="M6 4l14 8-14 8z"/>',
  bolt: '<path d="M13 2L4 14h7l-1 8 9-12h-7z"/>',
  db: '<ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v14c0 1.7 3.6 3 8 3s8-1.3 8-3V5M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3"/>',
  search: '<circle cx="11" cy="11" r="7"/><path d="M21 21l-4.3-4.3"/>',
  plus: '<path d="M12 5v14M5 12h14"/>',
  chev: '<path d="M9 6l6 6-6 6"/>',
  trash: '<path d="M4 7h16M10 11v6M14 11v6M6 7l1 13h10l1-13M9 7V4h6v3"/>',
  copy: '<rect x="9" y="9" width="12" height="12" rx="2"/><path d="M5 15V5a2 2 0 0 1 2-2h10"/>',
  logout: '<path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4M16 17l5-5-5-5M21 12H9"/>',
  stop: '<rect x="6" y="6" width="12" height="12" rx="2"/>',
  edit: '<path d="M12 20h9M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4z"/>',
  menu: '<path d="M3 6h18M3 12h18M3 18h18"/>',
  x: '<path d="M18 6L6 18M6 6l12 12"/>',
  key: '<circle cx="8" cy="15" r="4"/><path d="M10.8 12.2L21 2M16 7l3 3"/>',
  bell: '<path d="M18 8a6 6 0 0 0-12 0c0 7-3 9-3 9h18s-3-2-3-9M13.7 21a2 2 0 0 1-3.4 0"/>',
  alert: '<path d="M10.3 3.9L1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0zM12 9v4M12 17h.01"/>',
  info: '<circle cx="12" cy="12" r="9"/><path d="M12 8h.01M11 12h1v5h1"/>',
  list: '<path d="M8 6h13M8 12h13M8 18h13M3 6h.01M3 12h.01M3 18h.01"/>',
  up: '<path d="M12 19V5M5 12l7-7 7 7"/>',
  down: '<path d="M12 5v14M19 12l-7 7-7-7"/>',
  swap: '<path d="M7 4L3 8l4 4M3 8h14M17 20l4-4-4-4M21 16H7"/>',
  moon: '<path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/>'
};

// Event kinds with their own look; other kinds fall back to the level (info/warning/error).
var EV_KIND = {
  scaled_up: { cls: "green", icon: "up" },
  scaled_down: { cls: "blue", icon: "down" },
  unloaded_idle: { cls: "blue", icon: "moon" },
  cold_start: { cls: "amber", icon: "bolt" },
  preempted: { cls: "amber", icon: "alert" },
  rebalance_started: { cls: "blue", icon: "swap" },
  rebalanced: { cls: "green", icon: "swap" },
  rebalance_failed: { cls: "amber", icon: "alert" }
};

// Scaling part of the deploy form, derived from a model spec (defaults match the server's).
function scalingForm(spec) {
  var f = { mode: "fixed", replicas: 1, minR: 1, maxR: 2, target: 70, upAfter: 30, downAfter: 300, idleMin: 10 };
  if (!spec) return f;
  var as = spec.autoscale || {};
  if (as.target_busy != null) f.target = Math.round(as.target_busy * 100);
  if (as.up_after_s != null) f.upAfter = as.up_after_s;
  if (as.down_after_s != null) f.downAfter = as.down_after_s;
  if (spec.idle_unload_s != null) f.idleMin = Math.max(1, Math.round(spec.idle_unload_s / 60));
  if (spec.replicas >= 1) f.replicas = spec.replicas;
  var mn = spec.min_replicas, mx = spec.max_replicas;
  if (mn === 0 || spec.idle_unload_s != null) { f.mode = "demand"; f.maxR = mx != null ? mx : 1; }
  else if (mn != null && mx != null && mx > mn) { f.mode = "autoscale"; f.minR = mn; f.maxR = mx; }
  return f;
}

// Performance part of the deploy form (KV cache, attention, batching, speculative decoding), derived from a model spec.
function perfForm(spec) {
  var f = { kv: "f16", spec: "none", draftFile: "", draftN: 4, fa: "auto", ubatch: 512, batch: 2048, kvu: false };
  if (!spec) return f;
  if (spec.flash_attn) f.fa = spec.flash_attn;
  if (spec.kv_unified) f.kvu = true;
  if (spec.ubatch) f.ubatch = spec.ubatch;
  if (spec.batch) f.batch = spec.batch;
  if (spec.kv_cache_type) f.kv = spec.kv_cache_type;
  if (spec.speculative) f.spec = spec.speculative;
  if (spec.draft) f.draftFile = String(spec.draft).replace(/^coordinator:\/\//, "");
  if (spec.draft_n_max != null) f.draftN = spec.draft_n_max;
  return f;
}

// "Browse server folders" state of the Add model modal (GET /api/library/browse).
function emptyBrowse() {
  return { loading: false, loaded: false, err: "", roots: [], files: [], truncated: false };
}

// Conversion: stage order of a job and how each job state maps onto it.
// "calibrate" (the importance matrix) is only shown for jobs with imatrix_used.
var CONV_STAGES = [["download", "Download"], ["convert", "Convert"], ["calibrate", "Calibrate"], ["quantize", "Quantize"], ["validate", "Validate"]];
var CONV_STATE_KEY = { downloading: "download", converting: "convert", calibrating: "calibrate", quantizing: "quantize", validating: "validate" };
var CONV_ACTIVE = ["queued", "downloading", "converting", "calibrating", "quantizing", "validating"];
// Types the converter writes directly: no quantize step, no importance matrix.
var CONV_DIRECT = ["F16", "BF16", "Q8_0"];
// Types below this many bits per weight are tucked away behind "Show smaller, lower-quality types".
var CONV_LOW_BPW = 3.0;
var CONV_TIER = {
  lossless: ["Lossless", "accent"], near_lossless: ["Near-lossless", "green"], balanced: ["Balanced", "blue"],
  small: ["Small", "amber"], tiny: ["Tiny", ""]
};

// State of the "Convert a model" modal.
function emptyConv() {
  return {
    open: false, tab: "hf", repo: "", revision: "", path: "", inspecting: false, err: "", res: null, quant: "Q4_K_M",
    name: "", nameEdited: false, keepSource: false, advOpen: false, busy: false, showLow: false,
    adv: { intermediate: "auto", output_tensor_type: "", token_embedding_type: "", leave_output_tensor: false, pure: false, allow_remote_code: false, validate_generation: true, threads: 0,
           imatrix: "auto", calibration_path: "", imatrix_chunks: 0 }
  };
}

function app() {
  return {
    // ----- session -----
    key: "",
    authed: false,
    loginKey: "",
    loginErr: "",
    loginBusy: false,
    online: true,
    theme: "system", // "system" | "dark" | "light"
    bellPop: false, // brief pop animation of the bell badge when the unread count grows
    prevUnread: null,
    // ----- data -----
    st: null,
    flags: {}, // optimistic GPU toggles in flight: "node/dev" -> bool
    timer: null,
    // ----- ui state -----
    view: "overview",
    navOpen: false,
    search: "",
    collapsed: {}, // node_id -> true when collapsed
    sel: null, // { node: node_id, dev: device_id }
    detailTab: "overview",
    toasts: [],
    toastSeq: 0,
    rep: {}, // model name -> replicas input for Start
    openErr: {}, // model name -> error text expanded
    // events
    evOpen: false,
    evLast: null, // highest event id seen in this session (null until the first poll)
    evFilter: "all",
    evList: [],
    desktop: false, // desktop notifications enabled
    nowTs: Date.now() / 1000,
    // add server modal
    addSrv: { open: false, url: "", name: "gpu-node-1", busy: false },
    delSrv: null, // server object pending delete confirmation
    // add model modal
    addMdl: { open: false, tab: "hf", repo: "", files: [], file: "", path: "", busy: false, listing: false, err: "", noGguf: false, browse: emptyBrowse() },
    // conversion (Hugging Face -> GGUF)
    conv: emptyConv(),
    convOpts: { loaded: false, available: true, problem: null, imatrix_available: true, quant_options: [], cluster: { largest_gpu_mb: 0, pool_mb: 0 } },
    convJobs: [],
    convErr: "",
    convSeen: null, // job id -> last seen state (null until the first poll: the backlog never toasts)
    convStage: {}, // job id -> last stage key seen while polling: fallback for a failed job that carries no failed_stage
    convUi: {}, // job id -> { log, val } expanded panels
    convBusy: {}, // job id -> an action is in flight
    convPollAt: 0,
    convPolling: false,
    // deploy (new / edit) modal
    form: Object.assign({ open: false, edit: false, name: "", file: "", ctx: 4096, parallel: 1, priority: 50, preemptible: true, spread: "gpu", auto: true, pins: [], busy: false, plan: null, rec: null, recBusy: false, sim: null, simBusy: false }, scalingForm(null), perfForm(null)),
    rb: { busy: false, checked: false, moves: [] }, // "Placement health" panel: last check / rebalance result
    sc: {}, // model name -> { open, busy, data, err } for the "Scaling details" panel

    // ================= lifecycle =================
    init: function () {
      var hv = (location.hash || "").replace("#", "");
      if (["overview", "servers", "gpus", "models", "events", "settings"].indexOf(hv) >= 0) this.view = hv;
      this.theme = savedTheme();
      var saved = "";
      try { saved = localStorage.getItem(KEY_STORE) || ""; } catch (e) { saved = ""; }
      try { this.desktop = localStorage.getItem("gpupool.desktopAlerts") === "1" && window.Notification && Notification.permission === "granted"; } catch (e) { this.desktop = false; }
      if (saved) {
        this.key = saved;
        this.authed = true;
        this.refresh();
      }
      var self = this;
      setInterval(function () { self.nowTs = Date.now() / 1000; }, 5000);
      this.timer = setInterval(function () { if (!document.hidden && self.authed) self.refresh(); }, 2000);
      setInterval(function () { self.convTick(); }, 1000);
      document.addEventListener("visibilitychange", function () { if (!document.hidden && self.authed) self.refresh(); });
    },

    icon: function (name) {
      return '<svg class="ic" viewBox="0 0 24 24" aria-hidden="true">' + (ICONS[name] || "") + "</svg>";
    },

    // ================= HTTP =================
    api: async function (method, path, body) {
      var opts = { method: method, headers: { "Authorization": "Bearer " + this.key } };
      if (body !== undefined) {
        opts.headers["Content-Type"] = "application/json";
        opts.body = JSON.stringify(body);
      }
      var res;
      try {
        res = await fetch(path, opts);
      } catch (e) {
        this.online = false;
        throw new Error("Cannot reach the coordinator");
      }
      this.online = true;
      var data = null;
      var txt = await res.text();
      if (txt) { try { data = JSON.parse(txt); } catch (e) { data = txt; } }
      if (res.status === 401) {
        this.logout("Admin key rejected. Please sign in again.");
        var e401 = new Error("Unauthorized"); e401.status = 401; throw e401;
      }
      if (!res.ok) {
        var err = new Error(this.errText(data, res.status));
        err.status = res.status;
        throw err;
      }
      return data;
    },

    errText: function (data, status) {
      if (data && typeof data === "object" && data.detail !== undefined) {
        var d = data.detail;
        if (typeof d === "string") return d;
        if (Array.isArray(d)) {
          return d.map(function (x) {
            var loc = Array.isArray(x.loc) ? x.loc.filter(function (p) { return p !== "body"; }).join(".") : "";
            return (loc ? loc + ": " : "") + (x.msg || JSON.stringify(x));
          }).join("; ");
        }
        return JSON.stringify(d);
      }
      if (typeof data === "string" && data) return data.slice(0, 300);
      return "Request failed (HTTP " + status + ")";
    },

    refresh: async function () {
      try {
        this.st = await this.api("GET", "/api/state");
        this.processEvents();
        this.noteUnread();
        if (this.view === "events") this.loadEvents();
        this.refreshScaling();
      } catch (e) { /* offline or 401 already handled */ }
    },

    // ================= theme =================
    setTheme: function (t) {
      this.theme = t === "dark" || t === "light" ? t : "system";
      applyTheme(this.theme);
      try { localStorage.setItem(THEME_STORE, this.theme); } catch (e) { /* storage blocked */ }
    },

    // ================= events =================
    noteUnread: function () {
      var n = this.unread(), self = this;
      if (this.prevUnread !== null && n > this.prevUnread) {
        this.bellPop = false;
        setTimeout(function () { self.bellPop = true; setTimeout(function () { self.bellPop = false; }, 600); }, 20);
      }
      this.prevUnread = n;
    },
    events: function () { return (this.st && this.st.events) || []; },
    unread: function () { return (this.st && this.st.unread_events) || 0; },
    processEvents: function () {
      var evs = this.events(), self = this;
      var max = evs.reduce(function (m, e) { return Math.max(m, e.id); }, 0);
      if (this.evLast === null) { this.evLast = max; return; } // never toast the backlog
      var fresh = evs.filter(function (e) { return e.id > self.evLast; }).reverse();
      this.evLast = Math.max(this.evLast, max);
      fresh.forEach(function (e) {
        if (e.level === "error") self.toast(e.message, "error", true);
        else if (e.level === "warning") self.toast(e.message, "warning", false, 8000);
        if ((e.level === "error" || e.level === "warning") && self.desktop && document.hidden) {
          try { new Notification("gpupool: " + e.kind.replace(/_/g, " "), { body: e.message }); } catch (err) { /* ignore */ }
        }
      });
    },
    loadEvents: async function () {
      try { const r = await this.api("GET", "/api/events?limit=200"); this.evList = Array.isArray(r) ? r : ((r && r.events) || []); } catch (e) { this.fail(e); }
    },
    filteredEvents: function () {
      var f = this.evFilter;
      return this.evList.filter(function (e) { return f === "all" || e.level === f; });
    },
    markRead: async function () {
      var max = this.events().reduce(function (m, e) { return Math.max(m, e.id); }, 0);
      try { await this.api("POST", "/api/events/read", { up_to_id: max }); await this.refresh(); } catch (e) { this.fail(e); }
    },
    evClass: function (e) { var k = EV_KIND[e.kind]; return k ? k.cls : (e.level === "error" ? "red" : (e.level === "warning" ? "amber" : "blue")); },
    evIcon: function (e) { var k = EV_KIND[e.kind]; return k ? k.icon : (e.level === "info" ? "info" : "alert"); },
    ago: function (ts) {
      var d = Math.max(0, Math.round(this.nowTs - ts));
      if (d < 60) return d + "s ago";
      if (d < 3600) return Math.floor(d / 60) + "m ago";
      if (d < 86400) return Math.floor(d / 3600) + "h ago";
      return Math.floor(d / 86400) + "d ago";
    },
    clock: function (ts) { return new Date(ts * 1000).toLocaleTimeString(); },
    evGo: function (e, target) { this.evOpen = false; this.go(target); },
    latestEvent: function (kinds, field, value) {
      return this.events().find(function (e) { return kinds.indexOf(e.kind) >= 0 && e[field] === value; }) || null;
    },
    offlineInfo: function (s) {
      var down = this.latestEvent(["node_offline"], "node_id", s.node_id);
      var re = this.latestEvent(["realloc_started", "realloc_done", "realloc_failed"], "node_id", s.node_id);
      var since = down ? "Offline since " + this.clock(down.ts) : "Offline";
      var tail = " — ";
      if (!re) tail += "no models were running on it";
      else if (re.kind === "realloc_done") tail += "models re-allocated";
      else if (re.kind === "realloc_failed") tail += "waiting for capacity";
      else tail += "re-allocating models";
      return since + tail;
    },
    reallocError: function (m) {
      var e = this.latestEvent(["realloc_failed", "launch_failed"], "model", m.spec.name);
      return e ? e.message : "";
    },
    toggleDesktop: async function (on) {
      if (on) {
        if (!window.Notification) { this.toast("Desktop notifications are not supported here", "error"); this.desktop = false; return; }
        var p = Notification.permission === "granted" ? "granted" : await Notification.requestPermission();
        this.desktop = p === "granted";
        if (!this.desktop) this.toast("Permission denied by the browser", "error");
      } else this.desktop = false;
      try { localStorage.setItem("gpupool.desktopAlerts", this.desktop ? "1" : "0"); } catch (e) { /* ignore */ }
    },

    // ================= auth =================
    login: async function () {
      var k = this.loginKey.trim();
      if (!k) return;
      this.loginBusy = true;
      this.loginErr = "";
      this.key = k;
      try {
        this.st = await this.api("GET", "/api/state");
        try { localStorage.setItem(KEY_STORE, k); } catch (e) { /* storage blocked */ }
        this.authed = true;
        this.loginKey = "";
      } catch (e) {
        if (e.status !== 401) this.loginErr = e.message;
        else this.loginErr = "Invalid admin key.";
        this.key = "";
        this.authed = false;
      }
      this.loginBusy = false;
    },

    logout: function (msg) {
      try { localStorage.removeItem(KEY_STORE); } catch (e) { /* ignore */ }
      this.key = "";
      this.authed = false;
      this.st = null;
      this.loginErr = typeof msg === "string" ? msg : "";
    },

    // ================= toasts =================
    toast: function (msg, kind, sticky, ms) {
      var id = ++this.toastSeq;
      this.toasts.push({ id: id, msg: msg, kind: kind || "info" });
      if (!sticky) setTimeout(function (self) { self.dismiss(id); }, ms || (kind === "error" ? 7000 : 3500), this);
    },
    dismiss: function (id) { this.toasts = this.toasts.filter(function (t) { return t.id !== id; }); },
    fail: function (e) { if (e && e.status !== 401) this.toast(e.message || String(e), "error"); },

    // ================= formatting =================
    dash: "—",
    gb: function (mb) { return mb == null ? this.dash : (mb / 1024).toFixed(mb >= 10240 ? 0 : 1); },
    mem: function (d) {
      if (d.total_mb == null) return this.dash;
      return this.gb(d.total_mb - d.free_mb) + " / " + this.gb(d.total_mb) + " GB";
    },
    memPct: function (d) { return d.total_mb ? Math.round((d.total_mb - d.free_mb) * 100 / d.total_mb) : 0; },
    pct: function (v) { return v == null ? this.dash : Math.round(v) + "%"; },
    val: function (v, unit) { return v == null ? this.dash : v + unit; },
    bytes: function (n) {
      if (n == null) return this.dash;
      var u = ["B", "KB", "MB", "GB", "TB"], i = 0, x = n;
      while (x >= 1024 && i < u.length - 1) { x /= 1024; i++; }
      return x.toFixed(i >= 3 ? 2 : (i ? 1 : 0)) + " " + u[i];
    },
    level: function (p) { return p == null ? "" : (p >= 85 ? "red" : (p >= 60 ? "amber" : "green")); },
    clamp: function (p) { return Math.max(0, Math.min(100, p == null ? 0 : p)); },
    isGpu: function (d) { return d.kind !== "cpu"; },
    devName: function (d) { return d.kind === "cpu" ? "CPU (RAM)" : d.name; },
    busy: function (d) { return d.util_pct != null && d.util_pct >= 50; },
    ringDash: function (p) { var c = 2 * Math.PI * 60; return c + " " + c; },
    ringOffset: function (p) { var c = 2 * Math.PI * 60; return c * (1 - this.clamp(p) / 100); },
    ringColor: function (p) { var l = this.level(p); return l === "red" ? "var(--danger)" : (l === "amber" ? "var(--warning)" : "var(--success)"); },

    // ================= derived data =================
    servers: function () { return this.st ? this.st.servers : []; },
    models: function () { return this.st ? this.st.models : []; },
    library: function () { return this.st ? this.st.library : []; },
    settings: function () { return this.st ? this.st.settings : {}; },
    summary: function () { return this.st ? this.st.summary : {}; },
    devs: function (s) { return s.report ? s.report.devices : []; },
    gpus: function (s) { return this.devs(s).filter(function (d) { return d.kind !== "cpu"; }); },
    host: function (s) {
      if (s.report && s.report.host) return s.report.host;
      try { return new URL(s.agent_url).hostname; } catch (e) { return s.agent_url; }
    },
    gpuModel: function (s) { var g = this.gpus(s); return g.length ? g[0].name : this.dash; },
    freePoolGb: function () {
      var sum = this.summary();
      return sum.pool_usable_mb == null ? this.dash : this.gb(sum.pool_usable_mb);
    },
    filteredServers: function () {
      var q = this.search.trim().toLowerCase();
      if (!q) return this.servers();
      var self = this;
      return this.servers().filter(function (s) {
        var hay = [s.node_id, s.agent_url, self.host(s)].concat(self.devs(s).map(function (d) { return d.name; })).join(" ").toLowerCase();
        return hay.indexOf(q) >= 0;
      });
    },
    filteredModels: function () {
      var q = this.search.trim().toLowerCase();
      if (!q) return this.models();
      return this.models().filter(function (m) { return (m.spec.name + " " + (m.file || "")).toLowerCase().indexOf(q) >= 0; });
    },
    filteredLibrary: function () {
      var q = this.search.trim().toLowerCase();
      if (!q) return this.library();
      return this.library().filter(function (l) { return l.name.toLowerCase().indexOf(q) >= 0; });
    },
    allGpuRows: function () {
      var rows = [], q = this.search.trim().toLowerCase(), self = this;
      this.servers().forEach(function (s) {
        self.gpus(s).forEach(function (d) {
          if (!q || (s.node_id + " " + d.name + " " + d.device_id).toLowerCase().indexOf(q) >= 0) rows.push({ s: s, d: d });
        });
      });
      return rows;
    },
    readyLibrary: function () { return this.library().filter(function (l) { return l.status === "ready"; }); },
    enabled: function (s, d) {
      var k = s.node_id + "/" + d.device_id;
      if (k in this.flags) return this.flags[k];
      var f = s.gpu_enabled || {};
      return f[d.device_id] !== false; // default: in the pool
    },

    // ================= servers =================
    isOpen: function (s) { return !this.collapsed[s.node_id]; },
    toggleOpen: function (s) { this.collapsed[s.node_id] = !this.collapsed[s.node_id]; },
    select: function (s, d) {
      this.sel = { node: s.node_id, dev: d.device_id };
      this.detailTab = "overview";
    },
    isSel: function (s, d) { return !!this.sel && this.sel.node === s.node_id && this.sel.dev === d.device_id; },
    selected: function () {
      if (!this.sel) return null;
      var s = this.servers().find(function (x) { return x.node_id === this.sel.node; }, this);
      if (!s) return null;
      var d = this.devs(s).find(function (x) { return x.device_id === this.sel.dev; }, this);
      return d ? { s: s, d: d } : null;
    },

    toggleGpu: async function (s, d, on) {
      var k = s.node_id + "/" + d.device_id;
      this.flags[k] = on; // optimistic
      try {
        await this.api("PUT", "/api/servers/" + encodeURIComponent(s.node_id) + "/gpus/" + encodeURIComponent(d.device_id), { enabled: on });
        await this.refresh();
      } catch (e) {
        this.fail(e); // reverted below by dropping the override
      } finally {
        delete this.flags[k];
      }
    },

    // The URL servers use to reach this coordinator: the configured public URL, else the
    // address the browser used (inside Docker the server-side detection only sees a container IP).
    coordUrl: function () {
      return (this.settings().public_url || location.origin).replace(/\/+$/, "");
    },
    joinString: function () {
      return this.coordUrl() + "#" + (this.settings().cluster_token || "<cluster_token>");
    },
    agentCmd: function () {
      return "docker run -d --name gpupool-agent --gpus all --network host --pid host -v gpupool-agent:/data" +
        " -e GPUPOOL_JOIN='" + this.joinString() + "' ghcr.io/longduongbao29/gpupool-agent";
    },
    agentCmdUv: function () {
      return "uv run gpupool agent --join '" + this.joinString() + "' --llama-dir /path/to/llama.cpp/bin";
    },
    joinCmd: function (tab) { return tab === "uv" ? this.agentCmdUv() : this.agentCmd(); },
    openAddServer: function () {
      this.addSrv = { open: true, url: "", tab: "docker", busy: false };
    },
    addServer: async function () {
      var url = this.addSrv.url.trim();
      if (!url) return;
      this.addSrv.busy = true;
      try {
        var r = await this.api("POST", "/api/servers", { agent_url: url });
        this.toast("Server " + ((r && r.node_id) || url) + " added", "ok");
        this.addSrv.open = false;
        await this.refresh();
      } catch (e) { this.fail(e); }
      this.addSrv.busy = false;
    },
    deleteServer: async function () {
      var s = this.delSrv;
      if (!s) return;
      try {
        await this.api("DELETE", "/api/servers/" + encodeURIComponent(s.node_id));
        this.toast("Server " + s.node_id + " removed", "ok");
        if (this.sel && this.sel.node === s.node_id) this.sel = null;
        this.delSrv = null;
        await this.refresh();
      } catch (e) { this.fail(e); }
    },
    modelsOnServer: function (s) {
      var out = [];
      this.models().forEach(function (m) {
        var hit = (m.replicas || []).some(function (r) {
          return r.placement && r.placement.assignments.some(function (a) { return a.node_id === s.node_id; });
        });
        if (hit) out.push(m.spec.name);
      });
      return out;
    },

    copy: async function (text) {
      try {
        await navigator.clipboard.writeText(text);
      } catch (e) {
        var ta = document.createElement("textarea");
        ta.value = text; document.body.appendChild(ta); ta.select();
        try { document.execCommand("copy"); } catch (e2) { /* ignore */ }
        document.body.removeChild(ta);
      }
      this.toast("Copied", "ok");
    },

    // ================= library =================
    openAddModel: function () {
      this.addMdl = { open: true, tab: "hf", repo: "", files: [], file: "", path: "", busy: false, listing: false, err: "", noGguf: false, browse: emptyBrowse() };
    },
    // Files the coordinator can see under its model folders (inside Docker: only the mounted ones).
    loadBrowse: async function (force) {
      var b = this.addMdl.browse;
      if (b.loading || (b.loaded && !force)) return;
      b.loading = true;
      b.err = "";
      try {
        var r = await this.api("GET", "/api/library/browse");
        b.roots = (r && r.roots) || [];
        b.files = (r && r.files) || [];
        b.truncated = !!(r && r.truncated);
        b.loaded = true;
      } catch (e) {
        if (e.status !== 401) { b.err = "Could not list the server folders: " + e.message; b.loaded = true; b.roots = []; b.files = []; b.truncated = false; }
      }
      b.loading = false;
    },
    browseUsable: function (f) { return !f.in_library && !f.split_part && !f.broken_link; },
    browseCount: function () {
      var n = this.addMdl.browse.files.length;
      return n + (n === 1 ? " file" : " files") + (this.addMdl.browse.truncated ? "+" : "");
    },
    pickFile: function (f) {
      if (!this.browseUsable(f)) return;
      this.addMdl.path = f.path;
      this.addMdl.err = "";
    },
    folderOf: function (p) { var i = String(p).lastIndexOf("/"); return i > 0 ? p.slice(0, i) : "/"; },
    listHf: async function () {
      var repo = this.addMdl.repo.trim();
      if (!repo) return;
      this.addMdl.listing = true;
      this.addMdl.files = [];
      this.addMdl.file = "";
      this.addMdl.noGguf = false;
      try {
        var files = await this.api("GET", "/api/hf/files?repo=" + encodeURIComponent(repo).replace(/%2F/g, "/"));
        this.addMdl.files = files || [];
        if (!this.addMdl.files.length) { this.addMdl.noGguf = true; this.toast("No .gguf files in that repository", "error"); }
        else this.addMdl.file = this.addMdl.files[0].file;
      } catch (e) { this.fail(e); }
      this.addMdl.listing = false;
    },
    addToLibrary: async function () {
      var a = this.addMdl, body;
      if (a.tab === "hf") {
        if (!a.file) return;
        body = { hf_repo: a.repo.trim(), hf_file: a.file };
      } else {
        if (!a.path.trim()) return;
        body = { path: a.path.trim() };
      }
      a.busy = true;
      a.err = "";
      try {
        await this.api("POST", "/api/library", body);
        this.toast(a.tab === "hf" ? "Download started" : "Model added", "ok");
        a.open = false;
        await this.refresh();
      } catch (e) {
        // Shown in full inside the modal: these messages (missing file, mounted folders) are long and a toast would cut them.
        if (e.status !== 401) a.err = e.message || String(e);
      }
      a.busy = false;
    },
    deleteLibrary: async function (it) {
      if (!confirm("Remove " + it.name + " from the library?")) return;
      try {
        await this.api("DELETE", "/api/library/" + encodeURIComponent(it.name));
        await this.refresh();
      } catch (e) { this.fail(e); }
    },
    srcLabel: function (it) { return it.source === "hf" ? "Hugging Face" : (it.source === "convert" ? "Converted" : "Path"); },
    srcDetail: function (it) { return it.source === "path" ? it.path : (it.hf_repo || it.path); },
    libPct: function (it) {
      return it.bytes ? Math.min(100, Math.round((it.downloaded || 0) * 100 / it.bytes)) : 0;
    },
    libStatusClass: function (it) { return it.status === "ready" ? "green" : (it.status === "failed" ? "red" : "amber"); },

    // ================= conversion: Hugging Face / folder -> GGUF =================
    convActive: function () {
      return this.convJobs.some(function (j) { return CONV_ACTIVE.indexOf(j.state) >= 0; });
    },
    // One timer for everything: jobs are polled every 2 s while one is active, every 10 s on the Models page, else not at all.
    convTick: function () {
      if (!this.authed || document.hidden) return;
      if (this.view === "models" && !this.convOpts.loaded && !this.convOptsBusy) {
        this.convOptsBusy = true;
        var self = this;
        this.convLoadOptions().then(function () { self.convOptsBusy = false; });
      }
      var every = this.convActive() ? 2000 : (this.view === "models" ? 10000 : 0);
      if (this.convPollAt === 0 || (every && Date.now() - this.convPollAt >= every)) this.convPoll();
    },
    convPoll: async function () {
      if (this.convPolling || !this.authed) return;
      this.convPolling = true;
      this.convPollAt = Date.now();
      try {
        var jobs = await this.api("GET", "/api/convert");
        this.convApply(Array.isArray(jobs) ? jobs : []);
        this.convErr = "";
      } catch (e) {
        if (e.status === 404) this.convErr = "This coordinator does not support model conversion.";
        else if (e.status !== 401) this.convErr = e.message;
      }
      this.convPolling = false;
    },
    // Store the polled jobs; toast the transitions seen since the previous poll.
    convApply: function (jobs) {
      var seen = this.convSeen, next = {}, self = this, libDirty = false;
      jobs.forEach(function (j) {
        next[j.id] = j.state;
        if (CONV_STATE_KEY[j.state] !== undefined) self.convStage[j.id] = CONV_STATE_KEY[j.state];
        if (seen === null || seen[j.id] === undefined || seen[j.id] === j.state) return;
        if (j.state === "done") { self.toast("Converted " + j.output_name + ": it is in the library now", "ok"); libDirty = true; }
        else if (j.state === "needs_review") self.toast(j.output_name + " converted but needs your review", "warning", false, 9000);
        else if (j.state === "failed") self.toast("Conversion of " + j.output_name + " failed", "error");
      });
      this.convSeen = next;
      this.convJobs = jobs;
      if (libDirty) this.refresh();
    },
    convLoadOptions: async function () {
      try {
        var o = await this.api("GET", "/api/convert/options");
        this.convOpts = { loaded: true, available: !!o.available, problem: o.problem || null, imatrix_available: o.imatrix_available !== false,
          quant_options: o.quant_options || [], cluster: o.cluster || { largest_gpu_mb: 0, pool_mb: 0 } };
      } catch (e) {
        if (e.status === 404) this.convOpts = { loaded: true, available: false, problem: "This coordinator does not support model conversion.", imatrix_available: true, quant_options: [], cluster: { largest_gpu_mb: 0, pool_mb: 0 } };
        else { this.convOpts.loaded = true; this.fail(e); }
      }
    },
    // prefill: { repo } or { path }. The inspection starts at once.
    openConvert: function (prefill) {
      var c = emptyConv();
      c.open = true;
      if (prefill && prefill.repo) { c.tab = "hf"; c.repo = prefill.repo; }
      else if (prefill && prefill.path) { c.tab = "path"; c.path = prefill.path; }
      this.conv = c;
      this.convLoadOptions();
      if (prefill && (prefill.repo || prefill.path)) this.convInspect();
    },
    convertFromAdd: function () {
      var repo = this.addMdl.repo.trim();
      this.addMdl.open = false;
      this.openConvert({ repo: repo });
    },
    // Hand a ready-made GGUF repo from the inspection over to the normal "Add model" flow.
    convUseGguf: function (repo) {
      this.conv.open = false;
      this.openAddModel();
      this.addMdl.repo = repo;
      this.listHf();
    },
    convUseBase: function (repo) {
      this.conv.tab = "hf";
      this.conv.repo = repo;
      this.conv.revision = "";
      this.convInspect();
    },
    convSetTab: function (t) { this.conv.tab = t; this.conv.err = ""; this.conv.res = null; },
    convSpec: function () {
      var c = this.conv;
      if (c.tab === "path") return c.path.trim() ? { path: c.path.trim() } : null;
      var repo = c.repo.trim().replace(/^https?:\/\/huggingface\.co\//, "").replace(/\/+$/, "");
      if (!repo) return null;
      return { hf_repo: repo, revision: c.revision.trim() || "main" };
    },
    convInspect: async function () {
      var spec = this.convSpec(), c = this.conv;
      if (!spec) return;
      c.inspecting = true;
      c.err = "";
      c.res = null;
      try {
        var r = await this.api("POST", "/api/convert/inspect", spec);
        c.res = r;
        c.quant = r.recommended || "Q4_K_M";
        c.showLow = false;
        c.nameEdited = false;
        c.name = this.convDefaultName();
        c.advOpen = false;
        c.adv.allow_remote_code = false;
      } catch (e) {
        if (e.status !== 401) c.err = e.message || String(e);
      }
      c.inspecting = false;
    },
    // The backend default "<name_stem>-<QUANT>.gguf": the stem comes from inspect, so the UI never has its own naming rule.
    convDefaultName: function () {
      var c = this.conv;
      return ((c.res && c.res.name_stem) || "model") + "-" + c.quant + ".gguf";
    },
    convPick: function (t) {
      var o = ((this.conv.res && this.conv.res.options) || []).find(function (x) { return x.type === t; });
      if (o && this.convOptDisabled(o)) return;
      this.conv.quant = t;
      if (o && o.needs_imatrix && this.conv.adv.imatrix === "off") this.conv.adv.imatrix = "auto"; // Off is refused for these types
      if (!this.conv.nameEdited) this.conv.name = this.convDefaultName();
    },
    // ----- quantization picker: very-low-bit types folded away, importance-matrix badge and gate -----
    convIsLow: function (o) { return o.bpw < CONV_LOW_BPW; },
    // The chosen type always stays visible, even when the low-bit group is folded.
    convVisibleOptions: function () {
      var self = this, c = this.conv;
      return ((c.res && c.res.options) || []).filter(function (o) { return !self.convIsLow(o) || c.showLow || o.type === c.quant; });
    },
    convHiddenCount: function () {
      return (((this.conv.res && this.conv.res.options) || []).length) - this.convVisibleOptions().length;
    },
    convLowCount: function () { var self = this; return ((this.conv.res && this.conv.res.options) || []).filter(function (o) { return self.convIsLow(o); }).length; },
    convOptDisabled: function (o) { return !!o.needs_imatrix && !this.convOpts.imatrix_available; },
    // Does the importance matrix apply to the selected type at all (F16/BF16/Q8_0 are written directly)?
    convImatrixApplies: function () { var o = this.convOption(); return !!o && o.via !== "convert" && CONV_DIRECT.indexOf(o.type) < 0; },
    // Mode sent to the server: without llama-imatrix on the coordinator the only workable choice is off.
    convImatrixMode: function () {
      if (!this.convImatrixApplies()) return "auto";
      return this.convOpts.imatrix_available ? this.conv.adv.imatrix : "off";
    },
    // Will a calibration step run? Mirrors the server: auto = on for types that need it or are under 4 bits.
    convImatrixWillRun: function () {
      var o = this.convOption(), m = this.convImatrixMode();
      if (!o || !this.convImatrixApplies()) return false;
      return m === "on" || (m === "auto" && (!!o.needs_imatrix || o.bpw < 4.0));
    },
    convCalibPathErr: function () {
      var p = this.conv.adv.calibration_path.trim();
      if (!p) return "";
      if (!/^(\/|[A-Za-z]:[\\/])/.test(p)) return "The calibration text must be an absolute path on the server";
      if (!/\.txt$/i.test(p)) return "The calibration text must be a .txt file";
      return "";
    },
    convNameInput: function () {
      this.conv.nameEdited = this.conv.name.trim() !== "" && this.conv.name !== this.convDefaultName();
    },
    convNameOk: function () { return /^[A-Za-z0-9][A-Za-z0-9._-]*\.gguf$/i.test(this.conv.name.trim()); },
    convOption: function () {
      var c = this.conv;
      return c.res ? (c.res.options || []).find(function (o) { return o.type === c.quant; }) || null : null;
    },
    convViaConvert: function () { var o = this.convOption(); return !!o && o.via === "convert"; },
    // Can this source be converted at all (the picker is hidden otherwise)?
    convertible: function () {
      var r = this.conv.res;
      if (!r) return false;
      if (r.supported === false) return false;
      if (r.prequantized && r.prequant_supported === false) return false;
      return r.weight_format !== "none";
    },
    convBlocker: function () {
      var r = this.conv.res;
      if (!r) return "";
      if (r.supported === false) return "The converter does not support the architecture " + (r.architecture || "of this model") + ", so it cannot be converted.";
      if (r.prequantized && r.prequant_supported === false) return "This model is already quantized with " + r.prequantized.toUpperCase() + ", a format the converter cannot read back. Convert the original (unquantized) model instead.";
      if (r.weight_format === "none") return "No safetensors or PyTorch weight files were found, so there is nothing to convert.";
      return "";
    },
    convStartBlock: function () {
      var c = this.conv;
      if (!this.convOpts.available) return "Conversion is not available on this coordinator";
      if (!c.res || !this.convertible()) return "Inspect a convertible model first";
      if (!this.convNameOk()) return "The output name must end in .gguf and use only letters, digits, dot, dash and underscore";
      if (this.convImatrixApplies() && this.convCalibPathErr()) return this.convCalibPathErr();
      var o = this.convOption();
      if (o && this.convOptDisabled(o)) return o.type + " needs llama-imatrix, which is not installed on this coordinator";
      if (c.busy) return "Starting...";
      return "";
    },
    convBody: function () {
      var c = this.conv, a = c.adv, quantFlags = !this.convViaConvert();
      var th = parseInt(a.threads, 10), ch = parseInt(a.imatrix_chunks, 10), im = this.convImatrixApplies();
      return {
        source: this.convSpec(), quant: c.quant, name: c.name.trim(), keep_source: !!c.keepSource && c.tab === "hf",
        advanced: {
          intermediate: a.intermediate,
          output_tensor_type: quantFlags && a.output_tensor_type ? a.output_tensor_type : null,
          token_embedding_type: quantFlags && a.token_embedding_type ? a.token_embedding_type : null,
          leave_output_tensor: quantFlags && !!a.leave_output_tensor, pure: quantFlags && !!a.pure,
          allow_remote_code: !!a.allow_remote_code, validate_generation: !!a.validate_generation,
          imatrix: this.convImatrixMode(),
          calibration_path: im && a.calibration_path.trim() ? a.calibration_path.trim() : null,
          imatrix_chunks: im && !isNaN(ch) && ch > 0 ? ch : 0,
          threads: isNaN(th) || th < 0 ? 0 : th
        }
      };
    },
    convStart: async function () {
      var c = this.conv;
      if (this.convStartBlock()) return;
      c.busy = true;
      c.err = "";
      try {
        var j = await this.api("POST", "/api/convert", this.convBody());
        this.toast("Conversion started: " + ((j && j.output_name) || c.name), "ok");
        c.open = false;
        this.go("models");
        await this.convPoll();
      } catch (e) {
        if (e.status !== 401) c.err = e.message || String(e);
      }
      c.busy = false;
    },
    // ----- option display -----
    tierLabel: function (t) { return (CONV_TIER[t] || [String(t), ""])[0]; },
    tierClass: function (t) { return (CONV_TIER[t] || ["", ""])[1]; },
    isRecommended: function (o) { return !!this.conv.res && (o.recommended || o.type === this.conv.res.recommended); },
    // "fits one GPU" / "needs several GPUs" / "does not fit the cluster", or null when unknown.
    fitBadge: function (o) {
      if (o.fits_single_gpu === true) return { cls: "green", text: "Fits one GPU" };
      if (o.fits_pool === true) return { cls: "amber", text: "Needs several GPUs (slower, over network)" };
      if (o.fits_pool === false) return { cls: "red", text: "Does not fit the cluster" };
      return null;
    },
    humanParams: function (n) {
      if (n == null) return this.dash;
      if (n >= 1e9) return (n / 1e9).toFixed(1).replace(/\.0$/, "") + "B";
      if (n >= 1e6) return Math.round(n / 1e6) + "M";
      return String(n);
    },
    weightLabel: function (f) { return { safetensors: "safetensors", pytorch_bin: "PyTorch .bin", none: "none found" }[f] || this.dash; },
    hfUrl: function (repo) { return "https://huggingface.co/" + repo; },
    // ----- jobs -----
    convSource: function (j) {
      var s = (j.request && j.request.source) || {};
      return s.hf_repo ? s.hf_repo + (s.revision && s.revision !== "main" ? "@" + s.revision : "") : (s.path || "");
    },
    convStateClass: function (j) {
      return { queued: "", downloading: "amber", converting: "amber", calibrating: "amber", quantizing: "amber", validating: "amber", needs_review: "amber", done: "green", failed: "red", cancelled: "" }[j.state] || "";
    },
    convStateText: function (j) { return String(j.state).replace(/_/g, " "); },
    convIsActive: function (j) { return CONV_ACTIVE.indexOf(j.state) >= 0; },
    convIsRunning: function (j) { return CONV_STATE_KEY[j.state] !== undefined; },
    // Key of the stage a failed / cancelled job stopped in: the server's failed_stage, else the last stage seen while polling.
    convStoppedAt: function (j) {
      if (j.failed_stage) return CONV_STATE_KEY[j.failed_stage] || null; // "queued" -> before the first step
      return this.convStage[j.id] || null;
    },
    // Steps with a status class each: done / active / failed / stopped / skipped / todo.
    // Calibrate appears only when an importance matrix is computed; Quantize is skipped for the direct types.
    convStages: function (j) {
      var req = j.request || {}, hf = !!(req.source && req.source.hf_repo), direct = CONV_DIRECT.indexOf(req.quant) >= 0;
      var list = CONV_STAGES.filter(function (st) { return st[0] !== "calibrate" || j.imatrix_used; });
      var keys = list.map(function (st) { return st[0]; });
      var stopped = j.state === "failed" || j.state === "cancelled", over = j.state === "done" || j.state === "needs_review";
      var cur = -1;
      if (CONV_STATE_KEY[j.state] !== undefined) cur = keys.indexOf(CONV_STATE_KEY[j.state]);
      else if (stopped) cur = keys.indexOf(this.convStoppedAt(j));
      return list.map(function (st, i) {
        var cls = "todo";
        if (st[0] === "download" && !hf) cls = "skipped";
        else if (st[0] === "quantize" && direct) cls = "skipped";
        else if (over) cls = "done";
        else if (stopped && i === cur) cls = j.state === "failed" ? "failed" : "stopped";
        else if (cur >= 0 && i < cur) cls = "done";
        else if (!stopped && i === cur) cls = "active";
        return { key: st[0], label: st[1], cls: cls };
      });
    },
    // "1h 5m" / "3m 20s" / "45s".
    dur: function (sec) {
      var d = Math.max(0, Math.round(sec));
      if (d < 60) return d + "s";
      if (d < 3600) return Math.floor(d / 60) + "m " + (d % 60) + "s";
      return Math.floor(d / 3600) + "h " + Math.floor((d % 3600) / 60) + "m";
    },
    // Queued: waiting since; running: started (worker start) ago; ended: how long it ran and when it ended.
    convTimes: function (j) {
      if (j.state === "queued") return "queued " + this.ago(j.created_at);
      if (this.convIsActive(j)) return "started " + this.ago(j.started_at || j.created_at);
      var fin = j.finished_at;
      if (j.started_at != null && fin != null) return "ran " + this.dur(fin - j.started_at) + ", finished " + this.ago(fin);
      return fin != null ? "finished " + this.ago(fin) : "started " + this.ago(j.created_at);
    },
    // 0..100, or null when the stage has no measurable progress (shown as a moving bar).
    convPct: function (j) {
      if (j.state === "done" || j.state === "needs_review") return 100;
      if (j.state === "downloading") return j.bytes_total ? Math.min(100, j.bytes_done * 100 / j.bytes_total) : null;
      return j.stage_progress == null ? null : Math.max(0, Math.min(100, j.stage_progress * 100));
    },
    convProgressText: function (j) {
      var p = this.convPct(j);
      if (j.state === "downloading") return this.bytes(j.bytes_done) + (j.bytes_total ? " / " + this.bytes(j.bytes_total) : "") + (p == null ? "" : " (" + Math.round(p) + "%)");
      if (j.state === "calibrating") return "Computing the importance matrix on the CPU" + (p == null ? "..." : ": " + Math.round(p) + "%");
      return p == null ? "Working..." : Math.round(p) + "%";
    },
    convUiOf: function (j) { return this.convUi[j.id] || { log: false, val: null }; },
    convToggle: function (j, what) {
      var u = this.convUi[j.id] || { log: false, val: null };
      if (what === "log") u.log = !u.log;
      else u.val = !this.convValOpen(j);
      this.convUi[j.id] = u;
    },
    convValOpen: function (j) { var v = this.convUiOf(j).val; return v === null ? j.state === "needs_review" : v; },
    // Check chips of a validation result: ok / bad / skip.
    convChecks: function (v) {
      var tri = function (x, ok, bad, skip) { return x === true ? { s: "ok", t: ok } : (x === false ? { s: "bad", t: bad } : { s: "skip", t: skip }); };
      var c = tri(v.chat_template, "Chat template present", "No chat template", "Chat template unknown");
      if (v.chat_template === false) c.s = "skip"; // a missing template is informational, not a failure
      return [
        tri(v.header_ok, "GGUF header valid", "GGUF header invalid", "GGUF header not checked"),
        tri(v.tokenizer_ok, "Tokenizer matches the original", "Tokenizer differs from the original", "Tokenizer comparison could not run"),
        tri(v.generation_ok, "Generated text", "Generation failed", "Generation skipped"),
        c
      ].map(function (x, i) { return { key: i, s: x.s, t: x.t }; });
    },
    convCheckClass: function (s) { return s === "ok" ? "green" : (s === "bad" ? "red" : ""); },
    convMismatches: function (v) { return (v.tokenizer_cases || []).filter(function (x) { return !x.match; }).length; },
    idsText: function (ids) {
      var a = ids || [];
      return "[" + a.slice(0, 24).join(", ") + (a.length > 24 ? ", ... +" + (a.length - 24) : "") + "]";
    },
    quoteText: function (s) { return JSON.stringify(String(s == null ? "" : s)); },
    convAct: async function (j, action) {
      var mism = j.validation ? this.convMismatches(j.validation) : 0;
      if (action === "accept" && !confirm("Accept " + j.output_name + " anyway?\n\nThe automatic checks found a problem" +
        (mism ? " (the tokenizer produced different token ids than the original model on " + mism + " test string(s))" : "") +
        ". The model will be added to the library, but it may answer with garbled or wrong text. Only accept it if you plan to check it yourself.")) return;
      if (action === "delete" && !confirm("Delete this conversion job?\n\nIts temporary files are removed. A model already in the library is kept.")) return;
      if (action === "cancel" && !confirm("Cancel the conversion of " + j.output_name + "?")) return;
      this.convBusy[j.id] = true;
      try {
        if (action === "delete") await this.api("DELETE", "/api/convert/" + encodeURIComponent(j.id));
        else await this.api("POST", "/api/convert/" + encodeURIComponent(j.id) + "/" + action);
        if (action === "accept") this.toast(j.output_name + " added to the library", "ok");
        await this.convPoll();
        if (action === "accept") await this.refresh();
      } catch (e) { this.fail(e); }
      delete this.convBusy[j.id];
    },
    convDeploy: function (j) { this.openForm(null, j.output_name); },
    convInLibrary: function (j) { return this.readyLibrary().some(function (l) { return l.name === j.output_name; }); },

    // ================= deployments =================
    endpoint: function () {
      var base = this.settings().public_url || location.origin;
      return base.replace(/\/+$/, "") + "/v1";
    },
    stateClass: function (m) {
      return { running: "green", starting: "amber", stopping: "amber", failed: "red", idle: "blue" }[m.state] || "";
    },
    // ----- scaling (model card) -----
    scalingOf: function (m) { return m.scaling || null; },
    scaleRange: function (sc) { return sc.min === sc.max ? String(sc.min) : sc.min + "\u2013" + sc.max; },
    busyPct: function (v) { return v == null ? this.dash : Math.round(v * 100) + "%"; },
    scState: function (name) { return this.sc[name] || { open: false, busy: false, data: null, err: "" }; },
    toggleScaling: async function (m) {
      var n = m.spec.name, cur = this.sc[n] || { open: false, busy: false, data: null, err: "" };
      cur.open = !cur.open;
      this.sc[n] = cur;
      if (cur.open) await this.loadScaling(n);
    },
    loadScaling: async function (n) {
      var cur = this.sc[n];
      if (!cur) return;
      cur.busy = !cur.data;
      try {
        cur.data = await this.api("GET", "/api/models/" + encodeURIComponent(n) + "/scaling");
        cur.err = "";
      } catch (e) { if (e.status !== 401) cur.err = e.message; }
      cur.busy = false;
    },
    refreshScaling: function () {
      var self = this;
      Object.keys(this.sc).forEach(function (n) { if (self.sc[n].open) self.loadScaling(n); });
    },
    scStateLabel: function (s) { return s ? String(s).replace(/_/g, " ") : this.dash; },
    scStateClass: function (s) { return { scaling_up: "amber", scaling_down: "blue", unloaded: "blue", steady: "green" }[s] || ""; },
    decisionText: function (d) {
      if (!d) return "No scaling decision yet";
      return d.action + " \u2014 " + (d.reason || "") + " (" + this.ago(d.ts) + ")";
    },
    modelError: function (m) {
      if (m.error) return m.error;
      var errs = (m.replicas || []).map(function (r) { return r.error; }).filter(Boolean);
      return errs.join("\n");
    },
    placementLines: function (m) {
      var out = [], self = this;
      (m.replicas || []).forEach(function (r) {
        if (!r.placement) return;
        var parts = r.placement.assignments.map(function (a) {
          return a.node_id + "/" + a.device_id + ": " + a.layers + " layers";
        });
        out.push({ id: r.replica_id, state: r.state, tier: String(r.placement.tier).replace("_", " "), text: parts.join(", "),
          note: self.replicaNote(r), mark: self.replicaMark(m, r),
          devs: r.placement.assignments.map(function (a) { return a.node_id + "/" + a.device_id; }),
          tps: self.tps(r.placement.est_decode_tps), draft: r.placement.draft_est_mb ? "+draft " + Math.round(r.placement.draft_est_mb) + " MB" : "", reasons: (r.placement.reasons || []).join("; ") });
      });
      return out;
    },
    // ----- model card summary -----
    replicaClass: function (state) { return { ready: "green", starting: "amber", stopping: "amber", failed: "red" }[state] || ""; },
    readyCount: function (m) { return (m.replicas || []).filter(function (r) { return r.state === "ready"; }).length; },
    // "2 of 1–4": replicas serving now, out of the configured range.
    replicasText: function (m) {
      var sc = this.scalingOf(m);
      return this.readyCount(m) + " of " + (sc ? this.scaleRange(sc) : m.spec.replicas);
    },
    modeLabel: function (m) { return { fixed: "Fixed", autoscale: "Autoscale", demand: "On demand" }[scalingForm(m.spec).mode]; },
    // Summed estimated decode speed of the replicas that are serving.
    modelTps: function (m) {
      var sum = 0, any = false;
      (m.replicas || []).forEach(function (r) {
        if (r.state === "ready" && r.placement && r.placement.est_decode_tps != null) { sum += r.placement.est_decode_tps; any = true; }
      });
      return any ? this.tps(sum) : this.dash;
    },
    // ----- rebalancing -----
    rebalance: function () { return (this.st && this.st.rebalance) || { in_progress: null, next_run_ts: null }; },
    rbMove: function () { return this.rebalance().in_progress || null; },
    // "being replaced" / "replacement" for the two replicas of the move in progress, else "".
    replicaMark: function (m, r) {
      var mv = this.rbMove();
      if (!mv || mv.model !== m.spec.name) return "";
      return r.replica_id === mv.old ? "being replaced" : (r.replica_id === mv.new ? "replacement" : "");
    },
    rbDevices: function (list) {
      return (list || []).map(function (a) { return a.node_id + "/" + a.device_id; }).join(", ");
    },
    rbNextText: function () {
      var ts = this.rebalance().next_run_ts;
      return ts == null ? "Automatic rebalancing is off" : "Next automatic check at " + new Date(ts * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    },
    runRebalance: async function (dry) {
      if (this.rb.busy) return;
      if (!dry && !confirm("Rebalance now?\n\nThis starts a new replica on the better GPUs, then stops the old one once it is ready. Only one replica moves at a time.")) return;
      this.rb.busy = true;
      try {
        var r = await this.api("POST", "/api/rebalance", { dry_run: !!dry });
        this.rb.moves = (r && r.moves) || [];
        this.rb.checked = true;
        if (!dry) {
          if (r && r.started) { this.rb.checked = false; this.toast("Rebalance started: moving a replica of " + r.started.model, "ok"); }
          else if (r && r.in_progress) this.toast("Nothing started: a replica of " + r.in_progress.model + " is already being moved", "warning");
          else if (!this.rb.moves.length) this.toast("Nothing to do: all replicas are well placed", "ok");
          else this.toast("Nothing started: no better placement could be launched right now", "warning");
          await this.refresh();
        }
      } catch (e) { this.fail(e); }
      this.rb.busy = false;
    },
    replicasOf: function (m) { var v = this.rep[m.spec.name]; return v == null || v === "" ? 1 : v; },
    startModel: async function (m) {
      var n = parseInt(this.replicasOf(m), 10);
      if (!(n >= 1)) { this.toast("Replicas must be at least 1", "error"); return; }
      try {
        await this.api("POST", "/api/models/" + encodeURIComponent(m.spec.name) + "/start", { replicas: n });
        await this.refresh();
      } catch (e) { this.fail(e); }
    },
    stopModel: async function (m) {
      try {
        await this.api("POST", "/api/models/" + encodeURIComponent(m.spec.name) + "/stop");
        await this.refresh();
      } catch (e) { this.fail(e); }
    },
    deleteModel: async function (m) {
      if (!confirm("Stop and delete the deployment \"" + m.spec.name + "\"?")) return;
      try {
        await this.api("DELETE", "/api/models/" + encodeURIComponent(m.spec.name));
        await this.refresh();
      } catch (e) { this.fail(e); }
    },
    openForm: function (m, file) {
      if (m) {
        this.form = Object.assign({ open: true, edit: true, name: m.spec.name, file: m.file || "", ctx: m.spec.ctx_size, parallel: m.spec.parallel,
          priority: m.spec.priority == null ? 50 : m.spec.priority, preemptible: m.spec.preemptible !== false, spread: m.spec.spread || "gpu",
          auto: !(m.spec.pin_devices || []).length, pins: (m.spec.pin_devices || []).slice(), busy: false, plan: null, rec: null, recBusy: false, sim: null, simBusy: false }, scalingForm(m.spec), perfForm(m.spec));
      } else {
        var ready = this.readyLibrary();
        var f = file || (ready.length ? ready[0].name : "");
        this.form = Object.assign({ open: true, edit: false, name: f ? f.replace(/\.gguf$/i, "") : "", file: f, ctx: 4096, parallel: 1, priority: 50, preemptible: true, spread: "gpu", auto: true, pins: [], busy: false, plan: null, rec: null, recBusy: false, sim: null, simBusy: false }, scalingForm(null), perfForm(null));
      }
    },
    // ----- allowed servers / GPUs: "<node>/*" = the whole server (also GPUs added later), "<node>/<device>" = one GPU -----
    pinWhole: function (s) { return this.form.pins.indexOf(s.node_id + "/*") >= 0; },
    pinHas: function (s, d) { return this.pinWhole(s) || this.form.pins.indexOf(this.pinLabel(s, d)) >= 0; },
    // "all" (checked) / "some" (indeterminate) / "none" for a server row.
    pinNodeState: function (s) {
      if (this.pinWhole(s)) return "all";
      var self = this, g = this.gpus(s), n = g.filter(function (d) { return self.pinHas(s, d); }).length;
      return n === 0 ? "none" : (n === g.length ? "all" : "some");
    },
    pinReset: function () { this.form.plan = null; this.form.rec = null; this.form.sim = null; },
    // Checking a server stores "<node>/*" and drops that node's single-GPU pins; unchecking removes everything of the node.
    pinToggleServer: function (s, on) {
      var prefix = s.node_id + "/";
      this.form.pins = this.form.pins.filter(function (p) { return p.indexOf(prefix) !== 0; });
      if (on) this.form.pins.push(prefix + "*");
      this.pinReset();
    },
    // Unchecking one GPU of a whole-server pick turns it into the explicit remaining "<node>/<device>" pins.
    pinToggleGpu: function (s, d, on) {
      var key = this.pinLabel(s, d);
      if (this.pinWhole(s)) {
        if (on) return;
        var rest = this.gpus(s).filter(function (x) { return x.device_id !== d.device_id; }).map(function (x) { return s.node_id + "/" + x.device_id; });
        this.form.pins = this.form.pins.filter(function (p) { return p !== s.node_id + "/*"; }).concat(rest);
        this.pinReset();
        return;
      }
      this.togglePin(key, on);
    },
    // "3 of 7 GPUs on 2 servers"
    pinSummary: function () {
      var self = this, gpus = 0, total = 0, nodes = 0;
      this.servers().forEach(function (s) {
        var all = self.gpus(s), n = all.filter(function (d) { return self.pinHas(s, d); }).length;
        total += all.length;
        gpus += n;
        if (n > 0 || self.pinWhole(s)) nodes += 1;
      });
      return gpus + " of " + total + " GPUs on " + nodes + " server" + (nodes === 1 ? "" : "s");
    },
    togglePin: function (key, on) {
      var i = this.form.pins.indexOf(key);
      if (on && i < 0) this.form.pins.push(key);
      if (!on && i >= 0) this.form.pins.splice(i, 1);
      this.form.plan = null;
      this.form.rec = null;
      this.form.sim = null;
    },
    // Scaling fields of the PUT body for the selected mode, or { error } when the inputs are invalid.
    scalingBody: function (f) {
      var num = function (v) { var n = parseFloat(v); return isNaN(n) ? null : n; };
      var whole = function (v) { var n = num(v); return n != null && Math.floor(n) === n ? n : null; };
      if (f.mode === "autoscale") {
        var mn = whole(f.minR), mx = whole(f.maxR), tb = num(f.target), up = num(f.upAfter), dn = num(f.downAfter);
        if (mn == null || mn < 0) return { error: "Min replicas must be a whole number, 0 or more" };
        if (mx == null || mx < 1) return { error: "Max replicas must be at least 1" };
        if (mn > mx) return { error: "Min replicas cannot be greater than max replicas" };
        if (tb == null || tb <= 0 || tb > 100) return { error: "Target busy must be between 1 and 100 %" };
        if (up == null || up < 0 || dn == null || dn < 0) return { error: "Scale up/down delays must be 0 seconds or more" };
        return { body: { min_replicas: mn, max_replicas: mx, idle_unload_s: null,
          autoscale: { target_busy: tb / 100, up_after_s: up, down_after_s: dn } } };
      }
      if (f.mode === "demand") {
        var mx2 = whole(f.maxR), idle = num(f.idleMin);
        if (mx2 == null || mx2 < 1) return { error: "Max replicas must be at least 1" };
        if (idle == null || idle <= 0) return { error: "Unload after must be more than 0 minutes" };
        return { body: { min_replicas: 0, max_replicas: mx2, idle_unload_s: Math.round(idle * 60),
          autoscale: { target_busy: 0.7, up_after_s: 30, down_after_s: 300 } } };
      }
      var r = whole(f.replicas);
      if (r == null || r < 1) return { error: "Replicas must be at least 1" };
      return { body: { min_replicas: null, max_replicas: null, autoscale: null, idle_unload_s: null } };
    },
    // Performance fields (KV cache type, speculative decoding) of the request body, or { error } when invalid.
    perfBody: function (f) {
      var kv = f.kv || "f16", sp = f.spec || "none", fa = f.fa || "auto";
      var ub = parseInt(f.ubatch, 10) || 512, b = Math.max(parseInt(f.batch, 10) || 2048, ub);
      if (kv !== "f16" && fa === "off") return { error: "A quantized KV cache needs flash attention (Auto or On)" };
      var common = { kv_cache_type: kv, flash_attn: fa, ubatch: ub, batch: b, kv_unified: !!f.kvu };
      var n = parseFloat(f.draftN);
      if (sp === "mtp") {
        if (isNaN(n) || Math.floor(n) !== n || n < 1 || n > 16) return { error: "Draft tokens must be a whole number between 1 and 16" };
        return { body: Object.assign(common, { speculative: sp, draft_file: null, draft_n_max: n }) };
      }
      if (sp !== "draft") return { body: Object.assign(common, { speculative: sp, draft_file: null, draft_n_max: 4 }) };
      if (!f.draftFile) return { error: "Pick a draft model, or turn speculative decoding off" };
      if (f.draftFile === f.file) return { error: "The draft model must be a different file than the model" };
      if (isNaN(n) || Math.floor(n) !== n || n < 1 || n > 16) return { error: "Draft tokens must be a whole number between 1 and 16" };
      return { body: Object.assign(common, { speculative: sp, draft_file: f.draftFile, draft_n_max: n }) };
    },
    // Context each request gets: llama.cpp divides the context across the parallel slots.
    ctxPerSlot: function () {
      var c = parseInt(this.form.ctx, 10) || 0, p = parseInt(this.form.parallel, 10) || 1;
      return this.form.kvu ? c : Math.floor(c / Math.max(1, p));
    },
    // GPU generation from the compute capability NVML reports ("8.6" -> "Ampere · cc 8.6").
    archName: function (cc) {
      var m = /^(\d+)\.(\d+)$/.exec(cc || "");
      if (!m) return "";
      var v = parseInt(m[1], 10) * 10 + parseInt(m[2], 10);
      var t = [[100, "Blackwell"], [90, "Hopper"], [89, "Ada"], [80, "Ampere"], [75, "Turing"], [70, "Volta"], [60, "Pascal"], [50, "Maxwell"]];
      for (var i = 0; i < t.length; i++) if (v >= t[i][0]) return t[i][1];
      return "";
    },
    archLabel: function (d) {
      if (!d || !d.compute_cap) return "";
      var n = this.archName(d.compute_cap);
      return (n ? n + " · " : "") + "cc " + d.compute_cap;
    },
    tensorCores: function (d) {
      var m = /^(\d+)\./.exec((d && d.compute_cap) || "");
      return m ? parseInt(m[1], 10) >= 7 : null;
    },
    archHelp: function (d) {
      var tc = this.tensorCores(d);
      if (tc === null) return "";
      return tc ? "Tensor cores: fast flash attention and big micro-batches pay off"
        : "No tensor cores: flash attention uses fallback kernels, big micro-batches gain little";
    },
    // GPUs without tensor cores the form may use (all GPUs, or the selected ones).
    formOldGpus: function () {
      var f = this.form, out = [], self = this;
      this.servers().forEach(function (s) {
        self.gpus(s).forEach(function (d) {
          var allowed = f.auto || f.pins.indexOf(s.node_id + "/*") >= 0 || f.pins.indexOf(s.node_id + "/" + d.device_id) >= 0;
          if (allowed && self.tensorCores(d) === false) out.push(s.node_id + "/" + d.device_id + " (" + d.name + ")");
        });
      });
      return out;
    },
    // Suggested-settings cards of the recommendation.
    tipKind: function (t) { return { speed: "Faster", throughput: "More users", fix: "Fits better", memory: "Memory" }[t.kind] || t.kind; },
    tipClass: function (t) { return { speed: "green", throughput: "blue", fix: "amber" }[t.kind] || ""; },
    applyTip: async function (t) {
      var f = this.form, a = t.apply || {};
      var map = { kv_cache_type: "kv", speculative: "spec", draft_file: "draftFile", draft_n_max: "draftN", flash_attn: "fa",
                  ubatch: "ubatch", batch: "batch", ctx_size: "ctx", parallel: "parallel", file: "file", kv_unified: "kvu" };
      Object.keys(a).forEach(function (k) { if (map[k]) f[map[k]] = a[k]; });
      if (f.draftFile === f.file) f.draftFile = "";
      f.plan = null;
      f.sim = null;
      this.toast("Applied: " + t.title, "ok");
      await this.recommend();
    },
    // Ready library files that can serve as the draft model (not the model's own file).
    draftChoices: function () {
      var file = this.form.file;
      return this.readyLibrary().filter(function (l) { return l.name !== file; });
    },
    // Model card chips for non-default performance options.
    specChips: function (m) {
      var sp = m.spec || {}, out = [];
      if (sp.kv_cache_type && sp.kv_cache_type !== "f16") out.push("KV " + sp.kv_cache_type);
      if (sp.flash_attn && sp.flash_attn !== "auto") out.push("Flash attn " + sp.flash_attn);
      if (sp.ubatch && sp.ubatch !== 512) out.push("Micro-batch " + sp.ubatch);
      if (sp.parallel > 1) out.push(sp.parallel + " slots · " + (sp.kv_unified ? sp.ctx_size + " ctx shared" : Math.floor(sp.ctx_size / sp.parallel) + " ctx each"));
      if ((sp.pin_devices || []).length) out.push("Limited to " + sp.pin_devices.map(function (p) { return p.replace(/\/\*$/, ""); }).join(", "));
      if (sp.speculative === "ngram") out.push("Spec: n-gram");
      else if (sp.speculative === "mtp") out.push("Spec: MTP");
      else if (sp.speculative === "draft") out.push("Spec: draft " + (sp.draft ? String(sp.draft).replace(/^coordinator:\/\//, "") : "?"));
      return out;
    },
    startReplicas: function (f) {
      var n = parseInt(f.mode === "autoscale" ? f.minR : f.replicas, 10);
      return f.mode === "demand" || !(n >= 1) ? 1 : n;
    },
    saveForm: async function () {
      var f = this.form;
      if (!f.name.trim() || !f.file) { this.toast("Name and file are required", "error"); return false; }
      var sb = this.scalingBody(f);
      if (sb.error) { this.toast(sb.error, "error"); return false; }
      var pb = this.perfBody(f);
      if (pb.error) { this.toast(pb.error, "error"); return false; }
      if (!f.auto && !f.pins.length) { this.toast("Select at least one server or GPU, or use all servers and GPUs", "error"); return false; }
      f.busy = true;
      try {
        await this.api("PUT", "/api/models/" + encodeURIComponent(f.name.trim()), Object.assign({
          file: f.file, ctx_size: parseInt(f.ctx, 10) || 4096, parallel: parseInt(f.parallel, 10) || 1,
          priority: this.priorityOf(f), preemptible: !!f.preemptible, spread: f.spread || "gpu",
          pin_devices: f.auto ? [] : f.pins
        }, sb.body, pb.body));
        await this.refresh();
        f.busy = false;
        return true;
      } catch (e) { this.fail(e); }
      f.busy = false;
      return false;
    },
    saveOnly: async function () {
      if (await this.saveForm()) { this.toast("Saved " + this.form.name, "ok"); this.form.open = false; }
    },
    saveAndStart: async function () {
      var name = this.form.name.trim(), n = this.startReplicas(this.form);
      if (!(await this.saveForm())) return;
      this.form.open = false;
      try {
        await this.api("POST", "/api/models/" + encodeURIComponent(name) + "/start", { replicas: n });
        this.toast("Starting " + name, "ok");
        await this.refresh();
      } catch (e) { this.fail(e); }
    },
    checkPlan: async function () {
      var f = this.form;
      f.plan = null;
      if (!(await this.saveForm())) return;
      f.busy = true;
      try {
        var p = await this.api("POST", "/api/models/" + encodeURIComponent(f.name.trim()) + "/plan");
        f.plan = { ok: true, data: p };
      } catch (e) {
        if (e.status === 409) f.plan = { ok: false, msg: e.message };
        else this.fail(e);
      }
      f.busy = false;
    },
    pinLabel: function (s, d) { return s.node_id + "/" + d.device_id; },
    priorityOf: function (f) {
      var p = parseInt(f.priority, 10);
      return isNaN(p) ? 50 : Math.max(0, Math.min(100, p));
    },

    // ================= impact preview =================
    // Dry run of the whole cluster with this form applied: one change (existing model) or one add (new model).
    simulate: async function () {
      var f = this.form;
      if (!f.name.trim() || !f.file) { this.toast("Name and file are required", "error"); return; }
      var sb = this.scalingBody(f);
      if (sb.error) { this.toast(sb.error, "error"); return; }
      var pb = this.perfBody(f);
      if (pb.error) { this.toast(pb.error, "error"); return; }
      var item = Object.assign({}, pb.body, {
        ctx_size: parseInt(f.ctx, 10) || 4096, parallel: parseInt(f.parallel, 10) || 1,
        priority: this.priorityOf(f), preemptible: !!f.preemptible, spread: f.spread || "gpu", pin_devices: f.auto ? [] : f.pins
      });
      if (f.mode === "fixed") item.replicas = parseInt(f.replicas, 10);
      else { item.min_replicas = sb.body.min_replicas; item.max_replicas = sb.body.max_replicas; }
      var body = {};
      if (f.edit) body.changes = [Object.assign({ model: f.name.trim() }, item)];
      else body.add = [Object.assign({ name: f.name.trim(), file: f.file }, item)];
      f.simBusy = true;
      f.sim = null;
      try {
        f.sim = await this.api("POST", "/api/simulate", body);
      } catch (e) { this.fail(e); }
      f.simBusy = false;
    },
    simEmpty: function (r) {
      return !r || !["start", "stop", "preempt", "unplaced"].some(function (k) { return (r[k] || []).length; });
    },
    simDevices: function (s) {
      return (s.assignments || []).map(function (a) { return a.node_id + "/" + a.device_id; }).join(", ");
    },

    // ================= recommendation =================
    recommend: async function () {
      var f = this.form;
      if (!f.file) { this.toast("Pick a file first", "error"); return; }
      var pb = this.perfBody(f);
      if (pb.error) { this.toast(pb.error, "error"); return; }
      f.recBusy = true;
      f.rec = null;
      try {
        f.rec = await this.api("POST", "/api/recommend", Object.assign({
          file: f.file, ctx_size: parseInt(f.ctx, 10) || 4096, parallel: parseInt(f.parallel, 10) || 1,
          priority: this.priorityOf(f), spread: f.spread || "gpu", pin_devices: f.auto ? [] : f.pins, limit: 3
        }, pb.body));
      } catch (e) { this.fail(e); }
      f.recBusy = false;
    },
    optionPins: function (o) {
      var out = [];
      (o.assignments || []).forEach(function (a) {
        var k = a.node_id + "/" + a.device_id;
        if (out.indexOf(k) < 0) out.push(k);
      });
      return out;
    },
    pinOption: function (o) {
      this.form.auto = false;
      this.form.pins = this.optionPins(o);
      this.form.plan = null;
      this.toast("Pinned to " + this.form.pins.join(", "), "ok");
    },
    useCtx: async function (n) {
      this.form.ctx = n;
      this.form.plan = null;
      await this.recommend();
    },
    tpsNum: function (v) { return v == null ? this.dash : (v >= 100 ? Math.round(v) : v.toFixed(1)); },
    tps: function (v) { return v == null ? this.dash : (v >= 100 ? Math.round(v) : v.toFixed(1)) + " tok/s"; },
    bw: function (d) { return d && d.bandwidth_gbps != null ? Math.round(d.bandwidth_gbps) + " GB/s" : this.dash; },
    replicaNote: function (r) {
      var p = r && r.placement;
      if (!p) return "";
      var bits = [];
      if (p.est_decode_tps != null) bits.push("~" + this.tps(p.est_decode_tps));
      if (p.reasons && p.reasons.length) bits.push(p.reasons.join("; "));
      return bits.join(" · ");
    },

    // ================= snippets =================
    curlSnippet: function (modelName) {
      var name = modelName || (this.models().length ? this.models()[0].spec.name : "<model>");
      var auth = this.settings().api_keys_set ? "  -H \"Authorization: Bearer <api key>\" \\\n" : "";
      return "curl " + this.endpoint() + "/chat/completions \\\n" + auth + "  -H \"Content-Type: application/json\" \\\n" +
        "  -d '{\"model\": \"" + name + "\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello\"}]}'";
    },
    pySnippet: function () {
      var name = this.models().length ? this.models()[0].spec.name : "<model>";
      return "from openai import OpenAI\n\nclient = OpenAI(base_url=\"" + this.endpoint() + "\", api_key=\"" + (this.settings().api_keys_set ? "<api key>" : "no key required") + "\")\n" +
        "resp = client.chat.completions.create(\n    model=\"" + name + "\",\n    messages=[{\"role\": \"user\", \"content\": \"Hello\"}],\n)\nprint(resp.choices[0].message.content)";
    },

    go: function (v) { this.view = v; if (v === "models") { this.convPollAt = 0; this.convLoadOptions(); } try { history.replaceState(null, "", "#" + v); } catch (e) { /* ignore */ } if (v === "events") this.loadEvents(); this.navOpen = false; this.search = ""; },
    title: function () {
      return { overview: ["Overview", "Monitor your GPU pool at a glance"], servers: ["Servers", "Manage servers and the GPUs in the pool"],
        gpus: ["GPUs", "Every GPU across all servers"], events: ["Events", "Failures, re-allocations and other cluster activity"], models: ["Models", "Library and deployments"], settings: ["Settings", "Connection details and snippets"] }[this.view];
    }
  };
}
