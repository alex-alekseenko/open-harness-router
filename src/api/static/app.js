// The open-harness-router dashboard.
//
// The page is a thin shell (dashboard.html); everything below renders the
// state the router exposes at GET /dashboard/state. The state is polled
// every two seconds, so every figure on the page is live.
//
// The selected window (minute / hour / day presets, all or custom) governs every
// statistic except the live ones ("in flight", the event feed), because
// those describe now and not the window. The window lives in the URL
// query; localStorage is only a fallback, and the URL always wins.

import { h, render } from "preact";
import { useEffect, useMemo, useRef, useState } from "preact/hooks";
import htm from "htm";

const html = htm.bind(h);

const POLL_MS = 2000;
const MAX_BACKOFF_MS = 15000;
const VIEW_KEY = "ohr.window";
const CLOSED_KEY = "ohr.providers.closed";
const HIDDEN_KEY_PREFIX = "ohr.chart.hidden.";
const DEFAULT_VIEW = { window: "30d", start: "", end: "" };
const FALLBACK_WINDOWS = ["5m", "30m", "60m", "6h", "12h", "24h", "7d", "30d", "all", "custom"];
const SERIES_COUNT = 8;
// The second encoding once the colours run out: solid, dashed, dotted,
// dash-dotted (canvas dash arrays, in px).
const LINE_DASHES = [[], [8, 4], [2, 3], [8, 3, 2, 3]];
const STYLE_COUNT = SERIES_COUNT * LINE_DASHES.length;
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

// ---- formatting -------------------------------------------------------

const fmtInt = (n) => Number(n || 0).toLocaleString("en-US");

const trimZero = (s) => s.replace(/\.0+$/, "").replace(/(\.\d*?)0+$/, "$1");

// A compact number for the tiles: 1.2k, 59.6M, 1.07B.
const fmtCompact = (n) => {
  n = Number(n || 0);
  const abs = Math.abs(n);
  if (abs < 1000) return String(Math.round(n));
  if (abs < 1e6) return trimZero((n / 1e3).toFixed(1)) + "k";
  if (abs < 1e9) return trimZero((n / 1e6).toFixed(1)) + "M";
  return trimZero((n / 1e9).toFixed(2)) + "B";
};

const fmtUsd = (n) => {
  n = Number(n || 0);
  return "$" + (n < 0.01 ? n.toFixed(4) : n.toFixed(2));
};

const fmtMs = (ms) => {
  ms = Number(ms || 0);
  return ms >= 1000 ? (ms / 1000).toFixed(1) + " s" : Math.round(ms) + " ms";
};

const fmtDur = (s) => {
  s = Math.max(0, Math.floor(Number(s || 0)));
  const d = Math.floor(s / 86400);
  const h = Math.floor((s % 86400) / 3600);
  const m = Math.floor((s % 3600) / 60);
  if (d) return d + "d " + h + "h";
  if (h) return h + "h " + m + "m";
  return m + "m " + (s % 60) + "s";
};

const fmtTime = (iso) => {
  if (!iso) return "—";
  const t = new Date(iso);
  return isNaN(t) ? String(iso) : t.toLocaleTimeString("en-GB");
};

// A local date and time, short: "31 Oct 16:21".
const fmtLocalDateTime = (iso) => {
  if (!iso) return "—";
  const t = new Date(iso);
  if (isNaN(t)) return String(iso);
  return (
    t.getDate() + " " + MONTHS[t.getMonth()] + " " +
    String(t.getHours()).padStart(2, "0") + ":" + String(t.getMinutes()).padStart(2, "0")
  );
};

const fmtDate = (iso) => {
  if (!iso) return "—";
  const t = new Date(iso);
  return isNaN(t) ? String(iso) : t.toISOString().slice(0, 10);
};

const fmtTps = (v) => Number(v || 0).toFixed(1);

// USD per million tokens, without the trailing zeros.
const fmtRate = (n) => Number(n || 0).toFixed(4).replace(/\.?0+$/, "");

const cacheRate = (p) =>
  p.cache_write === p.cache_read
    ? fmtRate(p.cache_read)
    : fmtRate(p.cache_read) + " / " + fmtRate(p.cache_write);

const levelClass = (level) =>
  level === "error" || level === "critical" ? "err" : level === "warning" ? "warn" : "";

const sumBy = (rows, pick) =>
  rows.reduce((acc, row) => acc + (Number(pick(row)) || 0), 0);

// ---- the selected window ----------------------------------------------

// datetime-local fields show the reader's wall clock; URLs and API calls
// keep absolute timestamps so a shared link selects the same interval.
function rangeDraft(view) {
  const localInput = (value, endOfDay) => {
    if (!value) return "";
    const date = new Date(value);
    if (isNaN(date)) return "";
    // Preserve the inclusive end day of older date-only links.
    if (value.length === 10 && endOfDay) date.setUTCDate(date.getUTCDate() + 1);
    const pad = (n) => String(n).padStart(2, "0");
    return date.getFullYear() + "-" + pad(date.getMonth() + 1) + "-" + pad(date.getDate()) +
      "T" + pad(date.getHours()) + ":" + pad(date.getMinutes());
  };
  return { start: localInput(view.start, false), end: localInput(view.end, true) };
}

function normaliseView(view) {
  const next = {
    window: view && view.window ? String(view.window) : DEFAULT_VIEW.window,
    start: view && view.start ? String(view.start) : "",
    end: view && view.end ? String(view.end) : "",
  };
  if (next.window === "custom" && !next.start && !next.end) {
    // A custom range with no bounds reads nothing: seed it with the last
    // seven days so the first click shows something.
    const today = new Date();
    today.setSeconds(0, 0);
    const week = new Date(today.getTime() - 7 * 86400000);
    next.start = week.toISOString();
    next.end = today.toISOString();
  }
  return next;
}

// The URL wins over the stored choice: a link to a window is a request for
// that window, whatever the browser remembers.
function readInitialView() {
  try {
    const params = new URLSearchParams(window.location.search);
    const name = params.get("window");
    if (name) {
      return normaliseView({
        window: name,
        start: params.get("start") || "",
        end: params.get("end") || "",
      });
    }
  } catch (e) {
    /* no usable URL: fall through to the stored choice */
  }
  try {
    const saved = window.localStorage.getItem(VIEW_KEY);
    if (saved) {
      const parsed = JSON.parse(saved);
      if (parsed && parsed.window) return normaliseView(parsed);
    }
  } catch (e) {
    /* no storage: fall back to the default window */
  }
  return normaliseView(DEFAULT_VIEW);
}

function viewQuery(view) {
  const params = new URLSearchParams();
  params.set("window", view.window);
  if (view.window === "custom") {
    if (view.start) params.set("start", view.start);
    if (view.end) params.set("end", view.end);
  }
  return "?" + params.toString();
}

// "Today" is a page-side window: the server has no notion of the reader's
// timezone, so it is asked for an open custom range from the local midnight,
// recomputed on every poll so the window starts over at 00:00. Unlike the
// rolling 24h it only grows during the day.
function stateQuery(view) {
  if (view.window !== "today") return viewQuery(view);
  const midnight = new Date();
  midnight.setHours(0, 0, 0, 0);
  return viewQuery({ window: "custom", start: midnight.toISOString(), end: "" });
}

// The series a reader switched off in a chart legend, by series label, so
// the choice survives the next poll (which hands the chart new datasets)
// and a reload. One list per chart, keyed by its title.
function readHiddenSeries(chartName) {
  try {
    const saved = window.localStorage.getItem(HIDDEN_KEY_PREFIX + chartName);
    if (saved) {
      const parsed = JSON.parse(saved);
      if (Array.isArray(parsed)) return new Set(parsed.map(String));
    }
  } catch (e) {
    /* no storage: every series starts visible */
  }
  return new Set();
}

function writeHiddenSeries(chartName, hidden) {
  try {
    window.localStorage.setItem(HIDDEN_KEY_PREFIX + chartName, JSON.stringify([...hidden]));
  } catch (e) {
    /* no storage: the choice lasts until the page is reloaded */
  }
}

function readClosedProviders() {
  try {
    const saved = window.localStorage.getItem(CLOSED_KEY);
    if (saved) {
      const parsed = JSON.parse(saved);
      if (Array.isArray(parsed)) return parsed.map(String);
    }
  } catch (e) {
    /* no storage: every provider row stays open */
  }
  return [];
}

// ---- throughput series ------------------------------------------------

// Every bucket of the window, gaps included: a quiet hour must stay
// visible, otherwise the line would stretch over it and lie.
const windowBuckets = (startIso, endIso, bucket) => {
  const start = new Date(startIso);
  if (isNaN(start)) return [];
  let end = endIso ? new Date(endIso) : new Date();
  if (isNaN(end)) end = new Date();
  const keys = [];
  const hour = bucket === "hour";
  const d = new Date(
    Date.UTC(
      start.getUTCFullYear(),
      start.getUTCMonth(),
      start.getUTCDate(),
      hour ? start.getUTCHours() : 0
    )
  );
  const limit = end.getTime();
  while (endIso ? d.getTime() < limit : d.getTime() <= limit) {
    keys.push(d.toISOString().slice(0, hour ? 13 : 10));
    d.setTime(d.getTime() + (hour ? 3600000 : 86400000));
    if (keys.length > 2000) break;
  }
  return keys;
};

// The label of a bucket on the axis: the hours are UTC buckets drawn in the
// reader's local time, the days stay the date they are.
// An hourly bucket reads as its time, except local midnight, which names
// the day instead -- a 24h window spans 25 buckets across two dates, and a
// date on every tick crowded the axis.
const bucketLabel = (key, bucket) => {
  if (bucket === "hour") {
    const d = new Date(key + ":00:00Z");
    if (isNaN(d)) return key;
    if (d.getHours() === 0 && d.getMinutes() === 0) return d.getDate() + " " + MONTHS[d.getMonth()];
    return String(d.getHours()).padStart(2, "0") + ":" + String(d.getMinutes()).padStart(2, "0");
  }
  const d = new Date(key + "T00:00:00");
  if (isNaN(d)) return key;
  return d.getDate() + " " + MONTHS[d.getMonth()];
};

// The series of the throughput chart: one per model with data in the
// window, in the order the store lists the models (first appearance over
// the whole log). Every model is its own line -- none is ever folded away.
//
// A line's style follows its model: the model's place in that list picks
// it once and for good, so no line repaints when the range changes. The
// first eight models are solid lines in the eight palette colours; the
// next eight reuse the colours dashed, then dotted, then dash-dotted -- a
// second encoding rather than generated hues, which would blur into each
// other. Past 32 models the styles wrap round; the legend and the tooltip
// still name every line.
const tpsBuildSeries = (history) => {
  const tps = (history && history.tps) || {};
  const models = tps.models || [];
  const rowsByModel = {};
  (tps.series || []).forEach((row) => {
    const points = rowsByModel[row.model] || (rowsByModel[row.model] = {});
    points[row.bucket] = row;
  });
  const present = models.filter(
    (m) => rowsByModel[m] && Object.keys(rowsByModel[m]).length
  );
  // A model with data but missing from the list (a store older than the
  // list) still gets a line, after the listed ones.
  Object.keys(rowsByModel).forEach((m) => {
    if (!present.includes(m)) present.push(m);
  });
  return present.map((model) => {
    const listed = models.indexOf(model);
    const style = (listed >= 0 ? listed : models.length + present.indexOf(model)) % STYLE_COUNT;
    return {
      model,
      slot: style % SERIES_COUNT,
      dash: LINE_DASHES[Math.floor(style / SERIES_COUNT)],
      points: rowsByModel[model],
    };
  });
};

// ---- colours ----------------------------------------------------------

// The colours live in the stylesheet; the charts read them from the custom
// properties so the plots follow the page (and the colour scheme).
const FALLBACK_COLOURS = {
  text: "#1f2328",
  muted: "#6a737d",
  grid: "#e1e4e8",
  border: "#e1e4e8",
  panel: "#ffffff",
  crosshair: "#8b949e",
  series1: "#2a78d6",
  series2: "#eb6834",
  series3: "#1baf7a",
  series4: "#eda100",
  series5: "#e87ba4",
  series6: "#008300",
  series7: "#6250d6",
  series8: "#e34948",
};

function readColours() {
  const out = Object.assign({}, FALLBACK_COLOURS);
  try {
    const style = window.getComputedStyle(document.documentElement);
    const read = (name) => {
      const value = style.getPropertyValue(name);
      return value && value.trim() ? value.trim() : null;
    };
    const pairs = {
      text: "--fg",
      muted: "--muted",
      grid: "--line",
      border: "--line",
      panel: "--panel",
      crosshair: "--muted",
    };
    Object.keys(pairs).forEach((key) => {
      const value = read(pairs[key]);
      if (value) out[key] = value;
    });
    for (let i = 1; i <= SERIES_COUNT; i++) {
      const value = read("--series-" + i);
      if (value) out["series" + i] = value;
    }
  } catch (e) {
    /* no computed style (a bare test environment): keep the fallbacks */
  }
  return out;
}

// The colour scheme can change under the page (the system switched to
// dark); the charts are rebuilt then.
function useThemeColours() {
  const [version, setVersion] = useState(0);
  useEffect(() => {
    let query = null;
    try {
      query = window.matchMedia("(prefers-color-scheme: dark)");
    } catch (e) {
      return undefined;
    }
    if (!query || !query.addEventListener) return undefined;
    const onChange = () => setVersion((v) => v + 1);
    query.addEventListener("change", onChange);
    return () => query.removeEventListener("change", onChange);
  }, []);
  const colours = useMemo(() => readColours(), [version]);
  return { colours, version };
}

// A translucent version of a series colour, for the fill under a line.
function withAlpha(colour, alpha) {
  const hex = String(colour || "").trim();
  if (/^#[0-9a-f]{6}$/i.test(hex)) {
    const n = parseInt(hex.slice(1), 16);
    return (
      "rgba(" + ((n >> 16) & 255) + ", " + ((n >> 8) & 255) + ", " + (n & 255) + ", " + alpha + ")"
    );
  }
  if (/^#[0-9a-f]{3}$/i.test(hex)) {
    const r = parseInt(hex[1] + hex[1], 16);
    const g = parseInt(hex[2] + hex[2], 16);
    const b = parseInt(hex[3] + hex[3], 16);
    return "rgba(" + r + ", " + g + ", " + b + ", " + alpha + ")";
  }
  return hex;
}

// ---- charts -----------------------------------------------------------

// A one pixel vertical line at the hovered bucket. Chart.js owns the
// tooltip (and keeps it inside the canvas); the crosshair is the only
// thing drawn by hand.
const crosshairPlugin = {
  id: "crosshair",
  afterDatasetsDraw(chart, args, options) {
    const active = chart.tooltip && chart.tooltip._active;
    if (!active || !active.length) return;
    const element = active[0].element;
    const area = chart.chartArea;
    if (!element || !area) return;
    const ctx = chart.ctx;
    ctx.save();
    ctx.beginPath();
    ctx.lineWidth = 1;
    ctx.strokeStyle = (options && options.color) || "rgba(128,128,128,0.6)";
    ctx.moveTo(element.x, area.top);
    ctx.lineTo(element.x, area.bottom);
    ctx.stroke();
    ctx.restore();
  },
};

// A point with no neighbour on either side would be invisible under the
// line, so it gets a marker; every other point stays out of the way.
const isolatedPointRadius = (ctx) => {
  const data = ctx.dataset.data || [];
  const before = data[ctx.dataIndex - 1];
  const after = data[ctx.dataIndex + 1];
  return before == null && after == null ? 4 : 0;
};

function lineDataset(label, data, colour, colours, extra) {
  return Object.assign(
    {
      label,
      data,
      borderColor: colour,
      backgroundColor: withAlpha(colour, 0.12),
      borderWidth: 2,
      tension: 0.25,
      fill: false,
      spanGaps: false,
      pointRadius: isolatedPointRadius,
      pointHoverRadius: 4,
      pointBackgroundColor: colour,
      pointBorderColor: colours.panel,
      pointBorderWidth: 2,
    },
    extra || {}
  );
}

function chartConfig(labels, datasets, colours) {
  return {
    type: "line",
    data: { labels, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      interaction: { mode: "index", intersect: false },
      spanGaps: false,
      plugins: {
        legend: {
          display: true,
          position: "bottom",
          labels: {
            // A short stroke of the series' own line beside every label --
            // its colour and its dash -- so the legend tells the lines apart
            // without a hover, the dashed ones included. The default swatch
            // reads the series' point style, and the points are radius 0.
            usePointStyle: true,
            pointStyleWidth: 26,
            boxWidth: 10,
            boxHeight: 10,
            color: colours.text,
            font: { size: 12 },
            generateLabels: (chart) =>
              chart.data.datasets.map((dataset, index) => {
                const hidden = !chart.isDatasetVisible(index);
                const swatch = hidden ? withAlpha(dataset.borderColor, 0.3) : dataset.borderColor;
                return {
                  text: dataset.label,
                  fillStyle: swatch,
                  strokeStyle: swatch,
                  lineWidth: 4,
                  lineDash: dataset.borderDash || [],
                  pointStyle: "line",
                  fontColor: hidden ? colours.muted : colours.text,
                  hidden,
                  datasetIndex: index,
                };
              }),
          },
        },
        tooltip: {
          // Every series at the bucket, the busiest first.
          sort: (items) =>
            items.slice().sort((a, b) => (Number(b.raw) || 0) - (Number(a.raw) || 0)),
          usePointStyle: true,
          backgroundColor: colours.panel,
          borderColor: colours.border,
          borderWidth: 1,
          titleColor: colours.muted,
          bodyColor: colours.text,
          boxPadding: 6,
          caretSize: 5,
        },
        crosshair: { color: colours.crosshair },
      },
      scales: {
        x: {
          grid: { display: false },
          border: { color: colours.border },
          ticks: {
            color: colours.muted,
            maxRotation: 0,
            autoSkip: true,
            maxTicksLimit: 12,
            font: { size: 11 },
          },
        },
        y: {
          beginAtZero: true,
          grid: { color: colours.grid, drawBorder: false },
          border: { display: false },
          ticks: { color: colours.muted, font: { size: 11 } },
        },
      },
    },
    plugins: [crosshairPlugin],
  };
}

// One chart card. The chart is created once and then updated in place, and
// only when its input changed: the page polls every two seconds, and a
// rebuild would throw away the crosshair the reader is holding over the
// plot.
function ChartCard(props) {
  const canvasRef = useRef(null);
  const chartRef = useRef(null);
  const lastRef = useRef(null);
  // Legend toggles live here, by series label, not in Chart.js's per-index
  // dataset meta: every poll hands the chart new datasets, which reset that
  // meta and brought a hidden series back on the next tick.
  const hiddenRef = useRef(null);
  if (hiddenRef.current === null) hiddenRef.current = readHiddenSeries(props.title);

  const applyHidden = (data) => {
    (data.datasets || []).forEach((dataset) => {
      dataset.hidden = hiddenRef.current.has(dataset.label);
    });
    return data;
  };

  const onLegendClick = (event, item, legend) => {
    const chart = legend.chart;
    const dataset = chart.data.datasets[item.datasetIndex];
    if (!dataset) return;
    const hidden = hiddenRef.current;
    if (hidden.has(dataset.label)) hidden.delete(dataset.label);
    else hidden.add(dataset.label);
    writeHiddenSeries(props.title, hidden);
    dataset.hidden = hidden.has(dataset.label);
    chart.update("none");
  };

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !window.Chart) return;
    const Chart = window.Chart;
    if (!props.config) {
      if (chartRef.current) {
        chartRef.current.destroy();
        chartRef.current = null;
        lastRef.current = null;
      }
      return;
    }
    const last = lastRef.current;
    if (chartRef.current && last && last.key === props.configKey && last.version === props.version) {
      return;
    }
    if (
      chartRef.current &&
      last &&
      last.key !== props.configKey &&
      last.version === props.version
    ) {
      chartRef.current.data = applyHidden(props.config.data);
      chartRef.current.update("none");
    } else {
      if (chartRef.current) chartRef.current.destroy();
      applyHidden(props.config.data);
      props.config.options.plugins.legend.onClick = onLegendClick;
      chartRef.current = new Chart(canvas, props.config);
    }
    lastRef.current = { key: props.configKey, version: props.version };
  });

  useEffect(
    () => () => {
      if (chartRef.current) {
        chartRef.current.destroy();
        chartRef.current = null;
      }
    },
    []
  );

  return html`
    <section class="card chart-card">
      <div class="card-head">
        <h2>${props.title}</h2>
        ${props.subtitle ? html`<span class="muted">${props.subtitle}</span>` : null}
      </div>
      ${props.config
        ? html`<div class="chart-box"><canvas ref=${canvasRef}></canvas></div>`
        : html`<div class="empty">${props.emptyText}</div>`}
    </section>
  `;
}

// ---- small pieces -----------------------------------------------------

function Tile(props) {
  return html`
    <div class="tile">
      <div class="label">${props.label}</div>
      <div class=${"value" + (props.tone ? " " + props.tone : "")} title=${props.title || ""}>
        ${props.value}
      </div>
      <div class="sub" title=${props.subTitle || ""}>${props.sub}</div>
      ${props.href ? html`<a class="tile-link" href=${props.href}>View error details →</a>` : null}
    </div>
  `;
}

// Storage keeps disjoint billing buckets; display inclusive prompt totals.
function promptTokens(usage) {
  const u = usage || {};
  return (Number(u.input_tokens) || 0) +
    (Number(u.cache_read_input_tokens) || 0) +
    (Number(u.cache_creation_input_tokens) || 0);
}

function tokensCell(usage) {
  const u = usage || {};
  const cache = (u.cache_read_input_tokens || 0) + (u.cache_creation_input_tokens || 0);
  return html`
    <span>
      ${fmtInt(promptTokens(u))} / ${fmtInt(u.output_tokens)}
      ${cache ? html`<span class="muted"> (${fmtInt(cache)} cached)</span>` : null}
    </span>
  `;
}

// The Providers table's token columns -- in, out, cache -- each a compact
// number with the exact figure (and the cache's read/write split) in its
// title, so the columns stay narrow and line up.
function tokenCells(usage) {
  const u = usage || {};
  const read = Number(u.cache_read_input_tokens) || 0;
  const write = Number(u.cache_creation_input_tokens) || 0;
  return html`
    <td class="num" title=${fmtInt(promptTokens(u)) + " total input (including cache)"}>${fmtCompact(promptTokens(u))}</td>
    <td class="num" title=${fmtInt(u.output_tokens)}>${fmtCompact(u.output_tokens)}</td>
    <td class="num" title=${fmtInt(read) + " read · " + fmtInt(write) + " reported write"}>
      ${read + write ? fmtCompact(read + write) : html`<span class="muted">—</span>`}
    </td>
  `;
}

function costCell(stats) {
  const s = stats || {};
  return s.priced_requests
    ? html`<span>${fmtUsd(s.cost_usd)}</span>`
    : html`<span class="muted">n/a</span>`;
}

function priceTitle(pricing, source) {
  if (!pricing) return null;
  const where = source === "provider" ? "provider" : source === "rule" ? "rule" : "source";
  return (
    "in $" + fmtRate(pricing.input) +
    " · out $" + fmtRate(pricing.output) +
    " · cache $" + fmtRate(pricing.cache_read) +
    " per 1M (" + where + ")"
  );
}

// The prices of a route: "0.15 / 0.6 / 0.015 (rule)".
function pricesCell(pricing, source) {
  if (!pricing) return html`<span class="muted">n/a</span>`;
  return html`
    <span class="mono">
      ${fmtRate(pricing.input)} / ${fmtRate(pricing.output)} / ${cacheRate(pricing)}
      ${source === "provider" ? html`<span class="muted"> (provider)</span>` : null}
    </span>
  `;
}

// ---- the providers table ---------------------------------------------

// One row per provider, with its model rows under it. The provider totals
// come from the windowed provider counters; the throughput of a provider
// is the weighted sum over its model rows, never a mean of ratios.
function buildProviderRows(state, history) {
  const breakdown = history && Array.isArray(history.breakdown) ? history.breakdown : null;
  const rows = [];
  if (breakdown && breakdown.length) {
    const groups = new Map();
    breakdown.forEach((row) => {
      const key = row && row.provider != null ? String(row.provider) : "";
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(row);
    });
    const providers = (history && history.providers) || {};
    groups.forEach((modelRows, key) => {
      const models = modelRows
        .slice()
        .sort(
          (a, b) =>
            (Number(b.requests) || 0) - (Number(a.requests) || 0) ||
            String(a.model || "").localeCompare(String(b.model || ""))
        );
      const totals = providers[key] || sumRows(models);
      const outputTokens = sumBy(models, (r) => r.tps_output_tokens);
      const generationMs = sumBy(models, (r) => r.tps_generation_ms);
      const streamed = sumBy(models, (r) => r.tps_requests);
      const ttftWeighted = sumBy(models, (r) => (Number(r.avg_ttft_ms) || 0) * (Number(r.tps_requests) || 0));
      rows.push({
        name: key === "" ? null : key,
        totals,
        tps: generationMs > 0 ? (outputTokens * 1000) / generationMs : null,
        streamed,
        avgTtft: streamed > 0 ? ttftWeighted / streamed : null,
        // A provider may carry several rate cards (one per model rule); the
        // cell of the provider row names the first of them.
        pricing: models.length ? models[0].pricing : null,
        pricingSource: models.length ? models[0].pricing_source : null,
        models,
      });
    });
  } else {
    // An older backend (or no storage at all) has no breakdown: the
    // provider rows are all there is.
    const providers = (history && history.providers) || (state && state.providers) || {};
    Object.keys(providers).forEach((name) => {
      rows.push({
        name,
        totals: providers[name],
        tps: null,
        streamed: 0,
        avgTtft: null,
        models: [],
      });
    });
  }
  rows.sort(
    (a, b) =>
      (Number(b.totals && b.totals.requests) || 0) -
        (Number(a.totals && a.totals.requests) || 0) ||
      String(a.name || "").localeCompare(String(b.name || ""))
  );
  return rows;
}

// The totals of a provider that the windowed counters do not carry.
function sumRows(rows) {
  const usage = { input_tokens: 0, output_tokens: 0, cache_creation_input_tokens: 0, cache_read_input_tokens: 0 };
  let requests = 0;
  let errors = 0;
  let duration = 0;
  let priced = 0;
  let cost = 0;
  rows.forEach((row) => {
    requests += Number(row.requests) || 0;
    errors += Number(row.errors) || 0;
    duration += (Number(row.avg_duration_ms) || 0) * (Number(row.requests) || 0);
    priced += Number(row.priced_requests) || 0;
    cost += Number(row.cost_usd) || 0;
    const u = row.usage || {};
    usage.input_tokens += Number(u.input_tokens) || 0;
    usage.output_tokens += Number(u.output_tokens) || 0;
    usage.cache_creation_input_tokens += Number(u.cache_creation_input_tokens) || 0;
    usage.cache_read_input_tokens += Number(u.cache_read_input_tokens) || 0;
  });
  return {
    requests,
    errors,
    avg_duration_ms: requests ? duration / requests : 0,
    usage,
    priced_requests: priced,
    cost_usd: cost,
  };
}

function ProvidersTable(props) {
  const rows = props.rows;
  if (!rows.length) {
    return html`
      <section class="card">
        <div class="card-head"><h2>Providers</h2></div>
        <div class="empty">No routed requests in this range.</div>
      </section>
    `;
  }
  return html`
    <section class="card">
      <div class="card-head">
        <h2>Providers</h2>
        <span class="muted">${props.note}</span>
      </div>
      <div class="scroll">
        <table class="providers-table">
          <thead>
            <tr>
              <th>Provider / model</th>
              <th class="num">Requests</th>
              <th class="num">Errors</th>
              <th class="num">Avg latency</th>
              <th class="num">TPS</th>
              <th class="num">Tokens in</th>
              <th class="num">Tokens out</th>
              <th class="num" title="cache read + cache write">Cache</th>
              <th class="num">Cost</th>
            </tr>
          </thead>
          <tbody>
            ${rows.map((row) => renderProviderRow(row, props))}
          </tbody>
        </table>
      </div>
    </section>
  `;
}

function renderProviderRow(row, props) {
  const open = !props.closed.includes(row.name == null ? "" : String(row.name));
  const label = row.name == null ? "(no provider)" : String(row.name);
  const totals = row.totals || {};
  const tpsTitle =
    row.tps == null
      ? "no streamed requests"
      : "avg TTFT " + fmtMs(row.avgTtft) + " · " + row.streamed + " streamed requests";
  const costTitle = priceTitle(row.pricing, row.pricingSource);
  const out = [];
  out.push(html`
    <tr class="provider-row" key=${"p:" + label}>
      <td>
        <button
          type="button"
          class="provider-btn"
          aria-expanded=${open ? "true" : "false"}
          onClick=${() => props.toggle(row.name == null ? "" : String(row.name))}
        >
          <span class="caret">${open ? "▾" : "▸"}</span>
          <span>${label}</span>
        </button>
      </td>
      <td class="num">${fmtInt(totals.requests)}</td>
      <td class=${"num" + (totals.errors ? " err" : "")}>${fmtInt(totals.errors)}</td>
      <td class="num">${fmtMs(totals.avg_duration_ms)}</td>
      <td class="num" title=${tpsTitle}>${row.tps == null ? "—" : fmtTps(row.tps)}</td>
      ${tokenCells(totals.usage)}
      <td class="num" title=${costTitle || ""}>${costCell(totals)}</td>
    </tr>
  `);
  if (!open) return out;
  row.models.forEach((model) => {
    const modelTps = model.tps == null ? null : Number(model.tps);
    const modelTitle =
      modelTps == null
        ? "no streamed requests"
        : "avg TTFT " + fmtMs(model.avg_ttft_ms) + " · " + (model.tps_requests || 0) + " streamed requests";
    const modelCost = model.priced_requests
      ? fmtUsd(model.cost_usd)
      : html`<span class="muted">n/a</span>`;
    out.push(html`
      <tr class="model-row" key=${"m:" + label + ":" + (model.model || "")}>
        <td class="mono">${model.model == null ? "(no model)" : String(model.model)}</td>
        <td class="num">${fmtInt(model.requests)}</td>
        <td class=${"num" + (model.errors ? " err" : "")}>${fmtInt(model.errors)}</td>
        <td class="num">${fmtMs(model.avg_duration_ms)}</td>
        <td class="num" title=${modelTitle}>${modelTps == null ? "—" : fmtTps(modelTps)}</td>
        ${tokenCells(model.usage)}
        <td class="num" title=${priceTitle(model.pricing, model.pricing_source) || ""}>${modelCost}</td>
      </tr>
    `);
  });
  return out;
}

// ---- header -----------------------------------------------------------

const WINDOW_LABELS = {
  today: "Today",
  "5m": "Last 5 minutes",
  "30m": "Last 30 minutes",
  "60m": "Last 60 minutes",
  "6h": "Last 6 hours",
  "12h": "Last 12 hours",
  "24h": "Last 24 hours",
  "7d": "Last 7 days",
  "30d": "Last 30 days",
  all: "All available data",
  custom: "Custom range",
};

// The named windows that slide with the clock: their oldest requests drop out
// as time passes, so their totals can fall.
const ROLLING_WINDOWS = FALLBACK_WINDOWS.filter((name) => name !== "all" && name !== "custom");

function windowLine(history, selected) {
  if (!history || !history.window) return "storage off — the figures are since process start";
  const w = history.window;
  const parts = [];
  if (selected === "today") parts.push("today since 00:00");
  else if (ROLLING_WINDOWS.includes(w.label)) parts.push("rolling " + w.label);
  parts.push(
    (w.bucket === "hour" ? "hourly" : "daily") +
      " · " + fmtLocalDateTime(w.start) +
      " → " + (w.end ? fmtLocalDateTime(w.end) : "now")
  );
  const since = history.data_since;
  if (since && w.start && new Date(since) > new Date(w.start)) {
    parts.push("data since " + fmtLocalDateTime(since));
  }
  if (w.clamped) parts.push("clamped to retention");
  return parts.join(" · ");
}

function timeRangeLabel(view) {
  if (view.window !== "custom") return WINDOW_LABELS[view.window] || view.window;
  return fmtLocalDateTime(view.start) + " → " + (view.end ? fmtLocalDateTime(view.end) : "now");
}

function readRecentRanges() {
  try {
    const saved = JSON.parse(window.localStorage.getItem("ohr.recent-ranges") || "[]");
    return Array.isArray(saved) ? saved.filter((item) => item &&
      (FALLBACK_WINDOWS.includes(item.window) || item.window === "today") &&
      (item.window !== "custom" || (item.start && !isNaN(new Date(item.start)))))
      .slice(0, 5) : [];
  } catch (e) {
    return [];
  }
}

// Editing is local to the picker; only an applied range changes polling.
function TimePicker({ view, windows, history, onApply }) {
  const [open, setOpen] = useState(false);
  const [tab, setTab] = useState("quick");
  const [draft, setDraft] = useState(() => rangeDraft(view));
  const [error, setError] = useState(null);
  const [recent, setRecent] = useState(readRecentRanges);
  const root = useRef(null);
  const trigger = useRef(null);
  const zone = Intl.DateTimeFormat().resolvedOptions().timeZone || "Local time";

  useEffect(() => {
    if (!open) return;
    root.current.querySelector('[role="tab"][aria-selected="true"]').focus();
    const outside = (event) => {
      if (!root.current.contains(event.target)) setOpen(false);
    };
    const escape = (event) => {
      if (event.key === "Escape") {
        event.preventDefault();
        setOpen(false);
        trigger.current.focus();
      }
    };
    document.addEventListener("pointerdown", outside);
    document.addEventListener("keydown", escape);
    return () => {
      document.removeEventListener("pointerdown", outside);
      document.removeEventListener("keydown", escape);
    };
  }, [open]);

  const close = () => {
    setOpen(false);
    trigger.current.focus();
  };
  const toggle = () => {
    if (!open) {
      const bounds = view.window === "custom" ? view : {
        start: history && history.window ? history.window.start : "",
        end: new Date().toISOString(),
      };
      setDraft(rangeDraft(bounds));
      setError(null);
      setTab(view.window === "custom" ? "absolute" : "quick");
    }
    setOpen(!open);
  };
  const select = (next) => {
    const applied = normaliseView(next);
    const updated = [applied, ...recent.filter((item) => viewQuery(item) !== viewQuery(applied))]
      .slice(0, 5);
    setRecent(updated);
    try {
      window.localStorage.setItem("ohr.recent-ranges", JSON.stringify(updated));
    } catch (e) {
      /* the picker still works without browser storage */
    }
    onApply(applied);
    close();
  };
  const applyCustom = (event) => {
    event.preventDefault();
    const start = new Date(draft.start);
    const end = draft.end ? new Date(draft.end) : null;
    if (isNaN(start) || (end && isNaN(end))) {
      setError("Choose valid dates and times.");
      return;
    }
    if (start >= (end || new Date())) {
      setError("From must be before To (or now for an open range).");
      return;
    }
    select({ window: "custom", start: start.toISOString(), end: end ? end.toISOString() : "" });
  };
  const edit = (field, value) => {
    setError(null);
    setDraft((previous) => ({ ...previous, [field]: value }));
  };
  const tabKey = (event) => {
    if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
    event.preventDefault();
    const next = event.key === "Home" ? "quick" : event.key === "End" ? "absolute"
      : tab === "quick" ? "absolute" : "quick";
    setTab(next);
    root.current.querySelector('[data-tab="' + next + '"]').focus();
  };

  return html`
    <div class="time-picker" ref=${root}>
      <button type="button" class="time-range-trigger" ref=${trigger}
        aria-label=${"Time range: " + timeRangeLabel(view)} aria-haspopup="dialog"
        aria-expanded=${open ? "true" : "false"} aria-controls="time-range-panel"
        onClick=${toggle}>
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor"
          stroke-width="1.7" aria-hidden="true"><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></svg>
        <span>${timeRangeLabel(view)}</span><span class="range-chevron" aria-hidden="true"></span>
      </button>
      ${open ? html`
        <section id="time-range-panel" class="time-range-panel" role="dialog" aria-label="Select time range">
          <div class="range-panel-heading"><strong>Time range</strong>
            <button type="button" class="range-close" aria-label="Close time picker" onClick=${close}>×</button>
          </div>
          <div class="range-tabs" role="tablist" aria-label="Range selection mode">
            ${[["quick", "Quick select"], ["absolute", "Custom range"]].map(([key, label]) => html`
              <button type="button" role="tab" id=${"range-tab-" + key} data-tab=${key}
                aria-selected=${tab === key ? "true" : "false"} aria-controls=${"range-content-" + key}
                tabIndex=${tab === key ? 0 : -1} onKeyDown=${tabKey} onClick=${() => setTab(key)}>${label}</button>
            `)}
          </div>
          ${tab === "quick" ? html`
            <div id="range-content-quick" role="tabpanel" aria-labelledby="range-tab-quick" class="range-content">
              <div class="range-section-label">Common ranges</div>
              <div class="range-presets">
                ${windows.filter((name) => name !== "custom").map((name) => html`
                  <button type="button" aria-pressed=${view.window === name ? "true" : "false"}
                    onClick=${() => select({ window: name })}>
                    <span>${WINDOW_LABELS[name] || name}</span>
                    <span aria-hidden="true">${view.window === name ? "✓" : ""}</span>
                  </button>
                `)}
              </div>
              ${recent.length ? html`
                <div class="range-recent"><div class="range-section-label">Recently used</div>
                  ${recent.map((item) => html`
                    <button type="button" aria-label=${"Reuse " + timeRangeLabel(item)}
                      onClick=${() => select(item)}>${timeRangeLabel(item)}</button>
                  `)}
                </div>
              ` : null}
            </div>
          ` : html`
            <form id="range-content-absolute" role="tabpanel" aria-labelledby="range-tab-absolute"
              class="range-content" onSubmit=${applyCustom}>
              <div class="range-section-label">Choose an interval</div>
              <div class="range-date-fields">
                <label for="range-start">From
                  <input type="datetime-local" id="range-start" required step="60" value=${draft.start}
                    onInput=${(e) => edit("start", e.currentTarget.value)} />
                </label>
                <label for="range-end">To
                  <input type="datetime-local" id="range-end" step="60" value=${draft.end}
                    onInput=${(e) => edit("end", e.currentTarget.value)} />
                </label>
              </div>
              <p class="range-help">From is included; To is excluded. Leave To empty to follow activity through now.</p>
              ${error ? html`<p class="range-error" role="alert">${error}</p>` : null}
              <div class="range-actions">
                <button type="button" class="btn" onClick=${close}>Cancel</button>
                <button type="submit" class="btn range-apply">Apply range</button>
              </div>
            </form>
          `}
          <div class="range-panel-footer"><span>${zone}</span><span>Local time · minute precision</span></div>
        </section>
      ` : null}
    </div>
  `;
}

function Header(props) {
  const { state, status, view, rangeError } = props;
  const served = (state && state.windows) || FALLBACK_WINDOWS;
  const windows = served.includes("today") ? served : ["today"].concat(served);
  const proxy = (state && state.proxy) || {};
  const live = status.kind === "live";
  const stale = status.kind === "stale";
  return html`
    <header class="topbar">
      <div class="topbar-inner">
        <div class="brand">
          <span class="title">
            <img class="brand-icon" src="/dashboard/static/samurai.png" width="40" height="40" alt="" />
            open-harness-router
          </span>
          <span class="fact">version <b>${state ? state.version || "?" : "…"}</b></span>
          ${state && state.restart_required
            ? html`<span class="pill">checkout ${state.checkout_version} · restart required</span>`
            : null}
          <span class="fact">pid <b>${state ? state.pid : "…"}</b></span>
          <span class="fact">up <b>${state ? fmtDur(state.uptime_s) : "…"}</b></span>
        </div>
        <div class=${"live" + (stale ? " live-stale" : "")} role="status">
          <span class=${"dot" + (live ? " ok" : stale ? " warn" : "")}></span>
          <span>
            ${live
              ? "live · updated " + (status.at ? status.at.toLocaleTimeString("en-GB") : "")
              : stale
              ? "stale: " + status.reason
              : "connecting…"}
          </span>
        </div>
        <div class="time-controls">
          <${TimePicker} view=${view} windows=${windows} history=${state && state.history}
            onApply=${props.onApplyView} />
          <button type="button" class="refresh-button" aria-label="Refresh now" title="Refresh now"
            onClick=${props.onRefresh}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor"
              stroke-width="1.7" aria-hidden="true"><path d="M20 7v5h-5M4 17v-5h5"/><path d="M6 7a7 7 0 0 1 12-1l2 6M4 12l2 6a7 7 0 0 0 12-1"/></svg>
          </button>
        </div>
      </div>
      ${rangeError
        ? html`<div class="window-line"><span class="range-error" role="alert">${rangeError}</span></div>`
        : null}
      <div class="window-line">
        ${windowLine(state ? state.history : null, view.window)}
        ${proxy.enabled
          ? html` · forward proxy ${proxy.host}:${proxy.port}`
          : html` · forward proxy off`}
      </div>
    </header>
  `;
}

// ---- failure details and the activity cards ---------------------------

const isFailedRequest = (event) => event.error != null ||
  (event.status != null && event.status >= 400 && event.rejected == null);

function errorExplanation(event) {
  const explanations = {
    CancelledError: "Stream interrupted before completion: cancellation, client disconnect, or shutdown. The exact cause was not recorded.",
    GeneratorExit: "Stream closed before completion. The exact cause was not recorded.",
    StreamNotStarted: "Stream closed before its response body was read.",
    UpstreamStreamError: "The upstream reported a failure inside the response stream.",
    ReadTimeout: "Timed out waiting for data from the upstream.",
    ConnectTimeout: "Timed out connecting to the upstream.",
    PoolTimeout: "Timed out waiting for an available upstream connection.",
    UpstreamError: "The upstream request failed. Additional provider details were not stored.",
  };
  if (event.error) return explanations[event.error] || "Request failed with the recorded error type.";
  if (event.status === 429) return "The upstream rate limit was exceeded.";
  if (event.status === 401 || event.status === 403) return "The upstream rejected authentication or access.";
  if (event.status >= 500) return "The upstream returned a server error.";
  return "The upstream rejected the request. Response body details were not stored.";
}

function RequestErrors({ state, history }) {
  const panel = useRef(null);
  useEffect(() => {
    const revealHash = () => {
      if (window.location.hash === "#request-errors") panel.current.open = true;
    };
    const revealLink = (event) => {
      // A second click on the same anchor does not emit hashchange.
      if (event.target instanceof Element && event.target.closest('a[href="#request-errors"]')) {
        panel.current.open = true;
      }
    };
    revealHash();
    window.addEventListener("hashchange", revealHash);
    document.addEventListener("click", revealLink);
    return () => {
      window.removeEventListener("hashchange", revealHash);
      document.removeEventListener("click", revealLink);
    };
  }, []);
  const persisted = history && Array.isArray(history.error_events);
  const events = persisted ? history.error_events
    : (state.events || []).filter((event) => event.kind === "request" && isFailedRequest(event));
  const total = Number(((history && history.totals) || state.totals || {}).errors) || 0;
  return html`
    <details class="card request-errors" id="request-errors" ref=${panel}>
      <summary>Request errors
        <span class="error-summary-note">${fmtInt(total)} · ${persisted ? "selected range" : "since restart"}</span>
      </summary>
      ${events.length ? html`
        <p class="error-history-note muted">Showing ${events.length} of ${fmtInt(total)} errors.
          HTTP 200 can still have an interrupted or failed response stream.</p>
        <div class="scroll error-history">
          <table class="errors-table">
            <thead><tr><th>Time</th><th>Model / provider</th><th>Endpoint</th><th>HTTP</th><th>Reason</th><th class="num">Duration</th></tr></thead>
            <tbody>${events.map((event, index) => html`
              <tr key=${String(event.id || event.ts) + ":" + index}>
                <td class="mono" title=${event.ts}>${fmtLocalDateTime(event.ts)}<div class="muted">${fmtTime(event.ts)}</div></td>
                <td class="mono">${event.model || "—"}<div class="muted">${event.provider || "—"}</div></td>
                <td class="mono">${event.endpoint || "—"}<div class="muted">${event.stream ? "stream" : "unary"}</div></td>
                <td class="mono">${event.status == null ? "No response" : event.status}</td>
                <td class="error-reason"><strong class="err">${event.error || "HTTP " + event.status}</strong><div>${errorExplanation(event)}</div></td>
                <td class="num">${fmtMs(event.duration_ms)}</td>
              </tr>
            `)}</tbody>
          </table>
        </div>
      ` : html`<div class="empty">${total ? "No error details are available in the current event buffer." : "No request errors in this range."}</div>`}
    </details>
  `;
}

function Activity(props) {
  const state = props.state || {};
  const active = state.active || [];
  const events = state.events || [];
  return html`
    <div class="two">
      <section class="card">
        <div class="card-head">
          <h2>In flight</h2>
          <span class="muted">live · process</span>
        </div>
        ${active.length
          ? html`
              <div class="scroll feed">
                <table class="active-table">
                  <thead>
                    <tr>
                      <th>Model</th>
                      <th>Provider</th>
                      <th>Endpoint</th>
                      <th>Mode</th>
                      <th class="num">Age</th>
                    </tr>
                  </thead>
                  <tbody>
                    ${active.map(
                      (item) => html`
                        <tr key=${String(item.id)}>
                          <td class="mono">${item.model}</td>
                          <td class="mono">${item.provider}</td>
                          <td class="mono">${item.endpoint}</td>
                          <td>
                            <span class="pill">${item.stream ? "stream" : "unary"}</span>
                          </td>
                          <td class=${"num" + (item.age_s > 300 ? " warn" : "")}>
                            ${fmtDur(item.age_s)}
                          </td>
                        </tr>
                      `
                    )}
                  </tbody>
                </table>
              </div>
            `
          : html`<div class="empty">Nothing in flight.</div>`}
      </section>
      <section class="card">
        <div class="card-head">
          <h2>Events</h2>
          <span class="muted">live · process</span>
        </div>
        ${events.length
          ? html`
              <div class="scroll feed">
                <table class="events-table">
                  <thead>
                    <tr>
                      <th>Time</th>
                      <th>Model</th>
                      <th>Provider</th>
                      <th>Status</th>
                      <th class="num">Duration</th>
                      <th class="num">Tokens</th>
                    </tr>
                  </thead>
                  <tbody>
                    ${events.map((event, index) => renderEvent(event, index))}
                  </tbody>
                </table>
              </div>
            `
          : html`<div class="empty">No events yet.</div>`}
      </section>
    </div>
  `;
}

function renderEvent(event, index) {
  const isRequest = event.kind === "request";
  const failed = isRequest
    ? isFailedRequest(event)
    : levelClass(event.level) === "err";
  const what = isRequest
    ? event.endpoint + " " + (event.status == null ? "failed" : event.status) + (event.stream ? " stream" : "")
    : event.event;
  const details = isRequest
    ? null
    : Object.keys(event.fields || {})
        .map((key) => key + "=" + String(event.fields[key]))
        .join(" ");
  return html`
    <tr key=${String(event.ts) + ":" + index} class=${failed ? "event-failed" : ""}>
      <td class="mono muted" title=${event.ts}>${fmtTime(event.ts)}</td>
      <td class="mono">
        ${isRequest ? event.model : what}
        ${isRequest ? html`<div class="muted">${event.endpoint}${event.stream ? " · stream" : ""}</div>` : null}
        ${details ? html`<span class="muted"> ${details}</span>` : null}
      </td>
      <td class="mono">${isRequest ? event.provider : ""}</td>
      <td class=${isRequest ? (failed ? "err" : "ok") : levelClass(event.level)}>
        ${isRequest ? (event.status == null ? "failed" : event.status) : event.level}
        ${isRequest && event.error ? html`<div class="event-error">${event.error}</div>` : null}
      </td>
      <td class="num">${event.duration_ms != null ? fmtMs(event.duration_ms) : ""}</td>
      <td class="num">${isRequest ? tokensCell(event.usage) : ""}</td>
    </tr>
  `;
}

// ---- routing ----------------------------------------------------------

function Routing(props) {
  const state = props.state || {};
  const routes = state.routes || [];
  const routePricing = (state.pricing && state.pricing.routes) || [];
  const protocols = state.protocols || [];
  const proxy = state.proxy || {};
  return html`
    <details class="card routing">
      <summary>Routing</summary>
      <p class="routing-note">Active configuration loaded at startup. Editing routing.yaml requires a router restart.</p>
      ${protocols.length ? html`
        <h3 class="routing-heading">Endpoint defaults</h3>
        <div class="scroll">
          <table class="protocols-table">
            <thead><tr><th>API</th><th>Default provider</th><th>Selection</th></tr></thead>
            <tbody>
              ${protocols.map((protocol) => html`
                <tr key=${protocol.protocol}>
                  <td>
                    <strong>${{ messages: "Messages", responses: "Codex Responses", images: "Codex Images" }[protocol.protocol] || protocol.protocol}</strong>
                    <div class="routing-endpoints mono muted">${protocol.endpoints.join(" · ")}</div>
                  </td>
                  <td class="mono">${protocol.default_provider || "Not configured"}</td>
                  <td>
                    ${protocol.model_rules ? "First matching model rule, otherwise this default" : "Authenticated native default only; bypasses model rules"}
                    ${!protocol.available ? html`<div class="warn">Default unavailable for this API</div>` : null}
                  </td>
                </tr>
              `)}
            </tbody>
          </table>
        </div>
        <h3 class="routing-heading">Model rules</h3>
        <p class="routing-note">Checked in order for Messages and Responses. A matched provider must support the requested API.</p>
      ` : null}
      ${routes.length
        ? html`
            <div class="scroll">
              <table class="routes-table">
                <thead>
                  <tr>
                    <th>Match</th>
                    <th>Provider</th>
                    <th>Upstream model</th>
                    <th class="num">Prices $/M</th>
                  </tr>
                </thead>
                <tbody>
                  ${routes.map((route, index) => {
                    if (protocols.length && route.match_type === "default") return null;
                    const pricing = routePricing[index] || {};
                    return html`
                      <tr key=${String(index)}>
                        <td>
                          <span class="muted">${route.match_type}</span>
                          <span class="mono"> ${route.match_value || ""}</span>
                        </td>
                        <td class="mono">${route.provider}</td>
                        <td class="mono muted">${route.upstream_model || "Requested model"}</td>
                        <td class="num">${pricesCell(pricing.pricing, pricing.pricing_source)}</td>
                      </tr>
                    `;
                  })}
                </tbody>
              </table>
            </div>
          `
        : html`<div class="empty">No routes.</div>`}
      <p class="proxy-line">
        forward proxy${" "}
        <b>${proxy.enabled ? proxy.host + ":" + proxy.port : "off"}</b>
        ${proxy.mitm_hosts && proxy.mitm_hosts.length
          ? html` · MITM <b>${proxy.mitm_hosts.join(", ")}</b>`
          : null}
        ${proxy.root_certificate_not_after
          ? html` · root certificate valid until <b>${fmtDate(proxy.root_certificate_not_after)}</b>`
          : html` · root certificate <b>not generated</b>`}
      </p>
    </details>
  `;
}

// ---- the page ---------------------------------------------------------

function App() {
  const root = document.getElementById("app");
  const stateUrl = (root && root.getAttribute("data-state-url")) || "/dashboard/state";
  const [state, setState] = useState(null);
  const [status, setStatus] = useState({ kind: "connecting", reason: "", at: null });
  const [rangeError, setRangeError] = useState(null);
  const [refreshToken, setRefreshToken] = useState(0);
  const [view, setView] = useState(readInitialView);
  const [closed, setClosed] = useState(readClosedProviders);
  const { colours, version } = useThemeColours();

  const viewKey = useMemo(() => JSON.stringify(view), [view]);

  // The poll: every two seconds, with a backoff up to fifteen seconds when
  // the router does not answer.
  useEffect(() => {
    let cancelled = false;
    let timer = null;
    let failures = 0;
    const run = async () => {
      try {
        const response = await fetch(stateUrl + stateQuery(view), { cache: "no-store" });
        if (!response.ok) {
          let detail = "HTTP " + response.status;
          try {
            const body = await response.json();
            if (body && body.detail) detail = String(body.detail);
          } catch (e) {
            /* not a JSON error: keep the status line */
          }
          const error = new Error(detail);
          error.status = response.status;
          throw error;
        }
        const data = await response.json();
        if (cancelled) return;
        failures = 0;
        setRangeError(null);
        setState(data);
        setStatus({ kind: "live", reason: "", at: new Date() });
      } catch (error) {
        if (cancelled) return;
        failures += 1;
        // A rejected custom range keeps the previous figures on the page.
        if (error.status === 400) setRangeError(error.message);
        setStatus({ kind: "stale", reason: error.message, at: null });
      }
      if (!cancelled) {
        timer = setTimeout(run, failures ? Math.min(POLL_MS * failures, MAX_BACKOFF_MS) : POLL_MS);
      }
    };
    run();
    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, [viewKey, refreshToken]);

  // The window lives in the URL, so a reload (and a link) keep it.
  useEffect(() => {
    try {
      window.history.replaceState(null, "", window.location.pathname + viewQuery(view) + window.location.hash);
    } catch (e) {
      /* the history API is not always available */
    }
  }, [viewKey]);

  const applyView = (next) => {
    const normalised = normaliseView(next);
    setView(normalised);
    setRangeError(null);
    try {
      window.localStorage.setItem(VIEW_KEY, JSON.stringify(normalised));
    } catch (e) {
      /* no storage: the URL carries the choice */
    }
  };

  const toggleProvider = (name) => {
    setClosed((previous) => {
      const next = previous.includes(name)
        ? previous.filter((item) => item !== name)
        : previous.concat([name]);
      try {
        window.localStorage.setItem(CLOSED_KEY, JSON.stringify(next));
      } catch (e) {
        /* no storage: the choice lasts for this page only */
      }
      return next;
    });
  };

  const history = (state && state.history) || null;
  const totals = (history && history.totals) || (state && state.totals) || {};
  const usage = totals.usage || {};
  // "Today" reaches the server as a custom range, so its label is the page's.
  const windowLabel = !history
    ? "since start"
    : view.window === "today"
    ? "today"
    : (history.window && history.window.label) || "window";
  const requests = Number(totals.requests) || 0;
  const errors = Number(totals.errors) || 0;
  const active = (state && state.active) || [];
  const streaming = active.filter((item) => item.stream).length;
  const errorRate = requests > 0 ? ((errors / requests) * 100).toFixed(1) + "%" : "0%";
  const cacheRead = Number(usage.cache_read_input_tokens) || 0;
  const cacheWrite = Number(usage.cache_creation_input_tokens) || 0;

  // ---- the charts -----------------------------------------------------
  const bucket = history && history.window ? history.window.bucket : "day";
  const buckets = history ? windowBuckets(history.window.start, history.window.end, bucket) : [];
  const labels = buckets.map((key) => bucketLabel(key, bucket));

  let requestsConfig = null;
  let requestsKey = "";
  if (history) {
    const key = bucket === "hour" ? "bucket" : "day";
    const byBucket = {};
    (history.series || []).forEach((row) => {
      byBucket[row[key]] = row;
    });
    const requestsData = buckets.map((b) => {
      const row = byBucket[b];
      return row ? Number(row.requests) || 0 : null;
    });
    const errorsData = buckets.map((b) => {
      const row = byBucket[b];
      return row ? Number(row.errors) || 0 : null;
    });
    // No request in the window: the card shows its empty text, not bare axes.
    if (requestsData.some((value) => value != null)) {
      requestsConfig = chartConfig(
        labels,
        [
          lineDataset("Requests", requestsData, colours.series1, colours, { fill: true }),
          lineDataset("Errors", errorsData, colours.series2, colours),
        ],
        colours
      );
      requestsKey = JSON.stringify([history.series || [], buckets]);
    }
  }

  let tpsConfig = null;
  let tpsKey = "";
  if (history) {
    const series = tpsBuildSeries(history);
    if (series.length && buckets.length) {
      const datasets = series.map((item) => {
        const colour = colours["series" + (item.slot + 1)];
        const data = buckets.map((b) => {
          const point = item.points[b];
          return point ? Number(point.tps) || 0 : null;
        });
        return lineDataset(item.model, data, colour, colours, { borderDash: item.dash });
      });
      tpsConfig = chartConfig(labels, datasets, colours);
      tpsKey = JSON.stringify([history.tps || {}, buckets]);
    }
  }

  const providerRows = buildProviderRows(state, history);
  const providerNote = history
    ? history.breakdown && history.breakdown.length
      ? "window · provider and model rows"
      : "window · provider rows"
    : "since process start";

  return html`
    <${Header}
      state=${state}
      status=${status}
      view=${view}
      rangeError=${rangeError}
      onApplyView=${applyView}
      onRefresh=${() => setRefreshToken((previous) => previous + 1)}
    />
    <main class="wrap">
      <div class="tiles">
        <${Tile}
          label="Requests"
          value=${fmtCompact(requests)}
          title=${fmtInt(requests)}
          sub=${windowLabel + " · avg " + fmtMs(totals.avg_duration_ms)}
        />
        <${Tile}
          label="Errors"
          value=${fmtCompact(errors)}
          tone=${errors ? "err" : ""}
          title=${fmtInt(errors)}
          sub=${"error rate " + errorRate}
          href=${errors ? "#request-errors" : null}
        />
        <${Tile}
          label="In flight"
          value=${fmtInt(active.length)}
          title=${fmtInt(active.length)}
          sub=${streaming + " streaming · live"}
        />
        <${Tile}
          label="Tokens"
          value=${fmtCompact(promptTokens(usage)) + " / " + fmtCompact(usage.output_tokens)}
          title=${fmtInt(promptTokens(usage)) + " total input (including cache) / " + fmtInt(usage.output_tokens) + " output"}
          sub=${fmtCompact(cacheRead) + " cache read · " + fmtCompact(cacheWrite) + " reported write"}
          subTitle="Provider-reported cache usage. Missing write counters and older records count as zero; zero does not establish that no cache was written. Input above includes cache reads and writes."
        />
        <${Tile}
          label="Cost"
          value=${totals.priced_requests ? fmtUsd(totals.cost_usd) : html`<span class="muted">n/a</span>`}
          title=${fmtUsd(totals.cost_usd)}
          sub=${totals.priced_requests
            ? fmtInt(totals.priced_requests) + " priced of " + fmtInt(requests)
            : "set pricing in routing.yaml"}
        />
      </div>
      <div class="charts">
        <${ChartCard}
          title="Requests"
          subtitle="requests and errors per bucket"
          config=${requestsConfig}
          configKey=${requestsKey}
          version=${version}
          emptyText="No requests in this window."
        />
        <${ChartCard}
          title="Throughput (TPS)"
          subtitle="generation speed per model"
          config=${tpsConfig}
          configKey=${tpsKey}
          version=${version}
          emptyText="No timed streamed requests in this window. New streams appear after they finish."
        />
      </div>
      <${ProvidersTable}
        rows=${providerRows}
        closed=${closed}
        toggle=${toggleProvider}
        note=${providerNote}
      />
      <${RequestErrors} state=${state || {}} history=${history} />
      <${Activity} state=${state} />
      <${Routing} state=${state} />
    </main>
  `;
}

const root = document.getElementById("app");
if (root) render(html`<${App} />`, root);
