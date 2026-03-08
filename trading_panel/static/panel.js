const bootstrap = window.TRADING_PANEL_BOOTSTRAP;

const dateInput = document.getElementById("date-input");
const reloadBtn = document.getElementById("reload-btn");
const panelDateTitle = document.getElementById("panel-date-title");
const stationsGrid = document.getElementById("stations-grid");
const statusPill = document.getElementById("status-pill");

let loading = false;

function setStatus(text) {
  statusPill.textContent = text;
}

async function loadPanel(dateValue) {
  if (loading) {
    return;
  }
  loading = true;
  const requestStarted = performance.now();
  setStatus("Loading...");

  try {
    const url = new URL(bootstrap.panelDataUrl, window.location.origin);
    if (dateValue) {
      url.searchParams.set("date", dateValue);
    }

    const response = await fetch(url.toString(), { cache: "no-store" });
    if (!response.ok) {
      const text = await response.text();
      throw new Error(text || `Request failed with ${response.status}`);
    }

    const payload = await response.json();
    const renderStarted = performance.now();
    renderPanel(payload);
    const renderMs = performance.now() - renderStarted;
    const totalMs = performance.now() - requestStarted;
    const serverTimings = payload.timings_ms || {};
    setStatus(`Updated ${new Date().toLocaleTimeString()}`);
    console.log("trading_panel_timings", {
      server: serverTimings,
      client_render_ms: Number(renderMs.toFixed(2)),
      request_total_ms: Number(totalMs.toFixed(2)),
    });
  } catch (error) {
    stationsGrid.innerHTML = `<div class="station-card"><div class="empty-plot">${escapeHtml(String(error))}</div></div>`;
    setStatus("Load failed");
  } finally {
    loading = false;
  }
}

function renderPanel(payload) {
  panelDateTitle.textContent = payload.date;
  if (dateInput.value !== payload.date) {
    dateInput.value = payload.date;
  }

  stationsGrid.innerHTML = "";
  const pendingPlots = [];

  payload.stations.forEach((stationData, index) => {
    const card = document.createElement("article");
    card.className = "station-card";

    const title = document.createElement("h2");
    title.className = "station-title";
    title.textContent = stationData.station;
    card.appendChild(title);

    const subtitles = document.createElement("div");
    subtitles.className = "station-subtitles";

    const leftSubtitle = document.createElement("div");
    leftSubtitle.className = "station-subtitle-left";
    leftSubtitle.textContent = stationData.current_local_time;

    const rightSubtitle = document.createElement("div");
    rightSubtitle.className = "station-subtitle-right";
    rightSubtitle.textContent = stationData.last_observation;

    subtitles.appendChild(leftSubtitle);
    subtitles.appendChild(rightSubtitle);
    card.appendChild(subtitles);

    const plotId = `station-plot-${sanitizeId(stationData.station)}-${index}`;
    const plotDiv = document.createElement("div");
    plotDiv.id = plotId;
    plotDiv.className = "station-plot";
    card.appendChild(plotDiv);

    const resolvedMarketRow = document.createElement("div");
    resolvedMarketRow.className = "resolved-market-row";

    const resolvedLeft = document.createElement("div");
    resolvedLeft.className = "resolved-market-left";
    resolvedLeft.textContent =
      stationData.resolved_yes_market_left || "Max. Temp.: N/A";

    const resolvedRight = document.createElement("div");
    resolvedRight.className = "resolved-market-right";
    resolvedRight.textContent = stationData.resolved_yes_market_right || "";

    resolvedMarketRow.appendChild(resolvedLeft);
    resolvedMarketRow.appendChild(resolvedRight);
    card.appendChild(resolvedMarketRow);

    stationsGrid.appendChild(card);
    pendingPlots.push({ plotId, stationData });
  });

  requestAnimationFrame(() => {
    pendingPlots.forEach(({ plotId, stationData }) => {
      renderStationPlot(plotId, stationData);
    });
    requestAnimationFrame(resizeAllPlots);
  });
}

function renderStationPlot(plotId, stationData) {
  const nonEmptyTraces = stationData.traces.filter((trace) => trace.x.length > 0);
  if (nonEmptyTraces.length === 0) {
    const plotDiv = document.getElementById(plotId);
    plotDiv.className = "empty-plot";
    plotDiv.textContent = "No prediction data for this date.";
    return;
  }

  const tickValues = [...new Set(nonEmptyTraces.flatMap((trace) => trace.x))].sort();
  const tickTexts = tickValues.map(formatIssueToken);

  const traces = nonEmptyTraces.map((trace) => ({
    x: trace.x,
    y: trace.y,
    name: trace.label,
    mode: "lines+markers",
    line: {
      color: trace.color,
      width: 2,
    },
    marker: {
      size: 5,
    },
    hovertemplate: `${trace.label}: %{y}<extra></extra>`,
  }));

  const referenceLines = stationData.reference_lines || [];
  referenceLines.forEach((lineSpec) => {
    traces.push({
      x: tickValues,
      y: tickValues.map(() => lineSpec.value),
      name: lineSpec.label,
      mode: "lines",
      line: {
        color: lineSpec.color,
        width: lineSpec.width || 1.5,
        dash: lineSpec.dash || "solid",
      },
      hovertemplate: `${lineSpec.label}: ${lineSpec.value}<extra></extra>`,
    });
  });

  const layout = {
    showlegend: false,
    margin: {
      l: 40,
      r: 8,
      t: 10,
      b: 82,
    },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "#0b1624",
    font: {
      color: "#c8d7eb",
    },
    hovermode: "x unified",
    xaxis: {
      title: "",
      tickangle: 90,
      tickmode: "array",
      tickvals: tickValues,
      ticktext: tickTexts,
      tickfont: {
        size: 9,
      },
      showgrid: true,
      gridcolor: "#23344a",
      automargin: true,
    },
    yaxis: {
      title: "",
      showgrid: true,
      gridcolor: "#23344a",
      zeroline: false,
      automargin: true,
    },
  };

  const config = {
    responsive: true,
    displaylogo: false,
    modeBarButtonsToRemove: ["lasso2d", "select2d", "autoScale2d"],
  };

  Plotly.newPlot(plotId, traces, layout, config);
}

function formatIssueToken(isoUtc) {
  return isoUtc.replace(/[-:T]/g, "").slice(0, 10);
}

function sanitizeId(value) {
  return value.replace(/[^a-zA-Z0-9_-]/g, "_");
}

function escapeHtml(value) {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/\"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function resizeAllPlots() {
  document.querySelectorAll(".station-plot").forEach((plotDiv) => {
    if (plotDiv.data && plotDiv.layout) {
      Plotly.Plots.resize(plotDiv);
    }
  });
}

reloadBtn.addEventListener("click", () => loadPanel(dateInput.value));
dateInput.addEventListener("change", () => loadPanel(dateInput.value));
window.addEventListener("resize", resizeAllPlots);

loadPanel(bootstrap.defaultDate);
setInterval(() => loadPanel(dateInput.value), 120000);
