(() => {
  const DEFAULT_SPACE = "title";
  const chartState = new WeakMap();
  let resizeTimer = null;

  function escapeHtml(value) {
    return String(value ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  function endpointFor(panel) {
    const params = new URLSearchParams();
    params.set("q", panel.dataset.query || "");
    params.set("limit", panel.dataset.limit || "30");

    if (panel.dataset.categories) {
      params.set("categories", panel.dataset.categories);
    }
    if (panel.dataset.yearFrom) {
      params.set("year_from", panel.dataset.yearFrom);
    }
    if (panel.dataset.yearTo) {
      params.set("year_to", panel.dataset.yearTo);
    }

    return `/api/embedding-space?${params.toString()}`;
  }

  function setStatus(panel, message, isError = false) {
    const status = panel.querySelector("[data-embedding-status]");
    if (!status) {
      return;
    }
    status.textContent = message || "";
    status.hidden = !message;
    status.classList.toggle("is-error", isError);
  }

  function selectedSpace(panel) {
    const active = panel.querySelector("[data-space-tab].is-active");
    return active?.dataset.spaceTab || DEFAULT_SPACE;
  }

  function setSelectedSpace(panel, key) {
    panel.querySelectorAll("[data-space-tab]").forEach((tab) => {
      const isActive = tab.dataset.spaceTab === key;
      tab.classList.toggle("is-active", isActive);
      tab.setAttribute("aria-selected", isActive ? "true" : "false");
    });
  }

  function pointForSpace(data, spaceKey) {
    return (data.points || []).filter((point) => point.spaces && point.spaces[spaceKey]);
  }

  function tooltipHtml(point, spaceLabel) {
    if (point.is_query) {
      return `<strong>${escapeHtml(point.title)}</strong><span>${escapeHtml(spaceLabel)} space</span>`;
    }

    const parts = [
      `<strong>#${point.rank} ${escapeHtml(point.title)}</strong>`,
      point.year ? `<span>${point.year}</span>` : "",
      Number.isFinite(point.similarity)
        ? `<span>Similarity ${point.similarity.toFixed(3)}</span>`
        : "",
    ].filter(Boolean);
    return parts.join("<br>");
  }

  function positionTooltip(event, panel, tooltip) {
    const wrap = panel.querySelector(".embedding-chart-wrap");
    if (!wrap) {
      return;
    }
    const rect = wrap.getBoundingClientRect();
    let x = event.clientX - rect.left + 14;
    let y = event.clientY - rect.top + 14;
    const tooltipWidth = tooltip.offsetWidth || 280;
    if (x + tooltipWidth > rect.width - 8) {
      x = event.clientX - rect.left - tooltipWidth - 14;
    }
    tooltip.style.left = `${Math.max(8, x)}px`;
    tooltip.style.top = `${Math.max(8, y)}px`;
  }

  function render(panel) {
    const state = chartState.get(panel);
    if (!state?.data || !window.d3) {
      return;
    }

    const svgNode = panel.querySelector("[data-embedding-chart]");
    const tooltip = panel.querySelector("[data-embedding-tooltip]");
    if (!svgNode || !tooltip) {
      return;
    }

    const spaceKey = selectedSpace(panel);
    const space = state.data.spaces.find((item) => item.key === spaceKey) || state.data.spaces[0];
    const points = pointForSpace(state.data, space.key);
    const svgWidth = Math.max(svgNode.clientWidth || 760, 320);
    const svgHeight = svgWidth < 520 ? 340 : 390;
    const margin = { top: 24, right: 28, bottom: 34, left: 34 };

    const svg = d3.select(svgNode)
      .attr("viewBox", `0 0 ${svgWidth} ${svgHeight}`)
      .attr("aria-label", `Massimo ${space.label} embedding space projection`);

    svg.selectAll("*").remove();

    if (!points.length) {
      setStatus(panel, `No vectors available for the ${space.label.toLowerCase()} space.`);
      return;
    }

    setStatus(panel, "");

    const xExtent = d3.extent(points, (point) => point.spaces[space.key].x);
    const yExtent = d3.extent(points, (point) => point.spaces[space.key].y);

    function paddedExtent(extent) {
      const min = Number.isFinite(extent[0]) ? extent[0] : -1;
      const max = Number.isFinite(extent[1]) ? extent[1] : 1;
      if (Math.abs(max - min) < 1e-6) {
        return [min - 1, max + 1];
      }
      const pad = (max - min) * 0.16;
      return [min - pad, max + pad];
    }

    const xScale = d3.scaleLinear()
      .domain(paddedExtent(xExtent))
      .range([margin.left, svgWidth - margin.right]);
    const yScale = d3.scaleLinear()
      .domain(paddedExtent(yExtent))
      .range([svgHeight - margin.bottom, margin.top]);

    const grid = svg.append("g").attr("class", "embedding-grid");
    grid.selectAll("line.x")
      .data(xScale.ticks(6))
      .join("line")
      .attr("x1", (tick) => xScale(tick))
      .attr("x2", (tick) => xScale(tick))
      .attr("y1", margin.top)
      .attr("y2", svgHeight - margin.bottom)
      .attr("stroke", "rgba(170, 148, 121, 0.26)")
      .attr("stroke-width", 1);

    grid.selectAll("line.y")
      .data(yScale.ticks(5))
      .join("line")
      .attr("x1", margin.left)
      .attr("x2", svgWidth - margin.right)
      .attr("y1", (tick) => yScale(tick))
      .attr("y2", (tick) => yScale(tick))
      .attr("stroke", "rgba(170, 148, 121, 0.2)")
      .attr("stroke-width", 1);

    const pointGroups = svg.append("g")
      .selectAll("g")
      .data(points, (point) => point.corpus_id)
      .join("g")
      .attr("transform", (point) => {
        const coord = point.spaces[space.key];
        return `translate(${xScale(coord.x)},${yScale(coord.y)})`;
      });

    const paperGroups = pointGroups.filter((point) => !point.is_query);
    paperGroups.append("a")
      .attr("href", (point) => point.href)
      .attr("aria-label", (point) => `Open paper ${point.rank}: ${point.title}`)
      .append("circle")
      .attr("r", (point) => Math.max(4.5, 9 - point.rank * 0.1))
      .attr("fill", "#4f7fa8")
      .attr("fill-opacity", 0.78)
      .attr("stroke", "#fffdf9")
      .attr("stroke-width", 1.4);

    paperGroups.append("circle")
      .attr("r", 14)
      .attr("fill", "transparent")
      .on("mouseenter", (event, point) => {
        tooltip.innerHTML = tooltipHtml(point, space.label);
        tooltip.style.display = "block";
        positionTooltip(event, panel, tooltip);
      })
      .on("mousemove", (event) => positionTooltip(event, panel, tooltip))
      .on("mouseleave", () => {
        tooltip.style.display = "none";
      })
      .on("click", (_event, point) => {
        if (point.href) {
          window.location.href = point.href;
        }
      });

    const queryGroup = pointGroups.filter((point) => point.is_query);
    queryGroup.append("path")
      .attr("d", d3.symbol().type(d3.symbolDiamond).size(210))
      .attr("fill", "#714c2e")
      .attr("stroke", "#fffdf9")
      .attr("stroke-width", 1.8);

    queryGroup.append("text")
      .text("Query")
      .attr("x", 12)
      .attr("y", 4)
      .attr("fill", "#1f1a17")
      .attr("font-size", "12px")
      .attr("font-weight", 700)
      .attr("pointer-events", "none");

    queryGroup.append("circle")
      .attr("r", 16)
      .attr("fill", "transparent")
      .on("mouseenter", (event, point) => {
        tooltip.innerHTML = tooltipHtml(point, space.label);
        tooltip.style.display = "block";
        positionTooltip(event, panel, tooltip);
      })
      .on("mousemove", (event) => positionTooltip(event, panel, tooltip))
      .on("mouseleave", () => {
        tooltip.style.display = "none";
      });

    const variance = (space.explained_variance || [])
      .slice(0, 2)
      .map((value) => `${Math.round(value * 100)}%`)
      .join(" / ");
    svg.append("text")
      .attr("x", margin.left)
      .attr("y", svgHeight - 9)
      .attr("fill", "#62584d")
      .attr("font-size", "12px")
      .text(`PCA variance ${variance || "0% / 0%"}`);
  }

  function loadPanel(panel) {
    setStatus(panel, "Loading embedding space...");
    fetch(endpointFor(panel), { headers: { Accept: "application/json" } })
      .then((response) => response.json().then((body) => ({ response, body })))
      .then(({ response, body }) => {
        if (!response.ok || body.error) {
          throw new Error(body.error || "Embedding space unavailable.");
        }
        chartState.set(panel, { data: body });
        render(panel);
      })
      .catch((error) => {
        setStatus(panel, error.message || "Embedding space unavailable.", true);
      });
  }

  function initPanel(panel) {
    if (panel.dataset.embeddingSpaceReady === "true") {
      return;
    }
    panel.dataset.embeddingSpaceReady = "true";
    setSelectedSpace(panel, DEFAULT_SPACE);

    panel.querySelectorAll("[data-space-tab]").forEach((tab) => {
      tab.addEventListener("click", () => {
        setSelectedSpace(panel, tab.dataset.spaceTab || DEFAULT_SPACE);
        render(panel);
      });
    });

    if (!window.d3) {
      setStatus(panel, "D3 could not be loaded for this visualization.", true);
      return;
    }

    loadPanel(panel);
  }

  function init(root = document) {
    root.querySelectorAll("[data-embedding-space]").forEach(initPanel);
  }

  window.CephaloEmbeddingSpace = { init };

  document.addEventListener("DOMContentLoaded", () => init());

  document.body.addEventListener("htmx:afterSwap", (event) => {
    if (event.target instanceof HTMLElement) {
      init(event.target);
    }
  });

  window.addEventListener("resize", () => {
    window.clearTimeout(resizeTimer);
    resizeTimer = window.setTimeout(() => {
      document.querySelectorAll("[data-embedding-space]").forEach(render);
    }, 120);
  });
})();
