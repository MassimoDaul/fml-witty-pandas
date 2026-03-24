const svgEl = document.getElementById("graph");
const arxivId = svgEl.dataset.arxivId;
const centerTitle = svgEl.dataset.title;
const centerYear  = parseInt(svgEl.dataset.year) || null;

const EXT_ID_LINKS = {
  DOI:    id => ({ label: `DOI:${id}`,    href: `https://doi.org/${id}` }),
  PubMed: id => ({ label: `PubMed:${id}`, href: `https://pubmed.ncbi.nlm.nih.gov/${id}` }),
  DBLP:   id => ({ label: `DBLP`,         href: `https://dblp.org/rec/${id}` }),
};

function renderExternalIds(paperIds, s2Id) {
  const el = document.getElementById("paper-ext-ids");
  if (!el) return;
  const parts = [];
  for (const [type, fn] of Object.entries(EXT_ID_LINKS)) {
    if (paperIds[type]) {
      const { label, href } = fn(paperIds[type]);
      parts.push(`<a href="${href}" target="_blank">${label}</a>`);
    }
  }
  if (s2Id) {
    parts.push(`<a href="https://www.semanticscholar.org/paper/${s2Id}" target="_blank">S2</a>`);
  }
  if (parts.length) el.innerHTML = " &nbsp;·&nbsp; " + parts.join(" &nbsp;·&nbsp; ");
}

function loadGraph() {
  const statusEl = document.getElementById("graph-status");
  statusEl.textContent = "Loading references and citations from Semantic Scholar…";

  fetch(`/api/paper/${arxivId}/references`)
    .then(r => r.json())
    .then(data => {
      if (data.paper_ids || data.s2_paper_id) {
        renderExternalIds(data.paper_ids || {}, data.s2_paper_id);
      }
      if (data.error) {
        const is429 = data.error.toLowerCase().includes("rate limit");
        if (is429) {
          statusEl.innerHTML = `${data.error} &nbsp;<button onclick="loadGraph()" style="font-size:0.85em;padding:2px 10px;cursor:pointer;">Retry</button>`;
        } else {
          statusEl.textContent = data.error;
        }
        return;
      }
      if (data.nodes.length <= 1) {
        statusEl.textContent = "No references or citations found.";
        return;
      }
      statusEl.remove();
      const center = data.nodes.find(n => n.is_center);
      if (center) { center.title = centerTitle; center.year = centerYear; }
      renderGraph(data.nodes, data.links);
    })
    .catch(() => {
      statusEl.innerHTML = `Failed to load graph. &nbsp;<button onclick="loadGraph()" style="font-size:0.85em;padding:2px 10px;cursor:pointer;">Retry</button>`;
    });
}

loadGraph();

const tooltip = document.getElementById("tooltip");
const graphWrap = document.getElementById("graph-wrap");
let hideTimer = null;
let tooltipFocused = false;
let focusedNodeEl = null;

tooltip.addEventListener("mouseenter", () => clearTimeout(hideTimer));
tooltip.addEventListener("mouseleave", () => { if (!tooltipFocused) tooltip.style.display = "none"; });
tooltip.addEventListener("click", e => e.stopPropagation());

document.addEventListener("click", () => {
  if (tooltipFocused) {
    tooltipFocused = false;
    if (focusedNodeEl) {
      d3.select(focusedNodeEl).select("circle").attr("stroke", "#fff").attr("stroke-width", 1.5);
      focusedNodeEl = null;
    }
    tooltip.style.display = "none";
  }
});

function positionTooltip(e) {
  const rect = graphWrap.getBoundingClientRect();
  let x = e.clientX - rect.left + 14;
  let y = e.clientY - rect.top - 10;
  if (x + 295 > rect.width) x = e.clientX - rect.left - 295;
  tooltip.style.left = x + "px";
  tooltip.style.top  = y + "px";
}

function buildTooltip(d) {
  const titleHref = d.in_db && d.arxiv_id ? `/paper/${d.arxiv_id}`
    : d.arxiv_id ? `https://arxiv.org/abs/${d.arxiv_id}` : null;
  const titleHtml = titleHref
    ? `<strong><a href="${titleHref}"${d.in_db ? "" : ' target="_blank"'}>${d.title}</a></strong>`
    : `<strong>${d.title}</strong>`;
  const lines = [titleHtml];
  if (d.year) lines.push(`Year: ${d.year}`);
  if (d.citation_count != null) lines.push(`Citations: ${d.citation_count.toLocaleString()}`);
  if (d.authors && d.authors.length) {
    const authorLinks = d.authors.map(
      a => `<a href="/author?name=${encodeURIComponent(a)}">${a}</a>`
    );
    lines.push(authorLinks.join(", "));
  }
  const role = d.is_center ? "This paper"
    : d.is_citation ? "Cites this paper"
    : "Referenced by this paper";
  const loc = d.in_db ? " · In local database" : d.arxiv_id ? " · arXiv only" : "";
  lines.push(role + loc);
  const linkParts = [];
  if (d.arxiv_id) linkParts.push(
    d.in_db
      ? `<a href="/paper/${d.arxiv_id}">View paper</a>`
      : `<a href="https://arxiv.org/abs/${d.arxiv_id}" target="_blank">arXiv</a>`
  );
  const ids = d.ext_ids || {};
  if (ids.DOI)    linkParts.push(`<a href="https://doi.org/${ids.DOI}" target="_blank">DOI</a>`);
  if (ids.PubMed) linkParts.push(`<a href="https://pubmed.ncbi.nlm.nih.gov/${ids.PubMed}" target="_blank">PubMed</a>`);
  if (ids.DBLP)   linkParts.push(`<a href="https://dblp.org/rec/${ids.DBLP}" target="_blank">DBLP</a>`);
  if (d.s2_id)    linkParts.push(`<a href="https://www.semanticscholar.org/paper/${d.s2_id}" target="_blank">S2</a>`);
  if (linkParts.length) lines.push(linkParts.join(" · "));
  return lines.join("<br>");
}

function renderGraph(nodes, links) {
  const margin = { top: 20, right: 30, bottom: 36, left: 30 };
  const W = svgEl.clientWidth || 900;
  const H = 560;

  // Year scale bounded to actual data years
  const years = nodes.map(d => d.year).filter(Boolean);
  const minYear = d3.min(years) ?? 2000;
  const maxYear = d3.max(years) ?? new Date().getFullYear();
  const xScale = d3.scaleLinear()
    .domain([minYear - 0.5, maxYear + 0.5])
    .range([margin.left + 10, W - margin.right - 10]);

  // Citation count → node radius
  const maxCit = d3.max(nodes.filter(d => !d.is_center), d => d.citation_count ?? 0) || 1;
  const rScale = d3.scaleSqrt().domain([0, maxCit]).range([12, 48]).clamp(true);

  const svg = d3.select("#graph")
    .attr("viewBox", `0 0 ${W} ${H}`)
    .style("cursor", "grab");

  const yearSpan = maxYear - minYear + 1;
  const tickCount = Math.min(yearSpan, 14);

  // Vertical year gridlines — outside zoom group, updated on zoom
  const gridG = svg.append("g").attr("opacity", 0.5);
  function drawGridlines(scale) {
    gridG.selectAll("line")
      .data(scale.ticks(tickCount))
      .join("line")
      .attr("x1", d => scale(d)).attr("x2", d => scale(d))
      .attr("y1", margin.top).attr("y2", H - margin.bottom)
      .attr("stroke", "#e5e7eb")
      .attr("stroke-width", 1);
  }
  drawGridlines(xScale);

  // Nodes + links live in g (gets zoom transform)
  const g = svg.append("g");

  // X-axis — outside zoom group so it stays fixed at bottom
  const xAxisG = svg.append("g")
    .attr("transform", `translate(0,${H - margin.bottom})`);
  function drawXAxis(scale) {
    xAxisG.call(
      d3.axisBottom(scale).ticks(tickCount).tickFormat(d3.format("d"))
    )
    .call(ax => ax.select(".domain").attr("stroke", "#d1d5db"))
    .call(ax => ax.selectAll("text").attr("fill", "#6b7280").attr("font-size", "12px"))
    .call(ax => ax.selectAll(".tick line").attr("stroke", "#d1d5db"));
  }
  drawXAxis(xScale);

  // Zoom — minimum scale 1 prevents zooming out beyond the data year bounds
  svg.call(
    d3.zoom().scaleExtent([1, 5])
      .on("zoom", e => {
        g.attr("transform", e.transform);
        const newX = e.transform.rescaleX(xScale);
        drawXAxis(newX);
        drawGridlines(newX);
      })
  );

  const simulation = d3.forceSimulation(nodes)
    .force("link", d3.forceLink(links).id(d => d.id).distance(80).strength(0.3))
    .force("charge", d3.forceManyBody().strength(-150))
    .force("x", d3.forceX(d => d.year ? xScale(d.year) : W / 2).strength(d => d.year ? 0.8 : 0.05))
    .force("y", d3.forceY((H - margin.bottom + margin.top) / 2).strength(0.08))
    .force("collide", d3.forceCollide(d => d.is_center ? 22 : rScale(d.citation_count ?? 0) + 5));

  // Arrowhead markers
  const defs = svg.append("defs");
  defs.append("marker")
    .attr("id", "arrow-ref")
    .attr("viewBox", "0 -4 8 8").attr("refX", 8).attr("refY", 0)
    .attr("markerWidth", 6).attr("markerHeight", 6).attr("orient", "auto")
    .append("path").attr("d", "M0,-4L8,0L0,4").attr("fill", "#9ca3af");
  defs.append("marker")
    .attr("id", "arrow-cite")
    .attr("viewBox", "0 -4 8 8").attr("refX", 8).attr("refY", 0)
    .attr("markerWidth", 6).attr("markerHeight", 6).attr("orient", "auto")
    .append("path").attr("d", "M0,-4L8,0L0,4").attr("fill", "#f97316");

  const link = g.append("g")
    .selectAll("line")
    .data(links)
    .join("line")
    .attr("stroke", d => d.is_citation ? "#f97316" : "#9ca3af")
    .attr("stroke-width", d => d.is_citation ? 1.5 : 1)
    .attr("stroke-opacity", d => d.is_citation ? 0.7 : 0.5)
    .attr("marker-end", d => d.is_citation ? "url(#arrow-cite)" : "url(#arrow-ref)");

  const nodeG = g.append("g")
    .selectAll("g")
    .data(nodes)
    .join("g")
    .attr("cursor", "pointer")
    .on("click", (e, d) => {
      e.stopPropagation();
      const isSame = tooltipFocused && focusedNodeEl === e.currentTarget;
      // unfocus previous node ring
      if (focusedNodeEl) {
        d3.select(focusedNodeEl).select("circle").attr("stroke", "#fff").attr("stroke-width", 1.5);
      }
      if (isSame) {
        tooltipFocused = false;
        focusedNodeEl = null;
        tooltip.style.display = "none";
        return;
      }
      tooltipFocused = true;
      focusedNodeEl = e.currentTarget;
      clearTimeout(hideTimer);
      tooltip.innerHTML = buildTooltip(d);
      tooltip.style.display = "block";
      positionTooltip(e);
      d3.select(e.currentTarget).select("circle")
        .attr("stroke", "#1d4ed8")
        .attr("stroke-width", 2.5);
    })
    .on("mouseenter", (e, d) => {
      clearTimeout(hideTimer);
      if (!tooltipFocused) {
        tooltip.innerHTML = buildTooltip(d);
        tooltip.style.display = "block";
        positionTooltip(e);
      }
    })
    .on("mouseleave", () => { if (!tooltipFocused) hideTimer = setTimeout(() => { tooltip.style.display = "none"; }, 200); })
    .call(
      d3.drag()
        .on("start", (e, d) => { if (!e.active) simulation.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
        .on("drag",  (e, d) => { d.fx = e.x; d.fy = e.y; })
        .on("end",   (e, d) => { if (!e.active) simulation.alphaTarget(0); d.fx = null; d.fy = null; })
    );

  nodeG.append("circle")
    .attr("r", d => d.is_center ? 20 : rScale(d.citation_count ?? 0))
    .attr("fill", d => {
      if (d.is_center) return "#f59e0b";
      if (d.in_db)     return "#3b82f6";
      if (d.arxiv_id)  return "#9ca3af";
      return "#e5e7eb";
    })
    .attr("stroke", "#fff")
    .attr("stroke-width", 1.5)
    .attr("pointer-events", "none");

  // Invisible hit-area circle — minimum 16px radius for easy hovering
  nodeG.append("circle")
    .attr("r", d => Math.max(d.is_center ? 20 : rScale(d.citation_count ?? 0), 16))
    .attr("fill", "transparent")
    .attr("stroke", "none");

  // Label only the center node
  nodeG.filter(d => d.is_center)
    .append("text")
    .text(d => { const t = d.title || ""; return t.length > 40 ? t.slice(0, 40) + "…" : t; })
    .attr("x", 24).attr("y", 5)
    .attr("font-size", "13px")
    .attr("font-weight", "600")
    .attr("fill", "#374151")
    .attr("pointer-events", "none");

  simulation.on("tick", () => {
    link
      .attr("x1", d => d.source.x).attr("y1", d => d.source.y)
      .attr("x2", d => d.target.x).attr("y2", d => d.target.y);
    nodeG.attr("transform", d => `translate(${d.x},${d.y})`);
  });
}
