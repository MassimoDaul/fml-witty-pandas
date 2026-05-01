import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const wrap     = document.getElementById('cloud-canvas-wrap');
const canvas   = document.getElementById('cloud-canvas');
const statusEl = document.getElementById('cloud-status');
const tooltip  = document.getElementById('cloud-tooltip');
const statsEl  = document.getElementById('cloud-stats');
const legendEl = document.getElementById('cloud-legend');
const statTotal = document.getElementById('cloud-stat-total');
const statPc1   = document.getElementById('cloud-stat-pc1');
const statPc2   = document.getElementById('cloud-stat-pc2');
const statPc3   = document.getElementById('cloud-stat-pc3');
const yearMinEl = document.getElementById('cloud-year-min');
const yearMaxEl = document.getElementById('cloud-year-max');

// ── Scene ──────────────────────────────────────────────────────────────
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x100d0b);

const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

const camera = new THREE.PerspectiveCamera(58, 1, 0.01, 100);
camera.position.set(0, 0, 3.8);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping    = true;
controls.dampingFactor    = 0.06;
controls.autoRotate       = true;
controls.autoRotateSpeed  = 0.35;
controls.minDistance      = 0.4;
controls.maxDistance      = 14;

renderer.domElement.addEventListener('pointerdown', () => {
  controls.autoRotate = false;
}, { once: true });

// ── Shaders ────────────────────────────────────────────────────────────
// gl_PointSize = aSize * (f / -z) * C * pixelRatio
// where f = projectionMatrix[1][1] ≈ 1/tan(fov/2), C ≈ 4 → ~5–23 CSS px at z=3.8
const vertexShader = `
  attribute float aSize;
  attribute vec3  aColor;
  varying   vec3  vColor;
  varying   float vPixelSize;
  uniform   float uPixelRatio;

  void main() {
    vColor = aColor;
    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
    float f = projectionMatrix[1][1];
    vPixelSize = aSize * f * (uPixelRatio / -mvPosition.z) * 4.0;
    gl_PointSize = vPixelSize;
    gl_Position  = projectionMatrix * mvPosition;
  }
`;

const fragmentShader = `
  varying vec3  vColor;
  varying float vPixelSize;

  void main() {
    vec2  uv = gl_PointCoord - 0.5;
    float r  = length(uv);
    if (r > 0.5) discard;
    float softEdge  = 1.0 - smoothstep(0.30, 0.5, r);
    float fadeSmall = smoothstep(0.8, 2.5, vPixelSize);
    gl_FragColor = vec4(vColor, softEdge * fadeSmall * 0.90);
  }
`;

// Single material instance shared across rebuilds so resize can touch the uniform
const cloudMaterial = new THREE.ShaderMaterial({
  vertexShader,
  fragmentShader,
  transparent: true,
  depthWrite:  false,
  blending:    THREE.NormalBlending,
  uniforms: {
    uPixelRatio: { value: renderer.getPixelRatio() },
  },
});

// ── State ──────────────────────────────────────────────────────────────
const raycaster = new THREE.Raycaster();
raycaster.params.Points.threshold = 0.06;
const mouse = new THREE.Vector2(-999, -999);
let pointsObj  = null;
let pointMeta  = [];
let hoveredIdx = -1;

// ── Year → colour (blue → purple → amber → gold) ──────────────────────
function yearColor(year, minY, maxY) {
  const t = maxY > minY ? (year - minY) / (maxY - minY) : 0.5;
  const c = new THREE.Color();
  if (t < 0.5) {
    // hue 210° → 270° (blue → purple)
    c.setHSL(0.583 + t * 2 * 0.167, 0.60, 0.52 + t * 2 * 0.04);
  } else {
    // purple → amber/gold via RGB lerp (crosses the hue discontinuity)
    const from = new THREE.Color().setHSL(0.750, 0.60, 0.56);
    const to   = new THREE.Color().setHSL(0.083, 0.75, 0.58);
    c.lerpColors(from, to, (t - 0.5) * 2);
  }
  return c;
}

// ── Build / replace point cloud ────────────────────────────────────────
function buildCloud(data) {
  const pts = data.points;
  const n   = pts.length;

  let xMin = Infinity, xMax = -Infinity;
  let yMin = Infinity, yMax = -Infinity;
  let zMin = Infinity, zMax = -Infinity;
  let maxCit = 0, minYear = Infinity, maxYear = -Infinity;

  for (const p of pts) {
    if (p.x < xMin) xMin = p.x; if (p.x > xMax) xMax = p.x;
    if (p.y < yMin) yMin = p.y; if (p.y > yMax) yMax = p.y;
    if (p.z < zMin) zMin = p.z; if (p.z > zMax) zMax = p.z;
    if ((p.citation_count || 0) > maxCit) maxCit = p.citation_count;
    const y = p.year || 2022;
    if (y < minYear) minYear = y;
    if (y > maxYear) maxYear = y;
  }

  // Fit into a ~3-unit cube centred at origin
  const span = Math.max(xMax - xMin, yMax - yMin, zMax - zMin, 1e-6);
  const sc   = 3.0 / span;
  const cx   = (xMin + xMax) / 2;
  const cy   = (yMin + yMax) / 2;
  const cz   = (zMin + zMax) / 2;

  const maxSqrtCit = Math.sqrt(maxCit + 1);

  const positions = new Float32Array(n * 3);
  const colors    = new Float32Array(n * 3);
  const sizes     = new Float32Array(n);
  pointMeta = [];

  for (let i = 0; i < n; i++) {
    const p  = pts[i];
    const px = (p.x - cx) * sc;
    const py = (p.y - cy) * sc;
    const pz = (p.z - cz) * sc;

    positions[i * 3]     = px;
    positions[i * 3 + 1] = py;
    positions[i * 3 + 2] = pz;

    const col = yearColor(p.year || minYear, minYear, maxYear);
    colors[i * 3]     = col.r;
    colors[i * 3 + 1] = col.g;
    colors[i * 3 + 2] = col.b;

    // Size: sqrt-compressed citation count, mapped to [2.5, 12]
    sizes[i] = 2.5 + (Math.sqrt((p.citation_count || 0) + 1) / maxSqrtCit) * 9.5;

    pointMeta.push({ ...p });
  }

  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geo.setAttribute('aColor',   new THREE.BufferAttribute(colors,    3));
  geo.setAttribute('aSize',    new THREE.BufferAttribute(sizes,     1));

  if (pointsObj) scene.remove(pointsObj);
  pointsObj = new THREE.Points(geo, cloudMaterial);
  scene.add(pointsObj);

  // HUD
  yearMinEl.textContent = minYear;
  yearMaxEl.textContent = maxYear;
  statTotal.textContent = n.toLocaleString();
  const pct = (v) => `${Math.round((v || 0) * 100)}%`;
  statPc1.textContent = pct(data.explained_variance[0]);
  statPc2.textContent = pct(data.explained_variance[1]);
  statPc3.textContent = pct(data.explained_variance[2]);
  statsEl.hidden = false;
  legendEl.hidden = false;
  statusEl.classList.add('is-hidden');
}

// ── Hover ──────────────────────────────────────────────────────────────
function checkHover() {
  if (!pointsObj) return;
  raycaster.setFromCamera(mouse, camera);
  const hits = raycaster.intersectObject(pointsObj);
  if (hits.length > 0) {
    const idx = hits[0].index;
    if (idx !== hoveredIdx) {
      hoveredIdx = idx;
      const p   = pointMeta[idx];
      const yr  = p.year ?? 'n/a';
      const cit = (p.citation_count || 0).toLocaleString();
      tooltip.innerHTML =
        `<strong>${escHtml(p.title)}</strong>` +
        `<span class="tooltip-meta">${yr} &middot; ${cit} citations</span>`;
      tooltip.style.display = 'block';
      canvas.style.cursor   = 'pointer';
    }
  } else if (hoveredIdx >= 0) {
    hoveredIdx = -1;
    tooltip.style.display = 'none';
    canvas.style.cursor   = '';
  }
}

function escHtml(s) {
  return String(s ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

// ── Resize ─────────────────────────────────────────────────────────────
function resize() {
  const w = wrap.clientWidth;
  const h = wrap.clientHeight;
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h, false);
  cloudMaterial.uniforms.uPixelRatio.value = renderer.getPixelRatio();
}

// ── Render loop ────────────────────────────────────────────────────────
function animate() {
  requestAnimationFrame(animate);
  controls.update();
  checkHover();
  renderer.render(scene, camera);
}

// ── Mouse events ───────────────────────────────────────────────────────
wrap.addEventListener('mousemove', (e) => {
  const rect = wrap.getBoundingClientRect();
  mouse.x = ((e.clientX - rect.left) / rect.width)  *  2 - 1;
  mouse.y = ((e.clientY - rect.top)  / rect.height) * -2 + 1;
  tooltip.style.left = `${Math.min(e.clientX - rect.left + 16, rect.width  - 296)}px`;
  tooltip.style.top  = `${Math.min(e.clientY - rect.top  + 16, rect.height -  90)}px`;
});

wrap.addEventListener('mouseleave', () => {
  mouse.set(-999, -999);
  tooltip.style.display = 'none';
  hoveredIdx = -1;
  canvas.style.cursor = '';
});

wrap.addEventListener('click', () => {
  if (hoveredIdx >= 0 && pointMeta[hoveredIdx]?.href) {
    window.location.href = pointMeta[hoveredIdx].href;
  }
});

window.addEventListener('resize', resize);

// ── Boot ───────────────────────────────────────────────────────────────
resize();
animate();

fetch('/api/embedding-cloud', { headers: { Accept: 'application/json' } })
  .then((r) => r.json())
  .then((data) => {
    if (data.error) throw new Error(data.error);
    buildCloud(data);
  })
  .catch((err) => {
    statusEl.textContent = err.message || 'Failed to load point cloud.';
  });
