// Dron autónomo: el navegador solo dibuja; toda la simulación (y el mapa que construye el dron) vive en server.py.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const $ = (id) => document.getElementById(id);
const els = {
  profile: $("profile"), level: $("level"), seed: $("seed"), wind: $("wind"), wdir: $("wdir"), gusts: $("gusts"),
  noise: $("noise"), newBtn: $("btn-new"), play: $("btn-play"), speed: $("speed"), camera: $("camera"),
  precision: $("precision"), cp: $("cp"), rays: $("show-rays"), gps: $("show-gps"), viewport: $("viewport"),
  toast: $("toast"), banner: $("banner"), mode: $("mode"), terrain: $("terrain"), density: $("density"),
  goalKind: $("goal-kind"), rain: $("rain"), motion: $("motion"), mapMode: $("map-mode"), view: $("view"),
};
const PHASES = { "en tierra": "En tierra", despegue: "Despegue", crucero: "Crucero", "aproximación": "Aproximación",
  aterrizaje: "Aterrizaje", aterrizado: "Aterrizado", carrera: "Carrera", reintento: "Volviendo a la meta", meta: "¡Meta!", "persecución": "Persecución" };
const LABELS = {
  mode: { aterrizar: "Aterrizar en la meta", carrera: "Carrera (llegar el primero)" },
  terrain: { plano: "Plano", colinas: "Colinas", "montañoso": "Montañoso", precipicios: "Precipicios" },
  density: { baja: "Baja", normal: "Normal", alta: "Alta", extrema: "Extrema" },
  goal: { suelo: "En el suelo", azotea: "En una azotea", cima: "En una cima", aire: "En el aire (carrera)", azar: "Al azar" },
  rain: { no: "Sin lluvia", moderada: "Moderada", fuerte: "Fuerte" },
  motion: { fija: "Quieto", suave: "En movimiento: suave", medio: "En movimiento: medio", "rápido": "En movimiento: rápido",
    variable: "En movimiento: variable" },
  map: { conocido: "Conocido (fase 1)", desconocido: "Desconocido: lo construye (fase 2)" },
};
const LEVEL_LABEL = { bosque: "Bosque", ciudad: "Ciudad", mixto: "Mixto" };
const NOISE_LABEL = { ideal: "Ideales (sin ruido)", realista: "Realistas", alto: "Ruido alto" };

let options = null, frame = null, world = null, profile = null, playing = false, busy = false;
const telemetry = [];

async function api(path, body) {
  const res = await fetch(path, body === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || res.statusText);
  return data;
}
function toast(msg) {
  els.toast.textContent = msg;
  els.toast.classList.remove("hidden");
  clearTimeout(toast.t);
  toast.t = setTimeout(() => els.toast.classList.add("hidden"), 4500);
}

// ------------------------------------------------------------------ escena
// Simulador: x adelante, y izquierda, z arriba. three.js: y arriba. (x, y, z) -> (x, z, -y)
const V = (p) => new THREE.Vector3(p[0], p[2], -p[1]);
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
els.viewport.appendChild(renderer.domElement);
$("loading").classList.add("hidden");
const scene = new THREE.Scene();
scene.background = new THREE.Color("#0b1016");
scene.fog = new THREE.Fog("#0b1016", 60, 150);
const camera = new THREE.PerspectiveCamera(60, 1, 0.05, 400);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.enabled = false;
scene.add(new THREE.HemisphereLight("#cfe6ff", "#2a3a2a", 0.9));
const sun = new THREE.DirectionalLight("#fff4e0", 1.6);
sun.position.set(20, 45, 25);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
Object.assign(sun.shadow.camera, { left: -50, right: 50, top: 50, bottom: -50, near: 1, far: 140 });
scene.add(sun, sun.target);
const worldGroup = new THREE.Group();
scene.add(worldGroup);

function windowTexture(color) {
  const c = document.createElement("canvas");
  c.width = 64; c.height = 128;
  const g = c.getContext("2d");
  g.fillStyle = color; g.fillRect(0, 0, 64, 128);
  for (let y = 6; y < 128; y += 14) for (let x = 5; x < 64; x += 14) {
    g.fillStyle = Math.random() < 0.35 ? "rgba(255, 214, 140, .85)" : "rgba(160, 200, 230, .25)";
    g.fillRect(x, y, 8, 8);
  }
  const t = new THREE.CanvasTexture(c);
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

// altura del terreno en el navegador (misma rejilla que el servidor, algo más gruesa)
function terrainH(x, y) {
  if (!world) return 0;
  const T = world.terrain, r = T.res, h = T.h, nx = h.length, ny = h[0].length;
  const gx = Math.min(Math.max(x / r, 0), nx - 1.001), gy = Math.min(Math.max(y / r, 0), ny - 1.001);
  const i = Math.floor(gx), j = Math.floor(gy), fx = gx - i, fy = gy - j;
  return (h[i][j] * (1 - fx) + h[i + 1][j] * fx) * (1 - fy) + (h[i][j + 1] * (1 - fx) + h[i + 1][j + 1] * fx) * fy;
}
function terrainColor(z, maxZ) {  // verde -> ocre -> roca -> nieve
  const f = maxZ > 0.5 ? z / maxZ : 0;
  const stops = [[0, [0.12, 0.2, 0.14]], [0.35, [0.22, 0.32, 0.17]], [0.65, [0.42, 0.36, 0.25]], [0.88, [0.5, 0.49, 0.47]], [1, [0.9, 0.92, 0.95]]];
  for (let k = 1; k < stops.length; k++) if (f <= stops[k][0]) {
    const [f0, c0] = stops[k - 1], [f1, c1] = stops[k], t = (f - f0) / (f1 - f0);
    return c0.map((c, j) => c + (c1[j] - c) * t);
  }
  return stops[stops.length - 1][1];
}

function makeRover() {
  const g = new THREE.Group();
  const body = new THREE.Mesh(new THREE.BoxGeometry(1.6, 0.35, 1.1), new THREE.MeshStandardMaterial({ color: "#334155", metalness: 0.3 }));
  body.position.y = 0.35; body.castShadow = true;
  g.add(body);
  for (const [x, z] of [[0.55, 0.55], [0.55, -0.55], [-0.55, 0.55], [-0.55, -0.55]]) {
    const wh = new THREE.Mesh(new THREE.CylinderGeometry(0.2, 0.2, 0.14, 14), new THREE.MeshStandardMaterial({ color: "#111827" }));
    wh.rotation.x = Math.PI / 2; wh.position.set(x, 0.2, z);
    g.add(wh);
  }
  const pad = new THREE.Mesh(new THREE.CylinderGeometry(0.7, 0.7, 0.05, 32),
    new THREE.MeshStandardMaterial({ color: "#facc15", emissive: "#6b5200", roughness: 0.5 }));
  pad.position.y = 0.56;
  const light = new THREE.Mesh(new THREE.SphereGeometry(0.08, 10, 8), new THREE.MeshBasicMaterial({ color: "#ff3b3b" }));
  light.position.set(0.8, 0.5, 0);
  g.add(pad, light);
  return g;
}
let rover = null, gateMesh = null, trackLine = null;
const interceptMarker = new THREE.Mesh(new THREE.OctahedronGeometry(0.35), new THREE.MeshBasicMaterial({ color: "#facc15", wireframe: true }));
interceptMarker.visible = false;

let padMesh = null, fogMesh = null;
// --- mapa que construye el dron (fase 2): celdas ocupadas (instancias coloreadas por altura) y niebla
const fogCanvas = document.createElement("canvas");
const fogTex = new THREE.CanvasTexture(fogCanvas);
fogTex.flipY = false;
fogTex.magFilter = THREE.NearestFilter;
const MAX_VOX = 40000;
const voxMesh = new THREE.InstancedMesh(new THREE.BoxGeometry(0.94, 0.94, 0.94),
  new THREE.MeshLambertMaterial({ transparent: true, opacity: 0.6 }), MAX_VOX);
voxMesh.count = 0;
voxMesh.frustumCulled = false;
scene.add(voxMesh);
let mapData = null;
function voxelColor(z) {  // morado (bajo) -> azul -> turquesa -> lima -> ámbar (alto)
  const stops = [[0, [0.49, 0.23, 0.93]], [0.2, [0.15, 0.39, 0.92]], [0.45, [0.08, 0.72, 0.65]], [0.7, [0.52, 0.8, 0.09]], [1, [0.96, 0.62, 0.04]]];
  const f = Math.min(1, Math.max(0, z / 18));
  for (let k = 1; k < stops.length; k++) if (f <= stops[k][0]) {
    const [f0, c0] = stops[k - 1], [f1, c1] = stops[k], t = (f - f0) / (f1 - f0);
    return c0.map((c, j) => c + (c1[j] - c) * t);
  }
  return stops[stops.length - 1][1];
}
function drawMap(m) {
  mapData = m;
  const [nx, ny, nz] = m.shape, c = m.cell;
  const mat = new THREE.Matrix4(), col = new THREE.Color();
  const n = Math.min(m.occ.length, MAX_VOX);
  for (let k = 0; k < n; k++) {
    const idx = m.occ[k], z = idx % nz, y = Math.floor(idx / nz) % ny, x = Math.floor(idx / (ny * nz));
    const p = V([(x + 0.5) * c, (y + 0.5) * c, (z + 0.5) * c]);
    mat.makeTranslation(p.x, p.y, p.z);
    voxMesh.setMatrixAt(k, mat);
    voxMesh.setColorAt(k, col.setRGB(...voxelColor((z + 0.5) * c)));
  }
  voxMesh.count = n;
  voxMesh.instanceMatrix.needsUpdate = true;
  if (voxMesh.instanceColor) voxMesh.instanceColor.needsUpdate = true;
  fogCanvas.width = nx; fogCanvas.height = ny;
  const g = fogCanvas.getContext("2d"), img = g.createImageData(nx, ny);
  for (let i = 0; i < nx; i++) for (let j = 0; j < ny; j++) {
    const seen = m.seen[i * ny + j] === "1", o = (j * nx + i) * 4;
    img.data.set(seen ? [0, 0, 0, 0] : [6, 10, 16, 190], o);
  }
  g.putImageData(img, 0, 0);
  fogTex.needsUpdate = true;
  applyView();
}
function clearMap() {
  mapData = null;
  voxMesh.count = 0;
  applyView();
}
// qué se ve: el mundo real, lo que sabe el dron o los dos
function applyView() {
  const v = els.view.value, hasMap = !!mapData;
  worldGroup.children.forEach((c) => { if (c.userData.real) c.visible = !(hasMap && v === "map"); });
  voxMesh.visible = hasMap && v !== "world";
  voxMesh.material.opacity = v === "map" ? 0.95 : 0.55;
  if (fogMesh) fogMesh.visible = hasMap && v !== "world";
  document.querySelectorAll(".legend .map-only").forEach((e) => e.classList.toggle("hidden", !hasMap));
}

function buildWorld(w) {
  world = w;
  worldGroup.clear();
  const [L, W, H] = w.size;
  const T = w.terrain, nx = T.h.length, ny = T.h[0].length;
  // malla del terreno: un vértice por punto de la rejilla, en coordenadas de three (x, altura, -y)
  const verts = new Float32Array(nx * ny * 3), col = new Float32Array(nx * ny * 3);
  let maxZ = 0;
  T.h.forEach((row) => row.forEach((z) => { maxZ = Math.max(maxZ, z); }));
  for (let i = 0; i < nx; i++) for (let j = 0; j < ny; j++) {
    const k = i * ny + j, z = T.h[i][j];
    verts.set([i * T.res, z, -j * T.res], k * 3);
    col.set(terrainColor(z, maxZ), k * 3);
  }
  const idx = [];
  for (let i = 0; i < nx - 1; i++) for (let j = 0; j < ny - 1; j++) {
    const a = i * ny + j, b = (i + 1) * ny + j, c = (i + 1) * ny + j + 1, d = i * ny + j + 1;
    idx.push(a, c, b, a, d, c);
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(verts, 3));
  geo.setAttribute("color", new THREE.BufferAttribute(col, 3));
  geo.setIndex(idx);
  geo.computeVertexNormals();
  const ground = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 1, side: THREE.DoubleSide }));
  ground.receiveShadow = true;
  ground.userData.real = true;
  worldGroup.add(ground);
  // niebla de lo no explorado: la misma malla del terreno con una textura de 1 píxel por columna del mapa
  const uv = new Float32Array(nx * ny * 2);
  for (let i = 0; i < nx; i++) for (let j = 0; j < ny; j++) uv.set([(i * T.res) / L, (j * T.res) / W], (i * ny + j) * 2);
  geo.setAttribute("uv", new THREE.BufferAttribute(uv, 2));
  fogMesh = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ map: fogTex, transparent: true, depthWrite: false,
    polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -2, side: THREE.DoubleSide }));
  fogMesh.renderOrder = 1;
  fogMesh.visible = false;
  worldGroup.add(fogMesh);
  const edges = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(L, H, W)),
    new THREE.LineBasicMaterial({ color: "#3d5a6b", transparent: true, opacity: 0.25 }));
  edges.position.set(L / 2, H / 2, -W / 2);
  worldGroup.add(edges);
  for (const o of w.obstacles) {
    let mesh;
    if (o.type === "cylinder") {
      const color = new THREE.Color(o.kind === "tree" ? "#2f7d4a" : "#6aa84f").offsetHSL((Math.random() - 0.5) * 0.04, 0, (Math.random() - 0.5) * 0.08);
      mesh = new THREE.Mesh(new THREE.CylinderGeometry(o.r, o.r, o.h, 18), new THREE.MeshStandardMaterial({ color, roughness: 0.85 }));
      mesh.position.copy(V([o.x, o.y, o.base + o.h / 2]));
    } else {
      const low = o.kind === "low", w3 = o.x1 - o.x0, d3 = o.y1 - o.y0;
      const tex = low ? null : windowTexture("#40566b");
      if (tex) tex.repeat.set(Math.max(1, Math.round(w3 / 2)), Math.max(1, Math.round(o.h / 3)));
      mesh = new THREE.Mesh(new THREE.BoxGeometry(w3, o.h, d3),
        new THREE.MeshStandardMaterial({ color: low ? "#b8744a" : "#ffffff", map: tex, roughness: 0.7, metalness: 0.1 }));
      mesh.position.copy(V([(o.x0 + o.x1) / 2, (o.y0 + o.y1) / 2, o.base + o.h / 2]));
    }
    mesh.castShadow = mesh.receiveShadow = true;
    mesh.userData.real = true;  // lo que el dron NO sabe (en modo mapa desconocido)
    worldGroup.add(mesh);
  }
  const start = new THREE.Mesh(new THREE.RingGeometry(0.5, 0.65, 32), new THREE.MeshBasicMaterial({ color: "#19a7a0", side: THREE.DoubleSide }));
  start.rotation.x = -Math.PI / 2;
  start.position.copy(V([w.start[0], w.start[1], w.start[2] + 0.04]));
  padMesh = new THREE.Group();
  const pad = new THREE.Mesh(new THREE.CylinderGeometry(0.75, 0.75, 0.05, 40),
    new THREE.MeshStandardMaterial({ color: "#facc15", emissive: "#6b5200", roughness: 0.5 }));
  pad.position.y = 0.03;
  const hRing = new THREE.Mesh(new THREE.RingGeometry(0.28, 0.4, 32), new THREE.MeshBasicMaterial({ color: "#1c2630", side: THREE.DoubleSide }));
  hRing.rotation.x = -Math.PI / 2; hRing.position.y = 0.06;
  const moving = w.motion && w.motion !== "fija";
  if (w.goal_support && !moving) padMesh.add(pad, hRing);   // en el aire o sobre el vehículo: no hay plataforma fija
  padMesh.position.copy(V([w.goal[0], w.goal[1], w.goal[2] - 0.05]));
  const glow = new THREE.PointLight("#facc15", 15, 8);
  glow.position.copy(V([w.goal[0], w.goal[1], w.goal[2] + 1.5]));
  worldGroup.add(start, padMesh, glow);
  // puerta de meta en modo carrera (o marcador del punto en el aire)
  const race = frame ? frame.mode === "carrera" : false;
  rover = null; gateMesh = null;
  if (moving) {
    rover = makeRover();
    worldGroup.add(rover);
    trackLine = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineDashedMaterial({ color: "#facc15", dashSize: 0.6, gapSize: 0.4, transparent: true, opacity: 0.7 }));
    worldGroup.add(trackLine, interceptMarker);
  }
  if (race || !w.goal_support) {
    const gate = new THREE.Mesh(new THREE.TorusGeometry(1.0, 0.07, 10, 40),
      new THREE.MeshBasicMaterial({ color: "#facc15", transparent: true, opacity: 0.9 }));
    gate.position.copy(V([w.goal[0], w.goal[1], w.goal[2] + (w.goal_support ? 1.0 : 0)]));
    gate.userData.spin = true;
    worldGroup.add(gate);
    gateMesh = gate;
    if (!w.goal_support) {  // mástil hasta el suelo para ver dónde está
      const g0 = terrainH(w.goal[0], w.goal[1]);
      const mast = new THREE.Mesh(new THREE.CylinderGeometry(0.04, 0.04, w.goal[2] - g0, 6),
        new THREE.MeshBasicMaterial({ color: "#facc15", transparent: true, opacity: 0.35 }));
      mast.position.copy(V([w.goal[0], w.goal[1], (w.goal[2] + g0) / 2]));
      worldGroup.add(mast);
    }
  }
  sun.target.position.set(L / 2, 0, -W / 2);
}

// --- dron (la actitud llega del servidor: se inclina de verdad para acelerar)
const drone = new THREE.Group();
function buildDrone(radius) {
  drone.clear();
  const s = radius / 0.18 * 0.9;  // escala visual según el tamaño real
  const body = new THREE.Mesh(new THREE.BoxGeometry(0.2 * s, 0.07 * s, 0.14 * s), new THREE.MeshStandardMaterial({ color: "#e6edf2", metalness: 0.4, roughness: 0.4 }));
  body.castShadow = true;
  drone.add(body);
  const armMat = new THREE.MeshStandardMaterial({ color: "#26333d" });
  drone.userData.rotors = [];
  for (const [sx, sz] of [[1, 1], [1, -1], [-1, 1], [-1, -1]]) {
    const arm = new THREE.Mesh(new THREE.BoxGeometry(0.03 * s, 0.02 * s, 0.24 * s), armMat);
    arm.position.set(sx * 0.08 * s, 0.01 * s, sz * 0.08 * s);
    arm.rotation.y = Math.atan2(sx, sz);
    const rotor = new THREE.Mesh(new THREE.CylinderGeometry(0.075 * s, 0.075 * s, 0.006 * s, 20),
      new THREE.MeshBasicMaterial({ color: "#9fd8ff", transparent: true, opacity: 0.35 }));
    rotor.position.set(sx * 0.15 * s, 0.035 * s, sz * 0.15 * s);
    drone.userData.rotors.push(rotor);
    drone.add(arm, rotor);
  }
  const led = new THREE.Mesh(new THREE.SphereGeometry(0.02 * s, 8, 8), new THREE.MeshBasicMaterial({ color: "#ff3b3b" }));
  led.position.set(0.11 * s, 0, 0);
  drone.add(led);
  const halo = new THREE.Mesh(new THREE.RingGeometry(radius * 0.95, radius, 32),
    new THREE.MeshBasicMaterial({ color: "#19a7a0", side: THREE.DoubleSide, transparent: true, opacity: 0.45 }));
  halo.rotation.x = -Math.PI / 2;
  drone.add(halo);
}
scene.add(drone);
const shadowDot = new THREE.Mesh(new THREE.CircleGeometry(0.3, 24), new THREE.MeshBasicMaterial({ color: "#000", transparent: true, opacity: 0.35 }));
shadowDot.rotation.x = -Math.PI / 2;
scene.add(shadowDot);

// posición estimada (lo que cree el dron) y lecturas del GPS
const estMarker = new THREE.Mesh(new THREE.SphereGeometry(0.12, 12, 10), new THREE.MeshBasicMaterial({ color: "#f97316", transparent: true, opacity: 0.85 }));
scene.add(estMarker);
const GPS_N = 40;
const gpsGeo = new THREE.BufferGeometry();
gpsGeo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(GPS_N * 3), 3));
const gpsPts = new THREE.Points(gpsGeo, new THREE.PointsMaterial({ color: "#e879f9", size: 0.07, transparent: true, opacity: 0.7 }));
gpsPts.frustumCulled = false;
scene.add(gpsPts);
const gpsList = [];
let lastGps = null;

// trayectoria coloreada por velocidad
let trajLine = null;
function speedColor(f) {  // azul -> verde -> amarillo -> rojo
  const stops = [[0, [0.23, 0.51, 0.96]], [0.35, [0.13, 0.77, 0.37]], [0.7, [0.98, 0.8, 0.08]], [1, [0.94, 0.27, 0.27]]];
  f = Math.min(1, Math.max(0, f));
  for (let i = 1; i < stops.length; i++) if (f <= stops[i][0]) {
    const [f0, c0] = stops[i - 1], [f1, c1] = stops[i], k = (f - f0) / (f1 - f0);
    return c0.map((c, j) => c + (c1[j] - c) * k);
  }
  return stops[stops.length - 1][1];
}
function drawTrajectory(tr) {
  if (trajLine) { scene.remove(trajLine); trajLine.geometry.dispose(); }
  const pts = tr.points.map(V);
  const geo = new THREE.BufferGeometry().setFromPoints(pts);
  const col = new Float32Array(pts.length * 3);
  const vmax = profile ? profile.v_cruise : 8;
  tr.speed.forEach((s, i) => col.set(speedColor(s / vmax), i * 3));
  geo.setAttribute("color", new THREE.BufferAttribute(col, 3));
  trajLine = new THREE.Line(geo, new THREE.LineBasicMaterial({ vertexColors: true }));
  scene.add(trajLine);
  $("k-traj").textContent = `${tr.length.toFixed(0)} m · ${tr.duration.toFixed(1)} s`;
}

// rayos de los telémetros
const RAYS = 40;
const rayGeo = new THREE.BufferGeometry();
rayGeo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(RAYS * 6), 3));
rayGeo.setAttribute("color", new THREE.BufferAttribute(new Float32Array(RAYS * 6), 3));
const rayLines = new THREE.LineSegments(rayGeo, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.4 }));
rayLines.frustumCulled = false;
scene.add(rayLines);
let rayData = [];
function drawRays(center) {
  const P = rayGeo.attributes.position.array, C = rayGeo.attributes.color.array;
  P.fill(0); C.fill(0);
  const range = profile ? profile.sensor_range : 12;
  rayData.slice(0, RAYS).forEach((r, i) => {
    const dir = V(r.end).sub(V(frame.pos)).normalize();
    const e = center.clone().addScaledVector(dir, r.dist);
    P.set([center.x, center.y, center.z, e.x, e.y, e.z], i * 6);
    const c = r.dist >= range - 0.01 ? [0.2, 0.28, 0.33] : r.dist < 1.5 ? [1, 0.25, 0.2] : r.dist < 4 ? [1, 0.75, 0.2] : [0.3, 0.9, 0.5];
    C.set([...c, ...c], i * 6);
  });
  rayGeo.attributes.position.needsUpdate = rayGeo.attributes.color.needsUpdate = true;
}

// ecos de la cámara de profundidad (lo que ve ahora mismo)
const DEPTH_N = 600;  // cámara frontal (336) + inferior (192)
const depthGeo = new THREE.BufferGeometry();
depthGeo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(DEPTH_N * 3), 3));
const depthPts = new THREE.Points(depthGeo, new THREE.PointsMaterial({ color: "#22d3ee", size: 0.14, transparent: true, opacity: 0.85 }));
depthPts.frustumCulled = false;
scene.add(depthPts);
function drawDepth(pts) {
  const arr = depthGeo.attributes.position.array, n = Math.min(pts.length, DEPTH_N);
  for (let i = 0; i < n; i++) { const p = V(pts[i]); arr.set([p.x, p.y, p.z], i * 3); }
  depthGeo.setDrawRange(0, n);
  depthGeo.attributes.position.needsUpdate = true;
}

// partículas de viento
const WP = 500;
const windGeo = new THREE.BufferGeometry();
windGeo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(WP * 6), 3));
const windLines = new THREE.LineSegments(windGeo, new THREE.LineBasicMaterial({ color: "#93c5fd", transparent: true, opacity: 0.35 }));
windLines.frustumCulled = false;
scene.add(windLines);
const particles = Array.from({ length: WP }, () => new THREE.Vector3());
function respawn(p, center) {
  const x = center.x + (Math.random() - 0.5) * 50, z = center.z + (Math.random() - 0.5) * 50;
  p.set(x, terrainH(x, -z) + 0.3 + Math.random() * 12, z);
}
let windField = null, windCfg = null;
function fieldFactor(x, y, agl) {  // factor del viento medio por la estela de obstáculos y el relieve
  if (!windField) return 1;
  const st = windField.step, L = windField.f[0].length, W = windField.f[0][0].length;
  const i = Math.min(L - 1, Math.max(0, Math.floor(x / st))), j = Math.min(W - 1, Math.max(0, Math.floor(y / st)));
  const hs = windField.heights;
  let k = 0;
  while (k < hs.length - 1 && agl > (hs[k] + hs[k + 1]) / 2) k++;
  return windField.f[k][i][j];
}
particles.forEach((p) => respawn(p, new THREE.Vector3(35, 0, -20)));
function updateWind(dt, center) {
  const speed = windCfg ? windCfg.speed : 0;
  windLines.visible = speed > 0.3;
  if (!windLines.visible) return;
  const ang = (windCfg.dir * Math.PI) / 180;
  const dir = V([Math.cos(ang), Math.sin(ang), 0]);
  const arr = windGeo.attributes.position.array;
  particles.forEach((p, i) => {
    const sx = p.x, sy = -p.z, agl = p.y - terrainH(sx, sy);
    const v = speed * fieldFactor(sx, sy, agl) * Math.max(0.35, Math.min(1.3, Math.log(Math.max(agl, 0.3) / 0.046) / Math.log(6.1 / 0.046)));
    const len = Math.min(2.5, 0.25 * v + 0.2);
    p.addScaledVector(dir, v * dt);
    if (Math.abs(p.x - center.x) > 25 || Math.abs(p.z - center.z) > 25) respawn(p, center);
    const q = p.clone().addScaledVector(dir, -len);
    arr.set([p.x, p.y, p.z, q.x, q.y, q.z], i * 6);
  });
  windGeo.attributes.position.needsUpdate = true;
}

// lluvia
const RP = 1500;
const rainGeo = new THREE.BufferGeometry();
rainGeo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(RP * 6), 3));
const rainLines = new THREE.LineSegments(rainGeo, new THREE.LineBasicMaterial({ color: "#cbd5e1", transparent: true, opacity: 0.35 }));
rainLines.frustumCulled = false;
rainLines.visible = false;
scene.add(rainLines);
const drops = Array.from({ length: RP }, () => new THREE.Vector3(Math.random() * 60, Math.random() * 25, -Math.random() * 60));
function updateRain(dt, center) {
  const level = frame && frame.config ? frame.config.rain : (world && world.rain) || "no";
  rainLines.visible = level && level !== "no";
  if (!rainLines.visible) return;
  const n = level === "fuerte" ? RP : RP / 2, speed = 9;
  const arr = rainGeo.attributes.position.array;
  arr.fill(0);
  for (let i = 0; i < n; i++) {
    const d = drops[i];
    d.y -= speed * dt;
    if (d.y < terrainH(d.x, -d.z) || Math.abs(d.x - center.x) > 25 || Math.abs(d.z - center.z) > 25) {
      d.set(center.x + (Math.random() - 0.5) * 50, center.y + 4 + Math.random() * 16, center.z + (Math.random() - 0.5) * 50);
    }
    arr.set([d.x, d.y, d.z, d.x, d.y + 0.5, d.z], i * 6);
  }
  rainGeo.attributes.position.needsUpdate = true;
}

// --- reproducción suave: cola de poses con tiempo, reloj propio, Hermite para la posición y slerp para la actitud
const poses = [];
let playClock = 0;
function poseFrom(f) {
  const ex = V(f.x_body), ey = V(f.z_body), ez = new THREE.Vector3().crossVectors(ex, ey).normalize();
  ex.crossVectors(ey, ez).normalize();
  const q = new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().makeBasis(ex, ey, ez));
  return { t: f.t, p: V(f.pos), v: V(f.vel), q, est: V(f.est), goal: f.goal_now, tv: f.target_vel };
}
function pushPose(f, reset) {
  const pose = poseFrom(f);
  if (reset) { poses.length = 0; playClock = f.t; }
  poses.push(pose);
  while (poses.length > 60) poses.shift();
}
function hermite(a, b, t) {
  const dt = Math.max(1e-3, b.t - a.t), s = (t - a.t) / dt, s2 = s * s, s3 = s2 * s;
  return new THREE.Vector3().addScaledVector(a.p, 2 * s3 - 3 * s2 + 1).addScaledVector(a.v, (s3 - 2 * s2 + s) * dt)
    .addScaledVector(b.p, -2 * s3 + 3 * s2).addScaledVector(b.v, (s3 - s2) * dt);
}
function sampleAt(t) {
  if (!poses.length) return null;
  if (t <= poses[0].t) return poses[0];
  const last = poses[poses.length - 1];
  if (t >= last.t) return last;
  let i = poses.length - 2;
  while (i > 0 && poses[i].t > t) i--;
  const a = poses[i], b = poses[i + 1], f = (t - a.t) / Math.max(1e-3, b.t - a.t);
  const goal = a.goal && b.goal ? a.goal.map((x, k) => x + (b.goal[k] - x) * f) : a.goal;
  return { t, p: hermite(a, b, t), q: a.q.clone().slerp(b.q, f), est: a.est.clone().lerp(b.est, f), goal, tv: b.tv };
}

// cámara + bucle de dibujo
function resize() {
  const w = els.viewport.clientWidth, h = els.viewport.clientHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
window.addEventListener("resize", resize);
resize();
let lastRender = performance.now(), camYaw = 0;
// calidad adaptativa: si el navegador va a menos de ~35 fps durante 2 s, baja resolución interna y sombras
const perf = { acc: 0, n: 0, level: 0 };
function adaptQuality(dt) {
  perf.acc += dt; perf.n++;
  if (perf.acc < 2) return;
  const fps = perf.n / perf.acc;
  perf.acc = 0; perf.n = 0;
  if (fps < 35 && perf.level < 2) {
    perf.level++;
    renderer.setPixelRatio(perf.level === 1 ? 1 : 0.75);
    sun.shadow.mapSize.set(perf.level === 1 ? 1024 : 512, perf.level === 1 ? 1024 : 512);
    if (sun.shadow.map) { sun.shadow.map.dispose(); sun.shadow.map = null; }
    resize();
  }
}
function animate() {
  requestAnimationFrame(animate);
  const now = performance.now(), dt = Math.min(0.1, (now - lastRender) / 1000);
  lastRender = now;
  adaptQuality(dt);
  const simRate = Number(els.speed.value);
  if (poses.length) {
    const newest = poses[poses.length - 1].t;
    const target = newest - 0.3;  // colchón: se reproduce 0,3 s por detrás de lo último recibido
    if (playing || streamEnded) {
      playClock = Math.min(newest, Math.max(playClock + dt * simRate * (playClock < target - 0.3 ? 1.4 : 1), poses[0].t));
    }
    while (pending.length && pending[0].t <= playClock + 1e-6) applyFrame(pending.shift());  // panel al ritmo del vídeo
    const s = sampleAt(playClock);
    drone.position.copy(s.p);
    drone.quaternion.copy(s.q);
    estMarker.position.copy(s.est);
    if (rover && s.goal) {  // el vehículo con la plataforma y la puerta encima
      const g = V(s.goal);
      rover.position.set(g.x, g.y - 0.6, g.z);
      if (s.tv && Math.hypot(s.tv[0], s.tv[1]) > 0.2) rover.rotation.y = Math.atan2(s.tv[1], s.tv[0]);
      if (gateMesh) gateMesh.position.set(g.x, g.y + 1.0, g.z);
      padMesh.position.set(g.x, g.y - 0.05, g.z);
    }
    shadowDot.position.set(s.p.x, terrainH(s.p.x, -s.p.z) + 0.03, s.p.z);
    drawRays(s.p);
    const fwd = new THREE.Vector3(1, 0, 0).applyQuaternion(s.q);
    let dy = Math.atan2(-fwd.z, fwd.x) - camYaw;
    dy = Math.atan2(Math.sin(dy), Math.cos(dy));
    camYaw += dy * (1 - Math.exp(-dt * 2.0));
  }
  (drone.userData.rotors || []).forEach((r, i) => { r.rotation.y += (i % 2 ? 1 : -1) * dt * 60; });
  updateWind(dt * simRate, drone.position);
  updateRain(dt * simRate, drone.position);
  worldGroup.children.forEach((c) => { if (c.userData.spin) c.rotation.y += dt * 0.8; });
  const mode = els.camera.value, dp = drone.position, kc = 1 - Math.exp(-dt * 3);
  controls.enabled = mode === "orbit";
  if (mode === "chase") {
    const dist = profile ? Math.max(5, profile.radius * 14) : 6;
    camera.position.lerp(dp.clone().add(new THREE.Vector3(-Math.cos(camYaw) * dist, dist * 0.45, Math.sin(camYaw) * dist)), kc);
    camera.lookAt(dp.clone().add(new THREE.Vector3(Math.cos(camYaw) * 4, 0.3, -Math.sin(camYaw) * 4)));
  } else if (mode === "top" && world) {
    camera.position.lerp(new THREE.Vector3(world.size[0] / 2, 62, -world.size[1] / 2 + 0.01), kc);
    camera.lookAt(world.size[0] / 2, 0, -world.size[1] / 2);
  } else {
    controls.target.lerp(dp, kc);
    controls.update();
  }
  renderer.render(scene, camera);
}
camera.position.set(-6, 8, -20);
animate();

// ------------------------------------------------------------------ panel y gráficas
const fmt = (v, d = 1, s = "") => (v === null || v === undefined || Number.isNaN(v) ? "—" : v.toFixed(d) + s);

function drawCompass(f) {
  const cv = $("compass-c"), ctx = cv.getContext("2d"), c = 48;
  ctx.clearRect(0, 0, 96, 96);
  ctx.strokeStyle = "rgba(255,255,255,.25)"; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.arc(c, c, 40, 0, 2 * Math.PI); ctx.stroke();
  ctx.fillStyle = "#93a1ad"; ctx.font = "10px Segoe UI"; ctx.textAlign = "center";
  ctx.fillText("x+", c + 34, c + 3);
  const arrow = (ang, len, color, width) => {
    ctx.save(); ctx.translate(c, c); ctx.rotate(-ang);
    ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = width;
    ctx.beginPath(); ctx.moveTo(-len, 0); ctx.lineTo(len - 6, 0); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(len, 0); ctx.lineTo(len - 9, 5); ctx.lineTo(len - 9, -5); ctx.fill();
    ctx.restore();
  };
  const w = f.wind, ws = Math.hypot(w[0], w[1]);
  if (ws > 0.2) arrow(Math.atan2(w[1], w[0]), Math.min(36, 10 + ws * 2.5), "#93c5fd", 3);
  arrow(f.yaw, 22, "#ff5a5a", 2);
  ctx.fillStyle = "#e6edf2"; ctx.font = "bold 11px Segoe UI";
  ctx.fillText(`${ws.toFixed(1)} m/s`, c, 92);
}

function lineChart(id, series, yMax, labelEl, label) {
  const cv = $(id), dpr = window.devicePixelRatio || 1, W = cv.clientWidth, H = 120;
  cv.width = W * dpr; cv.height = H * dpr;
  const ctx = cv.getContext("2d");
  ctx.scale(dpr, dpr);
  const css = getComputedStyle(document.documentElement);
  ctx.strokeStyle = css.getPropertyValue("--line"); ctx.lineWidth = 1;
  [0.25, 0.5, 0.75].forEach((y) => { ctx.beginPath(); ctx.moveTo(0, H - y * H); ctx.lineTo(W, H - y * H); ctx.stroke(); });
  if (telemetry.length < 2) return;
  const t0 = telemetry[0].t, t1 = telemetry[telemetry.length - 1].t, span = Math.max(1, t1 - t0);
  for (const s of series) {
    ctx.strokeStyle = s.color; ctx.lineWidth = s.width || 1.8; ctx.setLineDash(s.dash || []);
    ctx.beginPath();
    telemetry.forEach((p, i) => {
      const x = ((p.t - t0) / span) * W, y = H - 4 - Math.min(1, p[s.key] / yMax) * (H - 8);
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    });
    ctx.stroke();
  }
  ctx.setLineDash([]);
  labelEl.textContent = label;
}

function drawPanel(f) {
  $("h-time").textContent = f.t.toFixed(1) + " s";
  $("h-phase").textContent = PHASES[f.phase] || f.phase;
  $("h-speed").textContent = f.speed.toFixed(1) + " m/s";
  $("h-alt").textContent = f.pos[2].toFixed(1) + " m";
  $("h-tilt").textContent = f.tilt_deg.toFixed(0) + "°";
  $("h-wind").textContent = Math.hypot(f.wind[0], f.wind[1]).toFixed(1) + " m/s";
  $("h-bat").textContent = f.battery.toFixed(1) + " %";
  $("phase-pill").textContent = f.blocked ? "Buscando camino" : PHASES[f.phase] || f.phase;
  const unknown = f.map_mode === "desconocido";
  $("k-map").textContent = unknown ? "lo construye" : "conocido";
  $("k-explored").textContent = unknown ? fmt(100 * f.explored, 0, " %") : "—";
  $("k-mapreplan").textContent = unknown ? f.map_replans : "—";
  $("k-dist").textContent = fmt(f.distance, 0, " m");
  $("k-speed").textContent = `${f.speed.toFixed(1)} / ${f.ref_speed.toFixed(1)} m/s`;
  $("k-vz").textContent = fmt(f.vz, 1, " m/s");
  $("k-acc").textContent = fmt(Math.hypot(...f.acc), 1, " m/s²");
  $("k-thrust").textContent = fmt(f.thrust_g, 2, " g");
  $("k-est").textContent = fmt(f.est_err, 2, " m");
  $("k-rej").textContent = f.rejected;
  $("k-replan").textContent = f.replans;
  $("k-power").textContent = fmt(f.power_w, 0, " W");
  $("k-vmax").textContent = fmt(f.max_speed, 1, " m/s");
  const gn = f.goal_now || f.pad;
  $("k-tdist").textContent = fmt(Math.hypot(f.pos[0] - gn[0], f.pos[1] - gn[1]), 1, " m");
  $("k-tvel").textContent = f.moving ? fmt(Math.hypot(...f.target_vel), 1, " m/s") : "quieto";
  drawCompass(f);
  const vmax = profile ? profile.v_cruise * 1.3 : 10;
  lineChart("chart-speed", [{ key: "ref", color: "#93a1ad", dash: [4, 3] }, { key: "speed", color: "#19a7a0", width: 2.2 }],
    vmax, $("c-speed"), `real ${f.speed.toFixed(1)} · plan ${f.ref_speed.toFixed(1)} m/s (escala ${vmax.toFixed(0)})`);
  lineChart("chart-err", [{ key: "wind", color: "#93c5fd" }, { key: "err", color: "#f97316", width: 2.2 }],
    Math.max(3, profile ? profile.wind_max : 10) / 3, $("c-err"),
    `error ${f.est_err.toFixed(2)} m · viento ${Math.hypot(f.wind[0], f.wind[1]).toFixed(1)} m/s`);
  const R = f.results || [], n = R.length;
  const cnt = (s) => R.filter((r) => r.status === s).length;
  const pct = (k) => (n ? `${k} (${Math.round((100 * k) / n)}%)` : "—");
  $("r-n").textContent = n;
  $("r-ok").textContent = pct(cnt("success"));
  $("r-miss").textContent = pct(cnt("missed"));
  $("r-bad").textContent = pct(n - cnt("success") - cnt("missed"));
  const ok = R.filter((r) => r.status === "success");
  $("r-time").textContent = ok.length ? (ok.reduce((a, r) => a + r.time, 0) / ok.length).toFixed(1) + " s" : "—";
  const landed = R.filter((r) => r.status === "success" || r.status === "missed");
  $("r-land").textContent = landed.length ? (landed.reduce((a, r) => a + r.land, 0) / landed.length).toFixed(2) + " m" : "—";
  const races = R.filter((r) => r.mode === "carrera" && r.status === "success");
  $("r-best").textContent = races.length ? Math.min(...races.map((r) => r.time)).toFixed(2) + " s" : "—";
  const b = els.banner;
  if (f.status !== "flying") {
    const race = f.mode === "carrera";
    const txt = { success: race ? "¡Meta cruzada!" : "¡Aterrizaje en la plataforma!", missed: "Aterrizó fuera de la plataforma",
      crash: "¡Choque!", timeout: "Se acabó el tiempo" }[f.status];
    const land = Math.hypot(f.pos[0] - f.pad[0], f.pos[1] - f.pad[1]);
    const sub = f.status === "success"
      ? (race ? `${f.t.toFixed(2)} s · velocidad máxima ${f.max_speed.toFixed(1)} m/s` : `${f.t.toFixed(1)} s · a ${(land * 100).toFixed(0)} cm del centro · batería ${f.battery.toFixed(1)} %`)
      : (f.cause || "");
    b.innerHTML = `${txt}<small>${sub}</small>`;
    b.className = "banner " + (f.status === "success" ? "success" : f.status === "missed" ? "timeout" : "crash");
    setPlaying(false);
  } else {
    b.className = "banner hidden";
  }
}

function showProfile(p) {
  profile = p;
  const rows = [
    ["Masa", `${p.mass < 1 ? (p.mass * 1000).toFixed(0) + " g" : p.mass.toFixed(2) + " kg"}`],
    ["Vel. máx. / crucero", `${p.v_max} / ${p.v_cruise} m/s`],
    ["Subida / bajada", `${p.v_up} / ${p.v_down} m/s`],
    ["Aceleración / tirón", `${p.acc_hor} m/s² · ${p.jerk} m/s³`],
    ["Inclinación máx.", `${p.tilt_max_deg}°`],
    ["Empuje/peso", `${p.twr} (estimado)`],
    ["Batería", `${p.battery_wh.toFixed(1)} Wh · ${(60 * p.battery_wh / p.hover_power_w).toFixed(0)} min`],
    ["Viento que soporta", `${p.wind_max} m/s`],
    ["GPS (1σ)", `${p.gps_sigma} m`],
    ["Telémetros", `${p.sensor_range} m`],
  ];
  $("profile-kv").innerHTML = rows.map(([k, v]) => `<div><span>${k}</span><b>${v}</b></div>`).join("");
  $("profile-src").textContent = "Fuentes: fichas técnicas de DJI (Mini 3, Matrice 350 RTK) y parámetros por defecto de PX4. Ver dron/params.py.";
  buildDrone(p.radius);
}

// fotogramas recibidos que aún no se han mostrado (llegan hasta 1 s antes de su momento)
const pending = [];
let lastT = -1, streamEnded = false;
function ingest(f) {
  pushPose(f, false);
  pending.push(f);
  lastT = f.t;
  if (f.status !== "flying") streamEnded = true;
}

function applyFrame(f) {
  frame = f;
  rayData = f.rays || [];
  rayLines.visible = els.rays.checked;
  depthPts.visible = els.rays.checked && f.map_mode === "desconocido";
  if (f.depth_pts) drawDepth(f.depth_pts);
  if (f.map) drawMap(f.map);
  const showGps = els.gps.checked;
  estMarker.visible = gpsPts.visible = showGps;
  if (f.gps && (!lastGps || f.gps.join() !== lastGps)) {
    lastGps = f.gps.join();
    gpsList.push(V(f.gps));
    if (gpsList.length > GPS_N) gpsList.shift();
    const arr = gpsGeo.attributes.position.array;
    arr.fill(0);
    gpsList.forEach((p, i) => arr.set([p.x, p.y, p.z], i * 3));
    gpsGeo.setDrawRange(0, gpsList.length);
    gpsGeo.attributes.position.needsUpdate = true;
  }
  if (f.trajectory) drawTrajectory(f.trajectory);
  if (trackLine && f.target_track) {
    const pts = f.target_track.map((q) => V([q[0], q[1], terrainH(q[0], q[1]) + 0.08]));
    trackLine.geometry.dispose();
    trackLine.geometry = new THREE.BufferGeometry().setFromPoints(pts);
    trackLine.computeLineDistances();
  }
  interceptMarker.visible = !!(f.intercept && f.moving && f.phase === "carrera");
  if (interceptMarker.visible) interceptMarker.position.copy(V(f.intercept));
  telemetry.push({ t: f.t, speed: f.speed, ref: f.ref_speed, err: f.est_err, wind: Math.hypot(f.wind[0], f.wind[1]) });
  while (telemetry.length && telemetry[0].t < f.t - 40) telemetry.shift();
  drawPanel(f);
}

// ------------------------------------------------------------------ acciones
function config() {
  return { profile: els.profile.value, level: els.level.value, seed: els.seed.value.trim(),
    wind_speed: Number(els.wind.value), wind_dir: Number(els.wdir.value), gusts: Number(els.gusts.value),
    noise: els.noise.value, precision_landing: els.precision.checked, collision_prevention: els.cp.checked,
    mode: els.mode.value, terrain: els.terrain.value, density: els.density.value, goal_kind: els.goalKind.value,
    rain: els.rain.value, motion: els.motion.value, map_mode: els.mapMode.value };
}
async function newFlight() {
  setPlaying(false);
  try {
    const f = await api("/api/reset", config());
    showProfile(f.profile);
    frame = f;
    windField = f.wind_field;
    windCfg = { speed: f.config.wind_speed, dir: f.config.wind_dir };
    world = f.world;
    world.rain = f.config.rain;
    buildWorld(f.world);
    world.rain = f.config.rain;
    clearMap();
    depthGeo.setDrawRange(0, 0);
    telemetry.length = 0; gpsList.length = 0; lastGps = null;
    gpsGeo.setDrawRange(0, 0);
    pending.length = 0; lastT = f.t; streamEnded = false;
    pushPose(f, true);
    applyFrame(f);
    const dp = V(f.pos);
    camYaw = f.yaw;
    camera.position.copy(dp.clone().add(new THREE.Vector3(-Math.cos(camYaw) * 6, 3, Math.sin(camYaw) * 6)));
    controls.target.copy(dp);
  } catch (e) { toast(e.message); }
}
async function tick() {  // recoge los fotogramas que el servidor ya ha simulado por delante
  if (busy || !frame || streamEnded) return !streamEnded;
  busy = true;
  try {
    const r = await api("/api/frames", { since: lastT, clock: playClock });
    r.frames.forEach(ingest);
    return !streamEnded;
  } catch (e) { toast(e.message); setPlaying(false); return false; } finally { busy = false; }
}
function setPlaying(on) {
  playing = on;
  els.play.textContent = on ? "⏸ Pausa" : "▶ Volar";
  if (on) loop();
}
async function loop() {
  while (playing) {
    if (!(await tick())) break;
    await new Promise((r) => setTimeout(r, 60));
  }
}
els.newBtn.onclick = newFlight;
els.play.onclick = () => setPlaying(!playing);
for (const el of [els.profile, els.level, els.gusts, els.noise, els.precision, els.cp, els.mode, els.terrain,
  els.density, els.goalKind, els.rain, els.motion, els.mapMode]) el.onchange = newFlight;
els.view.onchange = applyView;
els.wind.oninput = () => { $("wind-v").textContent = `${els.wind.value} m/s`; };
els.wdir.oninput = () => { $("wdir-v").textContent = `${els.wdir.value}°`; };
els.wind.onchange = els.wdir.onchange = newFlight;
els.rays.onchange = () => {
  rayLines.visible = els.rays.checked;
  depthPts.visible = els.rays.checked && !!mapData;
};
document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
  if (e.key === " ") { e.preventDefault(); setPlaying(!playing); }
});

(async function init() {
  try { options = await api("/api/options"); } catch (e) { toast("No hay servidor: ejecuta  python server.py"); return; }
  for (const [k, p] of Object.entries(options.profiles)) {
    const o = document.createElement("option"); o.value = k; o.textContent = p.name; els.profile.appendChild(o);
  }
  for (const l of options.levels) {
    const o = document.createElement("option"); o.value = l; o.textContent = LEVEL_LABEL[l] || l; els.level.appendChild(o);
  }
  for (const n of options.noise) {
    const o = document.createElement("option"); o.value = n; o.textContent = NOISE_LABEL[n] || n; els.noise.appendChild(o);
  }
  const fill = (sel, values, labels) => values.forEach((v) => {
    const o = document.createElement("option"); o.value = v; o.textContent = labels[v] || v; sel.appendChild(o);
  });
  fill(els.mode, options.modes, LABELS.mode);
  fill(els.terrain, options.terrains, LABELS.terrain);
  fill(els.density, options.densities, LABELS.density);
  fill(els.goalKind, options.goal_kinds, LABELS.goal);
  fill(els.rain, options.rain, LABELS.rain);
  fill(els.motion, options.motions, LABELS.motion);
  fill(els.mapMode, options.map_modes, LABELS.map);
  els.mapMode.value = "desconocido";
  els.profile.value = "mini"; els.level.value = "mixto"; els.noise.value = "realista";
  els.terrain.value = "colinas"; els.density.value = "normal"; els.goalKind.value = "suelo"; els.rain.value = "no";
  els.motion.value = "fija";
  await newFlight();
})();
