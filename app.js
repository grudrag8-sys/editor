import * as THREE from 'three';

const $ = id => document.getElementById(id), V3 = THREE.Vector3, Q = THREE.Quaternion;
const MAX = 1800;
const MP = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14';
const MODELS = 'https://storage.googleapis.com/mediapipe-models/';
// One tracked frame = Float32Array(N): time, ok, hip xy, 33 pose world xyz, 33 visibility, left hand, right hand, face
const N = 273, OK = 1, HIP = 2, POSE = 4, VIS = 103, LH = 136, RH = 200, FACE = 264;
const video = $('video'), canvas = $('stage');
const say = m => { $('status').textContent = m; };
const fmt = s => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`;
let frames = [], vrm = null, stop = false, ac, audioDest, sm = null, wake = null;

/* ---------- scene ---------- */
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
const scene = new THREE.Scene(), camera = new THREE.PerspectiveCamera(35, 16 / 9, .1, 50);
scene.add(new THREE.HemisphereLight(0xffffff, 0x667788, 1.6));
const sun = new THREE.DirectionalLight(0xffffff, 2); sun.position.set(1, 2, 3); scene.add(sun);
const FR = { full: [4.2, .95], upper: [2.4, 1.25], head: [1.4, 1.5] };
function setup() {
  const [w, h] = $('size').value.split('x').map(Number), [d, cy] = FR[$('frame').value];
  renderer.setSize(w, h, false); camera.aspect = w / h;
  camera.position.set(0, cy, d); camera.lookAt(0, cy, 0); camera.updateProjectionMatrix();
  scene.background = new THREE.Color($('bg').value);
  const H = 2 * d * Math.tan(THREE.MathUtils.degToRad(17.5));
  return { H, W: H * camera.aspect, cy };
}

/* ---------- mannequin ---------- */
const man = new THREE.Group(); scene.add(man);
const mat = new THREE.MeshStandardMaterial({ color: 0xd9dee8, roughness: .55 });
const cylG = new THREE.CylinderGeometry(1, 1, 1, 10), UP = new V3(0, 1, 0);
const mk = r => { const m = new THREE.Mesh(cylG, mat); m.userData.r = r; man.add(m); return m; };
// indexes 33 = mid shoulders, 34 = mid hips, 35 = head centre
const SEG = [[33, 34, .12], [11, 12, .05], [23, 24, .06], [11, 13, .045], [13, 15, .04], [12, 14, .045], [14, 16, .04],
  [23, 25, .07], [25, 27, .055], [24, 26, .07], [26, 28, .055], [27, 31, .03], [28, 32, .03], [33, 35, .04]]
  .map(([a, b, r]) => ({ a, b, m: mk(r) }));
const HAND = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], [10, 11], [11, 12],
  [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]];
const hseg = [0, 1].map(() => HAND.map(([a, b]) => ({ a, b, m: mk(.008) })));
const head = new THREE.Mesh(new THREE.SphereGeometry(.11, 24, 16), mat); man.add(head);
const vec = (s, o, mir, fz) => new V3(mir ? -s[o] : s[o], -s[o + 1], fz ? s[o + 2] : -s[o + 2]);
function place(m, a, b) {
  const d = b.clone().sub(a), L = d.length(); m.visible = L > 1e-4; if (!m.visible) return;
  m.position.copy(a).addScaledVector(d, .5); m.scale.set(m.userData.r, L, m.userData.r);
  m.quaternion.setFromUnitVectors(UP, d.divideScalar(L));
}
function poseMannequin(s, root, mir, fz) {
  const P = []; for (let i = 0; i < 33; i++) P[i] = vec(s, POSE + i * 3, mir, fz).add(root);
  P[33] = P[11].clone().add(P[12]).multiplyScalar(.5); P[34] = P[23].clone().add(P[24]).multiplyScalar(.5);
  P[35] = P[7].clone().add(P[8]).multiplyScalar(.5); head.position.copy(P[35]);
  SEG.forEach(({ a, b, m }) => { place(m, P[a], P[b]); if ((a < 33 && s[VIS + a] < .4) || (b < 33 && s[VIS + b] < .4)) m.visible = false; });
  [[LH, 15], [RH, 16]].forEach(([o, wi], k) => hseg[k].forEach(({ a, b, m }) => {
    if (!(s[o] > 0)) { m.visible = false; return; }
    const w0 = vec(s, o + 1, mir, fz), h = i => vec(s, o + 1 + i * 3, mir, fz).sub(w0).add(P[wi]);
    place(m, h(a), h(b));
  }));
}

/* ---------- VRM avatar ---------- */
const bn = n => vrm.humanoid.getNormalizedBoneNode(n), pq = new Q(), tq = new Q();
function setWorld(n, qw, a) { // set a bone's world rotation
  if (!n) return; n.parent.updateWorldMatrix(true, false); n.parent.getWorldQuaternion(pq);
  n.quaternion.slerp(tq.copy(pq).invert().multiply(qw), a);
}
function aim(name, child, dir, a) { // point a bone at a world direction
  const n = bn(name); if (!n || dir.lengthSq() < 1e-8) return;
  const c = child && bn(child), rest = (c ? c.position : n.position).clone().normalize();
  n.parent.updateWorldMatrix(true, false); n.parent.getWorldQuaternion(pq);
  n.quaternion.slerp(tq.setFromUnitVectors(rest, dir.clone().normalize().applyQuaternion(pq.invert())), a);
}
function basis(xv, yv) {
  const x = xv.clone().normalize(), z = new V3().crossVectors(x, yv).normalize(), y = new V3().crossVectors(z, x);
  return new Q().setFromRotationMatrix(new THREE.Matrix4().makeBasis(x, y, z));
}
const P3 = ['Proximal', 'Intermediate', 'Distal'];
const FING = [['Thumb', 1, ['Metacarpal', 'Proximal', 'Distal']], ['Index', 5, P3], ['Middle', 9, P3], ['Ring', 13, P3], ['Little', 17, P3]];
function poseVRM(s, mir, fz, a) {
  const g = i => vec(s, POSE + i * 3, mir, fz);
  const PL = [11, 13, 15, 23, 25, 27, 31], PR = [12, 14, 16, 24, 26, 28, 32], AL = mir ? PR : PL, AR = mir ? PL : PR;
  const up = g(11).add(g(12)).sub(g(23)).sub(g(24));
  const hipsQ = basis(g(AL[3]).sub(g(AR[3])), up), shQ = basis(g(AL[0]).sub(g(AR[0])), up);
  setWorld(bn('hips'), hipsQ, a);
  const rel = hipsQ.clone().invert().multiply(shQ), half = new Q().slerp(rel, .5), sp = bn('spine'), ch = bn('chest');
  sp.quaternion.slerp(ch ? half : rel, a); if (ch) ch.quaternion.slerp(half, a);
  if (s[FACE] > 0) setWorld(bn('head'), new Q(s[FACE + 1], mir ? -s[FACE + 2] : s[FACE + 2], mir ? -s[FACE + 3] : s[FACE + 3], s[FACE + 4]).normalize(), a);
  for (const [C, S] of [['left', AL], ['right', AR]]) {
    const [sh, el, wr, hp, kn, an, ft] = S, o = S === PL ? LH : RH;
    aim(`${C}UpperArm`, `${C}LowerArm`, g(el).sub(g(sh)), a);
    aim(`${C}LowerArm`, `${C}Hand`, g(wr).sub(g(el)), a);
    if (s[VIS + kn] > .5) {
      aim(`${C}UpperLeg`, `${C}LowerLeg`, g(kn).sub(g(hp)), a); aim(`${C}LowerLeg`, `${C}Foot`, g(an).sub(g(kn)), a);
      aim(`${C}Foot`, `${C}Toes`, g(ft).sub(g(an)), a);
    } else { const d = new V3(0, -1, 0); aim(`${C}UpperLeg`, `${C}LowerLeg`, d, a); aim(`${C}LowerLeg`, `${C}Foot`, d, a); }
    if (s[o] > 0) {
      const h = i => vec(s, o + 1 + i * 3, mir, fz);
      aim(`${C}Hand`, `${C}MiddleProximal`, h(9).sub(h(0)), a);
      for (const [f, b, names] of FING) names.forEach((nm, j) => aim(`${C}${f}${nm}`, j < 2 ? `${C}${f}${names[j + 1]}` : null, h(b + j + 1).sub(h(b + j)), a));
    } else aim(`${C}Hand`, `${C}MiddleProximal`, g(wr + 4).sub(g(wr)), a);
  }
  const em = vrm.expressionManager;
  if (em && s[FACE] > 0) { em.setValue('blink', Math.max(s[FACE + 5], s[FACE + 6])); em.setValue('aa', s[FACE + 7]); em.setValue('happy', s[FACE + 8] * .8); }
}
$('vrmFile').onchange = async e => {
  const f = e.target.files[0]; if (!f) return; say('Loading avatar…');
  try {
    const { VRMLoaderPlugin, VRMUtils } = await import('@pixiv/three-vrm');
    const { GLTFLoader } = await import('three/addons/loaders/GLTFLoader.js');
    const l = new GLTFLoader(); l.register(p => new VRMLoaderPlugin(p));
    const gl = await l.loadAsync(URL.createObjectURL(f));
    if (vrm) scene.remove(vrm.scene);
    vrm = gl.userData.vrm; VRMUtils.rotateVRM0(vrm);
    const box = new THREE.Box3().setFromObject(vrm.scene), sc = 1.7 / (box.max.y - box.min.y);
    vrm.scene.scale.setScalar(sc); vrm.baseY = -box.min.y * sc; vrm.scene.position.y = vrm.baseY;
    scene.add(vrm.scene); $('avatar').value = 'vrm'; say('Avatar loaded.');
  } catch (err) { say('Could not load the avatar: ' + err.message); }
};

/* ---------- tracking (pass 1) ---------- */
const cv = document.createElement('canvas'); cv.width = cv.height = 256;
const cx = cv.getContext('2d', { willReadFrequently: true });
function crop(x, y, size) { cx.clearRect(0, 0, 256, 256); cx.drawImage(video, x - size / 2, y - size / 2, size, size, 0, 0, 256, 256); return cv; }
async function loadTrackers() {
  const mp = await import(`${MP}/vision_bundle.mjs`), fs = await mp.FilesetResolver.forVisionTasks(`${MP}/wasm`);
  const make = async (C, path, o = {}) => {
    for (const delegate of ['GPU', 'CPU']) {
      try { return await C.createFromOptions(fs, { baseOptions: { modelAssetPath: MODELS + path, delegate }, runningMode: 'VIDEO', ...o }); }
      catch (e) { if (delegate === 'CPU') throw e; }
    }
  };
  const q = $('fast').checked ? 'lite' : 'full', hand = 'hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';
  return {
    pose: await make(mp.PoseLandmarker, `pose_landmarker/pose_landmarker_${q}/float16/1/pose_landmarker_${q}.task`),
    hl: await make(mp.HandLandmarker, hand, { numHands: 1 }), hr: await make(mp.HandLandmarker, hand, { numHands: 1 }),
    face: await make(mp.FaceLandmarker, 'face_landmarker/face_landmarker/float16/1/face_landmarker.task',
      { outputFaceBlendshapes: true, outputFacialTransformationMatrixes: true })
  };
}
function track(T, t) {
  const f = new Float32Array(N); f[0] = t;
  const ts = performance.now(), W = video.videoWidth, H = video.videoHeight;
  const r = T.pose.detectForVideo(video, ts), w = r.worldLandmarks[0], im = r.landmarks[0];
  if (!w) return f;
  f[OK] = 1; f[HIP] = (im[23].x + im[24].x) / 2; f[HIP + 1] = (im[23].y + im[24].y) / 2;
  w.forEach((p, i) => { f.set([p.x, p.y, p.z], POSE + i * 3); f[VIS + i] = im[i].visibility ?? 0; });
  const px = i => [im[i].x * W, im[i].y * H];
  [[15, 13, LH, T.hl], [16, 14, RH, T.hr]].forEach(([wr, el, o, lm]) => { // small crop around each wrist helps in wide shots
    if (im[wr].visibility < .5) return;
    const [x, y] = px(wr), [ex, ey] = px(el), len = Math.hypot(x - ex, y - ey);
    const hw = lm.detectForVideo(crop(x + (x - ex) * .35, y + (y - ey) * .35, Math.max(120, len * 2.4)), ts).worldLandmarks[0];
    if (!hw) return; f[o] = 1; hw.forEach((p, i) => f.set([p.x, p.y, p.z], o + 1 + i * 3));
  });
  if (im[0].visibility > .5) {
    const ear = Math.hypot((im[7].x - im[8].x) * W, (im[7].y - im[8].y) * H), shd = Math.hypot((im[11].x - im[12].x) * W, (im[11].y - im[12].y) * H);
    const fr = T.face.detectForVideo(crop(im[0].x * W, im[0].y * H, Math.max(100, ear * 3, shd * .8)), ts);
    if (fr.faceBlendshapes[0] && fr.facialTransformationMatrixes[0]) {
      const c = Object.fromEntries(fr.faceBlendshapes[0].categories.map(x => [x.categoryName, x.score])), q = new Q();
      new THREE.Matrix4().fromArray(fr.facialTransformationMatrixes[0].data).decompose(new V3(), q, new V3());
      f[FACE] = 1; f.set([q.x, q.y, q.z, q.w, c.eyeBlinkLeft, c.eyeBlinkRight, c.jawOpen, (c.mouthSmileLeft + c.mouthSmileRight) / 2], FACE + 1);
    }
  }
  return f;
}
const busy = b => { ['trackBtn', 'previewBtn', 'recordBtn'].forEach(i => $(i).disabled = b); $('stopBtn').disabled = !b; };
async function lock() { try { wake = await navigator.wakeLock?.request('screen'); } catch { } }
$('trackBtn').onclick = async () => {
  if (!video.src) return say('Choose a video first.');
  busy(true); stop = false; say('Loading tracking models (first time only, about 30 MB)…');
  try {
    const T = await loadTrackers(); frames = []; sm = null;
    video.muted = true; video.playbackRate = +$('speed').value; video.currentTime = 0; await video.play(); await lock();
    await new Promise(done => {
      let last = -1; video.onended = done;
      const step = () => {
        if (stop || video.ended) return done();
        if (video.currentTime !== last && video.readyState > 1) {
          last = video.currentTime; frames.push(track(T, last)); $('bar').value = last / video.duration;
          say(`Tracking ${fmt(last)} of ${fmt(video.duration)}`);
        }
        requestAnimationFrame(step);
      };
      step();
    });
    say(frames.length ? `Tracking finished: ${frames.length} frames. Preview or render now.` : 'No frames tracked.');
  } catch (e) { say('Tracking failed: ' + e.message); }
  video.pause(); wake?.release(); busy(false);
};

/* ---------- rendering (pass 2) ---------- */
function sample(t) {
  if (!frames.length) return null;
  let lo = 0, hi = frames.length - 1; while (hi - lo > 1) { const m = (lo + hi) >> 1; frames[m][0] <= t ? lo = m : hi = m; }
  const a = frames[lo], b = frames[hi], k = b[0] > a[0] ? Math.min(1, Math.max(0, (t - a[0]) / (b[0] - a[0]))) : 0, n = k < .5 ? a : b;
  if (!(a[OK] && b[OK])) return n;
  const s = new Float32Array(N); for (let i = 0; i < N; i++) s[i] = a[i] + (b[i] - a[i]) * k;
  for (const [o, l] of [[LH, 64], [RH, 64], [FACE, 9]]) if (!(a[o] && b[o])) s.set(n.subarray(o, o + l), o);
  return s;
}
function drawAt(t, d, first, dt) {
  const s = sample(t), a = first ? 1 : 1 - +$('smooth').value, mir = $('mirror').checked, fz = $('flipz').checked, fol = $('follow').checked;
  const useV = !!vrm && $('avatar').value === 'vrm'; man.visible = !useV; if (vrm) vrm.scene.visible = useV;
  if (s && s[OK]) {
    if (!useV && sm && !first) for (let i = POSE; i < VIS + 33; i++) s[i] = sm[i] + (s[i] - sm[i]) * a;
    sm = s;
    const x = fol ? (s[HIP] - .5) * d.W * (mir ? -1 : 1) : 0, y = fol ? d.cy + (.5 - s[HIP + 1]) * d.H : .95;
    if (useV) { vrm.scene.position.set(x, vrm.baseY + y - .95, 0); poseVRM(s, mir, fz, a); } else poseMannequin(s, new V3(x, y, 0), mir, fz);
  }
  if (vrm) vrm.update(dt || .016);
  renderer.render(scene, camera);
}
async function play(rec) {
  if (!frames.length) return say('Track the motion first.');
  const d = setup(); busy(true); stop = false; sm = null;
  if (!ac) { ac = new AudioContext(); const src = ac.createMediaElementSource(video); audioDest = ac.createMediaStreamDestination(); src.connect(audioDest); src.connect(ac.destination); }
  await ac.resume(); await lock();
  let recorder, wr, fh, chunks = [], chain = Promise.resolve(), mime, name;
  if (rec) {
    mime = ['video/mp4;codecs=avc1', 'video/webm;codecs=vp9,opus', 'video/webm'].find(m => MediaRecorder.isTypeSupported(m));
    name = `avatar-${Date.now()}.${mime.includes('mp4') ? 'mp4' : 'webm'}`;
    try { const dir = await navigator.storage.getDirectory(); fh = await dir.getFileHandle(name, { create: true }); wr = await fh.createWritable(); } catch { wr = null; }
    const stream = canvas.captureStream(30); audioDest.stream.getAudioTracks().forEach(t => stream.addTrack(t));
    recorder = new MediaRecorder(stream, { mimeType: mime, videoBitsPerSecond: 6e6 });
    recorder.ondataavailable = e => { if (e.data.size) { if (wr) chain = chain.then(() => wr.write(e.data)); else chunks.push(e.data); } };
    recorder.start(1000);
  }
  video.muted = false; video.playbackRate = 1; video.currentTime = 0; await video.play();
  await new Promise(done => {
    let prev = performance.now(), first = true; video.onended = done;
    const step = () => {
      if (stop || video.ended) return done();
      const now = performance.now(); drawAt(video.currentTime, d, first, (now - prev) / 1000); first = false; prev = now;
      $('bar').value = video.currentTime / video.duration; say(`${rec ? 'Rendering' : 'Previewing'} ${fmt(video.currentTime)} of ${fmt(video.duration)}`);
      requestAnimationFrame(step);
    };
    step();
  });
  video.pause(); wake?.release();
  if (rec) {
    await new Promise(r => { recorder.onstop = r; recorder.stop(); }); await chain;
    if (wr) await wr.close(); else { const url = URL.createObjectURL(new Blob(chunks, { type: mime })), a = document.createElement('a'); a.href = url; a.download = name; a.textContent = 'Download ' + name; $('lib').append(a); }
    say('Saved. Find your video under "Saved videos on this device".'); lib();
  } else say('Preview finished.');
  busy(false);
}
async function lib() {
  const box = $('lib'); box.textContent = '';
  try {
    const dir = await navigator.storage.getDirectory();
    for await (const [n, h] of dir.entries()) {
      const file = await h.getFile(), row = document.createElement('div'), a = document.createElement('a'), b = document.createElement('button');
      a.href = URL.createObjectURL(file); a.download = n; a.textContent = `${n} (${(file.size / 1e6).toFixed(0)} MB)`;
      b.textContent = 'Delete'; b.className = 'alt'; b.onclick = async () => { await dir.removeEntry(n); lib(); };
      row.append(a, b); box.append(row);
    }
    if (!box.children.length) box.textContent = 'Nothing saved yet.';
  } catch { box.textContent = 'This browser has no private storage, so the download link appears after rendering.'; }
}
$('previewBtn').onclick = () => play(false);
$('recordBtn').onclick = () => play(true);
$('stopBtn').onclick = () => { stop = true; };
$('file').onchange = e => {
  const f = e.target.files[0]; if (!f) return; frames = [];
  video.src = URL.createObjectURL(f);
  video.onloadedmetadata = () => {
    if (video.duration > MAX) { say('This video is longer than 30 minutes. Please trim it first.'); video.removeAttribute('src'); video.load(); }
    else say(`Video loaded (${fmt(video.duration)}). Next, track the motion.`);
  };
};
busy(false); $('stopBtn').disabled = true; setup(); lib();
