// vr/static/xr-client.js
// Dependency-free WebXR + WebGL client for xArm6 digital-twin teleop.
//
// Responsibilities:
//   * Display: decode incoming JPEG frames (mono panel, or per-eye stereo)
//     and draw them into the immersive-vr session (and a flat browser-tab
//     preview when not in VR).
//   * Input: every XRFrame, read the head pose + both controllers' grip pose,
//     buttons and axes, and stream them up the WebSocket (throttled).
//
// Frame wire format (down): 1-byte eye prefix + JPEG.
//   0x00 = mono, 0x01 = left eye, 0x02 = right eye.
// Status (down): text JSON {"type":"status", ...}.
// Input (up): text JSON, see vr/teleop_receiver.py.

(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const enterBtn = $("enter");
  const statusEl = $("status");
  const hudEl = $("hud");
  const previewCanvas = $("preview");
  const glCanvas = $("glcanvas");

  let CFG = { mode: "mono", ipd: 0.063, render_hz: 30, control_hz: 60 };

  // ---- frame state (latest-wins ImageBitmaps) ----------------------------
  const frames = { mono: null, left: null, right: null };
  let serverStatus = null;

  // ---- websocket ---------------------------------------------------------
  let ws = null;
  function wsUrl() {
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    return `${proto}//${location.host}/ws`;
  }
  function connectWS() {
    ws = new WebSocket(wsUrl());
    ws.binaryType = "arraybuffer";
    ws.onopen = () => setStatus("WebSocket connected.");
    ws.onclose = () => setStatus("WebSocket closed.");
    ws.onerror = () => setStatus("WebSocket error.");
    ws.onmessage = onWSMessage;
  }
  async function onWSMessage(ev) {
    if (typeof ev.data === "string") {
      try {
        const msg = JSON.parse(ev.data);
        if (msg.type === "status") { serverStatus = msg; updateHud(); }
      } catch (_) { /* ignore */ }
      return;
    }
    // Binary frame: first byte = eye prefix, rest = JPEG.
    const buf = new Uint8Array(ev.data);
    const eye = buf[0];
    const blob = new Blob([buf.subarray(1)], { type: "image/jpeg" });
    let bmp;
    try { bmp = await createImageBitmap(blob); } catch (_) { return; }
    if (eye === 0) { closeBmp(frames.mono); frames.mono = bmp; drawPreview(bmp); }
    else if (eye === 1) { closeBmp(frames.left); frames.left = bmp; }
    else if (eye === 2) { closeBmp(frames.right); frames.right = bmp; drawPreview(bmp); }
  }
  function closeBmp(b) { if (b && b.close) { try { b.close(); } catch (_) {} } }

  // ---- flat browser-tab preview (no headset needed) ----------------------
  const prevCtx = previewCanvas.getContext("2d");
  function drawPreview(bmp) {
    if (document.querySelector("#glcanvas").dataset.xr === "1") return; // in VR
    previewCanvas.width = bmp.width; previewCanvas.height = bmp.height;
    prevCtx.drawImage(bmp, 0, 0);
  }

  function setStatus(s) { statusEl.textContent = s; }
  function updateHud() {
    if (!serverStatus) { hudEl.textContent = ""; return; }
    const s = serverStatus;
    const rec = s.recording ? "● REC" : "○ idle";
    const grip = s.gripper_closed ? "grip:CLOSED" : "grip:open";
    const ik = s.ik_fail ? " IK-FAIL" : "";
    const cl = s.clutch ? " [clutch]" : "";
    hudEl.textContent = `${rec}  ${grip}  rail:${s.rail_mm}mm  ${s.servo_mode}${cl}${ik}`;
  }

  // ===========================================================================
  // WebGL: textured-quad renderer for the immersive session
  // ===========================================================================
  let gl = null;
  let prog = null, aPos = null, aUV = null, uTex = null;
  let quadBuf = null;
  const glTex = { mono: null, left: null, right: null };
  const texDirty = { mono: true, left: true, right: true };

  // HUD rendered to a 2D canvas, uploaded as a texture, drawn as a small quad.
  const hudCanvas = document.createElement("canvas");
  hudCanvas.width = 512; hudCanvas.height = 64;
  const hudCtx = hudCanvas.getContext("2d");
  let hudTex = null;

  const VS = `
    attribute vec2 aPos; attribute vec2 aUV; varying vec2 vUV;
    void main() { vUV = aUV; gl_Position = vec4(aPos, 0.0, 1.0); }`;
  const FS = `
    precision mediump float; varying vec2 vUV; uniform sampler2D uTex;
    void main() { gl_FragColor = texture2D(uTex, vUV); }`;

  function compile(type, src) {
    const s = gl.createShader(type);
    gl.shaderSource(s, src); gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS))
      throw new Error(gl.getShaderInfoLog(s));
    return s;
  }
  function initGL(glContext) {
    gl = glContext;
    prog = gl.createProgram();
    gl.attachShader(prog, compile(gl.VERTEX_SHADER, VS));
    gl.attachShader(prog, compile(gl.FRAGMENT_SHADER, FS));
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS))
      throw new Error(gl.getProgramInfoLog(prog));
    aPos = gl.getAttribLocation(prog, "aPos");
    aUV = gl.getAttribLocation(prog, "aUV");
    uTex = gl.getUniformLocation(prog, "uTex");
    quadBuf = gl.createBuffer();
    for (const k of ["mono", "left", "right"]) glTex[k] = newTex();
    hudTex = newTex();
  }
  function newTex() {
    const t = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    return t;
  }
  function uploadIfNeeded(key, bmp) {
    if (!bmp) return false;
    gl.bindTexture(gl.TEXTURE_2D, glTex[key]);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB, gl.RGB, gl.UNSIGNED_BYTE, bmp);
    return true;
  }

  // Draw a textured quad into a sub-rect of the *current* viewport. The rect
  // (rx,ry,rw,rh) is in normalized [0,1] viewport coords; default = full.
  function drawQuad(tex, rx = 0, ry = 0, rw = 1, rh = 1) {
    // clip-space corners for the sub-rect
    const x0 = rx * 2 - 1, x1 = (rx + rw) * 2 - 1;
    const y0 = ry * 2 - 1, y1 = (ry + rh) * 2 - 1;
    const verts = new Float32Array([
      // aPos      aUV
      x0, y0, 0, 0, x1, y0, 1, 0, x0, y1, 0, 1,
      x0, y1, 0, 1, x1, y0, 1, 0, x1, y1, 1, 1,
    ]);
    gl.bindBuffer(gl.ARRAY_BUFFER, quadBuf);
    gl.bufferData(gl.ARRAY_BUFFER, verts, gl.DYNAMIC_DRAW);
    gl.enableVertexAttribArray(aPos);
    gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 16, 0);
    gl.enableVertexAttribArray(aUV);
    gl.vertexAttribPointer(aUV, 2, gl.FLOAT, false, 16, 8);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.uniform1i(uTex, 0);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
  }

  function renderHudTex() {
    const s = serverStatus;
    hudCtx.clearRect(0, 0, hudCanvas.width, hudCanvas.height);
    hudCtx.fillStyle = "rgba(0,0,0,0.45)";
    hudCtx.fillRect(0, 0, hudCanvas.width, hudCanvas.height);
    hudCtx.font = "26px monospace";
    hudCtx.textBaseline = "middle";
    if (s) {
      hudCtx.fillStyle = s.recording ? "#f55" : "#6f8";
      const rec = s.recording ? "● REC" : "○ idle";
      const grip = s.gripper_closed ? "CLOSED" : "open";
      const ik = s.ik_fail ? " IK!" : "";
      hudCtx.fillText(`${rec}  grip:${grip}  rail:${s.rail_mm}mm${ik}`,
                      12, hudCanvas.height / 2);
    } else {
      hudCtx.fillStyle = "#6f8";
      hudCtx.fillText("connecting…", 12, hudCanvas.height / 2);
    }
    gl.bindTexture(gl.TEXTURE_2D, hudTex);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, hudCanvas);
  }

  // ===========================================================================
  // XR session
  // ===========================================================================
  let xrSession = null, xrRefSpace = null, xrGLLayer = null;
  let lastSendT = 0;
  const SEND_MIN_DT = 1 / 72;

  async function enterVR() {
    if (!navigator.xr) { setStatus("WebXR not available."); return; }
    try {
      xrSession = await navigator.xr.requestSession("immersive-vr", {
        requiredFeatures: ["local-floor"],
      });
    } catch (e) {
      setStatus("Could not start VR session: " + e.message);
      return;
    }
    $("ui").style.display = "none";
    glCanvas.dataset.xr = "1";

    const glCtx = glCanvas.getContext("webgl", {
      xrCompatible: true, alpha: false, antialias: true,
    });
    initGL(glCtx);

    xrGLLayer = new XRWebGLLayer(xrSession, gl);
    xrSession.updateRenderState({ baseLayer: xrGLLayer });
    xrRefSpace = await xrSession.requestReferenceSpace("local-floor");

    if (!ws || ws.readyState !== WebSocket.OPEN) connectWS();

    xrSession.addEventListener("end", onSessionEnd);
    xrSession.requestAnimationFrame(onXRFrame);
    setStatus("In VR.");
  }

  function onSessionEnd() {
    glCanvas.dataset.xr = "0";
    $("ui").style.display = "flex";
    setStatus("VR session ended.");
    xrSession = null;
  }

  function poseToObj(pose) {
    const p = pose.transform.position, o = pose.transform.orientation;
    return { pos: [p.x, p.y, p.z], quat: [o.x, o.y, o.z, o.w] };
  }

  function gatherInput(frame) {
    const viewerPose = frame.getViewerPose(xrRefSpace);
    const state = {};
    if (viewerPose) state.head = poseToObj(viewerPose);
    for (const src of xrSession.inputSources) {
      if (!src.gripSpace || (src.handedness !== "left" && src.handedness !== "right"))
        continue;
      const gp = frame.getPose(src.gripSpace, xrRefSpace);
      if (!gp) continue;
      const entry = poseToObj(gp);
      const pad = src.gamepad;
      entry.buttons = pad ? pad.buttons.map((b) => !!b.pressed) : [];
      entry.axes = pad ? Array.from(pad.axes) : [];
      state[src.handedness] = entry;
    }
    return state;
  }

  function onXRFrame(t, frame) {
    const session = frame.session;
    session.requestAnimationFrame(onXRFrame);

    const pose = frame.getViewerPose(xrRefSpace);
    gl.bindFramebuffer(gl.FRAMEBUFFER, xrGLLayer.framebuffer);
    gl.clearColor(0, 0, 0, 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    if (pose) {
      gl.useProgram(prog);
      // Refresh textures from the newest frames.
      uploadIfNeeded("mono", frames.mono);
      uploadIfNeeded("left", frames.left);
      uploadIfNeeded("right", frames.right);
      renderHudTex();

      const views = pose.views;
      for (let i = 0; i < views.length; i++) {
        const view = views[i];
        const vp = xrGLLayer.getViewport(view);
        gl.viewport(vp.x, vp.y, vp.width, vp.height);

        let tex;
        if (CFG.mode === "stereo") {
          // eye/index -> matching eye texture (fall back to mono if missing)
          if (view.eye === "right" || i === 1)
            tex = frames.right ? glTex.right : glTex.mono;
          else
            tex = frames.left ? glTex.left : glTex.mono;
        } else {
          tex = glTex.mono;
        }
        drawQuad(tex);                         // full head-locked panel
        drawQuad(hudTex, 0.30, 0.04, 0.40, 0.10); // HUD strip near the bottom
      }
    }

    // Stream input up (throttled).
    const now = t / 1000;
    if (ws && ws.readyState === WebSocket.OPEN && (now - lastSendT) >= SEND_MIN_DT) {
      lastSendT = now;
      const state = gatherInput(frame);
      try { ws.send(JSON.stringify(state)); } catch (_) {}
    }
  }

  // ===========================================================================
  // boot
  // ===========================================================================
  async function loadConfig() {
    try {
      const r = await fetch("/config.json");
      CFG = Object.assign(CFG, await r.json());
    } catch (_) { /* defaults */ }
  }

  async function boot() {
    await loadConfig();
    connectWS();               // start receiving frames immediately (tab preview)
    if (!navigator.xr) {
      enterBtn.textContent = "WebXR not supported";
      setStatus("Open this page in the Meta Quest Browser to enter VR. "
                + "This tab shows the live mono feed.");
      return;
    }
    let supported = false, why = "";
    try { supported = await navigator.xr.isSessionSupported("immersive-vr"); }
    catch (e) { why = `${e.name}: ${e.message}`; }
    if (supported) {
      enterBtn.disabled = false;
      enterBtn.textContent = `Enter VR (${CFG.mode})`;
      enterBtn.addEventListener("click", enterVR);
      setStatus("Ready. Put on the headset and press Enter VR.");
    } else {
      enterBtn.textContent = "immersive-vr unavailable";
      const secure = window.isSecureContext;
      setStatus(
        `immersive-vr not available.  xr=${!!navigator.xr}  `
        + `secureContext=${secure}  `
        + (why ? `error=[${why}]` : "isSessionSupported→false")
        + `.  If the headset is in Link/PCVR mode, fully QUIT Link `
        + `(it holds the VR compositor), then reload.`);
    }
  }

  boot();
})();
