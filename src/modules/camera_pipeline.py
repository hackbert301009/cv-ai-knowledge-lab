"""Wie eine Kamera ein Bild aufnimmt und verarbeitet."""
import numpy as np
import cv2
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from src.components import (
    hero,
    section_header,
    divider,
    info_box,
    lab_header,
    key_concept,
    video_embed,
    render_learning_block,
)


def _demo_scene(size: int = 320) -> np.ndarray:
    """Erzeugt ein synthetisches RGB-Testbild."""
    x = np.linspace(0, 1, size, dtype=np.float32)
    y = np.linspace(0, 1, size, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)

    base = np.zeros((size, size, 3), dtype=np.float32)
    base[..., 0] = 0.15 + 0.75 * xx
    base[..., 1] = 0.10 + 0.70 * yy
    base[..., 2] = 0.20 + 0.65 * (1.0 - xx * yy)

    # Geometrische Strukturen mit verschiedenen Frequenzen
    cv2.rectangle(base, (18, 18), (145, 145), (0.92, 0.25, 0.15), -1)
    cv2.circle(base, (230, 90), 55, (0.12, 0.92, 0.88), -1)
    cv2.line(base, (0, 260), (size, 260), (1.0, 1.0, 1.0), 3)
    cv2.putText(base, "CAM", (155, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0.95, 0.95, 0.2), 2, cv2.LINE_AA)

    # Feinmuster fuer Aliasing/Noise-Effekt
    pattern = (np.sin(xx * 90) * np.sin(yy * 90) > 0.0).astype(np.float32)
    base[..., 1] = np.clip(base[..., 1] + 0.14 * pattern, 0.0, 1.0)

    return np.clip(base * 255.0, 0, 255).astype(np.uint8)


def _radial_vignette(h: int, w: int, strength: float) -> np.ndarray:
    yy, xx = np.indices((h, w), dtype=np.float32)
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    rr = np.sqrt(((xx - cx) / max(1.0, cx)) ** 2 + ((yy - cy) / max(1.0, cy)) ** 2)
    mask = 1.0 - strength * np.clip(rr, 0.0, 1.0) ** 2
    return np.clip(mask, 0.05, 1.0)


def _simulate_camera_pipeline(
    rgb_input: np.ndarray,
    *,
    blur_sigma: float,
    vignette_strength: float,
    exposure_mult: float,
    iso: int,
    read_noise_e: float,
    bit_depth: int,
    wb_red_gain: float,
    wb_blue_gain: float,
    seed: int,
) -> dict[str, np.ndarray]:
    """Vereinfachte, aber physikalisch motivierte Kamera-Simulation."""
    rgb_gamma = rgb_input.astype(np.float32) / 255.0
    scene_linear = np.clip(rgb_gamma, 0.0, 1.0) ** 2.2  # inverse Display-Gamma

    optics_linear = cv2.GaussianBlur(scene_linear, (0, 0), sigmaX=max(0.0, blur_sigma))
    vignette = _radial_vignette(scene_linear.shape[0], scene_linear.shape[1], vignette_strength)
    optics_linear = np.clip(optics_linear * vignette[..., None], 0.0, 1.0)

    exposed = np.clip(optics_linear * exposure_mult, 0.0, 1.5)

    full_well = 16000.0  # Elektronen
    quantum_eff = 0.55   # Anteil Photonen -> Elektronen
    mean_electrons = np.clip(exposed * full_well * quantum_eff, 0.0, None)

    rng = np.random.default_rng(seed)
    shot_noise = rng.poisson(mean_electrons).astype(np.float32)
    read_noise = rng.normal(0.0, read_noise_e, mean_electrons.shape).astype(np.float32)
    sensor_electrons = np.clip(shot_noise + read_noise, 0.0, full_well)

    analog_gain = iso / 100.0
    gained_e = np.clip(sensor_electrons * analog_gain, 0.0, full_well)
    sensor_linear = np.clip(gained_e / full_well, 0.0, 1.0)

    # Bayer RGGB-Mosaik (ein Kanal pro Pixel).
    bayer = np.zeros(sensor_linear.shape[:2], dtype=np.float32)
    bayer[0::2, 0::2] = sensor_linear[0::2, 0::2, 0]  # R
    bayer[0::2, 1::2] = sensor_linear[0::2, 1::2, 1]  # G
    bayer[1::2, 0::2] = sensor_linear[1::2, 0::2, 1]  # G
    bayer[1::2, 1::2] = sensor_linear[1::2, 1::2, 2]  # B

    adc_max = float((2 ** bit_depth) - 1)
    bayer_adc = np.clip(np.round(bayer * adc_max), 0.0, adc_max).astype(np.uint16)

    demosaic = cv2.cvtColor(bayer_adc, cv2.COLOR_BAYER_RG2RGB).astype(np.float32) / adc_max

    # Vereinfachtes ISP: White Balance + Gamma + 8-bit Ausgabe.
    demosaic[..., 0] *= wb_red_gain
    demosaic[..., 2] *= wb_blue_gain
    isp_linear = np.clip(demosaic, 0.0, 1.0)
    isp_gamma = np.clip(isp_linear, 0.0, 1.0) ** (1.0 / 2.2)
    out_rgb = np.clip(np.round(isp_gamma * 255.0), 0.0, 255.0).astype(np.uint8)

    return {
        "scene_gamma": rgb_input,
        "scene_linear_vis": np.clip(scene_linear ** (1.0 / 2.2) * 255.0, 0.0, 255.0).astype(np.uint8),
        "optics_vis": np.clip(optics_linear ** (1.0 / 2.2) * 255.0, 0.0, 255.0).astype(np.uint8),
        "sensor_gray": np.clip(bayer * 255.0, 0.0, 255.0).astype(np.uint8),
        "output_rgb": out_rgb,
        "mean_electrons": mean_electrons,
        "sensor_electrons": sensor_electrons,
    }


def _render_pipeline_animation(*, stage_index: int, autoplay: bool, speed: float, cinematic: bool):
    """Animierter Flow mit Anime.js und Step-Mode."""
    html_template = """
    <div id="cam-flow" class="__CINE_CLASS__" style="font-family: Inter, system-ui, sans-serif; color: #E5E7EB;">
      <style>
        #cam-flow { background: linear-gradient(140deg, #0b1222 0%, #15112b 60%, #1f1230 100%);
                    border: 1px solid rgba(255,255,255,0.12); border-radius: 14px; padding: 14px; }
        #cam-flow.cinematic {
          background: radial-gradient(circle at 20% 0%, #25123f 0%, #111827 45%, #070b16 100%);
          border-color: rgba(167,139,250,0.45);
          box-shadow: 0 8px 38px rgba(124,58,237,0.28), inset 0 0 35px rgba(236,72,153,0.12);
          padding: 20px;
        }
        #cam-flow .toolbar { display: flex; justify-content: space-between; gap: 8px; margin-bottom: 10px; align-items: center; }
        #cam-flow .status { font-size: 0.75rem; color: #C4B5FD; background: rgba(124,58,237,0.15);
                            border: 1px solid rgba(124,58,237,0.35); border-radius: 999px; padding: 3px 10px; }
        #cam-flow .play-btn { border: 1px solid rgba(255,255,255,0.20); background: rgba(17,24,39,0.7);
                              color: #E5E7EB; border-radius: 8px; padding: 5px 10px; cursor: pointer; font-size: 0.78rem; }
        #cam-flow .stage-row { display: grid; grid-template-columns: repeat(5, minmax(110px, 1fr)); gap: 10px; align-items: center; }
        #cam-flow .stage {
          background: rgba(255,255,255,0.06);
          border: 1px solid rgba(255,255,255,0.12);
          border-radius: 12px;
          padding: 10px 8px;
          text-align: center;
          min-height: 82px;
          position: relative;
          overflow: hidden;
          transition: border-color 220ms ease, box-shadow 220ms ease, transform 220ms ease;
        }
        #cam-flow.cinematic .stage {
          min-height: 96px;
          background: rgba(255,255,255,0.08);
          border-color: rgba(255,255,255,0.20);
        }
        #cam-flow.cinematic .stage .ico { font-size: 1.4rem; }
        #cam-flow.cinematic .stage .ttl { font-size: 0.95rem; }
        #cam-flow.cinematic .stage .sub { font-size: 0.78rem; }
        #cam-flow .stage .ico { font-size: 1.2rem; margin-bottom: 0.25rem; display: block; }
        #cam-flow .stage .ttl { font-weight: 700; font-size: 0.88rem; }
        #cam-flow .stage .sub { color: #9CA3AF; font-size: 0.72rem; margin-top: 0.15rem; }
        #cam-flow .stage.active { border-color: rgba(236,72,153,0.8); box-shadow: 0 0 18px rgba(236,72,153,0.25); transform: translateY(-1px); }
        #cam-flow.cinematic .stage.active {
          border-color: rgba(245,158,11,0.92);
          box-shadow: 0 0 28px rgba(245,158,11,0.35), 0 0 44px rgba(236,72,153,0.24);
        }
        #cam-flow .arrow { text-align: center; opacity: 0.7; font-size: 1rem; }
        #cam-flow .pulse-bg {
          position: absolute; left: -20%; top: 0; width: 40%;
          height: 100%; background: linear-gradient(90deg, transparent, rgba(124,58,237,0.28), transparent);
          animation: movePulse 2.6s linear infinite;
        }
        #cam-flow .stage:nth-child(2) .pulse-bg { animation-delay: .2s; }
        #cam-flow .stage:nth-child(4) .pulse-bg { animation-delay: .4s; }
        #cam-flow .stage:nth-child(6) .pulse-bg { animation-delay: .6s; }
        #cam-flow .stage:nth-child(8) .pulse-bg { animation-delay: .8s; }
        #cam-flow .stage:nth-child(10) .pulse-bg { animation-delay: 1.0s; }
        @keyframes movePulse { 0% { left: -30%; } 100% { left: 130%; } }
        #beam-wrap { width: 100%; height: 42px; margin-top: 10px; border-radius: 8px; overflow: hidden;
                     background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.09); position: relative; }
        #cam-flow.cinematic #beam-wrap {
          height: 56px;
          border-color: rgba(255,255,255,0.2);
          box-shadow: inset 0 0 18px rgba(167,139,250,0.25);
        }
        #beam-wrap .photon {
          position: absolute; left: -12px; width: 6px; height: 6px; border-radius: 50%;
          background: radial-gradient(circle at 30% 30%, #fff7c0 0%, #f59e0b 38%, #ec4899 100%);
          box-shadow: 0 0 10px rgba(245,158,11,0.7);
          opacity: 0.7;
        }
        #cam-flow.cinematic #beam-wrap .photon {
          width: 8px;
          height: 8px;
          box-shadow: 0 0 14px rgba(245,158,11,0.95), 0 0 22px rgba(236,72,153,0.45);
        }
      </style>
      <div class="toolbar">
        <div class="status" id="stage-status">Stage 1/5</div>
        <div style="display:flex; gap:6px; align-items:center;">
          <button class="play-btn" id="play-btn">Pause</button>
          <button class="play-btn" id="fs-btn">Vollbild</button>
        </div>
      </div>
      <div class="stage-row">
        <div class="stage"><span class="ico">🌞</span><div class="ttl">Szene</div><div class="sub">Reflektiertes Licht</div><div class="pulse-bg"></div></div>
        <div class="arrow">➜</div>
        <div class="stage"><span class="ico">🔍</span><div class="ttl">Optik</div><div class="sub">PSF, Fokus, Vignette</div><div class="pulse-bg"></div></div>
        <div class="arrow">➜</div>
        <div class="stage"><span class="ico">🧪</span><div class="ttl">Sensor</div><div class="sub">Photonen -> Elektronen</div><div class="pulse-bg"></div></div>
        <div class="arrow">➜</div>
        <div class="stage"><span class="ico">🎨</span><div class="ttl">Bayer + ADC</div><div class="sub">Mosaik + Quantisierung</div><div class="pulse-bg"></div></div>
        <div class="arrow">➜</div>
        <div class="stage"><span class="ico">🧠</span><div class="ttl">ISP</div><div class="sub">WB, Gamma, Scharf</div><div class="pulse-bg"></div></div>
      </div>
      <div id="beam-wrap"></div>
    </div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/animejs/3.2.1/anime.min.js"></script>
    <script>
      const stages = Array.from(document.querySelectorAll("#cam-flow .stage"));
      const statusEl = document.getElementById("stage-status");
      const playBtn = document.getElementById("play-btn");
      const fsBtn = document.getElementById("fs-btn");
      const beam = document.getElementById("beam-wrap");
      const speed = __SPEED__;
      const cinematic = __CINE__;
      let current = Math.max(0, Math.min(stages.length - 1, __STAGE__));
      let playing = __PLAY__;
      playBtn.textContent = playing ? "Pause" : "Play";

      function renderStage(idx) {
        stages.forEach((el, i) => el.classList.toggle("active", i === idx));
        statusEl.textContent = `Stage ${idx + 1}/${stages.length}`;
      }

      for (let i = 0; i < 34; i += 1) {
        const p = document.createElement("span");
        p.className = "photon";
        p.style.top = `${4 + Math.random() * 30}px`;
        beam.appendChild(p);
      }

      const photons = Array.from(beam.querySelectorAll(".photon"));
      if (window.anime) {
        anime({
          targets: photons,
          translateX: [0, beam.clientWidth + 22],
          opacity: [{ value: 0.25 }, { value: 1.0 }, { value: 0.2 }],
          delay: anime.stagger(50, { start: 0 }),
          duration: () => ((cinematic ? 1650 : 1300) / speed) + Math.random() * (cinematic ? 1200 : 900),
          easing: "linear",
          loop: true,
        });
      }

      function pulseCurrent() {
        if (!window.anime) {
          return;
        }
        anime.remove(stages[current]);
        anime({
          targets: stages[current],
          scale: [1.0, 1.05, 1.0],
          borderColor: ["rgba(236,72,153,0.8)", "rgba(245,158,11,0.95)", "rgba(236,72,153,0.8)"],
          boxShadow: [
            "0 0 18px rgba(236,72,153,0.25)",
            "0 0 26px rgba(245,158,11,0.35)",
            "0 0 18px rgba(236,72,153,0.25)"
          ],
          easing: "easeInOutSine",
          duration: (cinematic ? 1200 : 850) / speed,
        });
      }

      function stepNext() {
        current = (current + 1) % stages.length;
        renderStage(current);
        pulseCurrent();
      }

      renderStage(current);
      pulseCurrent();
      const cadenceMs = Math.max(450, (cinematic ? 1900 : 1450) / speed);
      const timer = setInterval(() => {
        if (playing) {
          stepNext();
        }
      }, cadenceMs);

      playBtn.addEventListener("click", () => {
        playing = !playing;
        playBtn.textContent = playing ? "Pause" : "Play";
        if (playing) {
          pulseCurrent();
        }
      });

      function inFullscreen() {
        return !!document.fullscreenElement;
      }

      function updateFsLabel() {
        fsBtn.textContent = inFullscreen() ? "Exit Vollbild" : "Vollbild";
      }

      fsBtn.addEventListener("click", async () => {
        try {
          if (!inFullscreen()) {
            await document.getElementById("cam-flow").requestFullscreen();
          } else {
            await document.exitFullscreen();
          }
        } catch (err) {
          // Fallback: Popout in neuem Fenster, falls Fullscreen im Embed blockiert ist.
          const w = window.open("", "_blank", "noopener,noreferrer,width=1280,height=800");
          if (!w) {
            return;
          }
          const html = document.documentElement.outerHTML;
          w.document.open();
          w.document.write(html);
          w.document.close();
        }
        updateFsLabel();
      });
      document.addEventListener("fullscreenchange", updateFsLabel);
      updateFsLabel();

      window.addEventListener("beforeunload", () => clearInterval(timer));
    </script>
    """
    html = (
        html_template.replace("__STAGE__", str(max(0, stage_index)))
        .replace("__PLAY__", "true" if autoplay else "false")
        .replace("__SPEED__", f"{max(0.4, speed):.2f}")
        .replace("__CINE__", "true" if cinematic else "false")
        .replace("__CINE_CLASS__", "cinematic" if cinematic else "")
    )
    components.html(html, height=230 if cinematic else 190)


def _render_camera_3d_lab(*, stage_index: int, autoplay: bool, speed: float, shot_preset: int):
    """Interaktive 3D-Laborszene mit professioneller Animation."""
    # Robust fallback renderer ohne externe JS-Abhaengigkeiten (kein CDN noetig).
    safe_html = """
    <div id="cam3d-safe" style="font-family: Inter, system-ui, sans-serif; color:#E5E7EB;">
      <style>
        #cam3d-safe { border:1px solid rgba(255,255,255,.14); border-radius:14px; overflow:hidden;
                      background: radial-gradient(circle at 18% 0%, #26154a 0%, #0b1120 58%, #03060d 100%); }
        #cam3d-safe .bar { display:flex; justify-content:space-between; align-items:center; gap:8px;
                           padding:8px 10px; border-bottom:1px solid rgba(255,255,255,.08); background:rgba(255,255,255,.03); }
        #cam3d-safe .badge { font-size:.76rem; color:#DDD6FE; border:1px solid rgba(167,139,250,.42);
                             background:rgba(124,58,237,.14); border-radius:999px; padding:2px 9px; }
        #cam3d-safe .btns { display:flex; gap:6px; flex-wrap:wrap; }
        #cam3d-safe .btn { border:1px solid rgba(255,255,255,.20); background:rgba(17,24,39,.65); color:#E5E7EB;
                           border-radius:8px; padding:5px 10px; cursor:pointer; font-size:.78rem; }
        #cam3d-wrap { position:relative; width:100%; height:620px; }
        #cam3d-canvas { width:100%; height:100%; display:block; }
        #cam3d-explain { position:absolute; left:12px; bottom:12px; max-width:min(760px, 95%);
                         background:rgba(0,0,0,.48); border:1px solid rgba(255,255,255,.22); border-radius:12px;
                         padding:11px 13px; backdrop-filter:blur(4px); }
        #cam3d-explain .ttl { font-weight:700; margin-bottom:4px; color:#FDE68A; font-size:.95rem; }
        #cam3d-explain .txt { font-size:.84rem; color:#E5E7EB; line-height:1.36; }
      </style>
      <div class="bar">
        <div class="badge" id="cam3d-badge">Stage 1/5</div>
        <div class="btns">
          <button class="btn" id="cam3d-play">Pause</button>
          <button class="btn" id="cam3d-reset">Reset View</button>
          <button class="btn" id="cam3d-pop">Popout</button>
          <button class="btn" id="cam3d-fs">Vollbild</button>
        </div>
      </div>
      <div id="cam3d-wrap">
        <canvas id="cam3d-canvas"></canvas>
        <div id="cam3d-explain">
          <div class="ttl" id="cam3d-title">1) Szene & Licht</div>
          <div class="txt" id="cam3d-text">Licht aus der Szene trifft auf die Frontlinse.</div>
        </div>
      </div>
    </div>
    <script>
      (() => {
        const STAGE = Math.max(0, Math.min(4, __STAGE__));
        const SPEED = Math.max(0.4, __SPEED__);
        let autoPlay = __PLAY__;
        const SHOT = Math.max(0, Math.min(3, __SHOT__));

        const steps = [
          ["1) Szene & Licht", "Licht aus der Szene trifft auf die Frontlinse."],
          ["2) Optik fokussiert", "Die Linsengruppe fokussiert auf die Sensorebene (PSF/Defokus entsteht hier)."],
          ["3) Belichtung", "Shutter oeffnet, Sensor integriert Photonen waehrend der Belichtungszeit."],
          ["4) Bayer + ADC", "RGGB-Mosaik misst Farbanteile, ADC quantisiert sie in digitale Werte."],
          ["5) ISP Output", "Demosaicing, White-Balance und Tonkurve erzeugen das finale Bild."]
        ];

        const root = document.getElementById("cam3d-safe");
        const wrap = document.getElementById("cam3d-wrap");
        const canvas = document.getElementById("cam3d-canvas");
        const ctx = canvas.getContext("2d");
        const badge = document.getElementById("cam3d-badge");
        const title = document.getElementById("cam3d-title");
        const text = document.getElementById("cam3d-text");
        const playBtn = document.getElementById("cam3d-play");
        const resetBtn = document.getElementById("cam3d-reset");
        const popBtn = document.getElementById("cam3d-pop");
        const fsBtn = document.getElementById("cam3d-fs");

        function resize() {
          const dpr = window.devicePixelRatio || 1;
          const w = wrap.clientWidth;
          const h = wrap.clientHeight;
          canvas.width = Math.floor(w * dpr);
          canvas.height = Math.floor(h * dpr);
          ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        }
        resize();
        window.addEventListener("resize", resize);

        let stage = STAGE;
        let yaw = 0.0;
        let pitch = 0.25;
        let dist = 1.0;
        const presets = [
          { yaw: 0.0, pitch: 0.25, dist: 1.0 },
          { yaw: -0.55, pitch: 0.10, dist: 0.78 },
          { yaw: 0.9, pitch: 0.18, dist: 0.72 },
          { yaw: 0.0, pitch: 1.1, dist: 0.82 },
        ];
        yaw = presets[SHOT].yaw; pitch = presets[SHOT].pitch; dist = presets[SHOT].dist;
        const start = { yaw, pitch, dist };

        let dragging = false;
        let lx = 0;
        let ly = 0;
        canvas.addEventListener("pointerdown", (e) => { dragging = true; lx = e.clientX; ly = e.clientY; });
        window.addEventListener("pointerup", () => { dragging = false; });
        window.addEventListener("pointermove", (e) => {
          if (!dragging) return;
          const dx = e.clientX - lx;
          const dy = e.clientY - ly;
          lx = e.clientX; ly = e.clientY;
          yaw -= dx * 0.006;
          pitch = Math.max(-0.2, Math.min(1.25, pitch - dy * 0.004));
        });
        canvas.addEventListener("wheel", (e) => {
          e.preventDefault();
          dist = Math.max(0.62, Math.min(1.45, dist + e.deltaY * 0.0012));
        }, { passive: false });

        function updateStep() {
          badge.textContent = `Stage ${stage + 1}/5`;
          title.textContent = steps[stage][0];
          text.textContent = steps[stage][1];
        }
        updateStep();

        function inFullscreen() { return !!document.fullscreenElement; }
        function refreshFsLabel() { fsBtn.textContent = inFullscreen() ? "Exit Vollbild" : "Vollbild"; }
        document.addEventListener("fullscreenchange", refreshFsLabel);
        refreshFsLabel();

        playBtn.addEventListener("click", () => {
          autoPlay = !autoPlay;
          playBtn.textContent = autoPlay ? "Pause" : "Play";
        });
        playBtn.textContent = autoPlay ? "Pause" : "Play";

        resetBtn.addEventListener("click", () => {
          yaw = start.yaw;
          pitch = start.pitch;
          dist = start.dist;
        });

        popBtn.addEventListener("click", () => {
          const w = window.open("", "_blank", "noopener,noreferrer,width=1540,height=960");
          if (!w) return;
          w.document.open();
          w.document.write(document.documentElement.outerHTML);
          w.document.close();
        });

        fsBtn.addEventListener("click", async () => {
          try {
            if (document.fullscreenEnabled && root.requestFullscreen) {
              if (!inFullscreen()) await root.requestFullscreen(); else await document.exitFullscreen();
            } else {
              throw new Error("Fullscreen API unavailable");
            }
          } catch (err) {
            const w = window.open("", "_blank", "noopener,noreferrer,width=1540,height=960");
            if (!w) return;
            w.document.open();
            w.document.write(document.documentElement.outerHTML);
            w.document.close();
          }
          refreshFsLabel();
        });

        let tick = 0;
        let timer = setInterval(() => {
          if (!autoPlay) return;
          stage = (stage + 1) % 5;
          updateStep();
        }, Math.max(650, 2100 / SPEED));

        const particles = Array.from({ length: 40 }, (_, i) => ({
          x: -2.8 + Math.random() * 0.6,
          y: 0.15 + Math.random() * 0.9,
          z: 1.8 + Math.random() * 0.8,
          v: 0.008 + Math.random() * 0.018,
          phase: i * 0.31,
          hue: i % 2 ? 35 : 325,
        }));
        const bokeh = Array.from({ length: 22 }, (_, i) => ({
          x: -3.5 + Math.random() * 7.0,
          y: 0.3 + Math.random() * 2.4,
          z: -1.8 + Math.random() * 4.8,
          r: 12 + Math.random() * 34,
          a: 0.05 + Math.random() * 0.1,
          hue: i % 3 === 0 ? 220 : i % 3 === 1 ? 300 : 42,
        }));

        function drawRoundedRect(x, y, w, h, r, fill, stroke) {
          ctx.beginPath();
          ctx.moveTo(x + r, y);
          ctx.arcTo(x + w, y, x + w, y + h, r);
          ctx.arcTo(x + w, y + h, x, y + h, r);
          ctx.arcTo(x, y + h, x, y, r);
          ctx.arcTo(x, y, x + w, y, r);
          if (fill) { ctx.fillStyle = fill; ctx.fill(); }
          if (stroke) { ctx.strokeStyle = stroke; ctx.stroke(); }
        }

        function project(x, y, z, w, h) {
          const cy = Math.cos(yaw), sy = Math.sin(yaw);
          const cp = Math.cos(pitch), sp = Math.sin(pitch);
          let px = x * cy - z * sy;
          let pz = x * sy + z * cy;
          let py = y * cp - pz * sp;
          pz = y * sp + pz * cp;
          const f = 420 * dist / (3.9 + pz);
          return [w * 0.5 + px * f, h * 0.58 - py * f, f];
        }

        function draw() {
          tick += 0.016 * SPEED;
          const w = wrap.clientWidth;
          const h = wrap.clientHeight;
          ctx.clearRect(0, 0, w, h);

          const bg = ctx.createLinearGradient(0, 0, 0, h);
          bg.addColorStop(0, "#0b1120");
          bg.addColorStop(1, "#03060d");
          ctx.fillStyle = bg;
          ctx.fillRect(0, 0, w, h);

          // Cinematic vignette
          const vig = ctx.createRadialGradient(w * 0.5, h * 0.55, Math.min(w, h) * 0.15, w * 0.5, h * 0.55, Math.max(w, h) * 0.75);
          vig.addColorStop(0, "rgba(0,0,0,0)");
          vig.addColorStop(1, "rgba(0,0,0,0.55)");
          ctx.fillStyle = vig;
          ctx.fillRect(0, 0, w, h);

          // Background bokeh for cinematic depth
          for (const b of bokeh) {
            const bp = project(b.x, b.y, b.z, w, h);
            const rr = Math.max(3, (b.r * 0.01) * bp[2]);
            ctx.beginPath();
            ctx.arc(bp[0], bp[1], rr, 0, Math.PI * 2);
            if (b.hue === 220) ctx.fillStyle = `rgba(96,165,250,${b.a})`;
            else if (b.hue === 300) ctx.fillStyle = `rgba(244,114,182,${b.a})`;
            else ctx.fillStyle = `rgba(250,204,21,${b.a * 0.8})`;
            ctx.fill();
          }

          // Floor
          const f1 = project(-5, -1.15, -2.8, w, h);
          const f2 = project(5, -1.15, -2.8, w, h);
          const f3 = project(7, -1.15, 5.5, w, h);
          const f4 = project(-7, -1.15, 5.5, w, h);
          ctx.beginPath();
          ctx.moveTo(f1[0], f1[1]); ctx.lineTo(f2[0], f2[1]); ctx.lineTo(f3[0], f3[1]); ctx.lineTo(f4[0], f4[1]); ctx.closePath();
          ctx.fillStyle = "rgba(28,36,58,0.92)";
          ctx.fill();

          // Floor perspective grid
          ctx.strokeStyle = "rgba(173, 195, 228, 0.08)";
          ctx.lineWidth = 1;
          for (let gz = -2.5; gz <= 4.5; gz += 0.7) {
            const g1 = project(-5, -1.14, gz, w, h);
            const g2 = project(5, -1.14, gz, w, h);
            ctx.beginPath();
            ctx.moveTo(g1[0], g1[1]);
            ctx.lineTo(g2[0], g2[1]);
            ctx.stroke();
          }

          // Table top
          const t1 = project(-2.9, 0.0, -1.2, w, h);
          const t2 = project(2.9, 0.0, -1.2, w, h);
          const t3 = project(2.9, 0.0, 1.2, w, h);
          const t4 = project(-2.9, 0.0, 1.2, w, h);
          ctx.beginPath();
          ctx.moveTo(t1[0], t1[1]); ctx.lineTo(t2[0], t2[1]); ctx.lineTo(t3[0], t3[1]); ctx.lineTo(t4[0], t4[1]); ctx.closePath();
          ctx.fillStyle = "rgba(87,104,131,0.95)";
          ctx.fill();
          ctx.strokeStyle = "rgba(216,232,255,0.18)";
          ctx.lineWidth = 1;
          ctx.stroke();

          // Table legs
          const legSpec = [[-2.4, -0.55, -1.0], [2.4, -0.55, -1.0], [-2.4, -0.55, 1.0], [2.4, -0.55, 1.0]];
          for (const [lx3, ly3, lz3] of legSpec) {
            const top = project(lx3, 0.0, lz3, w, h);
            const bot = project(lx3, ly3, lz3, w, h);
            ctx.strokeStyle = "rgba(52,64,86,0.95)";
            ctx.lineWidth = Math.max(2, top[2] * 0.015);
            ctx.beginPath();
            ctx.moveTo(top[0], top[1]);
            ctx.lineTo(bot[0], bot[1]);
            ctx.stroke();
          }

          // Soft camera shadow on table
          const shadowP = project(0.25, 0.03, 0.55, w, h);
          const srx = Math.max(20, shadowP[2] * 0.25);
          const sry = srx * 0.45;
          const sgrad = ctx.createRadialGradient(shadowP[0], shadowP[1], srx * 0.12, shadowP[0], shadowP[1], srx);
          sgrad.addColorStop(0, "rgba(0,0,0,0.32)");
          sgrad.addColorStop(1, "rgba(0,0,0,0)");
          ctx.save();
          ctx.translate(shadowP[0], shadowP[1]);
          ctx.scale(1, sry / srx);
          ctx.beginPath();
          ctx.arc(0, 0, srx, 0, Math.PI * 2);
          ctx.fillStyle = sgrad;
          ctx.fill();
          ctx.restore();

          // Camera body as pseudo-3D boxes
          function drawBox(cx, cy, cz, bw, bh, bd, colFront, colSide) {
            const p000 = project(cx - bw, cy - bh, cz - bd, w, h);
            const p001 = project(cx - bw, cy - bh, cz + bd, w, h);
            const p010 = project(cx - bw, cy + bh, cz - bd, w, h);
            const p011 = project(cx - bw, cy + bh, cz + bd, w, h);
            const p100 = project(cx + bw, cy - bh, cz - bd, w, h);
            const p101 = project(cx + bw, cy - bh, cz + bd, w, h);
            const p110 = project(cx + bw, cy + bh, cz - bd, w, h);
            const p111 = project(cx + bw, cy + bh, cz + bd, w, h);

            // side
            ctx.beginPath();
            ctx.moveTo(p100[0], p100[1]); ctx.lineTo(p101[0], p101[1]); ctx.lineTo(p111[0], p111[1]); ctx.lineTo(p110[0], p110[1]); ctx.closePath();
            ctx.fillStyle = colSide; ctx.fill();
            // front
            ctx.beginPath();
            ctx.moveTo(p001[0], p001[1]); ctx.lineTo(p101[0], p101[1]); ctx.lineTo(p111[0], p111[1]); ctx.lineTo(p011[0], p011[1]); ctx.closePath();
            ctx.fillStyle = colFront; ctx.fill();
            // top
            ctx.beginPath();
            ctx.moveTo(p010[0], p010[1]); ctx.lineTo(p110[0], p110[1]); ctx.lineTo(p111[0], p111[1]); ctx.lineTo(p011[0], p011[1]); ctx.closePath();
            ctx.fillStyle = "rgba(58,67,88,0.85)"; ctx.fill();
          }

          drawBox(0, 0.56, 0.02, 0.95, 0.55, 0.52, "rgba(21,26,38,0.98)", "rgba(45,53,74,0.94)");
          drawBox(0, 1.25, -0.06, 0.32, 0.18, 0.30, "rgba(33,40,58,0.97)", "rgba(54,66,91,0.92)");
          drawBox(1.08, 0.50, 0.12, 0.18, 0.40, 0.26, "rgba(12,16,26,0.98)", "rgba(34,42,58,0.90)");
          drawBox(0.58, 1.33, 0.05, 0.16, 0.07, 0.16, "rgba(185,28,28,0.95)", "rgba(239,68,68,0.82)");
          drawBox(0.0, 1.43, -0.06, 0.20, 0.05, 0.12, "rgba(17,24,39,0.96)", "rgba(55,65,81,0.88)");

          // Brand plate
          const plate = project(-0.2, 0.85, 0.56, w, h);
          const pw = Math.max(38, plate[2] * 0.12), ph = pw * 0.25;
          drawRoundedRect(plate[0] - pw / 2, plate[1] - ph / 2, pw, ph, 4, "rgba(17,24,39,0.82)", "rgba(167,139,250,0.35)");
          ctx.fillStyle = "rgba(221,214,254,0.95)";
          ctx.font = `${Math.max(10, plate[2] * 0.03)}px Inter, sans-serif`;
          ctx.textAlign = "center";
          ctx.textBaseline = "middle";
          ctx.fillText("CV-CAM", plate[0], plate[1] + 0.5);

          // Lens (circles with depth)
          const lc = project(0, 0.56, 0.98, w, h);
          const lensR = Math.max(18, lc[2] * 0.09);
          const glow = stage >= 1 ? 0.7 : 0.35;
          const grad = ctx.createRadialGradient(lc[0] - lensR * 0.2, lc[1] - lensR * 0.2, 1, lc[0], lc[1], lensR);
          grad.addColorStop(0, `rgba(139,213,255,${0.85 * glow})`);
          grad.addColorStop(1, "rgba(14,18,31,0.95)");
          ctx.beginPath(); ctx.arc(lc[0], lc[1], lensR * 1.2, 0, Math.PI * 2); ctx.fillStyle = "rgba(33,43,63,0.96)"; ctx.fill();
          ctx.beginPath(); ctx.arc(lc[0], lc[1], lensR * 1.35, 0, Math.PI * 2); ctx.strokeStyle = "rgba(187,201,224,0.35)"; ctx.lineWidth = 1.2; ctx.stroke();
          ctx.beginPath(); ctx.arc(lc[0], lc[1], lensR, 0, Math.PI * 2); ctx.fillStyle = grad; ctx.fill();
          // Extra lens rings
          ctx.beginPath(); ctx.arc(lc[0], lc[1], lensR * 0.78, 0, Math.PI * 2); ctx.strokeStyle = "rgba(148,163,184,0.35)"; ctx.lineWidth = 1.0; ctx.stroke();
          ctx.beginPath(); ctx.arc(lc[0], lc[1], lensR * 0.56, 0, Math.PI * 2); ctx.strokeStyle = "rgba(99,102,241,0.32)"; ctx.lineWidth = 1.0; ctx.stroke();
          for (let m = 0; m < 16; m++) {
            const ma = (m / 16) * Math.PI * 2 + tick * 0.06;
            const r0 = lensR * 1.02;
            const r1 = lensR * 1.14;
            ctx.beginPath();
            ctx.moveTo(lc[0] + Math.cos(ma) * r0, lc[1] + Math.sin(ma) * r0);
            ctx.lineTo(lc[0] + Math.cos(ma) * r1, lc[1] + Math.sin(ma) * r1);
            ctx.strokeStyle = "rgba(203,213,225,0.3)";
            ctx.lineWidth = 0.8;
            ctx.stroke();
          }
          // Lens reflections
          ctx.beginPath(); ctx.arc(lc[0] - lensR * 0.22, lc[1] - lensR * 0.28, lensR * 0.18, 0, Math.PI * 2);
          ctx.fillStyle = "rgba(255,255,255,0.32)";
          ctx.fill();
          ctx.beginPath(); ctx.arc(lc[0] + lensR * 0.24, lc[1] + lensR * 0.20, lensR * 0.10, 0, Math.PI * 2);
          ctx.fillStyle = "rgba(167,139,250,0.28)";
          ctx.fill();

          // Shutter blades
          const bladeOpen = stage >= 3 ? 0.6 : stage >= 2 ? 0.35 + 0.15 * (0.5 + 0.5 * Math.sin(tick * 6)) : 0.25;
          for (let i = 0; i < 6; i += 1) {
            const a = (i / 6) * Math.PI * 2 + tick * 0.08;
            const r = lensR * (0.18 + bladeOpen * 0.28);
            const bx = lc[0] + Math.cos(a) * r;
            const by = lc[1] + Math.sin(a) * r;
            ctx.save();
            ctx.translate(bx, by);
            ctx.rotate(a + 0.8);
            drawRoundedRect(-lensR * 0.22, -lensR * 0.05, lensR * 0.44, lensR * 0.10, 2, "rgba(16,22,33,0.90)");
            ctx.restore();
          }

          // Sensor preview panel
          const sp = project(0, 0.56, 0.08, w, h);
          const sw = Math.max(32, sp[2] * 0.11), sh = sw * 0.64;
          ctx.save();
          ctx.translate(sp[0], sp[1]);
          ctx.fillStyle = "rgba(10,14,24,0.95)";
          drawRoundedRect(-sw / 2, -sh / 2, sw, sh, 3, "rgba(10,14,24,0.95)");
          if (stage >= 3) {
            const alpha = stage === 3 ? 0.55 : 0.85;
            for (let yy = 0; yy < 6; yy++) {
              for (let xx = 0; xx < 8; xx++) {
                const rx = xx % 2, ry = yy % 2;
                let c = `rgba(34,197,94,${alpha})`;
                if (ry === 0 && rx === 0) c = `rgba(239,68,68,${alpha})`;
                if (ry === 1 && rx === 1) c = `rgba(59,130,246,${alpha})`;
                ctx.fillStyle = c;
                ctx.fillRect(-sw/2 + xx * (sw/8), -sh/2 + yy * (sh/6), sw/8 - 1, sh/6 - 1);
              }
            }
          }
          ctx.restore();

          // Display output
          const dp = project(-1.7, 0.72, 0.05, w, h);
          const dw = Math.max(56, dp[2] * 0.20), dh = dw * 0.62;
          const reveal = stage >= 4 ? Math.min(1, (Math.sin(tick * 1.8) * 0.5 + 0.5) * 0.25 + 0.75) : 0;
          drawRoundedRect(dp[0] - dw/2, dp[1] - dh/2, dw, dh, 6, "rgba(7,10,18,0.95)", "rgba(127,153,196,0.28)");
          if (reveal > 0) {
            const g2 = ctx.createLinearGradient(dp[0] - dw/2, dp[1] - dh/2, dp[0] + dw/2, dp[1] + dh/2);
            g2.addColorStop(0, "rgba(30,41,59,0.95)");
            g2.addColorStop(1, "rgba(14,165,233,0.92)");
            drawRoundedRect(dp[0] - dw/2 + 4, dp[1] - dh/2 + 4, dw - 8, dh - 8, 4, g2);
          }

          // Light rays
          if (stage >= 0) {
            const n = 14;
            for (let i = 0; i < n; i++) {
              const sy = 0.15 + i * 0.06 + Math.sin(tick * 1.8 + i) * 0.01;
              const pA = project(-2.2, sy, 1.9, w, h);
              const pB = project(-0.2, 0.56, 1.1, w, h);
              const alpha = stage >= 3 ? 0.42 : stage >= 2 ? 0.30 : 0.20;
              ctx.strokeStyle = i % 2 ? `rgba(245,158,11,${alpha})` : `rgba(236,72,153,${alpha * 0.9})`;
              ctx.lineWidth = 1.0 + (stage >= 3 ? 0.4 : 0.0);
              ctx.beginPath();
              ctx.moveTo(pA[0], pA[1]);
              ctx.lineTo(pB[0], pB[1]);
              ctx.stroke();
            }
          }

          // Photon particles
          for (const p of particles) {
            p.x += p.v * SPEED;
            p.y += Math.sin(tick * 2.4 + p.phase) * 0.002;
            if (p.x > 0.15) {
              p.x = -2.9;
              p.y = 0.1 + Math.random() * 0.9;
            }
            const pp = project(p.x, p.y, p.z, w, h);
            const r = Math.max(1.2, pp[2] * 0.004);
            const alpha = stage >= 3 ? 0.85 : stage >= 2 ? 0.65 : 0.42;
            ctx.beginPath();
            ctx.arc(pp[0], pp[1], r, 0, Math.PI * 2);
            ctx.fillStyle = p.hue === 35 ? `rgba(245,158,11,${alpha})` : `rgba(236,72,153,${alpha * 0.9})`;
            ctx.fill();
          }

          // Stage glow halo
          const halo = ctx.createRadialGradient(lc[0], lc[1], lensR * 0.8, lc[0], lc[1], lensR * 2.8);
          const hs = stage >= 4 ? 0.30 : stage >= 3 ? 0.22 : stage >= 2 ? 0.14 : 0.08;
          halo.addColorStop(0, `rgba(236,72,153,${hs})`);
          halo.addColorStop(1, "rgba(236,72,153,0)");
          ctx.fillStyle = halo;
          ctx.fillRect(lc[0] - lensR * 3, lc[1] - lensR * 3, lensR * 6, lensR * 6);

          // Tripod under camera
          const tripodTop = project(0, -0.05, 0.08, w, h);
          const leg1 = project(-0.9, -1.1, -0.1, w, h);
          const leg2 = project(0.95, -1.15, -0.05, w, h);
          const leg3 = project(0.0, -1.2, 0.95, w, h);
          ctx.strokeStyle = "rgba(75,85,99,0.95)";
          ctx.lineWidth = Math.max(2, tripodTop[2] * 0.01);
          ctx.beginPath(); ctx.moveTo(tripodTop[0], tripodTop[1]); ctx.lineTo(leg1[0], leg1[1]); ctx.stroke();
          ctx.beginPath(); ctx.moveTo(tripodTop[0], tripodTop[1]); ctx.lineTo(leg2[0], leg2[1]); ctx.stroke();
          ctx.beginPath(); ctx.moveTo(tripodTop[0], tripodTop[1]); ctx.lineTo(leg3[0], leg3[1]); ctx.stroke();
          ctx.beginPath();
          ctx.arc(tripodTop[0], tripodTop[1], Math.max(4, tripodTop[2] * 0.012), 0, Math.PI * 2);
          ctx.fillStyle = "rgba(17,24,39,0.98)";
          ctx.fill();

          // Pro telemetry HUD
          const hudW = Math.min(300, w * 0.34), hudH = 108;
          const hudX = w - hudW - 16, hudY = 16;
          drawRoundedRect(hudX, hudY, hudW, hudH, 10, "rgba(2,6,23,0.62)", "rgba(148,163,184,0.28)");
          ctx.font = "12px Inter, sans-serif";
          ctx.textAlign = "left";
          ctx.fillStyle = "rgba(226,232,240,0.95)";
          ctx.fillText("CAMERA TELEMETRY", hudX + 12, hudY + 19);
          const teleRows = [
            `ISO: ${stage >= 4 ? 800 : stage >= 3 ? 400 : 200}`,
            `Exposure: ${(1 / (60 - stage * 8)).toFixed(4)} s`,
            `Aperture: f/${(2.8 + stage * 0.7).toFixed(1)}`,
            `ADC: ${stage >= 4 ? 12 : stage >= 3 ? 10 : 8} bit`,
            `Pipeline Stage: ${stage + 1}/5`,
          ];
          ctx.fillStyle = "rgba(191,219,254,0.95)";
          for (let i = 0; i < teleRows.length; i++) {
            ctx.fillText(teleRows[i], hudX + 12, hudY + 40 + i * 13);
          }

          requestAnimationFrame(draw);
        }

        draw();

        window.addEventListener("beforeunload", () => {
          if (timer) clearInterval(timer);
        });
      })();
    </script>
    """
    safe_html = (
        safe_html.replace("__STAGE__", str(max(0, stage_index)))
        .replace("__PLAY__", "true" if autoplay else "false")
        .replace("__SPEED__", f"{max(0.4, speed):.2f}")
        .replace("__SHOT__", str(max(0, shot_preset)))
    )
    components.html(safe_html, height=700)
    return
    html_template = """
    <div id="camera-3d-lab" style="font-family: Inter, system-ui, sans-serif; color: #E5E7EB;">
      <style>
        #camera-3d-lab {
          border: 1px solid rgba(255,255,255,0.15);
          border-radius: 14px;
          overflow: hidden;
          background: radial-gradient(circle at 15% 0%, #26154a 0%, #0b1120 58%, #03060d 100%);
          box-shadow: 0 10px 36px rgba(124,58,237,0.25), inset 0 0 28px rgba(236,72,153,0.09);
        }
        #camera-3d-lab .topbar {
          display: flex; justify-content: space-between; align-items: center;
          padding: 8px 10px; border-bottom: 1px solid rgba(255,255,255,0.08);
          background: rgba(255,255,255,0.03);
        }
        #camera-3d-lab .badge {
          font-size: 0.76rem; color: #DDD6FE; border: 1px solid rgba(167,139,250,0.42);
          background: rgba(124,58,237,0.14); border-radius: 999px; padding: 2px 9px;
        }
        #camera-3d-lab .btns { display: flex; gap: 6px; }
        #camera-3d-lab .btn {
          border: 1px solid rgba(255,255,255,0.20); background: rgba(17,24,39,0.65); color: #E5E7EB;
          border-radius: 8px; padding: 5px 10px; cursor: pointer; font-size: 0.78rem;
        }
        #camera-3d-lab #render-wrap { position: relative; width: 100%; height: 620px; }
        #camera-3d-lab #explain {
          position: absolute; left: 12px; bottom: 12px; max-width: min(780px, 95%);
          background: rgba(0,0,0,0.48); border: 1px solid rgba(255,255,255,0.22);
          border-radius: 12px; padding: 11px 13px; backdrop-filter: blur(5px);
        }
        #camera-3d-lab #explain .ttl { font-weight: 700; margin-bottom: 4px; color: #FDE68A; font-size: 0.95rem; }
        #camera-3d-lab #explain .txt { font-size: 0.84rem; color: #E5E7EB; line-height: 1.36; }
      </style>
      <div class="topbar">
        <div class="badge" id="stage-badge">Stage 1/5</div>
        <div class="btns">
          <button class="btn" id="lab-play">Pause</button>
          <button class="btn" id="lab-reset">Reset View</button>
          <button class="btn" id="lab-pop">Popout</button>
          <button class="btn" id="lab-fs">Vollbild</button>
        </div>
      </div>
      <div id="render-wrap">
        <div id="explain">
          <div class="ttl" id="exp-title">1) Szene & Licht</div>
          <div class="txt" id="exp-text">Licht von der Szene gelangt zur Optik. Die Kamera sieht zuerst nur Strahlung.</div>
        </div>
      </div>
    </div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r152/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.152.2/examples/js/controls/OrbitControls.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.152.2/examples/js/loaders/GLTFLoader.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.152.2/examples/js/postprocessing/EffectComposer.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.152.2/examples/js/postprocessing/RenderPass.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.152.2/examples/js/postprocessing/UnrealBloomPass.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.152.2/examples/js/shaders/CopyShader.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.152.2/examples/js/shaders/LuminosityHighPassShader.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.5/gsap.min.js"></script>
    <script>
      const STAGE = Math.max(0, Math.min(4, __STAGE__));
      const INIT_PLAY = __PLAY__;
      const SPEED = __SPEED__;
      const SHOT = __SHOT__;
      const wrap = document.getElementById("render-wrap");
      const badge = document.getElementById("stage-badge");
      const expTitle = document.getElementById("exp-title");
      const expText = document.getElementById("exp-text");
      const fsBtn = document.getElementById("lab-fs");
      const playBtn = document.getElementById("lab-play");
      const resetBtn = document.getElementById("lab-reset");
      const popBtn = document.getElementById("lab-pop");
      const root = document.getElementById("camera-3d-lab");

      const stepInfo = [
        ["1) Szene & Licht", "Licht aus der Umgebung trifft durch die Frontlinse auf das optische System."],
        ["2) Optik fokussiert", "Mehrere Linsengruppen fokusieren die Strahlen. PSF und Defokus entstehen hier."],
        ["3) Belichtung", "Die Blende/Shutter-Sequenz oeffnet. Der Sensor integriert Photonen ueber die Belichtungszeit."],
        ["4) Bayer + ADC", "Der Sensor misst mosaikartig (RGGB), danach quantisiert der ADC auf digitale Level."],
        ["5) ISP Output", "Demosaicing, White Balance, Tonmapping/Gamma erzeugen das finale, sichtbare RGB-Bild."]
      ];

      const scene = new THREE.Scene();
      scene.background = new THREE.Color(0x090d16);
      scene.fog = new THREE.Fog(0x090d16, 8, 35);

      const camera = new THREE.PerspectiveCamera(50, wrap.clientWidth / wrap.clientHeight, 0.1, 120);
      camera.position.set(6.0, 3.2, 7.2);
      camera.lookAt(0, 1.8, 0);

      const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
      renderer.setPixelRatio(window.devicePixelRatio || 1);
      renderer.setSize(wrap.clientWidth, wrap.clientHeight);
      renderer.outputColorSpace = THREE.SRGBColorSpace;
      renderer.toneMapping = THREE.ACESFilmicToneMapping;
      renderer.toneMappingExposure = 1.0;
      renderer.shadowMap.enabled = true;
      renderer.shadowMap.type = THREE.PCFSoftShadowMap;
      wrap.prepend(renderer.domElement);

      let composer = null;
      try {
        if (THREE.EffectComposer && THREE.RenderPass && THREE.UnrealBloomPass) {
          composer = new THREE.EffectComposer(renderer);
          const renderPass = new THREE.RenderPass(scene, camera);
          const bloomPass = new THREE.UnrealBloomPass(
            new THREE.Vector2(wrap.clientWidth, wrap.clientHeight),
            0.45,
            0.5,
            0.74
          );
          composer.addPass(renderPass);
          composer.addPass(bloomPass);
        }
      } catch (e) {
        composer = null;
      }

      let controls = null;
      if (THREE.OrbitControls) {
        controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        controls.target.set(0, 1.7, 0);
        controls.minDistance = 4;
        controls.maxDistance = 16;
        controls.maxPolarAngle = Math.PI * 0.48;
      }

      const shotPresets = [
        { name: "Hero",   pos: [6.0, 3.2, 7.2], target: [0, 1.7, 0] },
        { name: "Lens",   pos: [1.9, 2.1, 3.1], target: [0.0, 1.8, 1.0] },
        { name: "Sensor", pos: [-0.8, 2.0, 0.8], target: [0.0, 1.78, 0.12] },
        { name: "Top",    pos: [0.0, 7.2, 0.1], target: [0.0, 1.7, 0.0] },
      ];

      function applyShotPreset(idx, animated = true) {
        const s = shotPresets[Math.max(0, Math.min(shotPresets.length - 1, idx))];
        if (!s) return;
        if (!animated) {
          camera.position.set(s.pos[0], s.pos[1], s.pos[2]);
          if (controls) {
            controls.target.set(s.target[0], s.target[1], s.target[2]);
            controls.update();
          } else {
            camera.lookAt(s.target[0], s.target[1], s.target[2]);
          }
          return;
        }
        gsap.to(camera.position, {
          x: s.pos[0], y: s.pos[1], z: s.pos[2], duration: 1.2 / SPEED, ease: "power2.inOut",
        });
        if (controls) {
          gsap.to(controls.target, {
            x: s.target[0], y: s.target[1], z: s.target[2], duration: 1.2 / SPEED, ease: "power2.inOut",
            onUpdate: () => controls.update(),
          });
        }
      }

      // Lichtsetup
      const ambient = new THREE.HemisphereLight(0x7ea3ff, 0x171a24, 0.62);
      scene.add(ambient);
      const key = new THREE.SpotLight(0xffffff, 2.1, 35, Math.PI / 6, 0.32, 1.0);
      key.position.set(6, 8, 7);
      key.castShadow = true;
      scene.add(key);
      const rim = new THREE.PointLight(0xec4899, 1.0, 16, 2.0);
      rim.position.set(-4, 3, -3);
      scene.add(rim);

      // Laborraum
      const floor = new THREE.Mesh(
        new THREE.PlaneGeometry(36, 36),
        new THREE.MeshStandardMaterial({ color: 0x151a28, metalness: 0.25, roughness: 0.88 })
      );
      floor.rotation.x = -Math.PI / 2;
      floor.receiveShadow = true;
      scene.add(floor);

      const backWall = new THREE.Mesh(
        new THREE.PlaneGeometry(24, 10),
        new THREE.MeshStandardMaterial({ color: 0x0f1422, metalness: 0.1, roughness: 0.9 })
      );
      backWall.position.set(0, 5, -6);
      scene.add(backWall);

      const table = new THREE.Mesh(
        new THREE.BoxGeometry(6.2, 0.26, 2.6),
        new THREE.MeshStandardMaterial({ color: 0x424c5f, metalness: 0.45, roughness: 0.42 })
      );
      table.position.set(0, 1.05, 0.1);
      table.castShadow = true;
      table.receiveShadow = true;
      scene.add(table);

      // Kamera-Assembly
      const camRig = new THREE.Group();
      scene.add(camRig);

      const bodyMat = new THREE.MeshStandardMaterial({ color: 0x0f1117, metalness: 0.45, roughness: 0.38 });
      const accentMat = new THREE.MeshStandardMaterial({ color: 0x272d3b, metalness: 0.55, roughness: 0.33 });
      const rubberMat = new THREE.MeshStandardMaterial({ color: 0x08090d, metalness: 0.2, roughness: 0.84 });

      const camBody = new THREE.Mesh(new THREE.BoxGeometry(1.95, 1.2, 1.0), bodyMat);
      camBody.position.set(0, 1.78, 0.0);
      camBody.castShadow = true;
      camRig.add(camBody);

      const grip = new THREE.Mesh(new THREE.BoxGeometry(0.38, 0.95, 0.72), rubberMat);
      grip.position.set(0.95, 1.66, 0.08);
      grip.castShadow = true;
      camRig.add(grip);

      const topPrism = new THREE.Mesh(new THREE.BoxGeometry(0.66, 0.38, 0.66), accentMat);
      topPrism.position.set(0, 2.46, -0.02);
      topPrism.castShadow = true;
      camRig.add(topPrism);

      const button = new THREE.Mesh(new THREE.CylinderGeometry(0.08, 0.08, 0.06, 24), new THREE.MeshStandardMaterial({ color: 0xdc2626 }));
      button.rotation.x = Math.PI / 2;
      button.position.set(0.55, 2.41, 0.18);
      camRig.add(button);

      const lensGroup = new THREE.Group();
      camRig.add(lensGroup);

      const lensBarrel = new THREE.Mesh(
        new THREE.CylinderGeometry(0.45, 0.53, 0.88, 56),
        new THREE.MeshStandardMaterial({ color: 0x1b2232, metalness: 0.78, roughness: 0.28 })
      );
      lensBarrel.rotation.x = Math.PI / 2;
      lensBarrel.position.set(0, 1.79, 0.98);
      lensBarrel.castShadow = true;
      lensGroup.add(lensBarrel);

      const lensRing = new THREE.Mesh(
        new THREE.TorusGeometry(0.49, 0.045, 18, 60),
        new THREE.MeshStandardMaterial({ color: 0x3f4b63, metalness: 0.88, roughness: 0.24 })
      );
      lensRing.position.set(0, 1.79, 1.41);
      lensGroup.add(lensRing);

      const lensGlass = new THREE.Mesh(
        new THREE.CircleGeometry(0.33, 56),
        new THREE.MeshStandardMaterial({
          color: 0x8bd5ff,
          emissive: 0x08101f,
          emissiveIntensity: 0.4,
          metalness: 0.95,
          roughness: 0.04,
          transparent: true,
          opacity: 0.92,
        })
      );
      lensGlass.position.set(0, 1.79, 1.43);
      lensGroup.add(lensGlass);

      // Externes glTF-Kameramodell (optional, fallback bleibt prozedurale Kamera).
      let gltfCam = null;
      if (THREE.GLTFLoader) {
        const loader = new THREE.GLTFLoader();
        loader.load(
          "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/master/2.0/Camera/glTF-Binary/Camera.glb",
          (gltf) => {
            gltfCam = gltf.scene;
            gltfCam.scale.set(1.2, 1.2, 1.2);
            gltfCam.position.set(2.0, 1.6, -0.8);
            gltfCam.rotation.y = -Math.PI * 0.35;
            gltfCam.traverse((obj) => {
              if (obj.isMesh) {
                obj.castShadow = true;
                obj.receiveShadow = true;
              }
            });
            scene.add(gltfCam);
          },
          undefined,
          () => {
            // Falls Modell nicht erreichbar: kein Abbruch, prozedurale Kamera bleibt.
          }
        );
      }

      // Blendenlamellen / Shutter
      const shutterGroup = new THREE.Group();
      shutterGroup.position.set(0, 1.79, 1.23);
      camRig.add(shutterGroup);
      const bladeMat = new THREE.MeshStandardMaterial({ color: 0x111827, metalness: 0.5, roughness: 0.5, side: THREE.DoubleSide });
      const blades = [];
      for (let i = 0; i < 6; i += 1) {
        const blade = new THREE.Mesh(new THREE.BoxGeometry(0.34, 0.1, 0.016), bladeMat);
        const ang = (i / 6) * Math.PI * 2;
        blade.position.set(Math.cos(ang) * 0.13, Math.sin(ang) * 0.13, 0);
        blade.rotation.z = ang + Math.PI / 6;
        shutterGroup.add(blade);
        blades.push(blade);
      }

      // Bayer-Sensor-Texture
      const sensorCanvas = document.createElement("canvas");
      sensorCanvas.width = 128;
      sensorCanvas.height = 96;
      const sctx = sensorCanvas.getContext("2d");
      const px = 8;
      for (let y = 0; y < sensorCanvas.height; y += px) {
        for (let x = 0; x < sensorCanvas.width; x += px) {
          const xr = (x / px) % 2;
          const yr = (y / px) % 2;
          let col = "#22c55e"; // G
          if (yr === 0 && xr === 0) col = "#ef4444"; // R
          if (yr === 1 && xr === 1) col = "#3b82f6"; // B
          sctx.fillStyle = col;
          sctx.fillRect(x, y, px, px);
        }
      }
      const sensorTex = new THREE.CanvasTexture(sensorCanvas);
      sensorTex.colorSpace = THREE.SRGBColorSpace;

      const sensor = new THREE.Mesh(
        new THREE.PlaneGeometry(0.62, 0.42),
        new THREE.MeshStandardMaterial({
          color: 0x1f2937,
          map: sensorTex,
          emissive: 0x000000,
          emissiveMap: sensorTex,
          emissiveIntensity: 0.0,
          side: THREE.DoubleSide,
        })
      );
      sensor.position.set(0, 1.78, 0.14);
      sensor.rotation.y = Math.PI;
      camRig.add(sensor);

      // Display mit Render-Canvas
      const photoCanvas = document.createElement("canvas");
      photoCanvas.width = 320;
      photoCanvas.height = 200;
      const pctx = photoCanvas.getContext("2d");
      function paintPhoto(progress) {
        const g = pctx.createLinearGradient(0, 0, photoCanvas.width, photoCanvas.height);
        g.addColorStop(0, "#1e293b");
        g.addColorStop(1, "#0ea5e9");
        pctx.fillStyle = g;
        pctx.fillRect(0, 0, photoCanvas.width, photoCanvas.height);
        pctx.fillStyle = "rgba(255,255,255,0.16)";
        pctx.fillRect(20, 20, 110, 70);
        pctx.fillStyle = "rgba(244,114,182,0.6)";
        pctx.beginPath();
        pctx.arc(230, 96, 52, 0, Math.PI * 2);
        pctx.fill();
        pctx.fillStyle = "rgba(16,185,129,0.75)";
        pctx.fillRect(84, 128, 130, 46);
        pctx.fillStyle = "rgba(255,255,255,0.95)";
        pctx.font = "bold 24px Inter, sans-serif";
        pctx.fillText("CAPTURE", 20, 188);

        // Reveal overlay
        pctx.fillStyle = "rgba(6,9,15," + (1 - progress).toFixed(3) + ")";
        pctx.fillRect(0, 0, photoCanvas.width, photoCanvas.height);
      }
      paintPhoto(0);
      const photoTex = new THREE.CanvasTexture(photoCanvas);
      photoTex.colorSpace = THREE.SRGBColorSpace;

      const screen = new THREE.Mesh(
        new THREE.PlaneGeometry(1.05, 0.68),
        new THREE.MeshStandardMaterial({
          map: photoTex,
          emissiveMap: photoTex,
          emissive: 0x000000,
          emissiveIntensity: 0.0,
          metalness: 0.3,
          roughness: 0.4,
        })
      );
      screen.position.set(-1.74, 1.9, 0.05);
      screen.rotation.y = Math.PI / 2;
      camRig.add(screen);

      // Tripod
      const tripodMat = new THREE.MeshStandardMaterial({ color: 0x1f2937, metalness: 0.72, roughness: 0.34 });
      const stem = new THREE.Mesh(new THREE.CylinderGeometry(0.09, 0.12, 1.05, 24), tripodMat);
      stem.position.set(0, 0.96, 0);
      stem.castShadow = true;
      scene.add(stem);
      for (let i = 0; i < 3; i += 1) {
        const leg = new THREE.Mesh(new THREE.CylinderGeometry(0.035, 0.055, 1.3, 20), tripodMat);
        leg.position.set(Math.cos((i * 2 * Math.PI) / 3) * 0.6, 0.43, Math.sin((i * 2 * Math.PI) / 3) * 0.6);
        leg.rotation.z = (i - 1) * 0.58;
        leg.castShadow = true;
        scene.add(leg);
      }

      // Photonenstrahlen
      const beamGroup = new THREE.Group();
      scene.add(beamGroup);
      const beams = [];
      for (let i = 0; i < 26; i += 1) {
        const mat = new THREE.MeshBasicMaterial({
          color: i % 2 ? 0xf59e0b : 0xec4899,
          transparent: true,
          opacity: 0.0,
        });
        const mesh = new THREE.Mesh(new THREE.CylinderGeometry(0.008, 0.008, 1.9, 8), mat);
        mesh.rotation.x = Math.PI / 2;
        mesh.position.set(-2.6 + Math.random() * 5.2, 1.9 + Math.random() * 0.5, 2.0 + Math.random() * 0.35);
        beamGroup.add(mesh);
        beams.push(mesh);
      }

      // Floating explanatory hologram
      const holo = new THREE.Mesh(
        new THREE.RingGeometry(0.6, 0.67, 64),
        new THREE.MeshBasicMaterial({ color: 0xa78bfa, transparent: true, opacity: 0.28, side: THREE.DoubleSide })
      );
      holo.position.set(2.0, 2.25, -0.4);
      holo.rotation.x = Math.PI / 2.4;
      scene.add(holo);

      let stage = STAGE;
      let autoPlay = INIT_PLAY;
      let timer = null;
      let t = 0;

      function setStepText(idx) {
        badge.textContent = `Stage ${idx + 1}/5`;
        expTitle.textContent = stepInfo[idx][0];
        expText.textContent = stepInfo[idx][1];
      }

      function setBlades(openAmount) {
        blades.forEach((blade, i) => {
          const a = (i / blades.length) * Math.PI * 2;
          blade.position.x = Math.cos(a) * (0.05 + 0.20 * openAmount);
          blade.position.y = Math.sin(a) * (0.05 + 0.20 * openAmount);
          blade.rotation.z = a + (1.1 - openAmount) * 0.78;
        });
      }

      function setPhotoReveal(v) {
        paintPhoto(v);
        photoTex.needsUpdate = true;
      }

      function applyStage(idx, animated = true) {
        stage = idx;
        setStepText(idx);

        const dur = (animated ? 0.9 : 0.01) / SPEED;
        gsap.killTweensOf([lensGroup.scale, lensGlass.material, sensor.material, screen.material, beams.map((b) => b.material)]);

        // Baseline
        gsap.to(lensGroup.scale, { x: 1, y: 1, z: 1, duration: dur, ease: "power2.inOut" });
        gsap.to(lensGlass.material, { emissiveIntensity: 0.35, duration: dur });
        gsap.to(sensor.material, { emissiveIntensity: 0.0, duration: dur });
        gsap.to(screen.material, { emissiveIntensity: 0.0, duration: dur });
        setPhotoReveal(idx >= 4 ? 1 : 0);
        setBlades(0.25);
        beams.forEach((b, i) => {
          gsap.to(b.material, { opacity: idx >= 0 ? 0.12 + (i % 3) * 0.05 : 0.0, duration: dur * 0.8 });
        });

        if (idx >= 1) {
          gsap.to(lensGroup.scale, { x: 1.06, y: 1.06, z: 1.06, duration: dur, ease: "sine.inOut" });
          gsap.to(lensGlass.material, { emissiveIntensity: 0.7, duration: dur });
          setBlades(0.45);
        }
        if (idx >= 2) {
          gsap.to(sensor.material, { emissiveIntensity: 0.75, duration: dur });
          // Shutter close-open impulse
          gsap.to({}, {
            duration: 0.45 / SPEED,
            onUpdate: function() {
              const p = this.progress();
              const open = p < 0.5 ? 0.15 + p * 0.5 : 0.4 - (p - 0.5) * 0.4;
              setBlades(Math.max(0.08, open));
            },
          });
        }
        if (idx >= 3) {
          gsap.to(sensor.material, { emissiveIntensity: 1.1, duration: dur });
          beams.forEach((b, i) => gsap.to(b.material, { opacity: 0.2 + ((i % 4) * 0.06), duration: dur }));
        }
        if (idx >= 4) {
          gsap.to(screen.material, { emissiveIntensity: 0.95, duration: dur });
          gsap.to({}, {
            duration: 1.2 / SPEED,
            onUpdate: function() {
              setPhotoReveal(this.progress());
            },
          });
          setBlades(0.62);
        }
      }

      function tickStage() {
        applyStage((stage + 1) % 5, true);
      }

      function updatePlayLabel() {
        playBtn.textContent = autoPlay ? "Pause" : "Play";
      }

      function startAuto() {
        if (timer) {
          clearInterval(timer);
        }
        if (!autoPlay) {
          return;
        }
        timer = setInterval(() => tickStage(), Math.max(620, 2200 / SPEED));
      }

      playBtn.addEventListener("click", () => {
        autoPlay = !autoPlay;
        updatePlayLabel();
        startAuto();
      });

      resetBtn.addEventListener("click", () => {
        applyShotPreset(SHOT, true);
      });

      popBtn.addEventListener("click", () => {
        const w = window.open("", "_blank", "noopener,noreferrer,width=1540,height=960");
        if (!w) {
          return;
        }
        const html = document.documentElement.outerHTML;
        w.document.open();
        w.document.write(html);
        w.document.close();
      });

      // Vollbild
      function inFullscreen() {
        return !!document.fullscreenElement;
      }
      function updateFs() {
        fsBtn.textContent = inFullscreen() ? "Exit Vollbild" : "Vollbild";
      }
      fsBtn.addEventListener("click", async () => {
        try {
          if (document.fullscreenEnabled && root.requestFullscreen) {
            if (!inFullscreen()) {
              await root.requestFullscreen();
            } else {
              await document.exitFullscreen();
            }
          } else {
            throw new Error("Fullscreen API unavailable");
          }
        } catch (err) {
          const w = window.open("", "_blank", "noopener,noreferrer,width=1440,height=900");
          if (!w) {
            return;
          }
          const html = document.documentElement.outerHTML;
          w.document.open();
          w.document.write(html);
          w.document.close();
        }
        updateFs();
      });
      document.addEventListener("fullscreenchange", updateFs);
      updateFs();
      updatePlayLabel();

      function animate() {
        t += 0.008 * SPEED;
        holo.rotation.z += 0.003 * SPEED;
        holo.material.opacity = 0.22 + 0.10 * (0.5 + 0.5 * Math.sin(t * 3.0));
        beamGroup.children.forEach((b, i) => {
          b.position.z = 1.85 + Math.sin(t * 2.2 + i) * 0.24;
          b.position.y += Math.sin(t * 2.8 + i * 1.2) * 0.0018;
        });

        if (controls) {
          controls.update();
        }
        if (composer) {
          composer.render();
        } else {
          renderer.render(scene, camera);
        }
        requestAnimationFrame(animate);
      }

      applyShotPreset(SHOT, false);
      applyStage(stage, false);
      startAuto();
      animate();

      window.addEventListener("resize", () => {
        const w = wrap.clientWidth;
        const h = wrap.clientHeight;
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
        renderer.setSize(w, h);
        if (composer) {
          composer.setSize(w, h);
        }
      });
      window.addEventListener("beforeunload", () => {
        if (timer) {
          clearInterval(timer);
        }
      });
    </script>
    """
    html = (
        html_template.replace("__STAGE__", str(max(0, stage_index)))
        .replace("__PLAY__", "true" if autoplay else "false")
        .replace("__SPEED__", f"{max(0.4, speed):.2f}")
        .replace("__SHOT__", str(max(0, shot_preset)))
    )
    components.html(html, height=700)


def render():
    hero(
        eyebrow="Bildverarbeitung · Neues Kernmodul",
        title="Wie eine Kamera ein Bild macht",
        sub="Von Photonen auf der Szene bis zum fertigen RGB-Bild: Optik, Sensor, Bayer-Mosaik, "
            "Rauschen, ISP und die Grundlagen moderner Bildverarbeitung."
    )

    tabs = st.tabs([
        "🎬 Intuitive Animation",
        "🧪 3D Kamera-Labor",
        "🔬 Wissenschaftlich korrekt",
        "🧪 Kamera-Lab",
        "📈 Analyse & Messung",
        "🎥 Lernvideos",
        "🧭 Lernpfad & Uebungen",
    ])

    with tabs[0]:
        section_header("Pipeline in Bewegung", "Ein mentales Modell, das du in 20 Sekunden erfassen kannst.")
        c1, c2, c3, c4 = st.columns([1.0, 1.4, 1.2, 1.2])
        autoplay = c1.toggle("▶️ Play", value=True, key="cam_anim_play")
        stage_index = c2.select_slider(
            "Step-Mode (Stage)",
            options=[1, 2, 3, 4, 5],
            value=1,
            key="cam_anim_stage",
            disabled=autoplay,
        )
        cinematic = c3.toggle("🎥 Cinematic Mode", value=False, key="cam_anim_cinematic")
        speed_default = 1.0 if cinematic else 1.2
        speed = c4.slider("Animation Speed", 0.5, 2.4, speed_default, 0.1, key="cam_anim_speed")

        _render_pipeline_animation(stage_index=stage_index - 1, autoplay=autoplay, speed=speed, cinematic=cinematic)
        if cinematic:
            st.caption("Cinematic Mode aktiv: groesseres Visual, staerkerer Glow und langsamere Lehr-Transitions.")

        stage_cards = [
            ("🌞 Szene", "Die reale Welt liefert ein kontinuierliches Strahlungsfeld mit spektraler Verteilung."),
            ("🔍 Optik", "Die Linse faltet das Signal mit der PSF; Unschärfe und Vignette entstehen hier."),
            ("🧪 Sensor", "Photonen werden stochastisch zu Elektronen; Shot und Read Noise formen die Rohdaten."),
            ("🎨 Bayer + ADC", "CFA sampelt Farben mosaikartig, ADC quantisiert auf 8/10/12/14 bit."),
            ("🧠 ISP", "Demosaicing, White-Balance, Tonmapping/Gamma und Schärfung erzeugen das finale Bild."),
        ]
        step_title, step_desc = stage_cards[stage_index - 1]
        st.markdown(f"**Aktiver Schritt:** {step_title}")
        st.caption(step_desc)

        st.markdown(
            """
            **Merksatz:** Eine Kamera "sieht" nicht wie ein Mensch.  
            Sie **misst Licht** mit einer physikalischen Kette:
            Szene -> Optik -> Sensor -> ADC -> ISP -> Ausgabe.
            """
        )
        key_concept("💡", "Photonenstatistik", "Licht kommt diskret an. Shot Noise ist daher Poisson-verteilt und unvermeidbar.")
        key_concept("📉", "Signal-Rausch-Verhaeltnis", "Bei wenig Licht dominiert Rauschen. Hoeheres ISO verstaerkt Signal und Rauschen gleichzeitig.")
        key_concept("🎛️", "ISP", "Der Image Signal Processor macht aus rohen Sensordaten ein fuer Menschen ansehnliches Bild.")

    with tabs[1]:
        section_header(
            "3D Kamera im Labor",
            "Eine echte 3D-Ansicht mit Schritt-fuer-Schritt-Erklaerung des Aufnahmeprozesses.",
        )
        l1, l2, l3 = st.columns([1.2, 1.5, 1.3])
        lab_autoplay = l1.toggle("▶️ Auto-Demo", value=True, key="cam3d_auto")
        lab_stage = l2.select_slider(
            "Stage",
            options=[1, 2, 3, 4, 5],
            value=1,
            disabled=lab_autoplay,
            key="cam3d_stage",
        )
        lab_speed = l3.slider("Demo-Speed", 0.5, 2.0, 1.0, 0.1, key="cam3d_speed")
        shot_mode = st.selectbox(
            "3D Shot Preset",
            ["Hero Shot", "Lens Close-Up", "Sensor Focus", "Top-Down"],
            index=0,
            key="cam3d_shot",
        )
        shot_idx = ["Hero Shot", "Lens Close-Up", "Sensor Focus", "Top-Down"].index(shot_mode)
        _render_camera_3d_lab(stage_index=lab_stage - 1, autoplay=lab_autoplay, speed=lab_speed, shot_preset=shot_idx)

        step_titles = [
            "1) Szene beleuchtet Objekt",
            "2) Linse fokussiert auf Sensorebene",
            "3) Sensor integriert Photonen (Belichtung)",
            "4) Bayer-Messung + ADC-Quantisierung",
            "5) ISP erzeugt visuelles Endbild",
        ]
        st.caption(f"Aktiver Stage-Fokus: {step_titles[lab_stage - 1]}")

    with tabs[2]:
        section_header("Physik + Signalverarbeitung", "Was exakt passiert, Schritt fuer Schritt.")
        st.markdown(
            r"""
            ### 1) Optik und Abbildung
            Die Linse bildet die Szene auf den Sensor ab. Reale Optiken sind nicht perfekt:  
            Fokusfehler, Beugung und Aberrationen fuehren zu einer **Punktspreizfunktion (PSF)**.
            Im Modell ist das oft eine Faltung:

            $$
            I_{\mathrm{optik}} = I_{\mathrm{szene}} \star h_{\mathrm{PSF}}
            $$

            ### 2) Photoelektrischer Effekt am Sensor
            Pro Pixel entstehen Elektronen proportional zur Photonenzahl:

            $$
            \mu_e \approx E \cdot t_{\mathrm{exp}} \cdot \eta
            $$

            mit Bestrahlungsstaerke $E$, Belichtungszeit $t_{\mathrm{exp}}$ und Quanteneffizienz $\eta$.

            ### 3) Rauschen
            - **Shot Noise:** $\sigma_{\mathrm{shot}}^2 \approx \mu_e$ (Poisson)
            - **Read Noise:** additiver Elektronikanteil (nahezu Gauß)
            - **Dark Noise:** temperaturabhaengiger Leckstrom

            Gesamtmodell (vereinfacht):
            $$
            e = \mathrm{Poisson}(\mu_e) + \mathcal{N}(0, \sigma_{\mathrm{read}}^2)
            $$

            ### 4) CFA, ADC, ISP
            Ein Bayer-Pattern (RGGB) misst je Pixel nur einen Farbkanal. Danach:
            Demosaicing -> White Balance -> Tonkurve/Gamma -> Schaerfung/Entrauschen -> sRGB.
            """
        )
        info_box(
            "Wissenschaftlich wichtig: ISO erzeugt nicht mehr Photonen. ISO ist primaer Verstaerkung. "
            "Bei wenig Licht muss Signal und Rauschen gemeinsam verstaerkt werden.",
            kind="warn",
        )

    with tabs[3]:
        lab_header(
            "Kamera-Lab: Von Szene zu Kamera-Output",
            "Steuere Optik, Belichtung, ISO, Rauschen, ADC und White Balance live.",
        )

        source = st.radio("Bildquelle", ["Demo-Szene", "Eigenes Bild"], horizontal=True)
        if source == "Eigenes Bild":
            up = st.file_uploader("Bild hochladen (PNG/JPG)", type=["png", "jpg", "jpeg"])
            if up is not None:
                arr = np.frombuffer(up.read(), np.uint8)
                bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            else:
                rgb = _demo_scene(320)
        else:
            rgb = _demo_scene(320)

        c1, c2, c3 = st.columns(3)
        blur_sigma = c1.slider("Optik-Blur (PSF σ)", 0.0, 4.0, 1.0, 0.1)
        vignette_strength = c2.slider("Vignettierung", 0.0, 0.9, 0.25, 0.05)
        exposure_mult = c3.slider("Belichtung (Multiplikator)", 0.2, 3.0, 1.0, 0.1)

        c4, c5, c6 = st.columns(3)
        iso = c4.select_slider("ISO", options=[100, 200, 400, 800, 1600, 3200], value=400)
        read_noise_e = c5.slider("Read Noise (e- RMS)", 0.2, 12.0, 2.8, 0.1)
        bit_depth = c6.select_slider("ADC-Bittiefe", options=[8, 10, 12, 14], value=10)

        c7, c8, c9 = st.columns(3)
        wb_red_gain = c7.slider("White Balance Rot-Gain", 0.6, 2.2, 1.2, 0.05)
        wb_blue_gain = c8.slider("White Balance Blau-Gain", 0.6, 2.2, 1.1, 0.05)
        seed = c9.number_input("Noise-Seed", min_value=0, max_value=999999, value=42, step=1)

        sim = _simulate_camera_pipeline(
            rgb,
            blur_sigma=blur_sigma,
            vignette_strength=vignette_strength,
            exposure_mult=exposure_mult,
            iso=iso,
            read_noise_e=read_noise_e,
            bit_depth=bit_depth,
            wb_red_gain=wb_red_gain,
            wb_blue_gain=wb_blue_gain,
            seed=int(seed),
        )

        p1, p2, p3, p4, p5 = st.columns(5)
        p1.markdown("**1) Szene (Gamma)**")
        p1.image(sim["scene_gamma"], use_container_width=True)
        p2.markdown("**2) Szene linear**")
        p2.image(sim["scene_linear_vis"], use_container_width=True)
        p3.markdown("**3) Nach Optik**")
        p3.image(sim["optics_vis"], use_container_width=True)
        p4.markdown("**4) Sensor (Bayer, grau)**")
        p4.image(sim["sensor_gray"], clamp=True, use_container_width=True)
        p5.markdown("**5) ISP Output (sRGB)**")
        p5.image(sim["output_rgb"], use_container_width=True)

        info_box(
            "Experiment: Erhoehe ISO und Read Noise gleichzeitig. Du siehst, dass nicht nur Helligkeit steigt, "
            "sondern auch das wahrgenommene Korn/Rauschen.",
            kind="tip",
        )

    with tabs[4]:
        section_header("Messen statt raten", "Histogramme und SNR machen die Pipeline objektiv.")
        mean_e = sim["mean_electrons"].mean()
        var_shot = mean_e
        var_read = read_noise_e ** 2
        snr_linear = mean_e / np.sqrt(max(1e-6, var_shot + var_read))

        st.markdown(
            f"""
            - Geschätzte mittlere Elektronenzahl: **{mean_e:,.0f} e-**
            - Shot-Noise-Varianz (nahezu): **{var_shot:,.0f}**
            - Read-Noise-Varianz: **{var_read:,.2f}**
            - Geschätztes lineares SNR: **{snr_linear:,.2f}**
            """
        )

        fig = go.Figure()
        sensor_vals = sim["sensor_gray"].astype(np.float32).ravel()
        output_gray = cv2.cvtColor(sim["output_rgb"], cv2.COLOR_RGB2GRAY).astype(np.float32).ravel()
        fig.add_trace(go.Histogram(x=sensor_vals, nbinsx=64, name="Sensor (Bayer grau)", opacity=0.65))
        fig.add_trace(go.Histogram(x=output_gray, nbinsx=64, name="Output (ISP grau)", opacity=0.65))
        fig.update_layout(
            template="plotly_dark",
            barmode="overlay",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=10, b=10),
            height=280,
        )
        st.plotly_chart(fig, use_container_width=True)

        divider()
        st.code(
            """
# Minimalmodell fuer Sensor-Noise
mu_e = irradiance * exposure_time * quantum_eff
electrons = Poisson(mu_e) + Normal(0, sigma_read)
digital = round(clip(electrons * gain, 0, full_well) / full_well * (2**bit_depth - 1))
            """.strip(),
            language="python",
        )

    with tabs[5]:
        section_header("Kuratiertes Video-Material", "Fundament + Kamera-nahe Verarbeitung.")
        st.markdown("**How Images Work (Computerphile)**")
        video_embed(
            "LZNva7Kf9IM",
            "How Images Work — Computerphile",
            "Guter Einstieg in Pixel, Rasterung und digitale Bilddarstellung.",
        )
        divider()
        st.markdown("**Gaussian Blur (Computerphile)**")
        video_embed(
            "C_zFhWdM4ic",
            "Gaussian Blur — Computerphile",
            "Hilft den Optik-/PSF-Schritt intuitiv zu verankern.",
        )
        divider()
        st.markdown("**Sobel and Edge Detection (Computerphile)**")
        video_embed(
            "uihBwtPIBxM",
            "Sobel Operator — Computerphile",
            "Typischer erster Verarbeitungsschritt nach dem ISP in CV-Pipelines.",
        )

    with tabs[6]:
        render_learning_block(
            key_prefix="camera_pipeline",
            section_title="Lernpfad: Kameraaufnahme verstehen",
            section_sub="Konzept -> Messung -> Parametertuning -> Mini-Projekt",
            progression=[
                ("🟢", "Guided Lab", "Pipeline mit vorgegebenen Parametern nachvollziehen.", "Beginner", "green"),
                ("🟠", "Challenge", "SNR bei wenig Licht durch Parameterwahl verbessern.", "Intermediate", "amber"),
                ("🔴", "Debug Task", "Banding, Rauschen, Farbstich auf Ursache zurueckfuehren.", "Advanced", "pink"),
                ("🏁", "Mini-Projekt", "Eigenes Kamera-Sim-Lab mit 2 weiteren ISP-Schritten erweitern.", "Abschluss", "blue"),
            ],
            mcq_question="Welche Aussage ist korrekt?",
            mcq_options=[
                "Hoeheres ISO erzeugt mehr Photonen am Sensor.",
                "Shot Noise ist in erster Naeherung Poisson-basiert.",
                "Bayer misst pro Pixel direkt RGB gleichzeitig.",
                "ADC-Bittiefe beeinflusst nur Dateigroesse, nicht Bildqualitaet.",
            ],
            mcq_correct_option="Shot Noise ist in erster Naeherung Poisson-basiert.",
            mcq_success_message="Korrekt. Genau deshalb steigt das Rauschen typischerweise mit sqrt(Signal).",
            mcq_retry_message="Noch nicht. Pruefe Photonenmodell, Bayer und ISO-Rolle nochmals.",
            open_question="Welche zwei Pipeline-Stufen sind fuer dein Projekt am kritischsten und warum?",
            code_task="""# Erweiterungsidee:
# 1) Fuege ein Entrauschungsmodul vor dem Demosaicing ein.
# 2) Vergleiche PSNR/SSIM fuer ISO 100 vs 1600.
""",
            community_rows=[
                {"Format": "Peer Review", "Thema": "Ist die SNR-Interpretation korrekt?", "Output": "Kurzfeedback"},
                {"Format": "Lab Battle", "Thema": "Bestes Low-Light-Setting mit natuerlicher Farbe", "Output": "Parameter-Set"},
                {"Format": "Explain Like I am 12", "Thema": "Shot vs Read Noise", "Output": "1-min Erklaerung"},
            ],
            cheat_sheet=[
                "ISO ist Verstaerkung, kein Lichtgenerator.",
                "Bayer misst pro Pixel nur einen Farbkanal.",
                "Read Noise dominiert oft in sehr dunklen Bereichen.",
                "Mehr Bit-Tiefe reduziert Quantisierungsfehler/Banding.",
            ],
            key_takeaways=[
                "Kamera-Bildentstehung ist eine Kette aus Physik + DSP.",
                "Sichtbare Artefakte lassen sich meist einer Pipeline-Stufe zuordnen.",
            ],
            common_errors=[
                "ISO mit Belichtung verwechseln.",
                "Gamma-kodierte und lineare Daten mischen.",
                "Bayer-Mosaik als echtes RGB interpretieren.",
                "Quantisierungseffekte bei 8 bit unterschaetzen.",
            ],
        )
