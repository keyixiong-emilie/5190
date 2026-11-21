/*
 * app.js — interface for the DPO demo (controls, charts, tooltips)
 * CIS 4190/5190 Deep Dive (Fall 2026)
 * Authors: Keyi Xiong, Shengdong Yao, Yilun Lu
 * License: MIT (see LICENSE)
 *
 * All numerics live in dpo-core.js (window.DPO). This file only reads the
 * controls, calls the core, and draws SVG by hand. No external libraries.
 */
(function () {
  "use strict";
  var D = window.DPO;

  // ---------- the toy problem ----------
  var LABELS = ["A", "B", "C", "D", "E", "F"];
  var RESPONSES = [
    "It is like memorizing the practice exam instead of learning the subject.",
    "The model fits noise in its training data, so it does worse on new data.",
    "Overfitting is when training error is low but test error is high.",
    "It happens when a model is too complex.",
    "Overfitting means the model is very accurate.",
    "I am not sure, but more data usually makes it worse."
  ];
  var PI_REF = [0.05, 0.10, 0.30, 0.25, 0.20, 0.10]; // reference policy
  var R_TRUE = [1.2, 0.6, 0.2, -0.3, -0.7, -1.0];    // hidden "human" reward
  var CHAMPION = 0;                                   // response A

  var T_MAX = 10000;  // gradient steps
  var LR_R = 2;       // Route 1 step size on rewards
  var UNMATCHED = 0.5; // Route 2 uses 0.5 * LR_R / beta^2 unless "matched"

  // Steps we keep: 0, then log-spaced up to T_MAX (charts use a log axis).
  var STEPS = (function () {
    var s = [0], n = 110;
    for (var i = 0; i <= n; i++) {
      var v = Math.round(Math.pow(10, (i * Math.log10(T_MAX)) / n));
      if (v > s[s.length - 1]) s.push(v);
    }
    return s;
  })();

  var state = { mode: "noisy", n: 400, seed: 7, beta: 1, idx: STEPS.length - 1, matched: false, playing: false };
  var cache = { key: "", wins: null, route1: null };
  var run = null; // everything derived for the current settings
  var timer = null;

  var $ = function (id) { return document.getElementById(id); };

  // ---------- compute ----------
  function margin(centered) {
    var s = centered.slice().sort(function (a, b) { return b - a; });
    return s[0] - s[1];
  }

  function recompute() {
    var key = [state.mode, state.n, state.seed].join("|");
    if (cache.key !== key) {
      cache.key = key;
      cache.wins = D.samplePreferences(R_TRUE, state.n, state.seed, state.mode === "champ" ? CHAMPION : -1);
      // Route 1's reward model does not depend on beta, so fit it once per dataset.
      cache.route1 = D.trainRewardModel(cache.wins, T_MAX, LR_R, STEPS);
    }
    var beta = state.beta;
    var lrTheta = (state.matched ? 1 : UNMATCHED) * LR_R / (beta * beta);
    var route2 = D.trainDPO(cache.wins, PI_REF, beta, T_MAX, lrTheta, STEPS);

    var out = { wins: cache.wins, sep: D.separationReport(cache.wins), p1: [], p2: [], rew1: [], rew2: [],
                loss1: cache.route1.losses, loss2: route2.losses, g1: cache.route1.gradNorms, g2: route2.gradNorms, m1: [], m2: [], kl1: [], kl2: [], tv: [] };
    for (var t = 0; t < STEPS.length; t++) {
      var p1 = D.optimalPolicy(cache.route1.rewards[t], PI_REF, beta); // closed form
      var p2 = D.softmax(route2.logits[t]);                             // trained directly
      var w1 = D.center(cache.route1.rewards[t]);
      var w2 = D.implicitRewardFromLogits(route2.logits[t], PI_REF, beta);
      out.p1.push(p1); out.p2.push(p2); out.rew1.push(w1); out.rew2.push(w2);
      out.m1.push(margin(w1)); out.m2.push(margin(w2));
      out.kl1.push(D.kl(p1, PI_REF)); out.kl2.push(D.kl(p2, PI_REF));
      out.tv.push(D.totalVariation(p1, p2));
    }
    run = out;
  }

  // ---------- formatting ----------
  var SUP = { "-": "⁻", "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴", "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹" };
  function sci(v) {
    if (v < 1e-14) return "below 10⁻¹⁴";
    var parts = v.toExponential(0).split("e");
    var exp = String(parseInt(parts[1], 10)).split("").map(function (c) { return SUP[c]; }).join("");
    return parts[0] + "×10" + exp;
  }
  function fmtSmall(v) { return v >= 0.001 ? v.toFixed(3) : sci(v); }
  function fmtP(p) { return p >= 0.0005 ? p.toFixed(3) : "<0.001"; }
  function fmt2(v) { return (v < 0 ? "−" : "") + Math.abs(v).toFixed(2); }
  function fmtInt(v) { return v.toLocaleString("en-US"); }
  function fmtBeta(b) { return b < 1 ? b.toFixed(2) : b < 10 ? b.toFixed(1) : b.toFixed(0); }

  // ---------- svg helpers ----------
  function niceTicks(lo, hi, target) {
    var span = hi - lo || 1, raw = span / (target || 4);
    var mag = Math.pow(10, Math.floor(Math.log10(raw))), norm = raw / mag;
    var step = (norm <= 1 ? 1 : norm <= 2 ? 2 : norm <= 2.5 ? 2.5 : norm <= 5 ? 5 : 10) * mag;
    var start = Math.floor(lo / step + 1e-9) * step, end = Math.ceil(hi / step - 1e-9) * step, ticks = [];
    for (var v = start; v <= end + step * 1e-6; v += step) ticks.push(Math.abs(v) < step * 1e-9 ? 0 : v);
    return ticks;
  }
  function tickLabel(v, ticks) {
    // fewest decimals (up to 3) that print the tick step exactly
    var step = ticks.length > 1 ? Math.abs(ticks[1] - ticks[0]) : 1, d = 0;
    while (d < 3 && Math.abs(step * Math.pow(10, d) - Math.round(step * Math.pow(10, d))) > 1e-9) d++;
    return (v < 0 ? "−" : "") + Math.abs(v).toFixed(d);
  }
  // Column with a 4px rounded data-end and a square end on the baseline.
  function barPath(x, yBase, yVal, w) {
    var h = Math.abs(yVal - yBase);
    if (h < 0.5) return "";
    var r = Math.min(4, h, w / 2), up = yVal < yBase, s = up ? 1 : -1;
    return "M" + x + "," + yBase + "V" + (yVal + s * r) +
      "Q" + x + "," + yVal + " " + (x + r) + "," + yVal + "H" + (x + w - r) +
      "Q" + (x + w) + "," + yVal + " " + (x + w) + "," + (yVal + s * r) + "V" + yBase + "Z";
  }

  // ---------- bar charts (policy, reward) ----------
  function drawBars(el, opts) {
    var W = 560, H = 250, m = { l: 38, r: 8, t: 10, b: 28 };
    var iw = W - m.l - m.r, ih = H - m.t - m.b, K = LABELS.length, band = iw / K;
    var ticks = opts.ticks, lo = ticks[0], hi = ticks[ticks.length - 1];
    var y = function (v) { return m.t + ih * (1 - (Math.max(lo, Math.min(hi, v)) - lo) / (hi - lo)); };
    var bw = 20, gap = 2, s = "";
    s += '<svg viewBox="0 0 ' + W + " " + H + '" role="img" aria-label="' + opts.aria + '">';
    ticks.forEach(function (tv) {
      s += '<line class="' + (tv === 0 ? "baseline" : "gridline") + '" x1="' + m.l + '" x2="' + (W - m.r) + '" y1="' + y(tv) + '" y2="' + y(tv) + '"/>';
      s += '<text x="' + (m.l - 6) + '" y="' + (y(tv) + 4) + '" text-anchor="end">' + tickLabel(tv, ticks) + "</text>";
    });
    for (var i = 0; i < K; i++) {
      var cx = m.l + band * (i + 0.5), x1 = cx - bw - gap / 2, x2 = cx + gap / 2;
      s += '<path class="bar1" d="' + barPath(x1, y(0), y(opts.a[i]), bw) + '"/>';
      s += '<path class="bar2" d="' + barPath(x2, y(0), y(opts.b[i]), bw) + '"/>';
      if (opts.ref) s += '<line class="refmark" x1="' + (x1 - 5) + '" x2="' + (x2 + bw + 5) + '" y1="' + y(opts.ref[i]) + '" y2="' + y(opts.ref[i]) + '"/>';
      s += '<text class="cat" x="' + cx + '" y="' + (H - 8) + '" text-anchor="middle">' + LABELS[i] + "</text>";
      s += '<rect class="hit" data-kind="' + opts.kind + '" data-i="' + i + '" x="' + (m.l + band * i) + '" y="' + m.t + '" width="' + band + '" height="' + ih + '"/>';
    }
    s += "</svg>";
    el.innerHTML = s;
  }

  // ---------- line charts (loss, margin, KL) over a log step axis ----------
  var LM = { W: 360, H: 210, l: 42, r: 14, t: 12, b: 26 };
  function lx(step) { return LM.l + (LM.W - LM.l - LM.r) * (Math.log10(Math.max(step, 1)) / Math.log10(T_MAX)); }

  function drawLines(el, opts) {
    var ih = LM.H - LM.t - LM.b, n = state.idx;
    var all = opts.a.slice(1).concat(opts.b.slice(1));
    if (opts.ceiling != null) all.push(opts.ceiling);
    var mn = opts.zero ? 0 : Math.min.apply(null, all), mx = Math.max.apply(null, all);
    if (mx - mn < 1e-9) mx = mn + 1;
    var ticks = niceTicks(mn, mx, 4), lo = ticks[0], hi = ticks[ticks.length - 1];
    var y = function (v) { return LM.t + ih * (1 - (v - lo) / (hi - lo)); };
    var s = '<svg viewBox="0 0 ' + LM.W + " " + LM.H + '" role="img" aria-label="' + opts.aria + '">';
    ticks.forEach(function (tv) {
      s += '<line class="' + (tv === 0 ? "baseline" : "gridline") + '" x1="' + LM.l + '" x2="' + (LM.W - LM.r) + '" y1="' + y(tv) + '" y2="' + y(tv) + '"/>';
      s += '<text x="' + (LM.l - 6) + '" y="' + (y(tv) + 4) + '" text-anchor="end">' + tickLabel(tv, ticks) + "</text>";
    });
    [[1, "1"], [10, "10"], [100, "100"], [1000, "1,000"], [10000, "10,000"]].forEach(function (t) {
      s += '<text x="' + lx(t[0]) + '" y="' + (LM.H - 8) + '" text-anchor="' + (t[0] === 10000 ? "end" : t[0] === 1 ? "start" : "middle") + '">' + t[1] + "</text>";
    });
    if (opts.ceiling != null) {
      s += '<line class="ceiling" x1="' + LM.l + '" x2="' + (LM.W - LM.r) + '" y1="' + y(opts.ceiling) + '" y2="' + y(opts.ceiling) + '"/>';
      s += '<text class="endlabel" x="' + (LM.l + 4) + '" y="' + (y(opts.ceiling) - 5) + '">' + opts.ceilingLabel + "</text>";
    }
    var from = 1, to = Math.max(n, 1);
    var path = function (arr) {
      var d = "";
      for (var i = from; i <= to; i++) d += (i === from ? "M" : "L") + lx(STEPS[i]).toFixed(1) + "," + y(arr[i]).toFixed(1);
      return d;
    };
    s += '<path class="line1" d="' + path(opts.a) + '"/><path class="line2" d="' + path(opts.b) + '"/>';
    s += '<circle class="dot1" r="5" cx="' + lx(STEPS[to]) + '" cy="' + y(opts.a[to]) + '"/>';
    s += '<circle class="dot2" r="4" cx="' + lx(STEPS[to]) + '" cy="' + y(opts.b[to]) + '"/>';
    s += '<line class="hover" data-hover="1" x1="0" x2="0" y1="' + LM.t + '" y2="' + (LM.H - LM.b) + '" visibility="hidden"/>';
    s += '<rect class="hit" style="fill:transparent" data-kind="' + opts.kind + '" x="' + LM.l + '" y="' + LM.t + '" width="' + (LM.W - LM.l - LM.r) + '" height="' + ih + '"/>';
    s += "</svg>";
    el.innerHTML = s;
  }

  // ---------- tooltip ----------
  var tip = $("tooltip");
  function row(cls, name, val) {
    return '<div class="tt-row"><span class="tt-key"><span class="swatch ' + cls + '"></span>' + name + "</span><b>" + val + "</b></div>";
  }
  function showTip(html, ev) {
    tip.innerHTML = html; tip.hidden = false;
    var pad = 14, w = tip.offsetWidth, h = tip.offsetHeight;
    var x = ev.clientX + pad, yy = ev.clientY + pad;
    if (x + w > window.innerWidth - 8) x = ev.clientX - w - pad;
    if (yy + h > window.innerHeight - 8) yy = ev.clientY - h - pad;
    tip.style.left = Math.max(8, x) + "px"; tip.style.top = Math.max(8, yy) + "px";
  }
  function hideTip() {
    tip.hidden = true;
    document.querySelectorAll("[data-hover]").forEach(function (l) { l.setAttribute("visibility", "hidden"); });
  }
  var LINE_SERIES = { loss: ["loss1", "loss2", "Loss"], margin: ["m1", "m2", "Reward margin"], kl: ["kl1", "kl2", "KL from π_ref"] };

  function onPointer(ev) {
    var hit = ev.target.closest ? ev.target.closest(".hit") : null;
    if (!hit) { hideTip(); return; }
    var kind = hit.getAttribute("data-kind"), t = state.idx, html;
    if (kind === "policy" || kind === "reward") {
      var i = +hit.getAttribute("data-i");
      html = '<div class="tt-title">Response ' + LABELS[i] + "</div>";
      if (kind === "policy") {
        html += row("s1", "Route 1 (BT + closed form)", fmtP(run.p1[t][i])) + row("s2", "Route 2 (DPO)", fmtP(run.p2[t][i])) + row("ref", "π_ref", fmtP(PI_REF[i]));
      } else {
        html += row("s1", "Route 1 learned", fmt2(run.rew1[t][i])) + row("s2", "Route 2 implicit", fmt2(run.rew2[t][i]));
        if (state.mode === "noisy") html += row("ref", "True reward", fmt2(D.center(R_TRUE)[i]));
      }
      showTip(html, ev);
      return;
    }
    // line chart: find the nearest recorded step at or before the current one
    var svg = hit.ownerSVGElement, box = svg.getBoundingClientRect();
    var vx = ((ev.clientX - box.left) / box.width) * LM.W, best = 1, bd = Infinity;
    for (var k = 1; k <= Math.max(state.idx, 1); k++) {
      var d = Math.abs(lx(STEPS[k]) - vx);
      if (d < bd) { bd = d; best = k; }
    }
    var line = svg.querySelector("[data-hover]");
    line.setAttribute("x1", lx(STEPS[best])); line.setAttribute("x2", lx(STEPS[best])); line.setAttribute("visibility", "visible");
    var ser = LINE_SERIES[kind];
    html = '<div class="tt-title">' + ser[2] + " · step " + fmtInt(STEPS[best]) + "</div>" +
      row("s1", "Route 1", run[ser[0]][best].toFixed(4)) + row("s2", "Route 2 (DPO)", run[ser[1]][best].toFixed(4));
    showTip(html, ev);
  }

  // ---------- render ----------
  function render() {
    var t = state.idx, champ = state.mode === "champ";
    $("nPairsOut").textContent = fmtInt(state.n);
    $("betaOut").textContent = fmtBeta(state.beta);
    $("stepOut").textContent = fmtInt(STEPS[t]) + " of " + fmtInt(T_MAX);
    $("play").textContent = state.playing ? "❚❚ Pause" : "▶ Play training";
    $("modeNoisy").setAttribute("aria-pressed", String(!champ));
    $("modeChamp").setAttribute("aria-pressed", String(champ));

    $("tvVal").textContent = fmtSmall(run.tv[t]);
    $("klVal").textContent = run.kl2[t].toFixed(3);
    $("marginVal").textContent = run.m2[t].toFixed(2);
    $("pTopVal").textContent = fmtP(run.p2[t][CHAMPION]);
    $("pTopSub").textContent = "π_ref gives A " + PI_REF[CHAMPION].toFixed(2);

    // callout: say what the reader is looking at
    var c = $("callout"), sep = run.sep, msg, warn = false;
    if (champ) {
      warn = true;
      msg = "<b>Response A beats every other response, every time.</b> Raising A's reward always lowers the loss, so no finite reward fits best: A creates a separating direction in the logistic loss. The reward margin keeps growing. For any fixed β the policy drifts toward “always answer A”, and a larger β keeps it closer to π_ref at any given step. Both routes inherit this from the same logistic regression.";
    } else if (sep.neverLost.length || sep.neverWon.length) {
      warn = true;
      var who = sep.neverLost.concat(sep.neverWon).map(function (i) { return LABELS[i]; }).join(", ");
      msg = "<b>This random sample happens to be separable</b> (response " + who + " won or lost every comparison), so no finite reward fits best. Add more pairs or resample.";
    } else if (run.tv[t] < 1e-6) {
      msg = "<b>Same policy.</b> Total variation distance between the two routes: " + fmtSmall(run.tv[t]) + ". Gradient norms: " + sci(run.g1[t]) + " (Route 1) and " + sci(run.g2[t]) + " (Route 2), so both sit at the minimum of their loss. With r = β(θ − log π_ref) the two losses are the same function, which is why six free logits trained on the same data, π_ref and β reach the same optimal policy.";
    } else if (state.matched) {
      msg = "<b>Matched step sizes.</b> One DPO step of size η moves the implicit reward by η·β², so the two routes coincide at every step, not only at the end.";
    } else {
      msg = "<b>Training in progress.</b> The routes use different step sizes, so they take different paths. They are heading for the same policy.";
    }
    c.innerHTML = msg; c.className = "callout" + (warn ? " warn" : "");
    $("legendRef").innerHTML = champ ? "Reference policy π<sub>ref</sub>" : "Reference policy π<sub>ref</sub> (left chart) · true reward (right chart)";

    drawBars($("policyChart"), { kind: "policy", a: run.p1[t], b: run.p2[t], ref: PI_REF, ticks: [0, 0.25, 0.5, 0.75, 1],
      aria: "Probability of each response under Route 1, Route 2 and the reference policy" });
    var trueC = D.center(R_TRUE), mxR = 0;
    run.rew1[t].concat(run.rew2[t], champ ? [] : trueC).forEach(function (v) { mxR = Math.max(mxR, Math.abs(v)); });
    var R = mxR <= 1.5 ? 1.5 : mxR <= 3 ? 3 : mxR <= 6 ? 6 : mxR <= 12 ? 12 : 24;
    drawBars($("rewardChart"), { kind: "reward", a: run.rew1[t], b: run.rew2[t], ref: champ ? null : trueC, ticks: [-R, -R / 2, 0, R / 2, R],
      aria: "Centered reward of each response: learned by Route 1, implied by Route 2" });

    drawLines($("lossChart"), { kind: "loss", a: run.loss1, b: run.loss2, aria: "Loss against gradient steps for both routes" });
    drawLines($("marginChart"), { kind: "margin", a: run.m1, b: run.m2, zero: true, aria: "Reward margin against gradient steps for both routes" });
    drawLines($("klChart"), { kind: "kl", a: run.kl1, b: run.kl2, zero: true,
      ceiling: champ ? -Math.log(PI_REF[CHAMPION]) : null, ceilingLabel: "all mass on A: ln(1/π_ref(A)) = " + (-Math.log(PI_REF[CHAMPION])).toFixed(2),
      aria: "KL divergence from the reference policy against gradient steps for both routes" });

    // table view
    var h = "<table><thead><tr><th>Response</th><th>π_ref</th><th>Route 1 π</th><th>Route 2 π</th><th>" + (champ ? "" : "True r") + "</th><th>Route 1 r</th><th>Route 2 r</th></tr></thead><tbody>";
    for (var i = 0; i < LABELS.length; i++) {
      h += "<tr><td>" + LABELS[i] + "</td><td>" + PI_REF[i].toFixed(3) + "</td><td>" + fmtP(run.p1[t][i]) + "</td><td>" + fmtP(run.p2[t][i]) + "</td><td>" +
        (champ ? "" : fmt2(trueC[i])) + "</td><td>" + fmt2(run.rew1[t][i]) + "</td><td>" + fmt2(run.rew2[t][i]) + "</td></tr>";
    }
    $("valueTable").innerHTML = h + "</tbody></table>";
    var w = "<table><thead><tr><th>beat →</th>" + LABELS.map(function (l) { return "<th>" + l + "</th>"; }).join("") + "</tr></thead><tbody>";
    run.wins.forEach(function (r, a) {
      w += "<tr><td>" + LABELS[a] + "</td>" + r.map(function (v, b) { return "<td>" + (a === b ? "·" : v) + "</td>"; }).join("") + "</tr>";
    });
    $("winsTable").innerHTML = w + "</tbody></table>";
  }

  function update(needsCompute) { if (needsCompute) recompute(); render(); }

  // ---------- controls ----------
  function stop() { state.playing = false; if (timer) { clearInterval(timer); timer = null; } }
  function play() {
    if (state.playing) { stop(); render(); return; }
    state.playing = true; state.idx = 0; $("step").value = 0;
    timer = setInterval(function () {
      if (state.idx >= STEPS.length - 1) { stop(); render(); return; }
      state.idx += 1; $("step").value = state.idx; render();
    }, 45);
    render();
  }

  function init() {
    $("responseList").innerHTML = RESPONSES.map(function (r, i) { return "<li><b>" + LABELS[i] + "</b><span>" + r + "</span></li>"; }).join("");
    var step = $("step"); step.max = STEPS.length - 1; step.value = state.idx;

    $("modeNoisy").addEventListener("click", function () { state.mode = "noisy"; update(true); });
    $("modeChamp").addEventListener("click", function () { state.mode = "champ"; update(true); });
    $("nPairs").addEventListener("input", function (e) { state.n = +e.target.value; update(true); });
    $("beta").addEventListener("input", function (e) { state.beta = Math.pow(10, +e.target.value); update(true); });
    step.addEventListener("input", function (e) { stop(); state.idx = +e.target.value; render(); });
    $("resample").addEventListener("click", function () { state.seed += 1; update(true); });
    $("matched").addEventListener("change", function (e) { state.matched = e.target.checked; update(true); });
    $("play").addEventListener("click", play);

    document.addEventListener("pointermove", onPointer);
    document.addEventListener("pointerdown", onPointer);
    document.addEventListener("pointerleave", hideTip);
    window.addEventListener("scroll", hideTip, { passive: true });

    update(true);
  }

  init();
})();
