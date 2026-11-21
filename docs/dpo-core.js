/*
 * dpo-core.js — numerical core for "Can We Do RLHF Without RL?"
 * CIS 4190/5190 Deep Dive (Fall 2026)
 * Authors: Keyi Xiong, Shengdong Yao, Yilun Lu
 * License: MIT (see LICENSE)
 *
 * Toy setting: one prompt, K candidate responses. A "policy" is a softmax
 * over K logits. Everything here is plain JavaScript with no dependencies,
 * so the same file runs in the browser (index.html) and in Node (verify.js).
 */
(function (root) {
  "use strict";

  // ---------- small numeric helpers ----------

  function sigmoid(z) {
    // numerically stable logistic function
    if (z >= 0) return 1 / (1 + Math.exp(-z));
    var e = Math.exp(z);
    return e / (1 + e);
  }

  function logSigmoid(z) {
    // log(sigmoid(z)) without overflow
    return z >= 0 ? -Math.log1p(Math.exp(-z)) : z - Math.log1p(Math.exp(z));
  }

  function softmax(logits) {
    var m = Math.max.apply(null, logits);
    var ex = logits.map(function (v) { return Math.exp(v - m); });
    var s = ex.reduce(function (a, b) { return a + b; }, 0);
    return ex.map(function (v) { return v / s; });
  }

  function center(v) {
    // subtract the mean: rewards are only identified up to a constant
    var mean = v.reduce(function (a, b) { return a + b; }, 0) / v.length;
    return v.map(function (x) { return x - mean; });
  }

  function norm(v) {
    // Euclidean length of a vector
    return Math.sqrt(v.reduce(function (a, b) { return a + b * b; }, 0));
  }

  function kl(p, q) {
    // KL(p || q) for discrete distributions
    var s = 0;
    for (var i = 0; i < p.length; i++) {
      if (p[i] > 0) s += p[i] * Math.log(p[i] / q[i]);
    }
    return s;
  }

  function totalVariation(p, q) {
    var s = 0;
    for (var i = 0; i < p.length; i++) s += Math.abs(p[i] - q[i]);
    return 0.5 * s;
  }

  // Deterministic pseudo-random generator (mulberry32) so a seed
  // always reproduces the same dataset.
  function makeRng(seed) {
    var a = seed >>> 0;
    return function () {
      a = (a + 0x6D2B79F5) >>> 0;
      var t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  // ---------- data ----------

  /*
   * Sample n preference pairs. Each pair picks two distinct responses
   * uniformly at random; the winner is drawn from the Bradley–Terry model
   *   P(i beats j) = sigmoid(rTrue[i] - rTrue[j]).
   * If `champion` is a valid index, that response wins every comparison it
   * appears in (deterministic preferences — the failure case).
   * Returns a K x K matrix wins[w][l] = number of times w beat l.
   */
  function samplePreferences(rTrue, n, seed, champion) {
    var K = rTrue.length;
    var rng = makeRng(seed);
    var wins = [];
    for (var a = 0; a < K; a++) {
      wins.push(new Array(K).fill(0));
    }
    for (var t = 0; t < n; t++) {
      var i = Math.floor(rng() * K);
      var j = Math.floor(rng() * (K - 1));
      if (j >= i) j += 1;
      var u = rng();
      var iWins;
      if (champion === i) iWins = true;
      else if (champion === j) iWins = false;
      else iWins = u < sigmoid(rTrue[i] - rTrue[j]);
      if (iWins) wins[i][j] += 1; else wins[j][i] += 1;
    }
    return wins;
  }

  function countPairs(wins) {
    var n = 0;
    for (var i = 0; i < wins.length; i++)
      for (var j = 0; j < wins.length; j++) n += wins[i][j];
    return n;
  }

  // Responses that never lost / never won in this dataset. If either list is
  // non-empty, the Bradley–Terry maximum-likelihood estimate does not exist.
  function separationReport(wins) {
    var K = wins.length, neverLost = [], neverWon = [];
    for (var i = 0; i < K; i++) {
      var w = 0, l = 0;
      for (var j = 0; j < K; j++) { w += wins[i][j]; l += wins[j][i]; }
      if (w + l > 0 && l === 0) neverLost.push(i);
      if (w + l > 0 && w === 0) neverWon.push(i);
    }
    return { neverLost: neverLost, neverWon: neverWon };
  }

  // ---------- losses and gradients ----------

  /*
   * Bradley–Terry negative log-likelihood (mean over pairs) as a function of
   * the reward vector r, and its gradient.
   *   loss = -(1/N) * sum_{w,l} wins[w][l] * log sigmoid(r[w] - r[l])
   */
  function btLossGrad(r, wins) {
    var K = r.length, N = countPairs(wins);
    var loss = 0, grad = new Array(K).fill(0);
    for (var w = 0; w < K; w++) {
      for (var l = 0; l < K; l++) {
        var c = wins[w][l];
        if (!c) continue;
        var z = r[w] - r[l];
        loss -= c * logSigmoid(z);
        var g = c * sigmoid(-z); // weight: how wrong the current ranking is
        grad[w] -= g;
        grad[l] += g;
      }
    }
    return { loss: loss / N, grad: grad.map(function (v) { return v / N; }) };
  }

  /*
   * DPO loss (mean over pairs) as a function of the policy logits theta.
   *   margin = beta*log(pi(w)/piRef(w)) - beta*log(pi(l)/piRef(l))
   *   loss   = -(1/N) * sum wins[w][l] * log sigmoid(margin)
   * The gradient is taken through log-softmax explicitly (no shortcut), so
   * this function does not assume the equivalence it is used to demonstrate.
   */
  function dpoLossGrad(theta, wins, piRef, beta) {
    var K = theta.length, N = countPairs(wins);
    var pi = softmax(theta);
    var logPi = pi.map(Math.log);
    var logRef = piRef.map(Math.log);
    var loss = 0, grad = new Array(K).fill(0);
    for (var w = 0; w < K; w++) {
      for (var l = 0; l < K; l++) {
        var c = wins[w][l];
        if (!c) continue;
        var margin = beta * (logPi[w] - logRef[w]) - beta * (logPi[l] - logRef[l]);
        loss -= c * logSigmoid(margin);
        var coef = -c * beta * sigmoid(-margin);
        // d/dtheta_k [log pi(w) - log pi(l)] = (1[k=w] - pi_k) - (1[k=l] - pi_k)
        for (var k = 0; k < K; k++) {
          var dW = (k === w ? 1 : 0) - pi[k];
          var dL = (k === l ? 1 : 0) - pi[k];
          grad[k] += coef * (dW - dL);
        }
      }
    }
    return { loss: loss / N, grad: grad.map(function (v) { return v / N; }) };
  }

  // ---------- the two routes ----------

  // Closed-form KL-optimal policy: pi*(y) ∝ piRef(y) * exp(r(y) / beta)
  function optimalPolicy(r, piRef, beta) {
    return softmax(r.map(function (ri, i) { return Math.log(piRef[i]) + ri / beta; }));
  }

  // Implicit reward of a policy: r(y) = beta * log(pi(y) / piRef(y)), centered
  function implicitReward(pi, piRef, beta) {
    return center(pi.map(function (p, i) { return beta * Math.log(p / piRef[i]); }));
  }

  /*
   * Route 1 (RLHF without the RL): gradient descent on the Bradley–Terry loss
   * over rewards r, starting from r = 0. Returns the reward after every step.
   */
  // Which steps to keep: every step by default, or only those in `record`
  // (an ascending list of step numbers) so long runs stay cheap to store.
  function makeRecorder(record) {
    var ptr = 0;
    return function (t) {
      if (!record) return true;
      while (ptr < record.length && record[ptr] < t) ptr++;
      return ptr < record.length && record[ptr] === t;
    };
  }

  /*
   * Route 1 (RLHF without the RL): gradient descent on the Bradley–Terry loss
   * over rewards r, starting from r = 0. Returns the reward at every recorded
   * step (index 0 is always the starting point).
   */
  function trainRewardModel(wins, steps, lr, record) {
    var K = wins.length, r = new Array(K).fill(0), keep = makeRecorder(record);
    var traj = [], losses = [], gradNorms = [];
    for (var t = 0; t <= steps; t++) {
      var lg = btLossGrad(r, wins);
      if (keep(t)) { traj.push(r.slice()); losses.push(lg.loss); gradNorms.push(norm(lg.grad)); }
      if (t === steps) break;
      for (var i = 0; i < K; i++) r[i] -= lr * lg.grad[i];
    }
    return { rewards: traj, losses: losses, gradNorms: gradNorms };
  }

  /*
   * Route 2 (DPO): gradient descent on the DPO loss over policy logits theta,
   * starting from the reference policy (theta = log piRef).
   */
  function trainDPO(wins, piRef, beta, steps, lr, record) {
    var K = wins.length, theta = piRef.map(Math.log), keep = makeRecorder(record);
    var traj = [], losses = [], gradNorms = [];
    for (var t = 0; t <= steps; t++) {
      var lg = dpoLossGrad(theta, wins, piRef, beta);
      if (keep(t)) { traj.push(theta.slice()); losses.push(lg.loss); gradNorms.push(norm(lg.grad)); }
      if (t === steps) break;
      for (var i = 0; i < K; i++) theta[i] -= lr * lg.grad[i];
    }
    return { logits: traj, losses: losses, gradNorms: gradNorms };
  }

  // Implicit reward straight from logits (avoids log of a tiny probability):
  // beta * (theta - log piRef), centered. The softmax normalizer is a constant
  // shift, which centering removes.
  function implicitRewardFromLogits(theta, piRef, beta) {
    return center(theta.map(function (t, i) { return beta * (t - Math.log(piRef[i])); }));
  }

  var api = {
    sigmoid: sigmoid, logSigmoid: logSigmoid, softmax: softmax, center: center,
    norm: norm, kl: kl, totalVariation: totalVariation, makeRng: makeRng,
    samplePreferences: samplePreferences, countPairs: countPairs,
    separationReport: separationReport,
    btLossGrad: btLossGrad, dpoLossGrad: dpoLossGrad,
    optimalPolicy: optimalPolicy, implicitReward: implicitReward,
    implicitRewardFromLogits: implicitRewardFromLogits,
    trainRewardModel: trainRewardModel, trainDPO: trainDPO
  };

  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.DPO = api;
})(typeof window !== "undefined" ? window : this);
