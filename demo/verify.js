/*
 * verify.js — numerical checks behind the report. Run: node verify.js
 * CIS 4190/5190 Deep Dive (Fall 2026)
 * Authors: Keyi Xiong, Shengdong Yao, Yilun Lu
 * License: MIT (see LICENSE)
 */
const D = require("./dpo-core.js");
const piRef = [0.05, 0.10, 0.30, 0.25, 0.20, 0.10];
const rTrue = [1.2, 0.6, 0.2, -0.3, -0.7, -1.0];
const f = (v, d = 4) => v.map(x => x.toFixed(d)).join(" ");
const maxAbs = (a, b) => Math.max(...a.map((x, i) => Math.abs(x - b[i])));

// 0. finite-difference gradient check of both losses
{
  const wins = D.samplePreferences(rTrue, 400, 7, -1);
  const th = [0.3, -0.2, 0.5, 0.1, -0.4, 0.2], eps = 1e-6, beta = 0.7;
  const g = D.dpoLossGrad(th, wins, piRef, beta).grad;
  const gb = D.btLossGrad(th, wins).grad;
  let e1 = 0, e2 = 0;
  for (let k = 0; k < 6; k++) {
    const a = th.slice(), b = th.slice(); a[k] += eps; b[k] -= eps;
    e1 = Math.max(e1, Math.abs((D.dpoLossGrad(a, wins, piRef, beta).loss - D.dpoLossGrad(b, wins, piRef, beta).loss) / (2 * eps) - g[k]));
    e2 = Math.max(e2, Math.abs((D.btLossGrad(a, wins).loss - D.btLossGrad(b, wins).loss) / (2 * eps) - gb[k]));
  }
  console.log("grad check  dpo err", e1.toExponential(2), " bt err", e2.toExponential(2));
}

// 1. two routes, noisy data
for (const N of [400]) for (const seed of [7]) {
  const wins = D.samplePreferences(rTrue, N, seed, -1);
  console.log("\nwins matrix (row beat col):"); wins.forEach(r => console.log("  " + r.join("\t")));
  console.log("separation:", JSON.stringify(D.separationReport(wins)));
  for (const beta of [0.1, 0.5, 1, 3, 10]) {
    const T = 300, lrR = 2, mult = 0.5;
    const r1 = D.trainRewardModel(wins, T, lrR);
    const r2 = D.trainDPO(wins, piRef, beta, T, mult * lrR / (beta * beta));
    const rHat = D.center(r1.rewards[T]);
    const p1 = D.optimalPolicy(r1.rewards[T], piRef, beta);
    const p2 = D.softmax(r2.logits[T]);
    const pStar = D.optimalPolicy(rTrue, piRef, beta);
    const imp = D.implicitReward(p2, piRef, beta);
    console.log(`\nbeta=${beta}`);
    console.log("  route1 policy :", f(p1));
    console.log("  route2 policy :", f(p2));
    console.log("  true pi*      :", f(pStar));
    console.log("  TV(r1,r2)=", D.totalVariation(p1, p2).toExponential(2),
      " max|rhat-implicit|=", maxAbs(rHat, imp).toExponential(2),
      " loss1=", r1.losses[T].toFixed(6), " loss2=", r2.losses[T].toFixed(6),
      " KL(p2||ref)=", D.kl(p2, piRef).toFixed(4), " TV(p2,pstar)=", D.totalVariation(p2, pStar).toFixed(4));
    if (beta === 1) {
      console.log("  rhat(centered):", f(rHat, 3), " true(centered):", f(D.center(rTrue), 3));
      for (const t of [5, 20, 50, 100, 200, 300]) {
        const a = D.optimalPolicy(r1.rewards[t], piRef, beta), b = D.softmax(r2.logits[t]);
        console.log(`   step ${t}: TV=${D.totalVariation(a, b).toExponential(2)} loss1=${r1.losses[t].toFixed(5)} loss2=${r2.losses[t].toFixed(5)}`);
      }
      // matched step sizes -> identical paths
      const r3 = D.trainDPO(wins, piRef, beta, T, lrR / (beta * beta));
      let worst = 0;
      for (let t = 0; t <= T; t++) worst = Math.max(worst, D.totalVariation(D.optimalPolicy(r1.rewards[t], piRef, beta), D.softmax(r3.logits[t])));
      console.log("  matched steps: max TV over whole path =", worst.toExponential(2));
    }
  }
  // matched step with beta != 1
  {
    const beta = 0.5, T = 300, lrR = 2;
    const r1 = D.trainRewardModel(wins, T, lrR), r3 = D.trainDPO(wins, piRef, beta, T, lrR / (beta * beta));
    let worst = 0;
    for (let t = 0; t <= T; t++) worst = Math.max(worst, D.totalVariation(D.optimalPolicy(r1.rewards[t], piRef, beta), D.softmax(r3.logits[t])));
    console.log("\nmatched steps beta=0.5: max TV over path =", worst.toExponential(2));
  }
}

// 2. failure case: champion always wins
{
  const champ = 0;
  const wins = D.samplePreferences(rTrue, 400, 7, champ);
  console.log("\nFAILURE CASE separation:", JSON.stringify(D.separationReport(wins)));
  for (const beta of [0.5, 1, 3, 10]) {
    const T = 100000, lrR = 2;
    const r2 = D.trainDPO(wins, piRef, beta, T, 0.5 * lrR / (beta * beta));
    const out = [];
    for (const t of [10, 100, 300, 1000, 10000, 100000]) {
      const p = D.softmax(r2.logits[t]);
      const imp = D.implicitReward(p, piRef, beta);
      const margin = imp[champ] - Math.max(...imp.filter((_, i) => i !== champ));
      out.push(`t=${t}: margin=${margin.toFixed(2)} p(champ)=${p[champ].toFixed(4)} KL=${D.kl(p, piRef).toFixed(3)} loss=${r2.losses[t].toFixed(4)}`);
    }
    console.log(`beta=${beta}\n  ` + out.join("\n  "));
  }
  console.log("KL ceiling -log piRef(champ) =", (-Math.log(piRef[champ])).toFixed(3));
  // noisy comparison at same steps
  const winsN = D.samplePreferences(rTrue, 400, 7, -1);
  for (const beta of [1, 3, 10]) {
    const r2 = D.trainDPO(winsN, piRef, beta, 100000, 0.5 * 2 / (beta * beta));
    const p = D.softmax(r2.logits[100000]);
    console.log(`noisy beta=${beta} t=1e5: p(champ)=${p[0].toFixed(4)} KL=${D.kl(p, piRef).toFixed(3)}`);
  }
}

// 3. robustness: 20 seeds x 3 dataset sizes x 3 betas (Section 7 of the report)
{
  let worst = 0, runs = 0, skipped = 0;
  for (let s = 1; s <= 20; s++) for (const n of [100, 400, 2000]) for (const b of [0.3, 1, 3]) {
    const wins = D.samplePreferences(rTrue, n, s, -1), sep = D.separationReport(wins);
    if (sep.neverLost.length || sep.neverWon.length) { skipped++; continue; }
    const a = D.trainRewardModel(wins, 3000, 2, [3000]), c = D.trainDPO(wins, piRef, b, 3000, 1 / (b * b), [3000]);
    runs++; worst = Math.max(worst, D.totalVariation(D.optimalPolicy(a.rewards[0], piRef, b), D.softmax(c.logits[0])));
  }
  console.log(`\nrobustness: ${runs} runs, ${skipped} separable samples skipped, worst TV after 3000 steps = ${worst.toExponential(2)}`);
}

// 4. optimality: gradient norms at the end of training (beta = 1, default data)
{
  const wins = D.samplePreferences(rTrue, 400, 7, -1);
  const a = D.trainRewardModel(wins, 10000, 2, [0, 300, 10000]), c = D.trainDPO(wins, piRef, 1, 10000, 1, [0, 300, 10000]);
  console.log("\ngrad norms  route1:", a.gradNorms.map(x => x.toExponential(2)).join("  "), " route2:", c.gradNorms.map(x => x.toExponential(2)).join("  "));
  console.log("TV at 10000 (unmatched):", D.totalVariation(D.optimalPolicy(a.rewards[2], piRef, 1), D.softmax(c.logits[2])).toExponential(2));
}

// 5. failure case with a FIXED DPO step size (no 1/beta^2 scaling): margins now differ across beta
{
  const wins = D.samplePreferences(rTrue, 400, 7, 0);
  for (const beta of [1, 3, 10]) {
    const rec = [100, 1000, 10000], c = D.trainDPO(wins, piRef, beta, 10000, 1, rec);
    console.log(`fixed eta_theta=1, beta=${beta}: ` + rec.map((t, k) => {
      const p = D.softmax(c.logits[k]), imp = D.implicitRewardFromLogits(c.logits[k], piRef, beta);
      return `t=${t} margin=${(imp[0] - Math.max(...imp.slice(1))).toFixed(2)} pA=${p[0].toFixed(3)}`;
    }).join(" | "));
  }
}
