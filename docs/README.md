# Can We Do RLHF Without RL? — DPO demo

CIS 4190/5190 Deep Dive, Fall 2026
Authors: Keyi Xiong, Shengdong Yao, Yilun Lu
License: MIT (see `LICENSE`)

## Run it

Double-click `index.html`. There is no build step, no server, and no network
request. Everything is plain HTML, CSS and JavaScript with no external libraries.

## Files

| File | What it does |
| --- | --- |
| `index.html` | Page structure and explanatory text |
| `style.css` | Styling, light and dark themes |
| `dpo-core.js` | All numerics: data sampling, Bradley–Terry loss, DPO loss, both training routes |
| `app.js` | Controls, SVG charts, tooltips |
| `verify.js` | Optional Node script that reproduces the numbers quoted in the report (`node verify.js`) |

## The toy problem

One prompt, six candidate responses, a reference policy `pi_ref`, and a hidden
"human" reward. Preference pairs are sampled from the Bradley–Terry model.

- Route 1 (Bradley–Terry reward fit + closed-form KL-optimal policy): fit rewards by logistic regression, then apply the closed-form
  KL-optimal policy `pi*(y) ∝ pi_ref(y) exp(r(y)/beta)`.
- Route 2 (DPO): gradient descent on the DPO loss over the policy logits.

Fixed settings: 10,000 full-batch gradient steps; Route 1 step size 2;
Route 2 step size `0.5 * 2 / beta^2` (or `2 / beta^2` with "Match step sizes").
The default dataset is 400 pairs with seed 7.
