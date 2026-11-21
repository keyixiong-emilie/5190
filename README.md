# CIS 5190 DPO Deep Dive

Materials for the Fall 2026 CIS 4190/5190 Deep Dive presentation:
**“Can We Do RLHF Without RL?”**

Authors: Keyi Xiong, Shengdong Yao, and Yilun Lu.

## Repository contents

### `assets/`

- `DPO-Deep-Dive-slides-final.pdf` — final slides in PDF format
- `DPO-Deep-Dive-slides-final.pptx` — editable PowerPoint deck
- `DPO-script-by-speaker.md` — speaker script and running order
- `captions-draft.srt` — draft SRT captions
- `captions-draft.vtt` — draft WebVTT captions
- `dpo-demo.zip` — original packaged interactive demo

### `demo/`

Extracted interactive DPO demonstration. Open `demo/index.html` directly in a
browser; it uses plain HTML, CSS, and JavaScript with no build step or network
requests.

To verify the numerical examples with Node.js:

```bash
node demo/verify.js
```

See [`demo/README.md`](demo/README.md) for details.

## Topic

The presentation derives Direct Preference Optimization from the
KL-regularized RLHF objective, relates pairwise preferences to Bradley–Terry
logistic regression, and demonstrates why DPO and the two-stage reward-model
route converge to the same policy in a finite toy problem. It also covers the
separable-data failure case and the role of the regularization parameter
`beta`.

## License

The demo source is released under the MIT License; see
[`demo/LICENSE`](demo/LICENSE). Presentation materials remain attributed to
their authors.
