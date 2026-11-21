
## Keyi Xiong — 333 words

### Slide 1 · Title · 0:00–0:14

Hi, this is a CIS 5190 Deep Dive. Our question is can you align a language model with human feedback without any reinforcement learning?

### Slide 2 · RLHF Takes Two Stages. DPO Takes One Loss · 0:14–0:52

Here is the usual recipe. You collect pairs of responses and ask people which one is better. Stage one trains a reward model on those choices. Stage two runs reinforcement learning to earn high reward, with a penalty for drifting from where the model started. In 2023, a paper called Direct Preference Optimization showed that you can delete both stages and use a single loss instead. And it is a loss you already know from this class: logistic regression.

### Slide 3 · Where This Fits · 0:52–1:16

This builds on three parts of the course: logistic regression and maximum likelihood, language models, and the RL module, which gives us the objective we start from. By the end, you should be able to derive the DPO loss, explain what beta controls, and say when DPO has no finite optimum.

### Slide 4 · A Human Preference Is One Logistic Regression Example · 1:16–2:01

Start with the data. One prompt, two responses, and a person picks the one they prefer: y-w, the winner, and y-l, the loser. The Bradley–Terry model gives each response a hidden score, r. The probability that the winner is preferred is the sigmoid of the difference in scores. That is logistic regression, where the logit is a difference of rewards. So a reward model is trained with the usual negative log-likelihood. Remember two things. Only differences of reward matter. And like any logistic regression, this has no finite solution if the data are separable.

### Slide 10 · Takeaways · 7:43–8:13

Three takeaways. One: learning a reward from pairwise preferences is logistic regression. Two: the KL-regularized objective has a closed-form answer, and because preferences are pairwise, the normalizer cancels. That gives the DPO loss. Three: DPO inherits a failure of logistic regression on separable data. Beta controls how far the policy moves at a given step, but it cannot supply a finite optimum.

### Slide 11 · Check Your Understanding · 8:13–8:22

Here are two questions to check your understanding. The answers are in the slide notes. Thanks for watching.

## Shengdong Yao — 329 words

### Slide 5 · The RLHF Objective Has a Closed-Form Solution · 2:01–3:12

Now stage two. RLHF wants the policy pi that maximizes expected reward, minus beta times the KL divergence from a reference model, pi-ref. KL is zero only when the two distributions match, so a large beta keeps the policy near the reference. Usually this is optimized with PPO. But it has an exact solution. Write the reward as beta times the log of e to the r over beta, and fold it into the logarithm. The objective now looks almost like a KL divergence between pi and pi-ref times e to the r over beta. That does not sum to one, so we divide by its sum, Z. What is left is minus beta times the KL between pi and a new distribution, pi-star, plus a constant. KL is zero when pi equals pi-star. So the best policy is the reference model, reweighted by exponentiated reward.

### Slide 6 · From the Optimal Policy to the DPO Loss · 3:12–4:13

There is a catch. Z sums over every possible response, which is hopeless for a language model. Here is the key move. Step one: solve for the reward. It equals beta times the log-ratio of pi-star to pi-ref, plus beta log Z. So any policy secretly defines a reward. Step two: Bradley–Terry only needs the difference of two rewards for the same prompt. Both carry the same beta log Z. Subtract, and Z is gone. Step three: put what is left into the logistic loss, with our trainable model pi-theta in place of pi-star. This is the DPO loss. No reward network, no sampling, no Z. Its gradient pushes the chosen response up relative to the rejected one, weighted by how badly the model currently ranks that pair.

### Slide 7 · Demo: Both Routes Reach the Same Policy · 4:13–4:39

[Stay on the slide. Point at the first bullet.]

How close is DPO to the two-stage recipe? In a small problem, they are identical. Take six responses with one free logit, theta, each. Define r as beta times theta minus log pi-ref. Then the DPO loss is exactly the Bradley–Terry loss: same function, different coordinates. My teammate will now test that in a demo.

## Yilun Lu — 321 words

### Slide 7 · Demo: Both Routes Reach the Same Policy (on the demo) · 4:39–5:38

[Switch to the demo, default settings. Press Play training after the second sentence; tick Match step sizes and press Play again before the last sentence.]

This is our toy language model: six candidate answers and four hundred noisy preference pairs. Route 1, in blue, fits Bradley–Terry rewards and applies the closed-form policy. Route 2, in orange, is DPO, trained directly on the logits. Press play. They take different paths and land on the same policy. The gap between them falls from about five percent to below ten to the minus fourteen, and both gradients are essentially zero. If we match the step sizes, using eta times beta squared, the two paths coincide at every step.

### Slide 8 · What β Controls · 5:38–6:28

[Untick Match step sizes. Point at the Reward chart, then the Policy chart. Drag beta to 0.10, then to 10, as you say each value.]

Now look at beta equals one. Response A has the highest learned reward, but the most likely answer is still C. Why? The policy is the reference model reweighted by reward, and the reference gave C thirty percent and A only five. At beta equals zero point one, A jumps to about ninety-eight percent. At beta equals ten, the policy is almost exactly the reference. The reward bars never move. Beta does not change what is learned. It changes how strongly we act on it.

### Slide 9 · Failure Case: Response A Always Wins · 6:28–7:43

[Set beta back to 1.0. Click Response A always wins. Point at the Reward margin chart. For the second half, drag beta from 1 to 3 to 10 and point at “DPO probability of response A”.]

Now the failure case. Suppose A beats every other response, every single time. This is the separable case. Raising A's reward always lowers the loss, so there is no finite best answer. On noisy data, the reward margin levels off near one half. Here it keeps climbing, by about two point three for every tenfold increase in steps. Both routes do this, because they are the same logistic regression. Azar and co-authors pointed out this weakness in 2024. So what does beta do now? At any fixed step, it still matters. After ten thousand steps, the margin is the same for every beta, but the probability of A is ninety-nine percent at beta one, forty-six at beta three, and eleven at beta ten. What beta no longer gives you is a place to stop. For any fixed beta, the policy keeps drifting toward always answering A.

## Running order

| Time | Slide | Speaker |
|---|---|---|
| 0:00–0:14 | 1 · Title | Keyi Xiong |
| 0:14–0:52 | 2 · RLHF Takes Two Stages. DPO Takes One Loss | Keyi Xiong |
| 0:52–1:16 | 3 · Where This Fits | Keyi Xiong |
| 1:16–2:01 | 4 · A Human Preference Is One Logistic Regression Example | Keyi Xiong |
| 2:01–3:12 | 5 · The RLHF Objective Has a Closed-Form Solution | Shengdong Yao |
| 3:12–4:13 | 6 · From the Optimal Policy to the DPO Loss | Shengdong Yao |
| 4:13–4:39 | 7 · Demo: Both Routes Reach the Same Policy | Shengdong Yao |
| 4:39–5:38 | 7 · Demo: Both Routes Reach the Same Policy (on the demo) | Yilun Lu |
| 5:38–6:28 | 8 · What β Controls | Yilun Lu |
| 6:28–7:43 | 9 · Failure Case: Response A Always Wins | Yilun Lu |
| 7:43–8:13 | 10 · Takeaways | Keyi Xiong |
| 8:13–8:22 | 11 · Check Your Understanding | Keyi Xiong |
| 8:22–8:27 | 12 · Sources & Acknowledgements | no narration |
