### 1. Logistic-GLMM Link and Random Intercepts

Under a logistic-link generalized linear mixed model (GLMM), we model the probability of the binary outcome (e.g., payment = 1) in group $j$ given a binary treatment (click $t \in {0,1}$) and a group-specific random intercept $u_j$. Formally:

* **Linear predictor**:

 $$
  \begin{align*}
  \eta_{j}(t) = \beta_{0} + \beta_{1}t + u_{j}
  \end{align*}
 $$

  where:

* $\beta_{0}$ is the fixed intercept (MAP estimate),
* $\beta_{1}$ is the fixed log-odds coefficient for `click = 1` (MAP estimate),
* $u_{j}$ is the random intercept for score band $j$, drawn from a normal distribution $\mathcal{N}(0, \sigma_{u}^{2})$ and estimated via `BinomialBayesMixedGLM.fit_map()`.

* **Link function (sigmoid)**:

 $$
  \begin{align*}
  \sigma(x) = \frac{1}{1 + \exp(-x)}
  \end{align*}
 $$

* **Conditional probability**:
  Given $t \in {0,1}$ and group $j$,

  $$
    \begin{align*}
    P(\text{payment} = 1 \mid \text{click} = t, u_{j}) = \sigma(\beta_{0} + \beta_{1}t + u_{j})
    \end{align*}
  $$

  * When $t = 0$ (no click):

   $$
    \begin{align*}
    p_{0}=P(\text{payment}=1 \mid \text{click}=0,u_{j})=\sigma(\beta_{0} + u_{j})
    \end{align*}
   $$

  * When $t = 1$ (click):

   $$
    \begin{align*}
    p_{1}=P(\text{payment}=1 \mid \text{click}=1,u_{j})=\sigma(\beta_{0} + \beta_{1} + u_{j})
    \end{align*}
   $$

Because $u_{j}$ appears in both expressions, we isolate the treatment effect $\beta_{1}$ while conditioning on the same group-level random intercept.

---

### 2. Additive (Absolute) Lift

* **Definition**: The **additive lift** $L_{\text{add}}$ for group $j$ is the difference in predicted probabilities when `click` goes from $0$ to $1$, holding $u_{j}$ fixed:

 $$
  \begin{align*}
  L_{\text{add}}= p_{1} - p_{0}
  \end{align*}
 $$

* **Derivation**:

  1. Start with

    $$
    \begin{align*}
     p_{0} &= \sigma(\beta_{0} + u_{j}) \\
     p_{1} &= \sigma(\beta_{0} + \beta_{1} + u_{j})
    \end{align*}
    $$

  2. Subtract:

    $$
    \begin{align*}
     L_{\text{add}} = \sigma(\beta_{0} + \beta_{1} + u_{j}) - \sigma(\beta_{0} + u_{j})
    \end{align*}
    $$

  3. Since $p_{0}, p_{1} \in [0,1]$, it follows that

    $$
    \begin{align*}
     L_{\text{add}} \in [-1,1]
    \end{align*}
    $$

* **Interpretation**:

  *$L_{\text{add}}$ is measured in **percentage points**.

  * If $p_{0} = 0.10$ ($10 \%$) and $p_{1} = 0.15$ ($15\%$) for the same $u_{j}$, then

   $$
    \begin{align*}
    L_{\text{add}} = 0.15 - 0.10 = 0.05
    \end{align*}
   $$

    which means a **5 percentage-point** increase in payment probability when the customer clicks, conditional on that group’s random effect.

### 3. Multiplicative (Relative) Lift

* **Definition**: The **multiplicative lift** $L_{\text{rel}}$ for group $j$ is the relative change in predicted probability, again holding $u_{j}$ fixed:

 $$
  \begin{align*}
  L_{\text{rel}}= \frac{p_{1} - p_{0}}{p_{0}}
  \end{align*}
 $$

* **Derivation**:

  1. Note that

    $$
     p_{0} = \sigma(\beta_{0} + u_{j}),
     \quad
     p_{1} = \sigma(\beta_{0} + \beta_{1} + u_{j}).
    $$

  2. Compute the difference and scale by $p_{0}$:

    $$
     L_{\text{rel}}
     = \frac{\bigl[\sigma(\beta_{0} + \beta_{1} + u_{j}) - \sigma(\beta_{0} + u_{j})\bigr]}%
            {\sigma(\beta_{0} + u_{j})}.
    $$

  3. Provided $p_{0} = \sigma(\beta_{0} + u_{j}) \neq 0$, the ratio is finite. If $p_{0} \approx 0$, the ratio diverges or is undefined; in code, we guard this case via:

     ```python
     if np.isclose(p0, 0.0, atol=1e-12):
         return np.nan
     ```

     so that $L_{\text{rel}} = \text{NaN}$ when $p_{0}$ is effectively zero.

* **Interpretation (conditional on $u_{j}$)**:

  * By conditioning on the same $u_{j}$, we remove group-level variation from the comparison. The entire difference $(p_{1} - p_{0})$ stems from adding $\beta_{1}$ to the log-odds.

  * Dividing by $p_{0}$ yields a **proportional change** relative to the baseline probability. Concretely, if $p_{0} = 0.10$ and $p_{1} = 0.15$ for a given $u_{j}$, then:

   $$
   \begin{align*}
    L_{\text{rel}} = \frac{0.15 - 0.10}{0.10} = 0.50
    \end{align*}
   $$

    meaning a **50 % relative increase** in payment probability given a click, after accounting for that band’s random intercept.

* **Key point**: Because both $p_{0}$ and $p_{1}$ include the same $u_{j}$, the relative lift isolates exactly how much the fixed-effect increment $\beta_{1}$ magnifies the baseline probability $p_{0}$.

---

### Summary

* $\beta_{0}$: fixed intercept from the GLMM (MAP estimate).
* $\beta_{1}$: fixed log-odds coefficient for `click = 1` (MAP estimate).
* $u_{j}$: random intercept mean for score band $j$ (estimated by `BinomialBayesMixedGLM`).
* $p_{0} = \sigma(\beta_{0} + u_{j})$: predicted probability when `click = 0`.
* $p_{1} = \sigma(\beta_{0} + \beta_{1} + u_{j})$: predicted probability when `click = 1`.

* **Additive lift**:

 $$
  L_{\text{add}}
  = p_{1} - p_{0}.
 $$

* **Multiplicative lift**:

 $$
  L_{\text{rel}}
  = \frac{p_{1} - p_{0}}{p_{0}},
  \quad
  (\text{undefined if } p_{0} = 0)
 $$

All derivations and interpretations above are stated **conditional on** $u_{j}$, ensuring that both additive and multiplicative lifts measure only the effect of `click = 1` (via $\beta_{1}$) and do not conflate group-level heterogeneity.
