## Nested $K$-Fold CV

Let

* $N_{\text{total}}$ = total number of data samples
* $n_{\text{fold outer}}$ = number of outer folds
* $n_{\text{fold inner}}$ = number of inner folds

We denote

* $N_{\text{outer val}}$ = size of each outer **validation** fold
* $N_{\text{outer train}}$ = size of each outer **training** set
* $N_{\text{inner val}}$ = size of each inner **validation** fold (within an outer training set)
* $N_{\text{inner train}}$ = size of each inner **training** set (within an outer training set)

---

## General Formulas

* **Outer validation set size**

$$
\begin{align*}
N_{\text{outer val}} = \frac{N_{\text{total}}}{n_{\text{fold outer}}}
\end{align*}
$$

* **Outer training set size**

$$
\begin{align*}
N_{\text{outer train}} &= N_{\text{total}} - N_{\text{outer val}} \\
                        &= N_{\text{total}} - \frac{N_{\text{total}}}{n_{\text{fold outer}}} \\
                        &= N_{\text{total}} (1 - \frac{1}{n_{\text{fold outer}}}) \\
                        &= N_{\text{total}} (\frac{1}{1} - \frac{1}{n_{\text{fold outer}}}) \\
                        &= N_{\text{total}} (\frac{n_{\text{fold outer}}}{n_{\text{fold outer}}} - \frac{1}{n_{\text{fold outer}}}) \\
                        &= N_{\text{total}} \frac{n_{\text{fold outer}} - 1}{n_{\text{fold outer}}}
\end{align*}
$$

* **Inner validation set size**

$$
\begin{align*}
N_{\text{inner val}} &= \frac{N_{\text{outer train}}}{n_{\text{fold inner}}} \\
                      &= \frac{N_{\text{total}} \frac{n_{\text{fold outer}} - 1}{n_{\text{fold outer}}}}{n_{\text{fold inner}}} \\
                      &= N_{\text{total}}\frac{n_{\text{fold outer}} - 1}{n_{\text{fold outer}}}\frac{1}{n_{\text{fold inner}}} \\
                      &= N_{\text{total}}\frac{n_{\text{fold outer}} - 1}{n_{\text{fold outer}}n_{\text{fold inner}}}
\end{align*}
$$

* **Inner training set size**

$$  
\begin{align*}
N_{\text{inner train}} &= N_{\text{outer train}} - N_{\text{inner val}} \\
                        &= \big[N_{\text{total}} \frac{n_{\text{fold outer}} - 1}{n_{\text{fold outer}}}\big] - \big[N_{\text{total}}\frac{n_{\text{fold outer}} - 1}{n_{\text{fold outer}}n_{\text{fold inner}}}\big] \\
                        &= N_{\text{total}} \big[\frac{n_{\text{fold outer}} - 1}{n_{\text{fold outer}}} - \frac{n_{\text{fold outer}} - 1}{n_{\text{fold outer}}n_{\text{fold inner}}}\big] \\
                        &= N_{\text{total}} \big[\frac{(n_{\text{fold outer}} - 1)n_{\text{fold inner}} - (n_{\text{fold outer}} - 1)}{n_{\text{fold outer}}n_{\text{fold inner}}}\big] \\
                        &= N_{\text{total}} \big[\frac{n_{\text{fold outer}}n_{\text{fold inner}} - n_{\text{fold inner}} - n_{\text{fold outer}} + 1}{n_{\text{fold outer}}n_{\text{fold inner}}}\big] \\
                        &= N_{\text{total}} \big[\frac{(n_{\text{fold outer}} - 1)(n_{\text{fold inner}} - 1)}{n_{\text{fold outer}}n_{\text{fold inner}}}\big] \\
                        &= N_{\text{total}}\frac{n_{\text{fold outer}} - 1}{n_{\text{fold outer}}}\frac{n_{\text{fold inner}} - 1}{n_{\text{fold inner}}}
\end{align*}
$$

## Example

We simulate a dataset with

* $N {\text{total}} = 20000$ observations
* a binary target $y$
* a grouping variable (for **StratifiedGroupKFold**)
* nested CV with $n {\text{fold outer}} = 5$ outer folds and $n {\text{fold inner}} = 10$ inner folds

### Plug-in Values

With

$$
N_{\text{total}} = 20000, \quad n_{\text{fold outer}} = 5, \quad n_{\text{fold inner}} = 10
$$

we obtain:

1. $N_{\text{outer val}} = \frac{N {\text{total}}}{n {\text{fold outer}}} = \frac{20000}{5} = 4000$

2. $N_{\text{outer train}} = N {\text{total}} \frac{n_{\text{fold outer}} - 1}{n_{\text{fold outer}}} = 20000 \frac{5 - 1}{5} = 16000$

3. $N_{\text{inner val}} = N_{\text{total}} \frac{n_{\text{fold outer}} - 1}{n_{\text{fold outer}}n_{\text{fold inner}}} = 20000 \frac{5 - 1}{5 \cdot 10} = 1600$

   * Note: $N_{\text{inner val}}$ is the size of each inner validation fold within an outer training set.

4. $N_{\text{inner train}} = N_{\text{total}} \frac{n_{\text{fold outer}} - 1}{n_{\text{fold outer}}} \frac{n_{\text{fold inner}} - 1}{n_{\text{fold inner}}} = 20000 \frac{5 - 1}{5} \frac{10 - 1}{10} = 14400$

   * Note: $N_{\text{inner train}}$ is the size of each inner training set within an outer training set.

## Caveats

1. **Exact Divisibility** – Real-world splitters (e.g. `StratifiedKFold`, `GroupKFold`) may distribute a remainder of $\pm 1$ sample per fold.

2. **Grouping constraints** – With `StratifiedGroupKFold`, group boundaries can make fold sizes uneven even when the arithmetic divides perfectly.
