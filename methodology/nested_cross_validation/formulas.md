## Nested $K$-Fold CV

Let

* $N\_{\mathrm{total}}$ = total number of data samples
* $n\_{\mathrm{fold\_outer}}$ = number of outer folds
* $n\_{\mathrm{fold\_inner}}$ = number of inner folds

We denote

* $N\_{\mathrm{outer\_val}}$ = size of each outer **validation** fold
* $N\_{\mathrm{outer\_train}}$ = size of each outer **training** set
* $N\_{\mathrm{inner\_val}}$ = size of each inner **validation** fold (within an outer training set)
* $N\_{\mathrm{inner\_train}}$ = size of each inner **training** set (within an outer training set)

---

## General Formulas

* **Outer validation set size**

  $$
    N_{\mathrm{outer\_val}} = \frac{N_{\mathrm{total}}}{\,n_{\mathrm{fold\_outer}}\,}
  $$

* **Outer training set size**

  $$
    N_{\mathrm{outer\_train}} = N_{\mathrm{total}} - N_{\mathrm{outer\_val}} = N_{\mathrm{total}}\,\frac{n_{\mathrm{fold\_outer}} - 1}{\,n_{\mathrm{fold\_outer}}\,}
  $$

* **Inner validation set size**

  $$
    N_{\mathrm{inner\_val}} = \frac{N_{\mathrm{outer\_train}}}{\,n_{\mathrm{fold\_inner}}\,} = N_{\mathrm{total}}\,\frac{n_{\mathrm{fold\_outer}} - 1}{\,n_{\mathrm{fold\_outer}}\,}\,\frac{1}{\,n_{\mathrm{fold\_inner}}\,} = N_{\mathrm{total}}\,\frac{n_{\mathrm{fold\_outer}} - 1}{\,n_{\mathrm{fold\_outer}}\,n_{\mathrm{fold\_inner}}\,}
  $$

* **Inner training set size**

  $$
    N_{\mathrm{inner\_train}} = N_{\mathrm{outer\_train}} - N_{\mathrm{inner\_val}} = N_{\mathrm{total}}\,\frac{n_{\mathrm{fold\_outer}} - 1}{\,n_{\mathrm{fold\_outer}}\,}\,\frac{n_{\mathrm{fold\_inner}} - 1}{\,n_{\mathrm{fold\_inner}}\,} = N_{\mathrm{total}}\,\frac{(n_{\mathrm{fold\_outer}}-1)(n_{\mathrm{fold\_inner}}-1)}{\,n_{\mathrm{fold\_outer}}\,n_{\mathrm{fold\_inner}}\,}
  $$

---

## Example

We simulate a dataset with

* $N\_{\mathrm{total}} = 20000$ observations
* a binary target $y$
* a grouping variable (for **StratifiedGroupKFold**)
* nested CV with $n\_{\mathrm{fold\_outer}} = 5$ outer folds and $n\_{\mathrm{fold\_inner}} = 10$ inner folds

### Plug-in Values

With

$$
N_{\mathrm{total}} = 20000, \quad n_{\mathrm{fold\_outer}} = 5, \quad n_{\mathrm{fold\_inner}} = 10
$$

we obtain:

1. $N\_{\mathrm{outer\_val}} = 20000 / 5 = 4000$
2. $N\_{\mathrm{outer\_train}} = 20000 \times \tfrac{4}{5} = 16000$
3. $N\_{\mathrm{inner\_val}} = 16000 / 10 = 1600$
4. $N\_{\mathrm{inner\_train}} = 16000 - 1600 = 14400$
