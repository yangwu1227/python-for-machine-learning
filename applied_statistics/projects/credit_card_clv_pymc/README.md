
# Customer Lifetime Value (CLV) Project with PyMC Marketing

This repository demonstrates how to model **customer lifetime value** by combining:

- **Pareto/NBD** for predicting the *number of future transactions*.
- **Gamma-Gamma** for estimating *average spend* per transaction.

We then integrate these two models with **discounted cash flow** to produce probabilistic CLV estimates for each credit card account.

## Repository Structure

1. **`data/`**  
   - **raw/** contains original transaction data (e.g., zipped CSV or similar).  
   - **processed/** holds cleaned and feature-engineered data used for modeling.

2. **`models/`**  
   - Contains NetCDF (`.nc`) or other model artifact files from PyMC. These store posterior samples or MAP estimates for:
     - **Pareto/NBD** (purchase frequency and churn),
     - **Gamma-Gamma** (spend per transaction).

3. **`notebooks/`**  
   - **`clv_modeling.ipynb`**: Demonstrates training the Pareto/NBD and Gamma-Gamma models, then combining them to estimate CLV.
   - **`rfm_segments.ipynb`**: Explores Recency-Frequency-Monetary (RFM) segments and initial EDA.

4. **`src/`**  
   - **`model_utils.py`**: Functions and classes for building, fitting, and evaluating the CLV models, plus any data transformations or plotting utilities.

## References

### Quick Links

- [PyMC Marketing CLV Quickstart](https://www.pymc-marketing.io/en/0.10.0/notebooks/clv/clv_quickstart.html)

### Pareto/NBD Model Papers

1. Schmittlein, D. C., Morrison, D. G., & Colombo, R. (1987). [Counting Your Customers: Who Are They and What Will They Do Next](https://www.jstor.org/stable/2631608). *Management Science*, 33(1), 1-24.

2. Fader, P. S., & Hardie, B. G. S. (2005). [A Note on Deriving the Pareto/NBD Model and Related Expressions](http://brucehardie.com/notes/009/pareto_nbd_derivations_2005-11-05.pdf). Technical Note.

3. Fader, P. S., & Hardie, B. G. S. (2014). [Additional Results for the Pareto/NBD Model](https://www.brucehardie.com/notes/015/additional_pareto_nbd_results.pdf). Technical Note.

4. Fader, P. S., & Hardie, B. G. S. (2014). [Deriving the Conditional PMF of the Pareto/NBD Model](https://www.brucehardie.com/notes/028/pareto_nbd_conditional_pmf.pdf). Technical Note.

5. Fader, P. S., & Hardie, B. G. S. (2007). [Incorporating Time-Invariant Covariates into the Pareto/NBD and BG/NBD Models](https://www.brucehardie.com/notes/019/time_invariant_covariates.pdf). Technical Note.

### Gamma-Gamma Model Papers

1. Fader, P. S., & Hardie, B. G. S. (2013). [The Gamma-Gamma Model of Monetary Value](https://www.brucehardie.com/notes/025/gamma_gamma.pdf). Technical Note.

2. Fader, P. S., Hardie, B. G. S., & Lee, K. L. (2005). [RFM and CLV: Using Iso-Value Curves for Customer Base Analysis](https://journals.sagepub.com/doi/pdf/10.1509/jmkr.2005.42.4.415). *Journal of Marketing Research*, 42(4), 415-430.
