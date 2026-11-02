# Linear time-varying damped oscillator verification

This workspace contains an independent verification of the equation

  d/dt [A(t) y'(t)] + gamma y'(t) + omega0^2 y(t) = F(t)

with A(t) > 0 prescribed, gamma >= 0, omega0 > 0.

## Run order

1. Install the exact dependency set from `requirements.txt`.
2. Set the platform context to Windows 11 / x86_64.
3. Run the solver verification script exactly as:
   `c:/python313/python.exe oscillator_verification.py`
4. Inspect the generated files in the `output/` directory:
   - `results.json`
   - `convergence.png`
   - `verification_report.pdf`

## Exact dependency versions used

- Python: 3.13.7
- NumPy: 2.4.4
- SciPy: 1.17.1
- SymPy: 1.14.0
- Matplotlib: 3.10.9
- ReportLab: 4.4.4

## Reproducibility

- Platform: Windows 11 / x86_64
- Seed: 0
- Full run command: `c:/python313/python.exe oscillator_verification.py`
- Output directory: `output/`
- Results file: `output/results.json`
- Figure file: `output/convergence.png`
- PDF report: `output/verification_report.pdf`

The script writes all numerical outputs to JSON for audit and uses a fixed seed of `0` for the randomized stability study.

## Notes

- The symbolic residuals are checked using exact SymPy simplification.
- The manufactured forcing uses the exact second derivative of the chosen exact solution.
- The frequency formula used for underdamped constant-A limiting tests is the specified expression
  `sqrt(omega0^2 / a - gamma^2 / (4 a^2))`.
- The numerical check that did not pass is the energy-ledger accumulation test: the symbolic identity is zero, but the cumulative trapezoidal power residual remains above the fixed threshold of `1e-6` under refinement.
