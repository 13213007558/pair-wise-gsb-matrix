# frozen_string_literal: false
require 'test/unit'
require 'matrix'

class TestMatrixDARE < Test::Unit::TestCase
  def frobenius(m)
    Math.sqrt(m.each.inject(0.0) { |sum, e| sum + e * e })
  end

  def assert_matrix_in_delta(expected, actual, delta, message = nil)
    assert_equal(expected.row_count, actual.row_count, message)
    assert_equal(expected.column_count, actual.column_count, message)
    expected.row_count.times do |i|
      expected.column_count.times do |j|
        assert_in_delta(expected[i, j], actual[i, j], delta, message)
      end
    end
  end

  # Scalar system with a = 2, b = q = r = 1 has the exact stabilizing
  # solution X = 2 + sqrt(5), K = 2X/(1 + X), closed loop 2/(1 + X).
  def test_scalar_exact_solution
    result = Matrix.solve_discrete_are(Matrix[[2.0]], Matrix[[1.0]],
                                       Matrix[[1.0]], Matrix[[1.0]])
    assert(result.ok?, result.diagnostics[:message])
    assert_nil(result.failure_reason)

    x_exact = 2.0 + Math.sqrt(5.0)
    assert_in_delta(x_exact, result.x[0, 0], 1e-12)
    assert_in_delta(2.0 * x_exact / (1.0 + x_exact), result.gain[0, 0], 1e-12)
    assert_in_delta(2.0 / (1.0 + x_exact), result.closed_loop_matrix[0, 0], 1e-12)

    diag = result.diagnostics
    assert(diag[:converged])
    assert_operator(diag[:iterations], :<, 30)
    assert_operator(diag[:relative_residual], :<, 1e-12)
    assert_in_delta(2.0 / (1.0 + x_exact), diag[:closed_loop_spectral_radius], 1e-12)
    assert_operator(diag[:stability_margin], :>, 0.5)
    assert_operator(diag[:subspace_separation], :>, 0.5)
  end

  # 2x2 system built backwards from a known solution: pick X0, K0 is then
  # determined, and Q is chosen so that X0 solves the DARE exactly.
  def test_constructed_system_recovers_exact_solution
    a = Matrix[[1.2, 0.3], [0.1, 1.1]]
    b = Matrix[[1.0, 0.0], [0.5, 1.0]]
    x0 = Matrix[[2.0, 0.5], [0.5, 1.5]]
    r = Matrix[[2.0, 0.2], [0.2, 1.0]]
    k0 = (r + b.t * x0 * b).inverse * (b.t * x0 * a)
    q = x0 - a.t * x0 * a + k0.t * (r + b.t * x0 * b) * k0
    q = (q + q.t) / 2
    assert(q.eigen.eigenvalues.min > 0, "construction must give Q > 0")
    assert((a - b * k0).eigen.eigenvalues.map(&:abs).max < 1,
           "construction must be stabilizing")

    result = Matrix.solve_discrete_are(a, b, q, r)
    assert(result.ok?, result.diagnostics[:message])
    assert_matrix_in_delta(x0, result.x, 1e-10)
    assert_matrix_in_delta(k0, result.gain, 1e-10)
    assert_operator(result.diagnostics[:relative_residual], :<, 1e-12)
  end

  # A is singular (nilpotent delay chain); the doubling iteration handles it
  # without forming A**-1. Exact solution: X = diag(1, 2), deadbeat closed loop.
  def test_singular_a
    a = Matrix[[0.0, 1.0], [0.0, 0.0]]
    b = Matrix[[0.0], [1.0]]
    result = Matrix.solve_discrete_are(a, b, Matrix.I(2), Matrix[[1.0]])
    assert(result.ok?, result.diagnostics[:message])
    assert_matrix_in_delta(Matrix[[1.0, 0.0], [0.0, 2.0]], result.x, 1e-12)
    assert_in_delta(0.0, result.diagnostics[:closed_loop_spectral_radius], 1e-12)
  end

  # Discrete double integrator with scalar input: verify the Riccati equation
  # and closed-loop stability independently of the solver internals.
  def test_double_integrator
    a = Matrix[[1.0, 1.0], [0.0, 1.0]]
    b = Matrix[[0.0], [1.0]]
    q = Matrix.I(2)
    r = Matrix[[1.0]]
    result = Matrix.solve_discrete_are(a, b, q, r)
    assert(result.ok?, result.diagnostics[:message])

    x = result.x
    k = result.gain
    assert_equal(x, x.t)
    assert_operator(x.eigen.eigenvalues.min, :>, 0)
    assert_matrix_in_delta((r + b.t * x * b).inverse * (b.t * x * a), k, 1e-12)
    riccati = a.t * x * a - (a.t * x * b) * k + q
    assert_operator(frobenius(riccati - x), :<, 1e-10)
    assert_operator(result.closed_loop_matrix.eigen.eigenvalues.map(&:abs).max, :<, 1)
    assert_equal(result.closed_loop_matrix, a - b * k)
  end

  # Unstable mode not detectable in the cost: a positive semidefinite
  # solution exists but no stabilizing one.
  def test_unstabilizable_undetectable
    a = Matrix.diagonal(2.0, 0.5)
    b = Matrix[[1.0], [1.0]]
    q = Matrix.diagonal(0.0, 1.0)
    result = Matrix.solve_discrete_are(a, b, q, Matrix[[1.0]])
    assert(!result.ok?)
    assert_equal(:unstabilizable, result.failure_reason)
    assert_operator(result.diagnostics[:closed_loop_spectral_radius], :>, 1.0)
    assert_match(/stabiliz/, result.diagnostics[:message])
  end

  # Unstable mode not controllable: the iteration diverges.
  def test_unstabilizable_uncontrollable
    a = Matrix.diagonal(2.0, 0.5)
    b = Matrix[[0.0], [1.0]]
    result = Matrix.solve_discrete_are(a, b, Matrix.I(2), Matrix[[1.0]])
    assert(!result.ok?)
    assert_equal(:non_convergence, result.failure_reason)
    assert(!result.diagnostics[:converged])
  end

  # Optimal closed-loop pole at 1 - 1e-10: the symplectic spectrum cannot be
  # classified reliably this close to the unit circle.
  def test_spectrum_near_unit_circle
    result = Matrix.solve_discrete_are(Matrix[[1.0]], Matrix[[1.0]],
                                       Matrix[[1.0]], Matrix[[1e20]])
    assert(!result.ok?)
    assert_equal(:spectrum_near_unit_circle, result.failure_reason)
    assert_in_delta(1.0, result.diagnostics[:closed_loop_spectral_radius], 1e-8)
  end

  # Optimal closed-loop pole at 1 - 1e-5: still classifiable, but the
  # invariant subspace is too ill-conditioned to trust the solution.
  def test_ill_conditioned_subspace
    result = Matrix.solve_discrete_are(Matrix[[1.0]], Matrix[[1.0]],
                                       Matrix[[1.0]], Matrix[[1e10]])
    assert(!result.ok?)
    assert_equal(:ill_conditioned_subspace, result.failure_reason)
    assert_operator(result.diagnostics[:subspace_separation], :<, 1e-4)
  end

  # Same family with moderate weight: perfectly trustworthy.
  def test_well_conditioned_moderate_weight
    result = Matrix.solve_discrete_are(Matrix[[1.0]], Matrix[[1.0]],
                                       Matrix[[1.0]], Matrix[[1.0]])
    assert(result.ok?, result.diagnostics[:message])
    x_exact = (1.0 + Math.sqrt(5.0)) / 2.0
    assert_in_delta(x_exact, result.x[0, 0], 1e-12)
  end

  def test_input_validation
    a = Matrix.I(2)
    b = Matrix[[1.0], [1.0]]
    q = Matrix.I(2)
    r = Matrix[[1.0]]

    assert_raise(ArgumentError) { Matrix.solve_discrete_are("a", b, q, r) }
    assert_raise(ArgumentError) { Matrix.solve_discrete_are(Matrix[[1.0, 2.0]], b, q, r) }
    assert_raise(ArgumentError) { Matrix.solve_discrete_are(a, Matrix.I(3), q, r) }
    assert_raise(ArgumentError) { Matrix.solve_discrete_are(a, b, Matrix.I(3), r) }
    assert_raise(ArgumentError) { Matrix.solve_discrete_are(a, b, q, Matrix.I(2)) }
    assert_raise(ArgumentError) { Matrix.solve_discrete_are(a, b, Matrix[[1.0, 2.0], [0.0, 1.0]], r) }
    assert_raise(ArgumentError) { Matrix.solve_discrete_are(a, b, Matrix.diagonal(1.0, -1.0), r) }
    assert_raise(ArgumentError) { Matrix.solve_discrete_are(a, b, q, Matrix[[0.0]]) }
    assert_raise(ArgumentError) { Matrix.solve_discrete_are(a, b, q, Matrix[[-1.0]]) }
    assert_raise(ArgumentError) { Matrix.solve_discrete_are(a, b, q, Matrix[[1.0, 2.0], [0.0, 1.0]]) }
  end

  def test_integer_inputs_are_converted_to_float
    result = Matrix.solve_discrete_are(Matrix[[2]], Matrix[[1]],
                                       Matrix[[1]], Matrix[[1]])
    assert(result.ok?, result.diagnostics[:message])
    assert_in_delta(2.0 + Math.sqrt(5.0), result.x[0, 0], 1e-12)
  end

  def test_diagnostics_are_consistent
    a = Matrix[[1.0, 1.0], [0.0, 1.0]]
    b = Matrix[[0.0], [1.0]]
    result = Matrix.solve_discrete_are(a, b, Matrix.I(2), Matrix[[1.0]])
    diag = result.diagnostics
    eigs = diag[:closed_loop_eigenvalues]
    assert_equal(2, eigs.size)
    assert_in_delta(eigs.map(&:abs).max, diag[:closed_loop_spectral_radius], 1e-15)
    assert_in_delta(1.0 - diag[:closed_loop_spectral_radius], diag[:stability_margin], 1e-15)
    separation = eigs.flat_map { |li| eigs.map { |lj| (1 - li.conj * lj).abs } }.min
    assert_in_delta(separation, diag[:subspace_separation], 1e-15)
    assert_kind_of(String, diag[:message])
  end
end
