# frozen_string_literal: false
require 'test/unit'
require 'matrix'

class TestDiscreteAlgebraicRiccatiEquation < Test::Unit::TestCase
  def assert_close(expected, actual, tolerance = 1.0e-8)
    assert_equal(expected.row_count, actual.row_count)
    assert_equal(expected.column_count, actual.column_count)
    expected.row_count.times do |i|
      expected.column_count.times do |j|
        assert_in_delta(expected[i, j].to_f, actual[i, j].to_f, tolerance)
      end
    end
  end

  def test_scalar_exactly_constructed_system
    result = Matrix.solve_discrete_are(
      Matrix[[Rational(1, 2)]],
      Matrix[[1]],
      Matrix[[Rational(7, 8)]],
      Matrix[[1]]
    )

    assert(result.success?)
    assert_nil(result.reason)
    assert_close(Matrix[[1]], result.x)
    assert_close(Matrix[[Rational(1, 4)]], result.k)
    assert_close(Matrix[[Rational(1, 4)]], result.closed_loop)
    assert_operator(result.relative_residual, :<, 1.0e-12)
    assert_operator(result.spectral_radius, :<, 1.0)
    assert(result.diagnostics.key?(:stability_margin))
    assert_equal([result.x, result.k, result.diagnostics], result.to_a)
  end

  def test_non_symmetric_exactly_constructed_system
    transform = Matrix[[2, 1], [1, 1]]
    a0 = Matrix.diagonal(Rational(1, 2), 2)
    b0 = Matrix.identity(2)
    x0 = Matrix.diagonal(Rational(4, 3), 4)
    q0 = x0 - a0 * x0 * a0 +
         a0 * x0 * b0 * (Matrix.identity(2) + b0 * x0 * b0).inverse * b0 * x0 * a0
    a = transform * a0 * transform.inverse
    b = transform * b0
    q = transform.inverse.transpose * q0 * transform.inverse
    x = transform.inverse.transpose * x0 * transform.inverse

    result = Matrix.solve_discrete_are(a, b, q, Matrix.identity(2))

    assert(result.success?, result.reason)
    assert_close(x, result.x, 1.0e-9)
    assert_operator(result.relative_residual, :<, 1.0e-10)
    assert_operator(result.spectral_radius, :<, 0.45)
    result.closed_loop_eigenvalues.each do |value|
      assert_operator(value.abs, :<, 0.45)
    end
  end

  def test_singular_state_weight_uses_newton_refinement
    result = Matrix.solve_discrete_are(
      Matrix.diagonal(2, Rational(1, 2)),
      Matrix.identity(2),
      Matrix.diagonal(0, Rational(7, 8)),
      Matrix.identity(2)
    )

    assert(result.success?, result.reason)
    assert_close(Matrix.diagonal(3, 1), result.x, 1.0e-9)
    assert_close(Matrix[[1.5, 0], [0, 0.25]], result.k, 1.0e-9)
    assert(result.diagnostics[:newton_iterations])
    assert(result.diagnostics[:regularization])
  end

  def test_single_input_jordan_block
    a = Matrix[[2, 1], [0, 2]]
    b = Matrix.column_vector([0, 1])
    result = Matrix.solve_discrete_are(a, b, Matrix.identity(2), Matrix.identity(1))

    assert(result.success?, result.reason)
    assert_operator(result.relative_residual, :<, 1.0e-10)
    assert_operator(result.spectral_radius, :<, 0.45)
    assert_operator(result.minimum_eigenvalue_x, :>, 0.0)
  end

  def test_unstabilizable_system
    result = Matrix.solve_discrete_are(
      Matrix[[1.1]], Matrix[[0]], Matrix[[1]], Matrix[[1]]
    )

    assert(!result.success?)
    assert_equal(:not_stabilizable, result.reason)
    assert_nil(result.x)
    assert_nil(result.k)
  end

  def test_uncontrollable_mode_near_unit_circle
    result = Matrix.solve_discrete_are(
      Matrix[[1.0 + 1.0e-12]], Matrix[[0]], Matrix[[1]], Matrix[[1]]
    )

    assert(!result.success?)
    assert_equal(:spectrum_near_unit_circle, result.reason)
    assert_nil(result.x)
  end

  def test_weakly_controlled_mode_near_unit_circle
    result = Matrix.solve_discrete_are(
      Matrix[[1.0 + 1.0e-8]], Matrix[[1.0e-7]], Matrix[[1]], Matrix[[1]]
    )

    assert(!result.success?)
    assert_equal(:spectrum_near_unit_circle, result.reason)
    assert_in_delta(1.0e-7, result.diagnostics[:stabilizability_pivot], 1.0e-20)
  end

  def test_ill_conditioned_control_subspace_away_from_unit_circle
    result = Matrix.solve_discrete_are(
      Matrix[[2.0]], Matrix[[1.0e-8]], Matrix[[1]], Matrix[[1]]
    )

    assert(!result.success?)
    assert_equal(:ill_conditioned_subspace, result.reason)
    assert_nil(result.x)
  end

  def test_extreme_weight_scaling
    result = Matrix.solve_discrete_are(
      Matrix[[Rational(1, 2)]], Matrix[[1]], Matrix[[1]], Matrix[[1e200]]
    )

    assert(result.success?, result.reason)
    assert_in_delta(Rational(4, 3), result.x[0, 0], 1.0e-10)
    assert_operator(result.relative_residual, :<, 1.0e-10)
    assert_in_delta(0.5, result.spectral_radius, 1.0e-10)
  end

  def test_invalid_weight_matrices
    assert_raise(ArgumentError) do
      Matrix.solve_discrete_are(
        Matrix[[1]], Matrix[[1]], Matrix[[-1]], Matrix[[1]]
      )
    end
    assert_raise(ArgumentError) do
      Matrix.solve_discrete_are(
        Matrix[[1]], Matrix[[1]], Matrix[[1]], Matrix[[0]]
      )
    end
  end
end
