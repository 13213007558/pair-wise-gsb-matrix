# frozen_string_literal: false
require 'test/unit'
require 'matrix'

class TestNearestCorrelation < Test::Unit::TestCase
  TOLERANCE = 1e-8

  def assert_matrix_in_delta(expected, actual, delta = TOLERANCE)
    assert_equal(expected.row_count, actual.row_count)
    assert_equal(expected.column_count, actual.column_count)
    expected.each_with_index do |e, i, j|
      assert_in_delta(e, actual[i, j], delta, "entry [#{i}, #{j}]")
    end
  end

  def assert_valid_correlation_matrix(m)
    n = m.row_count
    n.times do |i|
      assert_in_delta(1.0, m[i, i], 1e-12, "diagonal entry #{i}")
      n.times do |j|
        assert_in_delta(m[j, i], m[i, j], 1e-12, "symmetry at [#{i}, #{j}]")
      end
    end
    m.eigensystem.eigenvalues.each_with_index do |ev, k|
      assert(ev.real >= -1e-9, "eigenvalue #{k} is negative: #{ev.real}")
    end
  end

  def test_fixed_point
    # A matrix that is already a valid correlation matrix needs no correction.
    c = Matrix[[1.0, 0.5, -0.3],
               [0.5, 1.0,  0.2],
               [-0.3, 0.2, 1.0]]
    result = Matrix.nearest_correlation(c)

    assert_equal(:converged, result.status)
    assert(result.converged?)
    assert_in_delta(0.0, result.residual, 1e-9)
    assert_in_delta(0.0, result.objective, 1e-9)
    assert_matrix_in_delta(c, result.matrix, 1e-9)
  end

  def test_fixed_point_with_consistent_fixed_entries
    c = Matrix[[1.0, 0.5, -0.3],
               [0.5, 1.0,  0.2],
               [-0.3, 0.2, 1.0]]
    result = Matrix.nearest_correlation(c, fixed: {[0, 1] => 0.5, [2, 1] => 0.2})

    assert(result.converged?)
    assert_in_delta(0.0, result.objective, 1e-9)
    assert_matrix_in_delta(c, result.matrix, 1e-9)
  end

  def test_conflicting_fixed_entries_are_not_reported_as_success
    # x12 = x13 = 1 forces rows 1..3 to be collinear, hence x23 = 1;
    # fixing x23 = -1 makes the constraints contradictory.
    a = Matrix[[1.0, 1.0,  1.0],
               [1.0, 1.0, -1.0],
               [1.0, -1.0, 1.0]]
    result = Matrix.nearest_correlation(a,
                                        fixed: {[0, 1] => 1.0, [0, 2] => 1.0, [1, 2] => -1.0},
                                        max_iterations: 50)

    assert_equal(:max_iterations_reached, result.status)
    assert(!result.converged?)
    assert_equal(50, result.iterations)
    assert(result.residual > 1e-3, "residual should stay away from zero: #{result.residual}")
  end

  def test_impossible_fixed_value_is_rejected
    a = Matrix[[1.0, 0.5], [0.5, 1.0]]
    assert_raise(ArgumentError) { Matrix.nearest_correlation(a, fixed: {[0, 1] => 1.5}) }
    assert_raise(ArgumentError) { Matrix.nearest_correlation(a, fixed: {[0, 0] => 0.5}) }
    assert_raise(ArgumentError) { Matrix.nearest_correlation(a, fixed: {[0, 2] => 0.5}) }
  end

  def test_iteration_budget_exhaustion_is_not_reported_as_success
    a = Matrix[[1.0, 2.0], [2.0, 1.0]]
    result = Matrix.nearest_correlation(a, max_iterations: 2)

    assert_equal(:max_iterations_reached, result.status)
    assert(!result.converged?)
    assert_equal(2, result.iterations)
  end

  def test_known_optimum_with_fixed_entry
    # Hand-assembled "correlation" matrix with a negative eigenvalue;
    # x23 = 0 must be preserved. With x23 fixed at 0, positive
    # semidefiniteness is equivalent to x12^2 + x13^2 <= 1, so the
    # optimum projects (2, 0.5) onto the unit circle.
    a = Matrix[[1.0, 2.0, 0.5],
               [2.0, 1.0, 0.0],
               [0.5, 0.0, 1.0]]
    assert(a.eigensystem.eigenvalues.any? { |ev| ev.real < 0 })

    result = Matrix.nearest_correlation(a, fixed: {[1, 2] => 0.0})

    assert_equal(:converged, result.status)
    assert(result.converged?)
    norm = Math.sqrt(2.0**2 + 0.5**2)
    x12 = 2.0 / norm
    x13 = 0.5 / norm
    expected = Matrix[[1.0, x12, x13],
                      [x12, 1.0, 0.0],
                      [x13, 0.0, 1.0]]
    assert_matrix_in_delta(expected, result.matrix)
    assert_in_delta(0.0, result.matrix[1, 2], 1e-12)
    expected_objective = Math.sqrt(2.0 * ((2.0 - x12)**2 + (0.5 - x13)**2))
    assert_in_delta(expected_objective, result.objective, TOLERANCE)
    assert_valid_correlation_matrix(result.matrix)
  end

  def test_result_exposes_residual_and_objective
    a = Matrix[[1.0, 2.0], [2.0, 1.0]]
    result = Matrix.nearest_correlation(a)

    assert(result.converged?)
    assert_kind_of(Float, result.residual)
    assert_kind_of(Float, result.objective)
    assert_in_delta(Math.sqrt(2.0), result.objective, TOLERANCE)
    assert_valid_correlation_matrix(result.matrix)
  end

  def test_invalid_input
    assert_raise(TypeError) { Matrix.nearest_correlation([[1.0]]) }
    assert_raise(Matrix::ErrDimensionMismatch) do
      Matrix.nearest_correlation(Matrix[[1.0, 0.5]])
    end
  end
end

