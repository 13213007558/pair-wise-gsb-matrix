# frozen_string_literal: false
require "test/unit"
require "matrix"

class TestNearestCorrelationMatrix < Test::Unit::TestCase
  def test_already_correlation_matrix_is_a_fixed_point
    matrix = Matrix[
      [1, Rational(1, 2), 0],
      [Rational(1, 2), 1, Rational(1, 2)],
      [0, Rational(1, 2), 1]
    ]

    result = Matrix.nearest_correlation_matrix(matrix)

    assert_equal :converged, result[:status]
    assert result[:converged]
    assert_equal matrix.to_a, result[:matrix].to_a
    assert_equal 0, result[:iterations]
    assert_equal 0.0, result[:residual]
    assert_equal 0.0, result[:objective_value]
    assert_equal 0.0, result[:squared_frobenius_distance]
    assert result[:minimum_eigenvalue] > 0
  end

  def test_conflicting_fixed_entries_are_not_successful
    matrix = Matrix[[1, 0.5, 0], [0.5, 1, 0.5], [0, 0.5, 1]]

    result = Matrix.nearest_correlation_matrix(
      matrix,
      fixed_entries: { [0, 1] => 0.25, [1, 0] => 0.75 }
    )

    assert_equal :infeasible, result[:status]
    refute result[:converged]
    assert_equal :conflicting_fixed_entries, result[:reason]
    assert_predicate result[:residual], :nan?
  end

  def test_psd_conflicting_fixed_entries_are_infeasible
    matrix = Matrix[[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    result = Matrix.nearest_correlation_matrix(
      matrix,
      fixed_entries: { [0, 1] => 1, [0, 2] => 0, [1, 2] => 1 }
    )

    assert_equal :infeasible, result[:status]
    refute result[:converged]
    assert_equal :fixed_entries_inconsistent_with_psd, result[:reason]
    assert_equal 0, result[:iterations]
    assert_predicate result[:residual], :nan?
  end

  def test_negative_correlation_equivalence_chain_is_feasible
    matrix = Matrix.identity(3)

    result = Matrix.nearest_correlation_matrix(
      matrix,
      fixed_entries: { [0, 1] => -1, [1, 2] => -1, [0, 2] => 1 }
    )

    assert_equal :converged, result[:status]
    assert result[:converged]
    assert_equal(-1.0, result[:matrix][0, 1])
    assert_equal(-1.0, result[:matrix][1, 2])
    assert_equal(1.0, result[:matrix][0, 2])
    assert result[:minimum_eigenvalue] >= -1e-10
  end

  def test_odd_negative_correlation_cycle_is_infeasible
    matrix = Matrix.identity(3)

    result = Matrix.nearest_correlation_matrix(
      matrix,
      fixed_entries: { [0, 1] => -1, [1, 2] => -1, [0, 2] => -1 }
    )

    assert_equal :infeasible, result[:status]
    refute result[:converged]
    assert_equal :fixed_entries_inconsistent_with_psd, result[:reason]
  end

  def test_third_order_fixed_entry_has_known_optimum
    matrix = Matrix[[1, 1, 0], [1, 1, 1], [0, 1, 1]]
    expected_squared_distance = 6.0 - 4.0 * Math.sqrt(2.0)
    expected_distance = Math.sqrt(expected_squared_distance)

    result = Matrix.nearest_correlation_matrix(
      matrix,
      fixed_entries: { [0, 2] => 0 }
    )

    assert_equal :converged, result[:status]
    assert result[:converged]
    assert_in_delta expected_squared_distance, result[:objective_value], 1e-9
    assert_in_delta expected_distance, result[:frobenius_distance], 1e-9
    assert_in_delta expected_squared_distance, result[:squared_frobenius_distance], 1e-9
    assert_in_delta 1.0 / Math.sqrt(2.0), result[:matrix][0, 1], 1e-8
    assert_equal 0.0, result[:matrix][0, 2]
    assert_in_delta 1.0 / Math.sqrt(2.0), result[:matrix][1, 2], 1e-8
    assert result[:residual] <= 1e-10
    assert result[:minimum_eigenvalue] >= -1e-10
  end

  def test_iteration_budget_is_not_reported_as_success
    matrix = Matrix[[1, 1, 0], [1, 1, 1], [0, 1, 1]]

    result = Matrix.nearest_correlation_matrix(
      matrix,
      max_iterations: 1
    )

    assert_equal :iteration_limit, result[:status]
    refute result[:converged]
    assert_equal 1, result[:iterations]
    assert result[:residual] > 1e-10
  end

  def test_unsupported_inputs_raise
    assert_raise(ExceptionForMatrix::ErrDimensionMismatch) do
      Matrix.nearest_correlation_matrix(Matrix[[1, 0], [0, 1, 1]])
    end

    assert_raise(ArgumentError) do
      Matrix.nearest_correlation_matrix(Matrix[[1, 0.2], [0.3, 1]])
    end

    assert_raise(ArgumentError) do
      Matrix.nearest_correlation_matrix(
        Matrix[[1, 0.2], [0.2, 1]],
        fixed_entries: { [0, 0] => 1 }
      )
    end
  end
end
