# frozen_string_literal: false
require "test/unit"
require "matrix"

class TestUpdatableQRDecomposition < Test::Unit::TestCase
  def residual_norms(matrix, rhs, solution)
    residual = matrix * solution - rhs
    rhs.column_count.times.map do |column|
      Math.sqrt(residual.column(column).magnitude ** 2)
    end
  end

  def assert_residual_matches(qr, rows, rhs_rows, epsilon = 1e-10)
    matrix = Matrix.rows(rows)
    rhs = Matrix.rows(rhs_rows)
    solution = qr.solve
    expected = residual_norms(matrix, rhs, solution)
    assert_equal(expected.size, qr.residual_norm.size)
    expected.zip(qr.residual_norm).each do |expected_value, actual_value|
      assert_in_delta(expected_value, actual_value, epsilon)
    end
    expected
  end

  def test_append_replace_and_delete_with_multiple_right_hand_sides
    rows = [
      [1.0, 2.0, 1.0],
      [3.0, 2.0, 0.0],
      [2.0, 5.0, 1.0],
      [0.0, 1.0, 4.0],
      [4.0, 0.0, 2.0]
    ]
    rhs = [
      [1.0, 0.0],
      [0.0, 1.0],
      [2.0, 1.0],
      [1.0, 2.0],
      [0.0, 0.0]
    ]
    qr = Matrix.empty(0, 3).updatable_qr(Matrix.empty(0, 2), block_size: 2)

    rows.zip(rhs).each do |row, right|
      qr.append_row(row, right)
      assert_residual_matches(qr, rows.first(qr.row_count), rhs.first(qr.row_count))
    end

    assert_equal(3, qr.rank)
    assert_equal(Matrix, qr.solve.class)
    assert_equal([3, 2], [qr.solve.row_count, qr.solve.column_count])
    assert(qr.rebuild_policy[:leaf_row_limits].all? { |size| size <= 2 })

    new_row = [7.0, -1.0, 2.0]
    new_rhs = [3.0, -2.0]
    qr.replace_row(1, new_row, new_rhs)
    rows[1] = new_row
    rhs[1] = new_rhs
    assert_residual_matches(qr, rows, rhs)

    qr.delete_row(0)
    rows.delete_at(0)
    rhs.delete_at(0)
    assert_residual_matches(qr, rows, rhs)
    assert_equal(4, qr.row_count)
  end

  def test_alternating_deletes_and_appends_do_not_rebuild_all_history
    rows = [
      [1.0, 0.0],
      [0.0, 1.0],
      [1.0, 1.0],
      [1.0, -1.0],
      [2.0, 1.0]
    ]
    rhs = [[1.0], [2.0], [3.0], [0.0], [4.0]]
    qr = Matrix.rows(rows).updatable_qr(Matrix.rows(rhs), block_size: 2)

    if qr.rebuild_count.positive?
      assert_match(/normal-equation-error/, qr.last_rebuild_reason)
      assert(qr.rebuild_policy[:normal_equation_residual] <= 1e-10)
    end
    qr.delete_row(0)
    rows.delete_at(0)
    rhs.delete_at(0)
    assert_residual_matches(qr, rows, rhs)

    qr.append_row([-1.0, 3.0], [5.0])
    rows << [-1.0, 3.0]
    rhs << [5.0]
    assert_residual_matches(qr, rows, rhs)

    qr.delete_row(1)
    rows.delete_at(1)
    rhs.delete_at(1)
    assert_residual_matches(qr, rows, rhs)

    qr.replace_row(-1, [0.5, 0.25], [1.25])
    rows[-1] = [0.5, 0.25]
    rhs[-1] = [1.25]
    assert_residual_matches(qr, rows, rhs)
    if qr.rebuild_count.positive?
      assert_match(/normal-equation-error/, qr.last_rebuild_reason)
      assert_operator(qr.rebuild_policy[:normal_equation_residual], :<=, 1e-10)
    end
    assert(qr.rebuild_policy[:leaf_row_limits].all? { |size| size <= 2 })
  end

  def test_empty_window_can_be_emptied_again_and_refilled
    qr = Matrix::UpdatableQRDecomposition.empty(3, 1, block_size: 2)
    assert_equal(0, qr.row_count)
    assert_equal([0.0], qr.residual_norm)

    qr.append_row([1.0, 2.0, 3.0], 1.0)
    qr.append_row([4.0, 5.0, 6.0], 2.0)
    assert_equal(2, qr.row_count)
    qr.delete_row(0)
    qr.delete_row(0)
    assert_equal(0, qr.row_count)

    qr.append_row(Vector[7.0, 8.0, 9.0], Vector[3.0])
    assert_residual_matches(qr, [[7.0, 8.0, 9.0]], [[3.0]])
  end

  def test_underdetermined_system_uses_minimum_norm_solution
    rows = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    rhs = [[1.0], [2.0]]
    qr = Matrix.rows(rows).updatable_qr(Matrix.rows(rhs), rank_tolerance: 1e-12)

    assert_equal(2, qr.rank)
    refute(qr.full_rank?)
    assert(qr.consistent?)
    assert_equal(Matrix::UpdatableQRDecomposition::MINIMUM_NORM_SOLUTION,
                 qr.solution_convention)
    assert_in_delta(-1.0 / 18.0, qr.solve[0], 1e-12)
    assert_in_delta(1.0 / 9.0, qr.solve[1], 1e-12)
    assert_in_delta(5.0 / 18.0, qr.solve[2], 1e-12)
    assert_residual_matches(qr, rows, rhs, 1e-12)
  end

  def test_rank_deficiency_reports_consistency_and_minimum_norm_choice
    rows = [[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]]
    rhs = [[1.0], [2.0]]
    qr = Matrix.rows(rows).updatable_qr(Matrix.rows(rhs), rank_tolerance: 1e-13)

    assert_equal(1, qr.rank)
    assert(qr.consistent?)
    assert_equal(Matrix::UpdatableQRDecomposition::MINIMUM_NORM_SOLUTION,
                 qr.solution_convention)
    assert_in_delta(1.0 / 14.0, qr.solve[0], 1e-12)
    assert_in_delta(1.0 / 7.0, qr.solve[1], 1e-12)
    assert_in_delta(3.0 / 14.0, qr.solve[2], 1e-12)
    assert_residual_matches(qr, rows, rhs, 1e-12)

    qr.replace_row(1, [2.0, 4.0, 6.0], [5.0])
    rhs[1] = [5.0]
    assert_equal(1, qr.rank)
    refute(qr.consistent?)
    assert_in_delta(3.0 / Math.sqrt(5.0), qr.residual_norm.first, 1e-10)
    assert_residual_matches(qr, rows, rhs, 1e-10)
  end

  def test_rank_drop_and_recovery_during_replacement
    rows = [[1.0, 0.0], [0.0, 1.0]]
    rhs = [[1.0, 2.0], [3.0, 4.0]]
    qr = Matrix.rows(rows).updatable_qr(Matrix.rows(rhs), rank_tolerance: 1e-12)
    assert_equal(2, qr.rank)

    qr.replace_row(1, [1.0, 1.0e-13], [3.0, 4.0])
    rows[1] = [1.0, 1.0e-13]
    assert_equal(1, qr.rank)
    refute(qr.consistent?)
    assert_residual_matches(qr, rows, rhs)

    qr.replace_row(1, [0.0, 1.0], [3.0, 4.0])
    rows[1] = [0.0, 1.0]
    assert_equal(2, qr.rank)
    assert(qr.consistent?)
    assert_residual_matches(qr, rows, rhs)
  end

  def test_nearly_linearly_dependent_rows_are_rank_one
    rows = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0 + 1e-15]]
    rhs = [[1.0], [2.0]]
    qr = Matrix.rows(rows).updatable_qr(Matrix.rows(rhs), rank_tolerance: 1e-13)
    assert_equal(1, qr.rank)
    assert_residual_matches(qr, rows, rhs)
    assert_kind_of(Numeric, qr.solution_status[:rank_tolerance])
  end

  def test_inputs_must_be_finite_real_numbers
    assert_raise(ArgumentError) do
      Matrix[[1.0, 2.0]].updatable_qr([Float::INFINITY])
    end
    qr = Matrix::UpdatableQRDecomposition.empty(2, 1)
    assert_raise(ArgumentError) { qr.append_row([1.0, Float::NAN], 1.0) }
    assert_raise(ArgumentError) { qr.append_row([Complex(1, 2), 0.0], 1.0) }
  end

  def test_dimension_mismatches_are_rejected
    qr = Matrix[[1.0, 2.0], [3.0, 4.0]].updatable_qr([1.0, 2.0])
    assert_raise(ExceptionForMatrix::ErrDimensionMismatch) do
      qr.append_row([1.0], 1.0)
    end
    assert_raise(ExceptionForMatrix::ErrDimensionMismatch) do
      qr.append_row([1.0, 2.0], [1.0, 2.0])
    end
    assert_raise(IndexError) { qr.delete_row(2) }
  end

  def test_zero_column_matrix_has_empty_solutions
    qr = Matrix.empty(3, 0).updatable_qr(Matrix[[1.0], [2.0], [3.0]])
    assert_equal(0, qr.rank)
    assert_equal(Vector[], qr.solve)
    assert_in_delta(Math.sqrt(14.0), qr.residual_norm.first, 1e-12)
  end
end
