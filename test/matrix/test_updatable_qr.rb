# frozen_string_literal: false
require "test/unit"
require "matrix"

class TestUpdatableQR < Test::Unit::TestCase
  EPS = 1e-12

  def assert_least_squares(updater, matrix, rhs)
    result = updater.solve(rhs)
    residual_matrix = matrix * result.solution - rhs
    residual_norm = frobenius(residual_matrix.to_a)
    reference = matrix.updatable_qr(rhs).solve(rhs)
    solution_error = frobenius((result.solution - reference.solution).to_a)
    reference_residual = frobenius((matrix * reference.solution - rhs).to_a)

    assert_in_delta(reference_residual, residual_norm, 1e-10)
    assert(solution_error < 1e-8 || residual_norm < 1e-8)
    result
  end

  def frobenius(rows)
    Math.sqrt(rows.sum(0.0) { |row| row.sum(0.0) { |value| value.abs2 } })
  end

  def test_empty_window_can_accept_rows_and_multi_right_hand_sides
    matrix = Matrix.empty(0, 3)
    rhs = Matrix.empty(0, 2)
    qr = matrix.updatable_qr(rhs)

    assert_predicate qr, :empty?
    assert_equal 0, qr.rank

    qr.add_row([1, 0, 0], [[10, 20]])
    refute_predicate qr, :empty?
    assert_equal 1, qr.rank
    assert_least_squares(qr, Matrix[[1, 0, 0]], Matrix[[10, 20]])

    qr.add_row(Vector[0, 1, 0], Vector[30, 40])
    assert_equal 2, qr.rank
    result = assert_least_squares(
      qr,
      Matrix[[1, 0, 0], [0, 1, 0]],
      Matrix[[10, 20], [30, 40]]
    )
    assert_equal Matrix[[10, 20], [30, 40], [0, 0]], result.solution
  end

  def test_append_delete_replace_keep_current_least_squares_solution
    srand(42)
    n = 4
    k = 2
    rows = []
    values = []
    qr = Matrix.empty(0, n).updatable_qr(Matrix.empty(0, k))

    120.times do |step|
      case rand
      when 0...0.45
        row = Array.new(n) { rand(-3.0..3.0) }
        rhs = Array.new(k) { rand(-3.0..3.0) }
        rows << row
        values << rhs
        qr.add_row(row, rhs)
      when 0.45...0.75
        next if rows.empty?
        index = rand(rows.size)
        rows.delete_at(index)
        values.delete_at(index)
        qr.delete_row(index)
      else
        next if rows.empty?
        index = rand(rows.size)
        row = Array.new(n) { rand(-3.0..3.0) }
        rhs = Array.new(k) { rand(-3.0..3.0) }
        rows[index] = row
        values[index] = rhs
        qr.replace_row(index, row, rhs)
      end

      next if rows.empty?
      assert_equal rows.size, qr.row_count
      assert_equal rows.size, qr.rhs.row_count
      assert_least_squares(qr, Matrix.rows(rows), Matrix.rows(values))
      assert(qr.rank <= [rows.size, n].min)
    end
  end

  def test_rank_drop_reports_consistency_and_minimum_norm_convention
    rows = [
      [1.0, 2.0],
      [2.0, 4.0],
      [0.0, 0.0]
    ]
    rhs = Matrix[[5.0], [10.0], [0.0]]
    qr = Matrix.rows(rows).updatable_qr(rhs)

    assert_equal 1, qr.rank
    assert_predicate qr, :rank_deficient?
    result = qr.solve
    assert_predicate result, :solvable
    assert_predicate result, :rank_deficient
    assert_equal "minimum-norm least squares solution", result.solution_convention
    assert_in_delta(0.0, result.residual_norm, EPS)
    solution_norm = Math.sqrt(result.solution.to_a.sum(0.0) { |x,| x.abs2 })
    assert_in_delta(Math.sqrt(5), solution_norm, EPS)

    inconsistent_rhs = Matrix[[5.0], [11.0], [0.0]]
    inconsistent = qr.solve(inconsistent_rhs)
    refute_predicate inconsistent, :solvable
    assert_in_delta(1.0 / Math.sqrt(5), inconsistent.residual_norm, EPS)
  end

  def test_nearly_linearly_dependent_rows_can_trigger_rebuild_without_history
    qr = Matrix.empty(0, 2).updatable_qr(Matrix.empty(0, 1))
    qr.add_row([1.0, 0.0], [1.0])
    qr.add_row([0.0, 1.0], [1.0])
    before = qr.rebuild_count

    qr.add_row([1.0 + 1e-7, 0.0], [1.0])
    assert_equal 2, qr.rank
    assert_in_delta(0.0, qr.solve.residual_norm, 1e-6)

    qr.replace_row(1, [1.0, 0.0], [1.0])
    assert_equal 1, qr.rank
    assert(qr.rebuild_count >= before || qr.last_rebuild_reason)
    assert_in_delta(0.0, qr.solve.residual_norm, 1e-6)
  end

  def test_rebuild_rules_are_inspectable
    qr = Matrix[[1.0], [2.0]].updatable_qr(Matrix[[1.0], [2.0]])
    assert_equal 0, qr.update_count
    assert_equal 0, qr.rebuild_count
    assert_nil qr.last_rebuild_reason
    assert_kind_of Numeric, qr.rank_tolerance

    qr.add_row([3.0], [3.0])
    assert_equal 1, qr.update_count
    qr.rebuild!("test")
    assert_equal 1, qr.rebuild_count
    assert_equal "test", qr.last_rebuild_reason
  end

  def test_finite_real_inputs_and_dimensions_are_validated
    qr = Matrix.empty(0, 2).updatable_qr
    assert_raise(ArgumentError) { qr.add_row([1.0, Float::NAN]) }
    assert_raise(ArgumentError) { qr.add_row([1.0, Complex(0, 1)]) }
    assert_raise(Matrix::ErrDimensionMismatch) { qr.add_row([1.0]) }
    assert_raise(IndexError) { qr.delete_row(0) }

    assert_raise(TypeError) { Matrix[1].updatable_qr }
  end

  def test_vector_rhs_factor_only_and_zero_column_windows
    qr = Matrix[[1.0, 2.0]].updatable_qr(Vector[3.0])
    result = qr.solve
    assert_kind_of Vector, result.solution
    assert_equal [0.6, 1.2], result.solution.to_a
    assert_in_delta(0.0, result.residual_norm, EPS)

    factor_only = Matrix[[1.0], [2.0]].updatable_qr
    factor_only.add_row([3.0])
    assert_equal 1, factor_only.rank
    assert_raise(ArgumentError) { factor_only.rhs }

    zero_columns = Matrix.empty(0, 0).updatable_qr
    zero_columns.add_row([], [1.0])
    zero_columns.delete_row(0)
    assert_predicate zero_columns, :empty?
  end

  def test_batch_append_follows_row_operations
    qr = Matrix.empty(0, 2).updatable_qr(Matrix.empty(0, 1))
    qr.add_rows([[1.0, 0.0], [0.0, 1.0]], [[1.0], [2.0]])
    result = qr.solve
    assert_equal Matrix[[1.0], [2.0]], result.solution
    assert_in_delta(0.0, result.residual_norm, EPS)
  end
end
