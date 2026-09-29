# frozen_string_literal: false
require 'test/unit'
require 'matrix'

class TestLDLTDecomposition < Test::Unit::TestCase
  def assert_matrix_close(expected, actual, delta = 1e-10)
    assert_equal(expected.row_count, actual.row_count)
    assert_equal(expected.column_count, actual.column_count)
    expected.each_with_index(:all) do |e, i, j|
      assert_in_delta(e, actual[i, j], delta, "element (#{i}, #{j})")
    end
  end

  def test_reconstruction_exact
    a = Matrix[[2, 1, 0], [1, 3, 1], [0, 1, 4]]
    l, d, p = a.ldlt.to_a
    assert_equal(l * d * l.transpose, p * a * p.transpose)
    assert_equal(a, p.transpose * l * d * l.transpose * p)
  end

  def test_reconstruction_with_2x2_pivots_exact
    a = Matrix[[0, 1, 2], [1, 0, 3], [2, 3, 0]]
    l, d, p = a.ldlt.to_a
    assert_equal(l * d * l.transpose, p * a * p.transpose)
    assert_equal(a, p.transpose * l * d * l.transpose * p)
  end

  def test_reconstruction_float
    m = Matrix[[1.0, -2.0, 3.0, 0.5],
               [4.0, 0.0, -1.0, 2.0],
               [2.0, 7.0, 1.0, -3.0],
               [0.0, 1.0, 5.0, 2.0]]
    a = m + m.transpose
    l, d, p = a.ldlt.to_a
    assert_matrix_close(l * d * l.transpose, p * a * p.transpose)
    assert_matrix_close(a, p.transpose * l * d * l.transpose * p)
  end

  def test_factor_structure
    a = Matrix[[0, 1, 2], [1, 0, 3], [2, 3, 0]]
    l, d, p = a.ldlt.to_a
    assert_predicate(l, :lower_triangular?)
    3.times {|i| assert_equal(1, l[i, i])}
    assert_predicate(p, :permutation?)
    assert_predicate(d, :symmetric?)
    d.each_with_index(:all) do |e, i, j|
      assert_equal(0, e) if (i - j).abs > 1
    end
  end

  def test_permutation_direction
    a = Matrix[[0.0, 2.0, 1.0], [2.0, 1.0, 0.0], [1.0, 0.0, 3.0]]
    ldlt = a.ldlt
    l, d, p = ldlt.to_a
    # P * A * P' == L * D * L'
    assert_matrix_close(l * d * l.transpose, p * a * p.transpose)
    # A == P' * L * D * L' * P
    assert_matrix_close(a, p.transpose * l * d * l.transpose * p)
    # pivots array is consistent with P: (P*A*P')[i][j] == A[pivots[i]][pivots[j]]
    permuted = p * a * p.transpose
    ldlt.pivots.each_with_index do |pi, i|
      ldlt.pivots.each_with_index do |pj, j|
        assert_equal(a[pi, pj], permuted[i, j])
      end
    end
    assert_equal((0...a.row_count).to_a, ldlt.pivots.sort)
  end

  def test_zero_diagonal_invertible
    a = Matrix[[0, 1], [1, 0]]
    ldlt = a.ldlt
    assert_not_predicate(ldlt, :singular?)
    assert_not_predicate(ldlt, :nearly_singular?)
    assert_equal(-1, ldlt.det)
    assert_equal([1, 1, 0], ldlt.inertia)
    assert_equal(Vector[4, 3], ldlt.solve(Vector[3, 4]))
  end

  def test_solve_vector_residual
    a = Matrix[[4.0, 1.0, 2.0], [1.0, 3.0, 0.0], [2.0, 0.0, 5.0]]
    b = Vector[1.0, 2.0, 3.0]
    x = a.ldlt.solve(b)
    assert_instance_of(Vector, x)
    assert_in_delta(0.0, (a * x - b).norm, 1e-10)
  end

  def test_solve_matrix_multiple_rhs_residual
    a = Matrix[[0.0, 1.0, 2.0], [1.0, 2.0, 0.0], [2.0, 0.0, 1.0]]
    b = Matrix[[1.0, 0.0, 4.0], [2.0, -1.0, 1.0], [3.0, 4.0, -2.0]]
    x = a.ldlt.solve(b)
    assert_instance_of(Matrix, x)
    assert_equal(3, x.column_count)
    assert_matrix_close(b, a * x)
  end

  def test_solve_exact
    a = Matrix[[2, 1], [1, 3]]
    assert_equal(Vector[1, 3], a.ldlt.solve(Vector[5, 10]))
  end

  def test_inertia
    assert_equal([2, 1, 0], Matrix.diagonal(2, -3, 1).ldlt.inertia)
    assert_equal([1, 1, 1], Matrix.diagonal(2, -3, 0).ldlt.inertia)
    assert_equal([0, 0, 2], Matrix.zero(2).ldlt.inertia)
    assert_equal([3, 0, 0], Matrix[[2, 1, 0], [1, 3, 1], [0, 1, 4]].ldlt.inertia)
  end

  def test_inertia_of_2x2_block_not_judged_by_diagonal_entries
    # eigenvalues are 4 and -2 although both diagonal entries are positive
    assert_equal([1, 1, 0], Matrix[[1, 3], [3, 1]].ldlt.inertia)
    # eigenvalues are 2 and -4 although both diagonal entries are negative
    assert_equal([1, 1, 0], Matrix[[-1, 3], [3, -1]].ldlt.inertia)
    # eigenvalues are 1 and -1 although both diagonal entries are zero
    assert_equal([1, 1, 0], Matrix[[0, 1], [1, 0]].ldlt.inertia)
  end

  def test_singular_exact
    a = Matrix[[1, 2], [2, 4]]
    ldlt = a.ldlt
    assert_predicate(ldlt, :singular?)
    assert_predicate(ldlt, :nearly_singular?)
    assert_equal(0, ldlt.det)
    assert_equal([1, 0, 1], ldlt.inertia)
    assert_raise(Matrix::ErrNotRegular) { ldlt.solve(Vector[1, 2]) }
    assert_raise(Matrix::ErrNotRegular) { ldlt.solve(Matrix[[1], [2]]) }
  end

  def test_nearly_singular_with_explicit_relative_tolerance
    e = 1e-12
    a = Matrix[[1.0, 1.0], [1.0, 1.0 + e]]
    ldlt = a.ldlt(tol: 1e-8)
    assert_equal(1e-8, ldlt.tol)
    assert_not_predicate(ldlt, :singular?)
    assert_predicate(ldlt, :nearly_singular?)
    assert(!ldlt.nearly_singular?(1e-15))
    assert_in_delta(e, ldlt.det, 1e-15)
  end

  def test_default_tolerance
    ldlt = Matrix[[1.0, 0.0], [0.0, 1.0]].ldlt
    assert_equal(Float::EPSILON * 2, ldlt.tol)
    assert_not_predicate(ldlt, :singular?)
    assert_not_predicate(ldlt, :nearly_singular?)
  end

  def test_det
    a = Matrix[[0, 1, 2], [1, 0, 3], [2, 3, 0]]
    assert_equal(a.det, a.ldlt.det)
    b = Matrix[[4.0, 1.0], [1.0, 3.0]]
    assert_in_delta(b.det, b.ldlt.det, 1e-10)
  end

  def test_errors
    assert_raise(Matrix::ErrDimensionMismatch) { Matrix[[1, 2, 3], [4, 5, 6]].ldlt }
    assert_raise(ArgumentError) { Matrix[[1, 2], [3, 4]].ldlt }
    assert_raise(TypeError) { Matrix::LDLTDecomposition.new([[1, 2], [2, 1]]) }
    assert_raise(Matrix::ErrDimensionMismatch) { Matrix[[1, 2], [2, 1]].ldlt.solve(Vector[1, 2, 3]) }
    assert_raise(Matrix::ErrDimensionMismatch) { Matrix[[1, 2], [2, 1]].ldlt.solve(Matrix[[1], [2], [3]]) }
  end
end
