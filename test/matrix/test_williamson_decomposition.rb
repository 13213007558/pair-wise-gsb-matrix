# frozen_string_literal: false
require 'test/unit'
require 'matrix'

class TestWilliamsonDecomposition < Test::Unit::TestCase
  def frobenius(m)
    Math.sqrt(m.entries.sum { |e| e * e })
  end

  def reconstruction_residual(w, v)
    frobenius(w.s * w.d * w.s.t - v) / frobenius(v)
  end

  def symplectic_residual(w)
    frobenius(w.s * w.j * w.s.t - w.j) / frobenius(w.s)**2
  end

  # Per-mode rotation: orthogonal symplectic matrix [[X, Y], [-Y, X]].
  def mode_rotation(angles)
    n = angles.size
    Matrix.build(2 * n) do |r, c|
      if r < n && c < n
        r == c ? Math.cos(angles[r]) : 0.0
      elsif r < n
        r == c - n ? Math.sin(angles[r]) : 0.0
      elsif c < n
        r - n == c ? -Math.sin(angles[r - n]) : 0.0
      else
        r == c ? Math.cos(angles[r - n]) : 0.0
      end
    end
  end

  # Squeeze: diag(s_1..s_n, 1/s_1..1/s_n).
  def squeeze(factors)
    Matrix.diagonal(*(factors + factors.map { |f| 1.0 / f }))
  end

  # Shear: [[I, B], [0, I]] with symmetric B.
  def shear(b)
    n = b.row_count
    Matrix.build(2 * n) do |r, c|
      if r == c
        1.0
      elsif r < n && c >= n
        b[r, c - n]
      else
        0.0
      end
    end
  end

  def setup
    @b = Matrix[[0.4, 0.1, 0.0], [0.1, -0.3, 0.2], [0.0, 0.2, 0.5]]
    @s0 = squeeze([1.3, 0.8, 2.1]) * mode_rotation([0.6, -0.9, 0.2]) * shear(@b)
    @j3 = Matrix::WilliamsonDecomposition.standard_symplectic_form(3)
  end

  def williamson_of(nus)
    delta = Matrix.diagonal(*(nus + nus))
    (@s0 * delta * @s0.t).williamson
  end

  def test_factors_are_symplectic
    [squeeze([1.3, 0.8, 2.1]), mode_rotation([0.6, -0.9, 0.2]), shear(@b), @s0].each do |s|
      assert_operator(frobenius(s * @j3 * s.t - @j3), :<, 1e-12)
    end
  end

  def test_standard_symplectic_form
    j = Matrix::WilliamsonDecomposition.standard_symplectic_form(2)
    assert_equal(Matrix[[0, 0, 1, 0], [0, 0, 0, 1], [-1, 0, 0, 0], [0, -1, 0, 0]], j)
    assert_equal(Matrix[[0, 1], [-1, 0]], Matrix[[2.0, 0.5], [0.5, 1.0]].williamson.j)
    assert_raise(ArgumentError) { Matrix::WilliamsonDecomposition.standard_symplectic_form(0) }
    assert_raise(ArgumentError) { Matrix::WilliamsonDecomposition.standard_symplectic_form(-2) }
  end

  def test_two_by_two
    v = Matrix[[2.0, 0.5], [0.5, 1.0]]
    w = v.williamson
    assert_equal(1, w.n)
    assert_in_delta(Math.sqrt(1.75), w.symplectic_eigenvalues[0], 1e-12)
    assert_operator(reconstruction_residual(w, v), :<, 1e-12)
    assert_operator(symplectic_residual(w), :<, 1e-12)
  end

  def test_constructed_matrix_identities
    nus = [0.5, 2.0, 7.0]
    delta = Matrix.diagonal(*(nus + nus))
    v = @s0 * delta * @s0.t
    w = v.williamson
    assert_equal(3, w.n)
    assert_operator(reconstruction_residual(w, v), :<, 1e-12)
    assert_operator(symplectic_residual(w), :<, 1e-12)
    nus.sort.each_with_index do |nu, i|
      assert_in_delta(nu, w.symplectic_eigenvalues[i], 1e-9)
    end
  end

  def test_identity_matrix
    v = Matrix.identity(6)
    w = v.williamson
    assert_equal([1.0, 1.0, 1.0], w.symplectic_eigenvalues)
    assert_operator(reconstruction_residual(w, v), :<, 1e-12)
    assert_operator(symplectic_residual(w), :<, 1e-12)
  end

  def test_diagonal_matrix
    v = Matrix.diagonal(3.0, 1.0, 2.0, 3.0, 1.0, 2.0)
    w = v.williamson
    assert_equal([1.0, 2.0, 3.0], w.symplectic_eigenvalues)
    assert_operator(reconstruction_residual(w, v), :<, 1e-12)
    assert_operator(symplectic_residual(w), :<, 1e-12)
  end

  def test_symplectic_eigenvalues_are_positive_and_sorted
    w = williamson_of([7.0, 0.5, 2.0])
    nus = w.symplectic_eigenvalues
    assert(nus.all? { |nu| nu > 0 })
    assert_equal(nus.sort, nus)
  end

  def test_repeated_symplectic_eigenvalues
    w = williamson_of([2.0, 2.0, 5.0])
    v = @s0 * Matrix.diagonal(*([2.0, 2.0, 5.0] * 2)) * @s0.t
    assert_operator(reconstruction_residual(w, v), :<, 1e-12)
    assert_operator(symplectic_residual(w), :<, 1e-12)
    assert_in_delta(2.0, w.symplectic_eigenvalues[0], 1e-9)
    assert_in_delta(2.0, w.symplectic_eigenvalues[1], 1e-9)
    assert_in_delta(5.0, w.symplectic_eigenvalues[2], 1e-9)
  end

  def test_fully_degenerate_symplectic_eigenvalues
    w = williamson_of([3.0, 3.0, 3.0])
    v = @s0 * Matrix.diagonal(*([3.0, 3.0, 3.0] * 2)) * @s0.t
    assert_operator(reconstruction_residual(w, v), :<, 1e-12)
    assert_operator(symplectic_residual(w), :<, 1e-12)
  end

  def test_nearly_degenerate_symplectic_eigenvalues
    nus = [2.0, 2.0 * (1 + 1e-9), 5.0]
    w = williamson_of(nus)
    v = @s0 * Matrix.diagonal(*(nus + nus)) * @s0.t
    assert_operator(reconstruction_residual(w, v), :<, 1e-12)
    assert_operator(symplectic_residual(w), :<, 1e-8)
  end

  def test_wide_scale_spread
    nus = [1e-6, 1.0, 1e6]
    w = williamson_of(nus)
    v = @s0 * Matrix.diagonal(*(nus + nus)) * @s0.t
    assert_operator(reconstruction_residual(w, v), :<, 1e-12)
    assert_operator(symplectic_residual(w), :<, 1e-6)
    nus.sort.each_with_index do |nu, i|
      assert_in_delta(nu, w.symplectic_eigenvalues[i], 1e-3 * nu)
    end
  end

  def test_repeated_with_wide_scale_spread
    nus = [1e-4, 1e-4, 1e4]
    w = williamson_of(nus)
    v = @s0 * Matrix.diagonal(*(nus + nus)) * @s0.t
    assert_operator(reconstruction_residual(w, v), :<, 1e-12)
    assert_operator(symplectic_residual(w), :<, 1e-8)
  end

  def test_extreme_overall_scaling
    nus = [0.5, 2.0, 7.0]
    delta = Matrix.diagonal(*(nus + nus))
    [1e12, 1e-12].each do |scale|
      v = @s0 * (delta * scale) * @s0.t
      w = v.williamson
      assert_operator(reconstruction_residual(w, v), :<, 1e-12)
      assert_operator(symplectic_residual(w), :<, 1e-12)
      nus.sort.each_with_index do |nu, i|
        assert_in_delta(nu * scale, w.symplectic_eigenvalues[i], 1e-6 * nu * scale)
      end
    end
  end

  def test_integer_and_rational_entries
    assert_equal([6.0], Matrix[[4, 0], [0, 9]].williamson.symplectic_eigenvalues)
    assert_equal([6.0], Matrix[[Rational(4), 0], [0, Rational(9)]].williamson.symplectic_eigenvalues)
  end

  def test_reproducibility
    nus = [0.5, 2.0, 2.0]
    delta = Matrix.diagonal(*(nus + nus))
    v = @s0 * delta * @s0.t
    assert_equal(v.williamson.s, v.williamson.s)
    assert_equal(v.williamson.symplectic_eigenvalues, v.williamson.symplectic_eigenvalues)
  end

  def test_accessors_and_aliases
    v = Matrix[[2.0, 0.5], [0.5, 1.0]]
    w = v.williamson
    assert_same(w.s, w.symplectic_matrix)
    assert_equal(w.d, w.diagonal_matrix)
    assert_equal(w.j, w.symplectic_form)
    assert_equal([w.s, w.d], w.to_ary)
    assert_equal([w.s, w.d], w.to_a)
    s, d = v.williamson
    assert_equal(w.s, s)
    assert_equal(w.d, d)
    assert_equal(v.williamson_decomposition.s, w.s)
    assert_equal(Matrix.diagonal(*w.symplectic_eigenvalues * 2), w.d)
  end

  def test_odd_order_matrix
    assert_raise(ExceptionForMatrix::ErrDimensionMismatch) do
      Matrix.identity(3).williamson
    end
  end

  def test_non_square_matrix
    assert_raise(ExceptionForMatrix::ErrDimensionMismatch) do
      Matrix[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]].williamson
    end
    assert_raise(ExceptionForMatrix::ErrDimensionMismatch) do
      Matrix.empty(0, 0).williamson
    end
  end

  def test_non_symmetric_matrix
    assert_raise(ArgumentError) { Matrix[[1.0, 2.0], [3.0, 4.0]].williamson }
  end

  def test_non_positive_definite_matrix
    assert_raise(ArgumentError) { Matrix[[1.0, 0.0], [0.0, -1.0]].williamson }
    assert_raise(ArgumentError) { Matrix[[1.0, 2.0], [2.0, 4.0]].williamson }
    assert_raise(ArgumentError) { Matrix.zero(4).williamson }
  end

  def test_invalid_argument_type
    assert_raise(TypeError) { Matrix::WilliamsonDecomposition.new([[1, 0], [0, 1]]) }
    assert_raise(TypeError) { Matrix[[Complex(1, 1), 0], [0, 1]].williamson }
    assert_raise(TypeError) { Matrix[[Object.new, 0], [0, 1]].williamson }
  end
end
