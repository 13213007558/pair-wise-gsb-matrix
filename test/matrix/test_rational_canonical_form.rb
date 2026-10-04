# frozen_string_literal: false
require 'test/unit'
require 'matrix'

class TestMatrixRationalCanonicalForm < Test::Unit::TestCase
  # Exact polynomial helpers (coefficient arrays, lowest degree first)
  def poly_trim(p)
    p = p.dup
    p.pop until p.empty? || p.last != 0
    p
  end

  def poly_divmod(a, b)
    a = a.dup
    q = Array.new([a.size - b.size + 1, 0].max, 0)
    until a.empty? || a.size < b.size
      c = a.last.quo(b.last)
      s = a.size - b.size
      q[s] = c
      b.each_with_index { |bj, j| a[s + j] -= c * bj }
      a.pop until a.empty? || a.last != 0
    end
    [poly_trim(q), a]
  end

  def poly_mul(a, b)
    return [] if a.empty? || b.empty?
    r = Array.new(a.size + b.size - 1, 0)
    a.each_with_index do |ai, i|
      b.each_with_index { |bj, j| r[i + j] += ai * bj }
    end
    poly_trim(r)
  end

  # Verifies every invariant the rational canonical form must satisfy,
  # exactly, and returns the decomposition for further checks.
  def assert_rcf(m)
    rcf = m.rational_canonical_form
    f = rcf.frobenius_form
    p = rcf.basis

    # The change of basis is exact and invertible
    assert_equal(m * p, p * f, "A*P == P*F for #{m.to_a.inspect}")
    assert_not_equal(0, p.determinant, "P is invertible")

    # The blocks reassemble into the Frobenius form
    unless rcf.blocks.empty?
      n = rcf.blocks.inject(0) { |s, b| s + b.row_count }
      rows = Array.new(n) { Array.new(n, 0) }
      offset = 0
      rcf.blocks.each do |b|
        b.row_count.times do |i|
          b.column_count.times { |j| rows[offset + i][offset + j] = b[i, j] }
        end
        offset += b.row_count
      end
      assert_equal(Matrix.rows(rows), f)
    end

    # Invariant factors are non-constant, monic, and each divides the next
    rcf.invariant_factors.each do |d|
      assert(d.size >= 2, "non-constant: #{d.inspect}")
      assert_equal(1, d.last, "monic: #{d.inspect}")
    end
    rcf.invariant_factors.each_cons(2) do |d1, d2|
      _, r = poly_divmod(d2, d1)
      assert_empty(r, "#{d1.inspect} divides #{d2.inspect}")
    end

    # Their degrees add up to the size of the matrix, and their product is
    # the characteristic polynomial
    assert_equal(m.row_count, rcf.invariant_factors.inject(0) { |s, d| s + d.size - 1 })
    char = rcf.invariant_factors.inject([1]) { |acc, d| poly_mul(acc, d) }
    assert_equal(char, rcf.characteristic_polynomial)
    rcf
  end

  def test_irrational_eigenvalues
    # Eigenvalues are +/-sqrt(2): no rational eigen decomposition exists,
    # but the rational canonical form does.
    rcf = assert_rcf(Matrix[[0, 2], [1, 0]])
    assert_equal([[-2, 0, 1]], rcf.invariant_factors) # x**2 - 2
    assert_equal([Matrix[[0, 2], [1, 0]]], rcf.blocks)
  end

  def test_repeated_irrational_factor
    # Two identical blocks with irrational eigenvalues, conjugated away
    # from Frobenius form so the change of basis is nontrivial.
    f0 = Matrix[[0, 2, 0, 0], [1, 0, 0, 0], [0, 0, 0, 2], [0, 0, 1, 0]]
    s = Matrix[[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 1]]
    rcf = assert_rcf(s * f0 * s.inverse)
    assert_equal([[-2, 0, 1], [-2, 0, 1]], rcf.invariant_factors)
    assert_equal(f0, rcf.frobenius_form)
  end

  def test_jordan_block_not_diagonalizable
    rcf = assert_rcf(Matrix[[2, 1, 0], [0, 2, 1], [0, 0, 2]])
    assert_equal([[-8, 12, -6, 1]], rcf.invariant_factors) # (x - 2)**3
    assert_equal([Matrix[[0, 0, 8], [1, 0, -12], [0, 1, 6]]], rcf.blocks)
  end

  def test_identical_jordan_blocks
    j = Matrix[[1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]]
    s = Matrix[[1, 2, 0, 0], [0, 1, 0, 0], [0, 0, 1, 3], [0, 0, 0, 1]]
    rcf = assert_rcf(s * j * s.inverse)
    assert_equal([[1, -2, 1], [1, -2, 1]], rcf.invariant_factors) # (x-1)**2 twice
  end

  def test_rational_entries
    rcf = assert_rcf(Matrix[[Rational(1, 2), 1], [0, Rational(1, 3)]])
    assert_equal([[Rational(1, 6), Rational(-5, 6), 1]], rcf.invariant_factors)
  end

  def test_mixed_size_factors
    rcf = assert_rcf(Matrix[[1, 2, 3], [4, 5, 6], [7, 8, 10]])
    assert_equal([[3, -12, -16, 1]], rcf.invariant_factors)
  end

  def test_zero_matrix
    rcf = assert_rcf(Matrix.zero(3))
    assert_equal([[0, 1], [0, 1], [0, 1]], rcf.invariant_factors)
  end

  def test_identity_matrix
    rcf = assert_rcf(Matrix.identity(3))
    assert_equal([[-1, 1], [-1, 1], [-1, 1]], rcf.invariant_factors)
  end

  def test_scalar_multiple_of_identity
    rcf = assert_rcf(Matrix[[2, 0], [0, 2]])
    assert_equal([[-2, 1], [-2, 1]], rcf.invariant_factors)
  end

  def test_single_entry
    rcf = assert_rcf(Matrix[[5]])
    assert_equal([[-5, 1]], rcf.invariant_factors)
    assert_equal(Matrix[[5]], rcf.frobenius_form)
  end

  def test_empty_matrix
    rcf = assert_rcf(Matrix.empty(0, 0))
    assert_equal([], rcf.invariant_factors)
    assert_equal([], rcf.blocks)
    assert_equal(Matrix.empty(0, 0), rcf.basis)
  end

  def test_similar_matrices_share_invariant_factors
    a = Matrix[[1, 2, 3], [4, 5, 6], [7, 8, 10]]
    s = Matrix[[1, 1, 0], [0, 1, 1], [0, 0, 1]]
    b = s * a * s.inverse
    assert_equal(a.rational_canonical_form.invariant_factors,
                 b.rational_canonical_form.invariant_factors)
    assert_rcf(b)
  end

  def test_same_characteristic_polynomial_but_not_similar
    # Both matrices have characteristic polynomial (x-1)**2 * (x-2), but
    # a is diagonalizable (minimal polynomial (x-1)(x-2)) while b is not
    # (minimal polynomial (x-1)**2 (x-2)): they are not similar.  The
    # characteristic polynomial alone cannot tell them apart; the
    # invariant factors can.
    a = Matrix[[1, 0, 0], [0, 0, -2], [0, 1, 3]]
    b = Matrix[[0, 0, 2], [1, 0, -5], [0, 1, 4]]
    rcf_a = assert_rcf(a)
    rcf_b = assert_rcf(b)

    assert_equal([-2, 5, -4, 1], rcf_a.characteristic_polynomial)
    assert_equal(rcf_a.characteristic_polynomial, rcf_b.characteristic_polynomial)

    assert_equal([[-1, 1], [2, -3, 1]], rcf_a.invariant_factors)
    assert_equal([[-2, 5, -4, 1]], rcf_b.invariant_factors)
    assert_not_equal(rcf_a.invariant_factors, rcf_b.invariant_factors)
    assert_not_equal(rcf_a.frobenius_form, rcf_b.frobenius_form)
  end

  def test_aliases
    rcf = Matrix[[0, 2], [1, 0]].rational_canonical_form
    assert_equal(rcf.basis, rcf.p)
    assert_equal(rcf.frobenius_form, rcf.f)
  end

  def test_no_float_contamination
    rcf = Matrix[[Rational(1, 2), 2], [3, Rational(-1, 3)]].rational_canonical_form
    (rcf.invariant_factors.flatten +
     rcf.basis.to_a.flatten +
     rcf.frobenius_form.to_a.flatten).each do |e|
      assert(e.is_a?(Integer) || e.is_a?(Rational), "exact entry, got #{e.class}")
    end
  end

  def test_invalid_input
    assert_raise(TypeError) { Matrix[[1.0, 2], [3, 4]].rational_canonical_form }
    assert_raise(TypeError) { Matrix[[Complex(1), 0], [0, 1]].rational_canonical_form }
    assert_raise(ExceptionForMatrix::ErrDimensionMismatch) { Matrix[[1, 2, 3], [4, 5, 6]].rational_canonical_form }
    assert_raise(TypeError) { Matrix::RationalCanonicalForm.new(Vector[1, 2]) }
  end
end
