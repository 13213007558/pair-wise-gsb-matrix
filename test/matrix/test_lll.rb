# frozen_string_literal: false
require 'test/unit'
require 'matrix'

class TestMatrixLLL < Test::Unit::TestCase
  # Exact Gram-Schmidt on row arrays; returns [mu, norms].
  def gso(rows)
    n = rows.size
    mu = Array.new(n) { Array.new(n, 0) }
    norms = []
    orth = []
    n.times do |i|
      v = rows[i].map(&:to_r)
      i.times do |j|
        mu[i][j] = rows[i].zip(orth[j]).sum { |x, y| x * y } / norms[j]
        v = v.zip(orth[j]).map { |x, y| x - mu[i][j] * y }
      end
      orth << v
      norms << v.zip(v).sum { |x, y| x * y }
    end
    [mu, norms]
  end

  def assert_lll_reduced(b, delta)
    mu, norms = gso(b.to_a)
    n = b.row_count
    n.times do |i|
      i.times do |j|
        assert_operator mu[i][j].abs, :<=, 1/2r,
          "not size-reduced: mu[#{i}][#{j}] = #{mu[i][j]}"
      end
      next if i == 0
      lhs = norms[i]
      rhs = (delta.to_r - mu[i][i - 1]**2) * norms[i - 1]
      assert_operator lhs, :>=, rhs,
        "Lovasz condition fails at #{i}: #{lhs} < #{rhs}"
    end
  end

  def assert_valid_result(a, b, u, delta)
    assert_equal a.row_count, b.row_count
    assert_equal a.column_count, b.column_count
    assert u.square?, "transform must be square"
    assert u.all? { |e| e.is_a?(Integer) }, "transform must be integral"
    assert_equal b, u * a, "expected B == U*A"
    assert_equal 1, u.determinant.abs, "transform must be unimodular"
    assert_lll_reduced b, delta
  end

  def test_classic_example
    a = Matrix[[1, 1, 1], [-1, 0, 2], [3, 5, 6]]
    b, u = a.lll_reduce
    assert_valid_result a, b, u, 3/4r
    assert_equal Matrix[[0, 1, 0], [1, 0, 1], [-2, 0, 1]], b
  end

  def test_custom_delta
    a = Matrix[[1, 1, 1], [-1, 0, 2], [3, 5, 6]]
    [1/4r + 1/10r**10, 1/2r, 3/4r, 9/10r, 99/100r].each do |delta|
      b, u = a.lll_reduce(delta)
      assert_valid_result a, b, u, delta
    end
  end

  def test_float_delta_is_exact
    a = Matrix[[1, 1, 1], [-1, 0, 2], [3, 5, 6]]
    b, u = a.lll_reduce(0.75)
    assert_valid_result a, b, u, 0.75
  end

  def test_invalid_delta
    a = Matrix[[1, 2], [3, 4]]
    [0, 1/4r, 1, 2, -1/2r, 0.25, 1.0].each do |delta|
      assert_raise(ArgumentError) { a.lll_reduce(delta) }
    end
    assert_raise(ArgumentError) { a.lll_reduce(nil) }
  end

  def test_non_integer_entries
    assert_raise(ArgumentError) { Matrix[[1.5, 2], [3, 4]].lll_reduce }
    assert_raise(ArgumentError) { Matrix[[1, 2], [3, 4r]].lll_reduce }
  end

  def test_dependent_rows
    assert_raise(Matrix::ErrNotRegular) { Matrix[[1, 2], [2, 4]].lll_reduce }
    assert_raise(Matrix::ErrNotRegular) { Matrix[[0, 0], [1, 2]].lll_reduce }
    assert_raise(Matrix::ErrNotRegular) { Matrix[[1, 2], [3, 4], [5, 6]].lll_reduce }
    assert_raise(Matrix::ErrNotRegular) { Matrix[[0, 0, 0]].lll_reduce }
  end

  def test_already_reduced
    b, u = Matrix[[1, 0], [0, 1]].lll_reduce
    assert_equal Matrix[[1, 0], [0, 1]], b
    assert_equal Matrix.identity(2), u
  end

  def test_single_row
    a = Matrix[[3, -4, 5]]
    b, u = a.lll_reduce
    assert_equal a, b
    assert_equal Matrix.identity(1), u
  end

  def test_rectangular_wide
    a = Matrix[[105, 821, 150, 209], [37, 289, 52, 74]]
    b, u = a.lll_reduce
    assert_valid_result a, b, u, 3/4r
  end

  def test_unimodular_scramble
    # A = U0 * B0 with U0 unimodular: A and B0 span the same lattice,
    # but the rows of A are huge. Reduction should recover short vectors.
    b0 = Matrix[[2, 0, 0], [0, 3, 0], [0, 0, 5]]
    u0 = Matrix[[1, 37, -52], [0, 1, -11], [0, 0, 1]] *
         Matrix[[1, 0, 0], [24, 1, 0], [-18, 0, 1]]
    assert_equal 1, u0.determinant
    a = u0 * b0
    b, u = a.lll_reduce(99/100r)
    assert_valid_result a, b, u, 99/100r
    # Same lattice as B0: determinants of the Gram matrices agree.
    assert_equal (b0 * b0.transpose).determinant, (b * b.transpose).determinant
    # Reduction actually made the rows much shorter.
    assert b.row_vectors.all? { |v| v.inner_product(v) <= 100 }
  end

  def test_random_unimodular
    rng = Random.new(20261002)
    n = 6
    u0 = Matrix.identity(n)
    30.times do
      i, j = rng.rand(n), rng.rand(n)
      redo if i == j
      e = Matrix.identity(n).to_a
      e[i][j] = rng.rand(-5..5)
      u0 = Matrix.rows(e) * u0
    end
    assert_equal 1, u0.determinant
    b0 = Matrix.diagonal(*Array.new(n) { rng.rand(1..4) })
    a = u0 * b0
    [1/2r, 3/4r, 99/100r].each do |delta|
      b, u = a.lll_reduce(delta)
      assert_valid_result a, b, u, delta
    end
  end

  def test_does_not_modify_receiver
    a = Matrix[[1, 1, 1], [-1, 0, 2], [3, 5, 6]]
    a.lll_reduce
    assert_equal Matrix[[1, 1, 1], [-1, 0, 2], [3, 5, 6]], a
  end
end

