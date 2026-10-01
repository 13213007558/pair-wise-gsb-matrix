# encoding: utf-8
# frozen_string_literal: false

class Matrix
  #
  # Performs LLL (Lenstra-Lenstra-Lovasz) lattice basis reduction, treating
  # the rows of the matrix as row vectors spanning a lattice.
  #
  # Returns an array <tt>[reduced_basis, transform]</tt> where:
  # * +reduced_basis+ is a Matrix whose rows are LLL-reduced with respect to
  #   +delta+: they are size-reduced and satisfy the Lovasz condition,
  # * +transform+ is a unimodular integer Matrix (determinant +/-1) such that
  #   <tt>transform * self == reduced_basis</tt>. It certifies that both
  #   bases generate the same lattice and can be checked independently.
  #
  # The optional +delta+ must lie in the open interval (1/4, 1); values
  # closer to 1 yield a more strongly reduced basis at the cost of more
  # swaps. All computations use exact rational arithmetic, so the Lovasz
  # condition is decided exactly, even at or near equality.
  #
  # The matrix must have Integer entries and linearly independent rows.
  #
  #   b, u = Matrix[[1, 1, 1], [-1, 0, 2], [3, 5, 6]].lll_reduce
  #   b                             # => Matrix[[0, 1, 0], [1, 0, 1], [-2, 0, 1]]
  #   u * Matrix[[1, 1, 1], [-1, 0, 2], [3, 5, 6]] == b
  #                                 # => true
  #   u.determinant.abs             # => 1
  #
  # Raises ArgumentError if +delta+ is out of range or if any entry is not
  # an Integer, and ErrNotRegular if the rows are linearly dependent.
  #
  def lll_reduce(delta = 3/4r)
    delta = begin
      delta.to_r
    rescue NoMethodError, RangeError, FloatDomainError
      raise ArgumentError, "delta must be a real number, got #{delta.inspect}"
    end
    unless delta > 1/4r && delta < 1
      raise ArgumentError, "delta must be in the open interval (1/4, 1), got #{delta.inspect}"
    end
    each do |entry|
      unless entry.is_a?(Integer)
        raise ArgumentError, "LLL reduction requires Integer entries, got #{entry.inspect}"
      end
    end

    n = row_count
    return [self, Matrix.identity(n)] if n == 0

    basis = @rows.map(&:dup)
    transform_rows = Matrix.identity(n).to_a

    dot = ->(v, w) { v.zip(w).sum { |x, y| x * y } }

    # Exact Gram-Schmidt orthogonalization of the given rows.
    # Returns [mu, norms] where mu[i][j] = <rows[i], rows*[j]> / |rows*[j]|^2
    # and norms[i] = |rows*[i]|^2.
    gram_schmidt = ->(rows) {
      mu = Array.new(n) { Array.new(n, 0) }
      norms = Array.new(n)
      orthogonal = []
      n.times do |i|
        v = rows[i].map(&:to_r)
        i.times do |j|
          mu[i][j] = dot.call(rows[i], orthogonal[j]) / norms[j]
          v = v.zip(orthogonal[j]).map { |x, y| x - mu[i][j] * y }
        end
        orthogonal << v
        norms[i] = dot.call(v, v)
        if norms[i] == 0
          raise ErrNotRegular, "LLL reduction requires linearly independent rows"
        end
      end
      [mu, norms]
    }

    mu, norms = gram_schmidt.call(basis)

    k = 1
    while k < n
      # Size-reduce basis[k] against basis[0..k-1], updating mu incrementally.
      (k - 1).downto(0) do |j|
        q = mu[k][j].round
        next if q == 0
        basis[k] = basis[k].zip(basis[j]).map { |x, y| x - q * y }
        transform_rows[k] = transform_rows[k].zip(transform_rows[j]).map { |x, y| x - q * y }
        j.times { |i| mu[k][i] -= q * mu[j][i] }
        mu[k][j] -= q
      end
      # Lovasz condition, decided with exact rational arithmetic.
      if norms[k] >= (delta - mu[k][k - 1]**2) * norms[k - 1]
        k += 1
      else
        basis[k - 1], basis[k] = basis[k], basis[k - 1]
        transform_rows[k - 1], transform_rows[k] = transform_rows[k], transform_rows[k - 1]
        mu, norms = gram_schmidt.call(basis)
        k = k > 2 ? k - 1 : 1
      end
    end

    reduced = Matrix.rows(basis)
    transform = Matrix.rows(transform_rows)

    # Independent exact verification of the result.
    unless transform * self == reduced
      raise "LLL internal error: transform * self != reduced basis"
    end
    unless transform.determinant.abs == 1
      raise "LLL internal error: transform is not unimodular"
    end
    check_mu, check_norms = gram_schmidt.call(reduced.to_a)
    n.times do |i|
      i.times do |j|
        if check_mu[i][j].abs > 1/2r
          raise "LLL internal error: basis is not size-reduced"
        end
      end
      next if i == 0
      unless check_norms[i] >= (delta - check_mu[i][i - 1]**2) * check_norms[i - 1]
        raise "LLL internal error: Lovasz condition violated"
      end
    end

    [reduced, transform]
  end
end
