# frozen_string_literal: false
class Matrix
  #
  # Matrix::RationalCanonicalForm
  #
  # The rational canonical form (Frobenius normal form) of a square matrix
  # over the rationals.
  #
  # Every square matrix +A+ with rational entries is similar, over the
  # rationals, to a unique block diagonal matrix +F+ whose diagonal blocks
  # are the companion matrices of monic polynomials +d1, d2, ..., dk+ with
  # +d1 | d2 | ... | dk+.  Those polynomials are the invariant factors of
  # +A+, and:
  #
  #   A * P == P * F,  i.e.  A == P * F * P.inverse
  #
  # for some invertible rational matrix +P+.
  #
  # The computation is entirely exact: only Integer and Rational arithmetic
  # is used, and no floating point approximation is ever involved.  This
  # makes it usable for matrices whose eigenvalues are not rational (or not
  # even real), as well as for non-diagonalizable matrices, unlike
  # EigenvalueDecomposition.
  #
  # Two rational matrices are similar over the rationals if and only if
  # they have the same invariant factors.  In particular this decomposition
  # distinguishes matrices that share the same characteristic polynomial
  # but are not similar:
  #
  #   a = Matrix[[1, 0, 0], [0, 0, -2], [0, 1, 3]]
  #   b = Matrix[[0, 0, 2], [1, 0, -5], [0, 1, 4]]
  #   a.rational_canonical_form.characteristic_polynomial ==
  #     b.rational_canonical_form.characteristic_polynomial # => true
  #   a.rational_canonical_form.invariant_factors # => [[-1, 1], [2, -3, 1]]
  #   b.rational_canonical_form.invariant_factors # => [[-2, 5, -4, 1]]
  #
  # Polynomials are represented as arrays of exact coefficients in
  # increasing order of degree: +[c0, c1, ..., cd]+ stands for
  # +c0 + c1*x + ... + cd*x**d+.
  #
  class RationalCanonicalForm
    # Returns the Frobenius blocks, i.e. the companion matrices of the
    # invariant factors, in the order they appear on the diagonal of the
    # Frobenius form.
    attr_reader :blocks

    # Returns the invertible change-of-basis matrix +P+ such that
    # +A * P == P * F+, where +F+ is the Frobenius form of +A+.
    # +P+ is exact, but not unique.
    attr_reader :basis
    alias_method :p, :basis

    # Returns the non-constant invariant factors of the matrix, as arrays
    # of coefficients, sorted so that each one divides the next.
    attr_reader :invariant_factors

    # Returns the Frobenius normal form +F+ of the matrix: the block
    # diagonal matrix of the companion matrices of the invariant factors.
    attr_reader :frobenius_form
    alias_method :f, :frobenius_form

    def initialize(a)
      raise TypeError, "Expected Matrix but got #{a.class}" unless a.is_a?(Matrix)
      raise Matrix::ErrDimensionMismatch, "Rational canonical form is only defined for square matrices" unless a.square?
      a.each do |e|
        next if e.is_a?(Integer) || e.is_a?(Rational)
        raise TypeError, "Rational canonical form requires exact entries (Integer or Rational), got #{e.class}"
      end

      @size = a.row_count
      @invariant_factors = compute_invariant_factors(a)
      @blocks = @invariant_factors.map { |d| companion_matrix(d) }
      @frobenius_form = block_diagonal(@blocks)
      @basis = @size == 0 ? Matrix.empty(0, 0) : find_basis(a)
      unless a * @basis == @basis * @frobenius_form
        raise Matrix::ErrNotRegular, "Failed to compute a change of basis to the Frobenius form"
      end
    end

    # Returns the characteristic polynomial of the matrix (the product of
    # its invariant factors), as an array of coefficients.
    def characteristic_polynomial
      @invariant_factors.inject([1]) { |acc, d| Poly.mul(acc, d) }
    end

    # Exact polynomial arithmetic over the rationals.
    # Polynomials are arrays of coefficients, lowest degree first,
    # normalized so that the last coefficient is nonzero; +[]+ is the
    # zero polynomial.
    module Poly # :nodoc:
      module_function

      def normalize(p)
        p = p.dup
        p.pop until p.empty? || p.last != 0
        simplify(p)
      end

      def simplify(p)
        p.map! { |c| c.is_a?(Rational) && c.denominator == 1 ? c.numerator : c }
      end

      def add(a, b)
        n = a.size >= b.size ? a.size : b.size
        r = Array.new(n, 0)
        i = 0
        while i < n
          r[i] = (i < a.size ? a[i] : 0) + (i < b.size ? b[i] : 0)
          i += 1
        end
        normalize(r)
      end

      def sub(a, b)
        n = a.size >= b.size ? a.size : b.size
        r = Array.new(n, 0)
        i = 0
        while i < n
          r[i] = (i < a.size ? a[i] : 0) - (i < b.size ? b[i] : 0)
          i += 1
        end
        normalize(r)
      end

      def mul(a, b)
        return [] if a.empty? || b.empty?
        r = Array.new(a.size + b.size - 1, 0)
        a.each_with_index do |ai, i|
          next if ai == 0
          b.each_with_index do |bj, j|
            r[i + j] = r[i + j] + ai * bj
          end
        end
        normalize(r)
      end

      def divmod(a, b)
        raise ZeroDivisionError, "divided by the zero polynomial" if b.empty?
        a = a.dup
        q = Array.new(a.size >= b.size ? a.size - b.size + 1 : 0, 0)
        until a.empty? || a.size < b.size
          coeff = a.last.quo(b.last)
          shift = a.size - b.size
          q[shift] = coeff
          i = 0
          while i < b.size
            a[shift + i] = a[shift + i] - coeff * b[i]
            i += 1
          end
          a.pop until a.empty? || a.last != 0
        end
        [normalize(q), normalize(a)]
      end

      def monic(p)
        return p if p.empty? || p.last == 1
        lc = p.last
        simplify(p.map { |c| c.quo(lc) })
      end

      def gcd(a, b)
        until b.empty?
          _, r = divmod(a, b)
          a = b
          b = r
        end
        monic(a)
      end
    end
    private_constant :Poly

    private

    # Non-constant invariant factors of +a+, computed as the non-unit
    # diagonal entries of the Smith normal form of +x*I - a+ over Q[x].
    def compute_invariant_factors(a)
      n = a.row_count
      return [] if n == 0
      pm = Array.new(n) do |i|
        Array.new(n) do |j|
          v = a[i, j]
          if i == j
            [-v, 1]
          else
            v == 0 ? [] : [-v]
          end
        end
      end
      smith_invariants(pm).reject { |p| p.size <= 1 }
    end

    # Diagonal entries of the Smith normal form of a square matrix of
    # polynomials, using exact row and column operations over Q[x].
    def smith_invariants(mat)
      n = mat.size
      m = mat.map { |row| row.map { |p| Poly.normalize(p) } }
      k = 0
      while k < n
        # Locate a nonzero entry of minimal degree in the k..n-1 submatrix
        pi = pj = min_deg = nil
        (k...n).each do |i|
          (k...n).each do |j|
            p = m[i][j]
            next if p.empty?
            d = p.size - 1
            if min_deg.nil? || d < min_deg
              pi = i
              pj = j
              min_deg = d
            end
          end
        end
        break unless pi

        loop do
          # Move the chosen entry to the pivot position (k, k)
          if pi != k
            m[pi], m[k] = m[k], m[pi]
            pi = k
          end
          if pj != k
            (k...n).each { |i| m[i][pj], m[i][k] = m[i][k], m[i][pj] }
            pj = k
          end
          pivot = m[k][k]

          # Reduce column k and row k modulo the pivot
          (k + 1...n).each do |i|
            q, = Poly.divmod(m[i][k], pivot)
            next if q.empty?
            (k...n).each { |j| m[i][j] = Poly.sub(m[i][j], Poly.mul(q, m[k][j])) }
          end
          (k + 1...n).each do |j|
            q, = Poly.divmod(m[k][j], pivot)
            next if q.empty?
            (k...n).each { |i| m[i][j] = Poly.sub(m[i][j], Poly.mul(q, m[i][k])) }
          end

          # Any nonzero remainder has a smaller degree than the pivot;
          # move the smallest one into the pivot position and repeat
          pi = pj = min_deg = nil
          (k + 1...n).each do |i|
            p = m[i][k]
            if !p.empty? && (min_deg.nil? || p.size - 1 < min_deg)
              pi = i
              pj = k
              min_deg = p.size - 1
            end
            p = m[k][i]
            if !p.empty? && (min_deg.nil? || p.size - 1 < min_deg)
              pi = k
              pj = i
              min_deg = p.size - 1
            end
          end
          next if pi

          # Row and column k are clear; the pivot must also divide every
          # entry of the remaining submatrix
          di = nil
          (k + 1...n).each do |i|
            (k + 1...n).each do |j|
              _, r = Poly.divmod(m[i][j], pivot)
              unless r.empty?
                di = i
                break
              end
            end
            break if di
          end
          break unless di

          # Bring the offending entry into row k; the next reduction
          # strictly decreases the degree of the pivot
          (k...n).each { |j| m[k][j] = Poly.add(m[k][j], m[di][j]) }
          pi = pj = k
        end

        m[k][k] = Poly.monic(m[k][k])
        k += 1
      end
      (0...n).map { |i| m[i][i] }
    end

    # Companion matrix of a monic polynomial +[c0, c1, ..., 1]+: ones on
    # the subdiagonal and the negated coefficients in the last column.
    def companion_matrix(poly)
      d = poly.size - 1
      rows = Array.new(d) { Array.new(d, 0) }
      (1...d).each { |i| rows[i][i - 1] = 1 }
      d.times { |i| rows[i][d - 1] = -poly[i] }
      Matrix.send(:new, rows)
    end

    def block_diagonal(blocks)
      return Matrix.empty(0, 0) if blocks.empty?
      n = blocks.inject(0) { |sum, b| sum + b.row_count }
      rows = Array.new(n) { Array.new(n, 0) }
      offset = 0
      blocks.each do |b|
        b.row_count.times do |i|
          b.column_count.times do |j|
            rows[offset + i][offset + j] = b[i, j]
          end
        end
        offset += b.row_count
      end
      Matrix.send(:new, rows)
    end

    # Invertible matrix +P+ with +a * P == P * f+, computed by the
    # classical cyclic decomposition: repeatedly split off a maximal
    # cyclic (Krylov) subspace together with an invariant complement.
    # The factors obtained along the way must match the invariant factors
    # computed through the Smith normal form, which is verified here.
    def find_basis(a)
      n = a.row_count
      return Matrix.identity(n) if a == @frobenius_form
      basis, factors = krylov_decompose(a)
      unless factors == @invariant_factors.reverse
        raise Matrix::ErrNotRegular, "Failed to compute a change of basis to the Frobenius form"
      end
      # The Krylov decomposition peels off the largest invariant factor
      # first; reorder the blocks of columns to match the order of the
      # invariant factors in the Frobenius form.
      cols = basis.column_vectors
      groups = []
      offset = 0
      factors.each do |f|
        size = f.size - 1
        groups << cols[offset, size]
        offset += size
      end
      Matrix.columns(groups.reverse.flatten(1).map(&:to_a))
    end

    # Decomposes +a+ as +P * F * P.inverse+, where +F+ is the block
    # diagonal matrix of the companion matrices of the returned monic
    # polynomials +[m1, m2, ...]+ with m_{i+1} dividing m_i.
    # Returns +[P, [m1, m2, ...]]+.
    def krylov_decompose(a)
      n = a.row_count
      return [Matrix.empty(0, 0), []] if n == 0
      mu = minimal_polynomial(a)
      d = mu.size - 1
      v = cyclic_vector(a, mu)
      cols = [v]
      (d - 1).times { cols << (a * cols[-1]) }
      n.times do |i|
        e = Vector.basis(size: n, index: i)
        candidate = cols + [e]
        cols << e if Matrix.columns(candidate.map(&:to_a)).rank == candidate.size
      end
      b = Matrix.columns(cols.map(&:to_a))
      return [b, [mu]] if d == n

      # In the basis +b+, +a+ is block upper triangular with the companion
      # matrix of +mu+ in the top left corner; an invariant complement is
      # obtained by solving the Sylvester equation C*X - X*A2 = -G.
      aprime = b.inverse * a * b
      g = aprime.minor(0...d, d...n)
      a2 = aprime.minor(d...n, d...n)
      x = solve_sylvester(companion_matrix(mu), a2, -g)
      p2, factors = krylov_decompose(a2)
      m = Matrix.I(d).hstack(x * p2).vstack(Matrix.zero(n - d, d).hstack(p2))
      [b * m, [mu] + factors]
    end

    # Minimal polynomial of +a+: the least common multiple of the
    # annihilators of the standard basis vectors.
    def minimal_polynomial(a)
      n = a.row_count
      mu = [1]
      n.times do |i|
        mu = poly_lcm(mu, annihilator(a, Vector.basis(size: n, index: i)))
      end
      mu
    end

    def poly_lcm(a, b)
      q, = Poly.divmod(b, Poly.gcd(a, b))
      Poly.monic(Poly.mul(a, q))
    end

    # The unique monic polynomial +p+ of minimal degree with +p(a)*v == 0+.
    def annihilator(a, v)
      cols = [v]
      loop do
        w = a * cols[-1]
        k = cols.size
        mat = Array.new(a.row_count) { |i| Array.new(k) { |j| cols[j][i] } }
        sol = solve_linear_system(mat, w.to_a)
        return Poly.normalize(sol.map { |c| -c } + [1]) if sol
        cols << w
      end
    end

    # A vector whose annihilator is exactly the given minimal polynomial
    # +mu+.  Such vectors always exist; the enumeration is guaranteed to
    # find one, because the annihilator of almost every vector is +mu+
    # (the exceptional vectors lie in a union of at most +n+ proper
    # subspaces, which cannot cover the grid {-n..n}**n).
    def cyclic_vector(a, mu)
      n = a.row_count
      bound = 0
      loop do
        bound += 1
        each_candidate_vector(n, bound) do |coords|
          v = Vector.elements(coords)
          return v if annihilator(a, v) == mu
        end
        raise Matrix::ErrNotRegular, "Failed to find a cyclic vector" if bound >= n
      end
    end

    # Yields vectors of length +n+ with entries in -bound..bound whose
    # largest absolute value is exactly +bound+, ordered by increasing
    # number of nonzero entries.
    def each_candidate_vector(n, bound)
      (1..n).each do |weight|
        positions = (0...weight).to_a
        loop do
          values = Array.new(weight, -bound)
          loop do
            if values.any? { |val| val.abs == bound }
              vec = Array.new(n, 0)
              positions.each_with_index { |pos, k| vec[pos] = values[k] }
              yield vec
            end
            pos = 0
            pos += 1 while pos < weight && values[pos] == bound && (values[pos] = -bound)
            break if pos == weight
            values[pos] += 1
          end
          break unless next_combination(positions, n)
        end
      end
    end

    def next_combination(positions, n)
      k = positions.size
      i = k - 1
      i -= 1 while i >= 0 && positions[i] == n - k + i
      return false if i < 0
      positions[i] += 1
      (i + 1...k).each { |j| positions[j] = positions[j - 1] + 1 }
      true
    end

    # Solves the Sylvester equation +c * x - x * a2 == b+ exactly over the
    # rationals.  A solution is guaranteed to exist in this context.
    def solve_sylvester(c, a2, b)
      d = c.row_count
      m = a2.row_count
      sys = []
      rhs = []
      d.times do |i|
        m.times do |j|
          eq = Array.new(d * m, 0)
          d.times { |l| eq[l * m + j] = eq[l * m + j] + c[i, l] unless c[i, l] == 0 }
          m.times { |l| eq[i * m + l] = eq[i * m + l] - a2[l, j] unless a2[l, j] == 0 }
          sys << eq
          rhs << b[i, j]
        end
      end
      sol = solve_linear_system(sys, rhs)
      raise Matrix::ErrNotRegular, "Failed to solve a Sylvester equation" unless sol
      Matrix.rows(Array.new(d) { |i| sol[i * m, m] })
    end

    # Solves the linear system +mat * x == rhs+ exactly over the rationals.
    # Returns a particular solution (free variables set to zero), or nil
    # if the system is inconsistent.
    def solve_linear_system(mat, rhs)
      nrows = mat.size
      ncols = mat[0].size
      aug = mat.each_with_index.map { |row, i| row.dup << rhs[i] }
      pivot_cols = []
      r = 0
      (0...ncols).each do |c|
        break if r == nrows
        piv = nil
        (r...nrows).each do |i|
          if aug[i][c] != 0
            piv = i
            break
          end
        end
        next unless piv
        aug[r], aug[piv] = aug[piv], aug[r]
        (0...nrows).each do |i|
          next if i == r
          factor = aug[i][c]
          next if factor == 0
          q = factor.quo(aug[r][c])
          (c..ncols).each { |j| aug[i][j] = aug[i][j] - q * aug[r][j] }
        end
        pivot_cols << c
        r += 1
      end
      (r...nrows).each { |i| return nil unless aug[i][ncols] == 0 }
      sol = Array.new(ncols, 0)
      pivot_cols.each_with_index do |pc, i|
        sol[pc] = aug[i][ncols].quo(aug[i][pc])
      end
      sol
    end
  end
end
