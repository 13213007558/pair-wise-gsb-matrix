# frozen_string_literal: false
class Matrix
  #
  # For an n-by-n real symmetric matrix A, the LDLt decomposition with
  # Bunch-Kaufman pivoting is a unit lower triangular matrix L, a block
  # diagonal matrix D with 1x1 and 2x2 diagonal blocks, and a permutation
  # matrix P so that
  #
  #   L * D * L.transpose == P * A * P.transpose
  #
  # The 2x2 blocks of D allow the decomposition of invertible matrices
  # whose diagonal entries are all zero or tiny, e.g.
  # Matrix[[0, 1], [1, 0]], which must not be reported as singular.
  #
  # The decomposition always exists, so the constructor will never fail.
  # Its primary uses are solving symmetric indefinite systems of linear
  # equations (possibly with several right hand sides) and computing the
  # inertia of A, i.e. the numbers of positive, negative and zero
  # eigenvalues.
  #
  class LDLTDecomposition
    include Matrix::ConversionHelper

    # Bunch-Kaufman pivoting threshold: (1 + sqrt(17)) / 8
    ALPHA = (1.0 + Math.sqrt(17.0)) / 8.0

    # Returns the unit lower triangular factor +L+

    def l
      Matrix.build(@n, @n) {|i, j| i >= j ? @l[i][j] : 0}
    end

    # Returns the block diagonal factor +D+, with 1x1 and 2x2 blocks

    def d
      Matrix.build(@n, @n) do |i, j|
        if i == j
          @d_diag[i]
        elsif i == j + 1 && @d_sub[j]
          @d_sub[j]
        elsif j == i + 1 && @d_sub[i]
          @d_sub[i]
        else
          0
        end
      end
    end

    # Returns the permutation matrix +P+, such that
    # <tt>L*D*L.transpose == P*A*P.transpose</tt>

    def p
      rows = Array.new(@n){Array.new(@n, 0)}
      @pivots.each_with_index{|pivot, i| rows[i][pivot] = 1}
      Matrix.send :new, rows, @n
    end

    # Returns +L+, +D+, +P+ in an array

    def to_ary
      [l, d, p]
    end
    alias_method :to_a, :to_ary

    # Returns the pivoting indices; row +i+ of the permuted matrix
    # <tt>P*A*P.transpose</tt> corresponds to row <tt>pivots[i]</tt> of +A+.

    attr_reader :pivots

    # Returns the relative tolerance used by nearly_singular?

    attr_reader :tol

    # Returns +true+ if +D+, and hence +A+, is exactly singular,
    # i.e. some 1x1 pivot is zero or some 2x2 pivot block has a
    # zero determinant.

    def singular?
      k = 0
      while k < @n
        if @d_sub[k]
          return true if block_det(k) == 0
          k += 2
        else
          return true if @d_diag[k] == 0
          k += 1
        end
      end
      false
    end

    # Returns +true+ if +A+ is numerically near-singular, i.e. the
    # smallest pivot magnitude is at most +tol+ times the largest one.
    # Pivot magnitudes of 2x2 blocks are the magnitudes of the two
    # eigenvalues of the block, not of its diagonal entries.
    # Strictly singular matrices are also nearly singular.

    def nearly_singular?(tolerance = @tol)
      return false if @n == 0
      min_scale = Float::INFINITY
      max_scale = 0.0
      k = 0
      while k < @n
        if @d_sub[k]
          lo, hi = block_eigenvalue_magnitudes(k)
          k += 2
        else
          lo = hi = @d_diag[k].abs
          k += 1
        end
        min_scale = lo if lo < min_scale
        max_scale = hi if hi > max_scale
      end
      return true if max_scale == 0
      min_scale <= tolerance * max_scale
    end

    # Returns the determinant of +A+, calculated efficiently from the
    # factorization. Since <tt>det(P) ** 2 == 1</tt>, the determinant of
    # +A+ is simply the product of the determinants of the blocks of +D+.

    def det
      d = 1
      k = 0
      while k < @n
        if @d_sub[k]
          d *= block_det(k)
          k += 2
        else
          d *= @d_diag[k]
          k += 1
        end
      end
      d
    end
    alias_method :determinant, :det

    # Returns the inertia of +A+ as an array
    # <tt>[positive, negative, zero]</tt> with the numbers of positive,
    # negative and zero eigenvalues. The signs contributed by a 2x2 block
    # are determined from the eigenvalues of the block (via its
    # determinant and trace), never from its two diagonal entries
    # separately.

    def inertia
      positive = negative = zero = 0
      k = 0
      while k < @n
        if @d_sub[k]
          det = block_det(k)
          if det < 0
            # eigenvalues of opposite signs
            positive += 1
            negative += 1
          else
            trace = @d_diag[k] + @d_diag[k + 1]
            if det == 0
              zero += 1
              if trace > 0
                positive += 1
              elsif trace < 0
                negative += 1
              else
                zero += 1
              end
            elsif trace > 0
              positive += 2
            else
              negative += 2
            end
          end
          k += 2
        else
          case @d_diag[k] <=> 0
          when 1 then positive += 1
          when -1 then negative += 1
          else zero += 1
          end
          k += 1
        end
      end
      [positive, negative, zero]
    end

    # Returns +m+ so that <tt>A*m = b</tt>, or equivalently so that
    # <tt>L*D*L.transpose*m = P*b*P.transpose</tt>.
    # +b+ can be a Matrix (each column is a right hand side) or a Vector.
    # Raises ErrNotRegular if +A+ is strictly singular; check
    # nearly_singular? to detect numerically troublesome systems.

    def solve(b)
      if singular?
        raise Matrix::ErrNotRegular, "Matrix is singular."
      end
      if b.is_a? Matrix
        if b.row_count != @n
          raise Matrix::ErrDimensionMismatch
        end
        rhs = Array.new(@n){|i| b.row(@pivots[i]).to_a}
        Matrix.send :new, backsolve(rhs, b.column_count), b.column_count
      else # same algorithm, specialized for simpler case of a vector
        b = convert_to_array(b)
        if b.size != @n
          raise Matrix::ErrDimensionMismatch
        end
        rhs = @pivots.map{|pivot| [b[pivot]]}
        Vector.elements(backsolve(rhs, 1).map(&:first), false)
      end
    end

    def initialize(a, tol: nil)
      raise TypeError, "Expected Matrix but got #{a.class}" unless a.is_a?(Matrix)
      raise Matrix::ErrDimensionMismatch unless a.square?
      unless a.symmetric?
        raise ArgumentError, "LDLt decomposition requires a symmetric matrix"
      end
      @n = a.row_count
      @tol = tol || Float::EPSILON * @n
      @a = a.to_a
      @l = Array.new(@n) do |i|
        row = Array.new(@n, 0)
        row[i] = 1
        row
      end
      @d_diag = Array.new(@n, 0)
      @d_sub = Array.new(@n, nil)
      @pivots = (0...@n).to_a
      factorize
    end

    private

    def factorize
      k = 0
      while k < @n
        if choose_pivot(k) == 2
          eliminate_2x2(k)
          k += 2
        else
          eliminate_1x1(k)
          k += 1
        end
      end
    end

    # Bunch-Kaufman pivot choice at column +k+; returns the pivot size
    # (1 or 2), after swapping rows/columns if necessary.
    def choose_pivot(k)
      n = @n
      a = @a
      return 1 if k == n - 1

      # largest off-diagonal magnitude in column k of the reduced matrix
      r = k + 1
      lambda1 = a[r][k].abs
      (k + 2...n).each do |i|
        m = a[i][k].abs
        if m > lambda1
          lambda1 = m
          r = i
        end
      end
      return 1 if a[k][k].abs >= ALPHA * lambda1

      # largest off-diagonal magnitude in row/column r of the reduced matrix
      sigma = 0
      (k...n).each do |j|
        next if j == r
        m = a[j > r ? j : r][j > r ? r : j].abs
        sigma = m if m > sigma
      end
      return 1 if a[k][k].abs * sigma >= ALPHA * lambda1 * lambda1
      if a[r][r].abs >= ALPHA * sigma
        swap(k, r, k)
        return 1
      end
      swap(k + 1, r, k)
      2
    end

    # Swaps rows/columns +p+ and +q+ of the reduced matrix, the
    # corresponding entries of the first +k+ columns of +L+, and the
    # permutation indices.
    def swap(p, q, k)
      return if p == q
      @a[p], @a[q] = @a[q], @a[p]
      @n.times {|j| @a[j][p], @a[j][q] = @a[j][q], @a[j][p]}
      k.times {|j| @l[p][j], @l[q][j] = @l[q][j], @l[p][j]}
      @pivots[p], @pivots[q] = @pivots[q], @pivots[p]
    end

    def eliminate_1x1(k)
      n = @n
      a = @a
      d = a[k][k]
      @d_diag[k] = d
      return if d == 0
      (k + 1...n).each do |i|
        @l[i][k] = a[i][k].quo(d)
      end
      (k + 1...n).each do |i|
        cik = a[i][k]
        next if cik == 0
        (k + 1...n).each do |j|
          a[i][j] -= cik * @l[j][k]
        end
      end
    end

    def eliminate_2x2(k)
      n = @n
      a = @a
      e11 = a[k][k]
      e21 = a[k + 1][k]
      e22 = a[k + 1][k + 1]
      det = e11 * e22 - e21 * e21
      @d_diag[k] = e11
      @d_diag[k + 1] = e22
      @d_sub[k] = e21
      return if det == 0
      (k + 2...n).each do |i|
        b1 = a[i][k]
        b2 = a[i][k + 1]
        @l[i][k]     = (e22 * b1 - e21 * b2).quo(det)
        @l[i][k + 1] = (e11 * b2 - e21 * b1).quo(det)
      end
      (k + 2...n).each do |i|
        c1 = a[i][k]
        c2 = a[i][k + 1]
        next if c1 == 0 && c2 == 0
        (k + 2...n).each do |j|
          a[i][j] -= c1 * @l[j][k] + c2 * @l[j][k + 1]
        end
      end
    end

    def block_det(k)
      @d_diag[k] * @d_diag[k + 1] - @d_sub[k] * @d_sub[k]
    end

    # Returns the magnitudes of the two eigenvalues of the 2x2 block
    # starting at index +k+, sorted ascending.
    def block_eigenvalue_magnitudes(k)
      a = @d_diag[k]
      b = @d_sub[k]
      c = @d_diag[k + 1]
      mid = (a + c) / 2.0
      dev = Math.sqrt(((a - c) / 2.0)**2 + b**2)
      [(mid - dev).abs, (mid + dev).abs].minmax
    end

    # Solves L*D*L' * x = rhs, where +rhs+ holds one row per equation and
    # +width+ right hand sides, and returns the rows of +x+.
    def backsolve(y, width)
      n = @n
      # Solve L * z = y (L is unit lower triangular)
      n.times do |i|
        li = @l[i]
        i.times do |j|
          lij = li[j]
          next if lij == 0
          yj = y[j]
          yi = y[i]
          width.times {|c| yi[c] -= lij * yj[c]}
        end
      end
      # Solve D * w = z (block diagonal)
      k = 0
      while k < n
        if @d_sub[k]
          e11 = @d_diag[k]
          e21 = @d_sub[k]
          e22 = @d_diag[k + 1]
          det = e11 * e22 - e21 * e21
          yk = y[k]
          yk1 = y[k + 1]
          width.times do |c|
            b1 = yk[c]
            b2 = yk1[c]
            yk[c]  = (e22 * b1 - e21 * b2).quo(det)
            yk1[c] = (e11 * b2 - e21 * b1).quo(det)
          end
          k += 2
        else
          d = @d_diag[k]
          yk = y[k]
          width.times {|c| yk[c] = yk[c].quo(d)}
          k += 1
        end
      end
      # Solve L' * x = w
      (n - 1).downto(0) do |i|
        (i + 1...n).each do |j|
          lji = @l[j][i]
          next if lji == 0
          yj = y[j]
          yi = y[i]
          width.times {|c| yi[c] -= lji * yj[c]}
        end
      end
      # Un-permute: x = P' * y
      out = Array.new(n)
      n.times {|i| out[@pivots[i]] = y[i]}
      out
    end
  end
end

