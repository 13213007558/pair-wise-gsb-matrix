# frozen_string_literal: false
class Matrix
  # Williamson decomposition of a real symmetric positive-definite matrix
  # of even order.
  #
  # For a symmetric positive-definite 2n x 2n matrix +V+, Williamson's
  # theorem guarantees a symplectic matrix +S+ and positive scalars
  # nu_1, ..., nu_n (the symplectic eigenvalues) such that
  #
  #   V = S * D * S.transpose,   D = diagonal(nu_1, ..., nu_n, nu_1, ..., nu_n)
  #
  # Coordinates are ordered as (q1, ..., qn, p1, ..., pn) and the standard
  # symplectic matrix is fixed to the block form
  #
  #   J = [[0, I_n], [-I_n, 0]]
  #
  # so that +S+ satisfies the symplectic constraint
  #
  #   S * J * S.transpose == J
  #
  # Example:
  #   v = Matrix[[2.0, 0.5], [0.5, 1.0]]
  #   w = v.williamson
  #   w.symplectic_eigenvalues # => [1.3228756555322945]
  #   (w.s * w.d * w.s.t - v).map(&:abs).max < 1e-12 # => true
  #   (w.s * w.j * w.s.t - w.j).map(&:abs).max < 1e-12 # => true
  #
  # The symplectic eigenvalues are returned in ascending order. When some
  # of them coincide (or nearly coincide), the corresponding columns of +S+
  # span a degenerate subspace: a reproducible, legally paired basis is
  # chosen, but that basis is not unique and should not be relied upon.
  #
  # The computation uses only real arithmetic: the symmetric square root
  # +W+ of +V+, the antisymmetric matrix K = W*J*W, and the symmetric
  # doubled matrix H = [[0, K], [-K, 0]] whose eigenvalues are exactly
  # the signed symplectic eigenvalues. Avoiding K**2 keeps very different
  # scales resolvable; even so, symplectic eigenvalues smaller than about
  # Float::EPSILON * (K norm) are inherently limited to absolute, not
  # relative, accuracy in floating point.
  #
  # This code only supports real matrices; it does not require any
  # external linear algebra backend.
  class WilliamsonDecomposition
    # Relative asymmetry tolerated before a matrix is rejected as
    # non-symmetric.
    SYMMETRY_TOLERANCE = 1e-10

    # A candidate eigenvector whose component orthogonal to the
    # already-used directions is smaller than this fraction is treated as
    # the conjugate partner of an earlier candidate and skipped. Genuine
    # new directions leave a residual of norm ~1, partners leave ~0.
    INDEPENDENCE_THRESHOLD = 0.5

    # Minimum acceptable norm of a position/momentum half of a candidate
    # eigenvector; below this the direction is considered unreliable.
    MIN_DIRECTION_NORM = 1e-2

    # Constructs the Williamson decomposition for a symmetric
    # positive-definite matrix +a+ of even order.
    #
    # Raises TypeError if +a+ is not a Matrix or has non-real entries,
    # ExceptionForMatrix::ErrDimensionMismatch if +a+ is not square, is
    # empty or has odd order, and ArgumentError if +a+ is not symmetric or
    # not positive definite.
    def initialize(a)
      raise TypeError, "Expected Matrix but got #{a.class}" unless a.is_a?(Matrix)
      raise ExceptionForMatrix::ErrDimensionMismatch, "Williamson decomposition requires a square matrix" unless a.square?
      @size = a.row_count
      raise ExceptionForMatrix::ErrDimensionMismatch, "Williamson decomposition is not defined for an empty matrix" if @size == 0
      raise ExceptionForMatrix::ErrDimensionMismatch, "Williamson decomposition requires an even order matrix (got order #{@size})" if @size.odd?
      @n = @size / 2

      decompose(real_float_matrix(a))
    end

    # Returns the number of (q, p) mode pairs, i.e. half the matrix order.
    attr_reader :n

    # Returns the symplectic eigenvalues [nu_1, ..., nu_n] as an Array of
    # positive Floats, sorted in ascending order.
    attr_reader :symplectic_eigenvalues

    # Returns the symplectic matrix +S+ such that
    #   self == s * d * s.transpose  and  s * j * s.transpose == j
    def symplectic_matrix
      @s
    end
    alias_method :s, :symplectic_matrix

    # Returns the Williamson diagonal form
    #   D = Matrix.diagonal(nu_1, ..., nu_n, nu_1, ..., nu_n)
    def diagonal_matrix
      Matrix.diagonal(*@symplectic_eigenvalues, *@symplectic_eigenvalues)
    end
    alias_method :d, :diagonal_matrix

    # Returns the standard symplectic matrix J = [[0, I_n], [-I_n, 0]] for
    # the (q1, ..., qn, p1, ..., pn) coordinate ordering used throughout.
    def symplectic_form
      self.class.standard_symplectic_form(@n)
    end
    alias_method :j, :symplectic_form

    # Returns [symplectic_matrix, diagonal_matrix]
    def to_ary
      [s, d]
    end
    alias_method :to_a, :to_ary

    # Returns the standard symplectic matrix J = [[0, I_n], [-I_n, 0]] of
    # order 2n for the (q1, ..., qn, p1, ..., pn) coordinate ordering.
    def self.standard_symplectic_form(n)
      raise ArgumentError, "expected a positive number of modes" unless n.is_a?(Integer) && n > 0
      Matrix.build(2 * n) do |row, column|
        if column == row + n
          1
        elsif row == column + n
          -1
        else
          0
        end
      end
    end

    private def real_float_matrix(a)
      a.map do |entry|
        begin
          value = Float(entry)
        rescue TypeError, ArgumentError, RangeError
          raise TypeError, "Williamson decomposition requires real numeric entries, got #{entry.inspect}"
        end
        raise TypeError, "Williamson decomposition requires real entries, got #{entry.inspect}" unless value.real?
        value
      end
    end

    private def decompose(v)
      scale0 = v.entries.map(&:abs).max || 0.0
      asymmetry = (v - v.transpose).entries.map(&:abs).max || 0.0
      unless asymmetry <= SYMMETRY_TOLERANCE * @size * scale0
        raise ArgumentError, "matrix is not symmetric (max asymmetry #{asymmetry})"
      end
      v = (v + v.transpose) * 0.5

      eig = v.eigensystem
      eigenvalues = eig.eigenvalues
      lambda_max = eigenvalues.last
      lambda_min = eigenvalues.first
      unless lambda_min > 0
        raise ArgumentError, "matrix is not positive definite (smallest eigenvalue is #{lambda_min})"
      end

      # Normalize the spectrum to O(1) so that the products below neither
      # overflow nor underflow; the symplectic transformation is
      # scale-invariant and the eigenvalues are rescaled at the end.
      q = eig.eigenvector_matrix
      root = Matrix.diagonal(*eigenvalues.map { |lam| Math.sqrt(lam / lambda_max) })
      w = q * root * q.transpose

      k = w * symplectic_form * w # antisymmetric: i*K has eigenvalues +/-nu

      # H is symmetric with eigenvalues +/-nu_i; working with H instead of
      # K**2 keeps small symplectic eigenvalues resolvable.
      h = Matrix.build(2 * @size) do |row, column|
        if row < @size && column >= @size
          k[row, column - @size]
        elsif row >= @size && column < @size
          -k[row - @size, column]
        else
          0.0
        end
      end
      h = (h + h.transpose) * 0.5

      heig = h.eigensystem
      vectors = heig.eigenvectors
      # The spectrum of H is symmetric about zero, so the upper half of
      # the sorted spectrum holds one conjugate pair of eigenvectors per
      # mode (tiny eigenvalues may straddle zero; |K*a| below repairs the
      # value and the sign convention repairs the orientation).
      pairs = build_pairs(vectors[@size, @size], k)
      pairs = pairs.each_with_index.sort_by { |pair, index| [pair[2], index] }.map(&:first)

      modes = pairs.map { |a, _b, _nu| a }
      conjugates = pairs.map { |_a, b, _nu| b }
      @symplectic_eigenvalues = pairs.map { |_a, _b, nu| nu * lambda_max }
      @symplectic_eigenvalues.freeze

      o = Matrix.columns(modes + conjugates)
      inv_root = Matrix.diagonal(*(pairs.map { |_a, _b, nu| 1.0 / Math.sqrt(nu) } * 2))
      @s = w * o * inv_root
    end

    # Builds the n conjugate pairs (a, b) with K*a = -nu*b, K*b = nu*a
    # from the positive-spectrum eigenvectors of H. Each eigenvector
    # u = (x, y) of H contributes the pair a ~ x, b ~ y; its mirror
    # J*u = (-y, x) spans the same conjugate pair and is skipped.
    private def build_pairs(candidates, k)
      used = []
      chosen = []
      pairs = []
      candidates.each do |u|
        2.times { used.each { |q| u -= q * q.inner_product(u) } }
        next if u.magnitude <= INDEPENDENCE_THRESHOLD
        u = u / u.magnitude
        x = Vector.elements(u.to_a.first(@size))
        y = Vector.elements(u.to_a.last(@size))
        ju = Vector.elements((-y).to_a + x.to_a)
        used.push(u, ju)

        next if x.magnitude <= MIN_DIRECTION_NORM
        a = x / x.magnitude
        if a.component(a.each_with_index.max_by { |value, _| value.abs }.last) < 0
          a = -a
          y = -y
        end

        ka = k * a
        nu = ka.magnitude
        b = y.magnitude > MIN_DIRECTION_NORM ? y / y.magnitude : ka / -nu
        2.times { (chosen + [a]).each { |q| b -= q * q.inner_product(b) } }
        raise ExceptionForMatrix::ErrNotRegular, "degenerate symplectic subspace" if b.magnitude <= Float::EPSILON * @size
        b = b / b.magnitude
        b = -b if a.inner_product(k * b) < 0

        pairs << [a, b, nu]
        chosen.push(a, b)
        break if pairs.size == @n
      end
      unless pairs.size == @n
        raise ExceptionForMatrix::ErrNotRegular, "could not resolve #{@n} symplectic mode pairs (matrix is numerically singular?)"
      end
      pairs
    end
  end
end
