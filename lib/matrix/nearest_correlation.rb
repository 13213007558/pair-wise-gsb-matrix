# frozen_string_literal: false

class Matrix
  # The result of Matrix.nearest_correlation.
  #
  # Attributes:
  # * +matrix+:     the nearest correlation matrix found (always symmetric,
  #                 with unit diagonal and the requested fixed entries).
  # * +status+:     +:converged+ or +:max_iterations_reached+.
  # * +iterations+: number of Dykstra sweeps performed.
  # * +residual+:   Frobenius distance from +matrix+ to the positive
  #                 semidefinite cone at termination (feasibility gap).
  # * +objective+:  Frobenius distance between +matrix+ and the input.
  #
  # A result must only be considered successful when +converged?+ is true;
  # when the constraints are contradictory or the iteration budget is
  # exhausted, +converged?+ is false and +matrix+ is the last iterate,
  # which is not guaranteed to be positive semidefinite.
  class NearestCorrelationResult
    attr_reader :matrix, :status, :iterations, :residual, :objective

    def initialize(matrix:, status:, iterations:, residual:, objective:)
      @matrix = matrix
      @status = status
      @iterations = iterations
      @residual = residual
      @objective = objective
      freeze
    end

    # Whether the iteration converged to a positive semidefinite
    # correlation matrix within the requested tolerance.
    def converged?
      @status == :converged
    end
  end

  # Computes the correlation matrix nearest to +matrix+ in the Frobenius
  # norm, i.e. the symmetric positive semidefinite matrix with unit
  # diagonal minimizing the Frobenius distance to +matrix+.
  #
  #   a = Matrix[[1.0, 2.0, 0.5], [2.0, 1.0, 0.0], [0.5, 0.0, 1.0]]
  #   r = Matrix.nearest_correlation(a, fixed: {[1, 2] => 0.0})
  #   r.converged?        # => true
  #   r.matrix            # => nearest correlation matrix
  #   r.objective         # => Frobenius distance to +a+
  #
  # Options:
  # * +fixed+: optional Hash mapping <code>[i, j]</code> index pairs to
  #   off-diagonal entries that must be preserved exactly. Both +[i, j]+
  #   and +[j, i]+ are accepted; values must be real and in [-1, 1].
  # * +tolerance+: convergence tolerance on the Frobenius distance to the
  #   positive semidefinite cone (default 1e-9).
  # * +max_iterations+: iteration budget (default 1000).
  #
  # The input is symmetrized first, which does not change the solution of
  # the Frobenius-norm problem. Uses Dykstra's alternating projections
  # between the positive semidefinite cone and the affine set of matrices
  # with unit diagonal and the requested fixed entries.
  #
  # When the constraints are contradictory (e.g. fixed entries that no
  # correlation matrix can have) or the budget is exhausted, the returned
  # result has +status+ +:max_iterations_reached+ and +converged?+ false.
  def self.nearest_correlation(matrix, fixed: nil, tolerance: 1e-9, max_iterations: 1000)
    raise TypeError, "Expected Matrix but got #{matrix.class}" unless matrix.is_a?(Matrix)
    raise ErrDimensionMismatch, "Expected a square matrix" unless matrix.square?
    raise ArgumentError, "tolerance must be positive" unless tolerance > 0
    raise ArgumentError, "max_iterations must be positive" unless max_iterations >= 1

    n = matrix.row_count
    fixed_entries = normalize_fixed_entries(fixed, n)

    # Symmetrize; only the symmetric part matters for the Frobenius norm.
    a = matrix.to_a.map { |row| row.map { |e| Float(e.real) } }
    n.times do |i|
      (i + 1...n).each do |j|
        s = (a[i][j] + a[j][i]) / 2
        a[i][j] = a[j][i] = s
      end
    end

    project_u = lambda do |m|
      r = m.map(&:dup)
      n.times { |i| r[i][i] = 1.0 }
      fixed_entries.each do |(i, j), value|
        r[i][j] = r[j][i] = value
      end
      r
    end

    project_s = lambda do |m|
      n.times do |i|
        (i + 1...n).each do |j|
          s = (m[i][j] + m[j][i]) / 2
          m[i][j] = m[j][i] = s
        end
      end
      eigen = Matrix.rows(m, false).eigensystem
      v = eigen.eigenvector_matrix
      d = eigen.eigenvalues.map { |ev| ev.real > 0.0 ? ev.real : 0.0 }
      (v * Matrix.diagonal(*d) * v.transpose).to_a
    end

    add = lambda { |x, y| x.each_index.map { |i| x[i].each_index.map { |j| x[i][j] + y[i][j] } } }
    sub = lambda { |x, y| x.each_index.map { |i| x[i].each_index.map { |j| x[i][j] - y[i][j] } } }
    frobenius = lambda { |x| Math.sqrt(x.sum { |row| row.sum { |e| e * e } }) }

    # Dykstra's algorithm with corrections p_u and p_s.
    x = project_u.call(a)
    p_u = Array.new(n) { Array.new(n, 0.0) }
    p_s = Array.new(n) { Array.new(n, 0.0) }
    residual = Float::INFINITY
    iterations = 0
    while iterations < max_iterations
      corrected = add.call(x, p_s)
      y = project_s.call(corrected)
      p_s = sub.call(corrected, y)
      corrected = add.call(y, p_u)
      x = project_u.call(corrected)
      p_u = sub.call(corrected, x)
      iterations += 1
      residual = frobenius.call(sub.call(x, project_s.call(x.map(&:dup))))
      break if residual <= tolerance
    end

    status = residual <= tolerance ? :converged : :max_iterations_reached
    NearestCorrelationResult.new(
      matrix: Matrix.rows(x, false),
      status: status,
      iterations: iterations,
      residual: residual,
      objective: frobenius.call(sub.call(x, a))
    )
  end

  def self.normalize_fixed_entries(fixed, n) # :nodoc:
    entries = {}
    (fixed || {}).each do |key, value|
      i, j = key
      unless i.is_a?(Integer) && j.is_a?(Integer) &&
             0 <= i && i < n && 0 <= j && j < n
        raise ArgumentError, "invalid fixed entry index #{key.inspect}"
      end
      raise ArgumentError, "diagonal entries cannot be fixed (they are always 1)" if i == j
      unless value.is_a?(Numeric) && value.real?
        raise ArgumentError, "fixed entry #{key.inspect} must be a real number"
      end
      value = Float(value)
      if value.abs > 1.0
        raise ArgumentError, "fixed entry #{key.inspect}=#{value} is outside [-1, 1]; " \
                             "no correlation matrix can satisfy it"
      end
      i, j = j, i if i > j
      if entries.key?([i, j]) && entries[[i, j]] != value
        raise ArgumentError, "conflicting values for fixed entry [#{i}, #{j}]"
      end
      entries[[i, j]] = value
    end
    entries
  end
  private_class_method :normalize_fixed_entries
end

