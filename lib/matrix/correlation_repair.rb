# frozen_string_literal: false

class Matrix
  # Result of Matrix.repair_correlation.
  class CorrelationRepair
    STATUSES = [:converged, :not_converged, :contradictory_locks].freeze

    attr_reader :matrix, :iterations, :error, :status

    def initialize(matrix, iterations, error, status)
      raise ArgumentError, "unknown status #{status.inspect}" unless STATUSES.include?(status)

      @matrix = matrix
      @iterations = iterations
      @error = error
      @status = status
    end

    def converged?
      @status == :converged
    end

    def contradictory_locks?
      @status == :contradictory_locks
    end

    def not_converged?
      @status == :not_converged
    end
  end

  # :call-seq:
  #   Matrix.repair_correlation(matrix, weights: nil, locked: [], tolerance: 1e-10,
  #                             max_iterations: 1000)
  #
  # Returns the positive semidefinite correlation matrix nearest to +matrix+ in
  # a weighted least-squares sense. The objective is
  # <tt>0.5 * sum_{i,j} weight[i,j] * (result[i,j] - matrix[i,j])**2</tt>.
  #
  # Diagonal entries are constrained to 1. Entries in +locked+, given as
  # <tt>[[row, column, value], ...]</tt> or <tt>{[row, column] => value}</tt>,
  # constrain both symmetric positions.
  #
  # The returned Matrix::CorrelationRepair distinguishes convergence,
  # non-convergence within +max_iterations+, and mutually impossible locks.
  def self.repair_correlation(matrix, weights: nil, locked: [], tolerance: 1e-10,
                              max_iterations: 1000)
    tolerance = positive_repair_float(tolerance, "tolerance")
    max_iterations = positive_repair_int(max_iterations, "max_iterations")

    original = require_repair_matrix(matrix, "matrix")
    raise ErrDimensionMismatch, "matrix must be square" unless original.square?
    raise ArgumentError, "matrix must have at least one dimension" if original.row_count.zero?
    raise ArgumentError, "matrix must be symmetric" unless original.symmetric?

    size = original.row_count
    target = float_repair_rows(original)
    target.each_with_index do |row, i|
      raise ArgumentError, "correlation matrices must have 1 on the diagonal" unless row[i] == 1.0
    end

    weight_matrix =
      case weights
      when nil
        Matrix.build(size) { 1.0 }
      when Matrix
        weights
      when Array
        Matrix.rows(weights)
      else
        raise TypeError, "weights must be a Matrix or an Array"
      end
    require_repair_matrix(weight_matrix, "weights")
    unless weight_matrix.row_count == size && weight_matrix.column_count == size
      raise ErrDimensionMismatch, "weights must have the same dimensions as matrix"
    end
    raise ArgumentError, "weights must be symmetric" unless weight_matrix.symmetric?
    weight_rows = float_repair_rows(weight_matrix)
    weight_rows.each do |row|
      row.each do |weight|
        unless weight > 0.0
          raise ArgumentError, "weights must be strictly positive"
        end
      end
    end

    constraints = []
    add_repair_constraint = lambda do |row_index, column_index, value|
      i = CoercionHelper.coerce_to_int(row_index)
      j = CoercionHelper.coerce_to_int(column_index)
      unless i.between?(0, size - 1) && j.between?(0, size - 1)
        raise IndexError, "locked position [#{i}, #{j}] is outside a #{size}x#{size} matrix"
      end

      locked_value = repair_float(value, "locked value")
      unless locked_value.between?(-1.0, 1.0)
        raise ArgumentError, "locked values must be between -1 and 1"
      end

      i, j = j, i if i > j
      if i == j && locked_value != 1.0
        raise ArgumentError, "locked diagonal entries must be 1"
      end

      existing = constraints.find { |entry| entry[0] == i && entry[1] == j }
      if existing && existing[2] != locked_value
        raise ArgumentError, "conflicting locks for symmetric position [#{i}, #{j}]: " \
                             "#{existing[2].inspect} and #{locked_value.inspect}"
      end
      constraints << [i, j, locked_value] unless existing
    end

    case locked
    when Hash
      locked.each do |position, value|
        unless position.is_a?(Array) && position.length == 2
          raise ArgumentError, "locked positions must be [row, column]"
        end
        add_repair_constraint.call(position[0], position[1], value)
      end
    when Array
      locked.each do |lock|
        unless lock.is_a?(Array) && lock.length == 3
          raise ArgumentError, "locks must be [row, column, value]"
        end
        add_repair_constraint.call(lock[0], lock[1], lock[2])
      end
    else
      raise TypeError, "locked must be a Hash or an Array"
    end

    size.times do |i|
      add_repair_constraint.call(i, i, 1)
    end
    solver = CorrelationRepairSolver.new(target, weight_rows, constraints, tolerance)
    solver.solve(max_iterations)
  end

  def self.require_repair_matrix(value, name)
    unless value.is_a?(Matrix)
      raise TypeError, "#{name} must be a Matrix"
    end
    value
  end
  private_class_method :require_repair_matrix

  def self.float_repair_rows(matrix)
    matrix.to_a.map do |row|
      row.map { |value| repair_float(value, "matrix value") }
    end
  end
  private_class_method :float_repair_rows

  def self.repair_float(value, name)
    unless value.is_a?(Numeric) && value.real?
      raise TypeError, "#{name} must be a real number"
    end
    float_value = value.to_f
    unless float_value.finite?
      raise ArgumentError, "#{name} must be finite"
    end
    float_value
  end
  private_class_method :repair_float

  def self.positive_repair_float(value, name)
    float_value = repair_float(value, name)
    unless float_value > 0.0
      raise ArgumentError, "#{name} must be greater than 0"
    end
    float_value
  end
  private_class_method :positive_repair_float

  def self.positive_repair_int(value, name)
    integer_value = CoercionHelper.coerce_to_int(value)
    unless integer_value > 0
      raise ArgumentError, "#{name} must be greater than 0"
    end
    integer_value
  end
  private_class_method :positive_repair_int

  class CorrelationRepairSolver # :nodoc: all
    def initialize(target, weights, constraints, tolerance)
      @target = target
      @weights = weights
      @constraints = constraints
      @tolerance = tolerance
      @size = target.length
      @maximum_weight = weights.flatten.max
    end

    def solve(max_iterations)
      feasible, feasibility_iterations, feasibility_error = find_feasible_point
      unless feasible
        return CorrelationRepair.new(nil, feasibility_iterations, feasibility_error,
                                     :contradictory_locks)
      end

      x = affine_projection(@target)
      z = x.map(&:dup)
      u = zero_rows
      rho = @maximum_weight
      primal_residual = 0.0
      dual_residual = 0.0

      initial_error = positive_semidefinite_error(x)
      return repair_result(x, 0, initial_error, :converged) if initial_error <= @tolerance

      max_iterations.times do |iteration|
        previous_z = z.map(&:dup)
        x = minimize_augmented_lagrangian(z, u, rho)
        z, = project_positive_semidefinite(add_matrices(x, u))
        u = subtract_matrices(add_matrices(x, u), z)

        primal_residual = frobenius_distance(x, z)
        dual_residual = rho * frobenius_distance(z, previous_z)
        error = [primal_residual, dual_residual].max
        if error <= @tolerance
          return repair_result(x, iteration + 1, error, :converged)
        end
      end

      final_error = [primal_residual, dual_residual,
                     positive_semidefinite_error(x)].max
      CorrelationRepair.new(nil, max_iterations, final_error, :not_converged)
    end

    private

    def find_feasible_point
      point = affine_projection(zero_rows)
      psd_correction = zero_rows
      affine_correction = zero_rows
      iterations = [200, 20 * @size].max

      iterations.times do |iteration|
        projection_input = add_matrices(point, psd_correction)
        projected, = project_positive_semidefinite(projection_input)
        psd_correction = subtract_matrices(projection_input, projected)

        affine_input = add_matrices(projected, affine_correction)
        point = affine_projection(affine_input)
        affine_correction = subtract_matrices(affine_input, point)

        error = frobenius_distance(projected, point)
        return [true, iteration + 1, error] if error <= @tolerance
      end

      projected, = project_positive_semidefinite(add_matrices(point, psd_correction))
      final_point = affine_projection(add_matrices(projected, affine_correction))
      [false, iterations, frobenius_distance(projected, final_point)]
    end

    def minimize_augmented_lagrangian(z, u, rho)
      rows = Array.new(@size) do |i|
        Array.new(@size) do |j|
          row, column = i < j ? [i, j] : [j, i]
          (@weights[row][column] * @target[row][column] + rho * (z[i][j] - u[i][j])) /
            (@weights[row][column] + rho)
        end
      end
      apply_constraints(rows)
    end

    def affine_projection(source)
      rows = source.map(&:dup)
      apply_constraints(rows)
    end

    def apply_constraints(rows)
      @constraints.each do |i, j, value|
        rows[i][j] = value
        rows[j][i] = value
      end
      rows
    end

    def project_positive_semidefinite(rows)
      symmetric = symmetrize(rows)
      values = eigenvalues(symmetric)
      return [symmetric, values] if values.all? { |value| value >= 0.0 }

      matrix = Matrix.send(:new, symmetric)
      eigensystem = matrix.eigensystem
      vectors = eigensystem.eigenvector_matrix
      clipped = Matrix.diagonal(*values.map { |value| value > 0.0 ? value : 0.0 })
      [(vectors * clipped * vectors.transpose).to_a, values]
    end

    def eigenvalues(rows)
      Matrix.send(:new, symmetrize(rows)).eigensystem.eigenvalues
    end

    def symmetrize(rows)
      Array.new(@size) do |i|
        Array.new(@size) do |j|
          i <= j ? rows[i][j] : rows[j][i]
        end
      end
    end

    def zero_rows
      Array.new(@size) { Array.new(@size, 0.0) }
    end

    def add_matrices(left, right)
      Array.new(@size) do |i|
        Array.new(@size) { |j| left[i][j] + right[i][j] }
      end
    end

    def subtract_matrices(left, right)
      Array.new(@size) do |i|
        Array.new(@size) { |j| left[i][j] - right[i][j] }
      end
    end

    def positive_semidefinite_error(rows)
      [0.0, -eigenvalues(rows).min].max
    end

    def frobenius_distance(left, right)
      Math.sqrt(@size.times.sum do |i|
        @size.times.sum do |j|
          difference = left[i][j] - right[i][j]
          difference * difference
        end
      end)
    end

    def repair_result(rows, iterations, error, status)
      feasible_rows = polish(rows) if status == :converged
      CorrelationRepair.new(Matrix.send(:new, feasible_rows || rows), iterations, error, status)
    end

    def polish(rows)
      point = rows.map(&:dup)
      psd_correction = zero_rows
      affine_correction = zero_rows
      50.times do
        projection_input = add_matrices(point, psd_correction)
        projected, = project_positive_semidefinite(projection_input)
        psd_correction = subtract_matrices(projection_input, projected)

        affine_input = add_matrices(projected, affine_correction)
        point = affine_projection(affine_input)
        affine_correction = subtract_matrices(affine_input, point)
      end
      point
    end
  end
end
