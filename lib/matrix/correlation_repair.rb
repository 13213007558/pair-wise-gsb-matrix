# frozen_string_literal: false

class Matrix
  CorrelationRepairResult = Struct.new(:status, :matrix, :iterations, :error, :objective, keyword_init: true) do
    def converged?
      status == :converged
    end

    def lock_conflict?
      status == :lock_conflict
    end

    alias_method :infeasible?, :lock_conflict?

    def iteration_limit_reached?
      status == :iteration_limit_reached
    end

    def not_converged?
      status == :iteration_limit_reached
    end

    alias_method :success?, :converged?
  end

  class CorrelationRepair
    def self.repair(matrix, weights, locked, tolerance, max_iterations)
      new(matrix, weights, locked, tolerance, max_iterations).repair
    end

    def initialize(matrix, weights, locked, tolerance, max_iterations)
      @target = validate_matrix(matrix)
      @size = @target.size
      @weights = validate_weights(weights)
      @locked = validate_locked(locked)
      @tolerance = validate_tolerance(tolerance)
      @max_iterations = validate_max_iterations(max_iterations)
    end

    def repair
      return conflict if diagonal_lock_conflict? || duplicate_lock_conflict? || fully_locked_conflict?

      x = constrained_initial_matrix
      z = project_to_positive_semidefinite(x)
      u = zero_matrix
      rho = initial_rho
      error = residual_error(x, z, zero_matrix, rho)
      return result(:converged, x, 0, error) if error <= @tolerance / 100
      certificate_streak = 0
      @max_iterations.times do |iteration|
        previous_z = z
        x = update_x(z, u, rho)
        x = symmetrize(x)
        z = project_to_positive_semidefinite(symmetrize(add(x, u)))
        u = add(u, subtract(x, z))
        error = residual_error(x, z, subtract(z, previous_z), rho)

        if error <= @tolerance / 100
          return result(:converged, x, iteration + 1, error)
        end

        if certificate_check_iteration?(iteration)
          if infeasibility_certificate?(u)
            certificate_streak += 1
            return certificate_conflict(iteration + 1) if certificate_streak == 3
          else
            certificate_streak = 0
          end
        end

        rho, u = adapt_rho(rho, u, x, z, previous_z)
      end

      result(:iteration_limit_reached, x, @max_iterations, error)
    end

    def certificate_check_iteration?(iteration)
      count = iteration + 1
      (count == @max_iterations) || (count >= 25 && (count - 25) % 10 == 0)
    end

    private

    def result(status, matrix, iterations, error)
      CorrelationRepairResult.new(
        status: status,
        matrix: Matrix.send(:new, deep_dup(matrix)),
        iterations: iterations,
        error: error,
        objective: objective(matrix)
      )
    end

    def conflict
      CorrelationRepairResult.new(
        status: :lock_conflict,
        matrix: nil,
        iterations: 0,
        error: Float::INFINITY,
        objective: nil
      )
    end

    def certificate_conflict(iterations)
      CorrelationRepairResult.new(
        status: :lock_conflict,
        matrix: nil,
        iterations: iterations,
        error: Float::INFINITY,
        objective: nil
      )
    end

    def validate_matrix(matrix)
      unless matrix.is_a?(Matrix)
        raise TypeError, "expected Matrix, got #{matrix.class}"
      end
      raise ExceptionForMatrix::ErrDimensionMismatch unless matrix.square?
      if matrix.row_count.zero?
        raise ArgumentError, "matrix must have at least one dimension"
      end

      values = matrix.to_a
      values.each do |row|
        row.each do |value|
          validate_real_value(value, "matrix entry")
        end
      end
      values.map { |row| row.map(&:to_f) }
    end

    def validate_weights(weights)
      values = case weights
               when nil
                 Array.new(@size) { Array.new(@size, 1.0) }
               when Matrix
                 weights.to_a
               when Array
                 weights
               else
                 raise TypeError, "weights must be a Matrix or an Array of Arrays"
               end

      unless values.is_a?(Array) &&
        values.size == @size &&
        values.all? { |row| row.is_a?(Array) && row.size == @size }
        raise ExceptionForMatrix::ErrDimensionMismatch, "weights must be a #{@size}x#{@size} matrix"
      end

      values.each_with_index do |row, i|
        row.each_with_index do |value, j|
          validate_real_value(value, "weight")
          if value.to_f.negative?
            raise ArgumentError, "weight at [#{i}, #{j}] must be non-negative"
          end
        end
      end
      values.map { |row| row.map(&:to_f) }
    end

    def validate_locked(locked)
      entries = case locked
                when nil
                  []
                when Hash
                  locked.map do |position, value|
                    unless position.is_a?(Array) && position.size == 2
                      raise ArgumentError, "locked position must be [row, column]"
                    end
                    [*position, value]
                  end
                when Array
                  locked.map do |entry|
                    unless entry.is_a?(Array) && entry.size == 3
                      raise ArgumentError, "locked entry must be [row, column, value]"
                    end
                    entry
                  end
                else
                  raise ArgumentError, "locked entries must be a Hash or an Array of triples"
                end

      entries.each_with_object({}) do |(row, column, value), map|
        row = validate_index(row, :row)
        column = validate_index(column, :column)
        validate_real_value(value, "locked value")
        value = value.to_f
        if value.abs > 1
          raise ArgumentError, "locked value at [#{row}, #{column}] must be in [-1, 1]"
        end

        if map.key?([row, column]) && map[[row, column]] != value
          @duplicate_conflict = true
        end
        map[[row, column]] = value
      end
    end

    def validate_index(value, direction)
      unless value.is_a?(Integer) || value.respond_to?(:to_int)
        raise TypeError, "locked #{direction} must be an Integer"
      end
      integer = value.to_int
      unless integer >= 0 && integer < @size
        raise ExceptionForMatrix::ErrDimensionMismatch, "locked #{direction} #{integer} is outside the matrix"
      end
      integer
    end

    def validate_real_value(value, name)
      unless value.is_a?(Numeric) && value.real? && value.to_f.finite?
        raise ArgumentError, "#{name} must be a finite real number"
      end
    end

    def validate_tolerance(value)
      validate_real_value(value, "tolerance")
      tolerance = value.to_f
      unless tolerance.positive?
        raise ArgumentError, "tolerance must be positive"
      end
      tolerance
    end

    def validate_max_iterations(value)
      integer = Matrix::CoercionHelper.coerce_to_int(value)
      unless integer >= 0
        raise ArgumentError, "max_iterations must be non-negative"
      end
      integer
    end

    def diagonal_lock_conflict?
      @locked.any? { |(row, column), value| row == column && value != 1.0 }
    end

    def duplicate_lock_conflict?
      return true if @duplicate_conflict

      @locked.any? do |(row, column), value|
        row != column && @locked.key?([column, row]) && @locked[[column, row]] != value
      end
    end

    def fully_locked_conflict?
      off_diagonal = @locked.keys.count { |row, column| row < column }
      off_diagonal == @size * (@size - 1) / 2 && eigenvalues(fixed_matrix).min < -@tolerance
    end

    def constrained_initial_matrix
      matrix = symmetric_target
      apply_locked_values(matrix)
    end

    def fixed_matrix
      matrix = zero_matrix
      apply_locked_values(matrix)
    end

    def apply_locked_values(matrix)
      @locked.each do |(row, column), value|
        matrix[row][column] = value
        matrix[column][row] = value
      end
      @size.times { |i| matrix[i][i] = 1.0 }
      matrix
    end

    def symmetric_target
      Array.new(@size) do |i|
        Array.new(@size) do |j|
          denominator = @weights[i][j] + @weights[j][i]
          if denominator.zero?
            0.0
          else
            (@weights[i][j] * @target[i][j] + @weights[j][i] * @target[j][i]) / denominator
          end
        end
      end
    end

    def update_x(z, u, rho)
      target = symmetric_target
      Array.new(@size) do |i|
        Array.new(@size) do |j|
          next locked_value(i, j) if locked?(i, j)

          numerator = @weights[i][j] * target[i][j] + @weights[j][i] * target[j][i] +
            rho * (z[i][j] - u[i][j])
          denominator = @weights[i][j] + @weights[j][i] + rho
          numerator / denominator
        end
      end
    end

    def locked?(row, column)
      row == column || @locked.key?([row, column]) || @locked.key?([column, row])
    end

    def locked_value(row, column)
      return 1.0 if row == column
      @locked.fetch([row, column]) { @locked[[column, row]] }
    end

    def project_to_positive_semidefinite(matrix)
      symmetric = symmetrize(matrix)
      decomposition = Matrix.send(:new, deep_dup(symmetric)).eigensystem
      vectors = decomposition.eigenvector_matrix.to_a
      values = decomposition.eigenvalues.map { |value| [value.to_f, 0.0].max }

      projected = Array.new(@size) do |i|
        Array.new(@size) do |j|
          (0...@size).sum(0.0) do |k|
            vectors[i][k] * values[k] * vectors[j][k]
          end
        end
      end

      symmetrize(projected)
    end

    def eigenvalues(matrix)
      Matrix.send(:new, deep_dup(symmetrize(matrix))).eigensystem.eigenvalues.map(&:to_f)
    end

    def symmetrize(matrix)
      Array.new(@size) do |i|
        Array.new(@size) do |j|
          (matrix[i][j] + matrix[j][i]) / 2.0
        end
      end
    end

    def infeasibility_certificate?(u)
      normalized = symmetrize(multiply(u, 1.0 / [1.0, max_abs(u)].max))
      return false unless free_entries_small?(normalized)
      return false unless eigenvalues(normalized).max <= 1e-6

      trace_product = 0.0
      @size.times do |i|
        @size.times do |j|
          trace_product += fixed_matrix[i][j] * normalized[i][j]
        end
      end
      trace_product >= 1e-6
    end

    def free_entries_small?(matrix)
      @size.times do |i|
        @size.times do |j|
          unless locked?(i, j)
            return false if matrix[i][j].abs > [1e-6, @tolerance].max
          end
        end
      end
      true
    end

    def initial_rho
      values = @weights.flatten
      [values.sum / values.size, 1e-8].max
    end

    def adapt_rho(rho, u, x, z, previous_z)
      primal = frobenius_norm(subtract(x, z))
      dual = frobenius_norm(subtract(z, previous_z)) * rho
      return [rho, u] if primal.zero? || dual.zero?

      if primal > 10 * dual
        new_rho = [rho * 2, 1e12].min
        [new_rho, multiply(u, rho / new_rho)]
      elsif dual > 10 * primal
        new_rho = [rho / 2, 1e-12].max
        [new_rho, multiply(u, rho / new_rho)]
      else
        [rho, u]
      end
    end

    def residual_error(x, z, z_change, rho)
      primal = frobenius_norm(subtract(x, z))
      dual = frobenius_norm(z_change) * rho
      scale = [frobenius_norm(x), frobenius_norm(z), 1.0].max
      Math.sqrt(primal * primal + dual * dual) / scale
    end

    def objective(matrix)
      (0...@size).sum do |i|
        (0...@size).sum do |j|
          difference = matrix[i][j] - @target[i][j]
          @weights[i][j] * difference * difference
        end
      end
    end

    def add(left, right)
      Array.new(@size) do |i|
        Array.new(@size) { |j| left[i][j] + right[i][j] }
      end
    end

    def subtract(left, right)
      Array.new(@size) do |i|
        Array.new(@size) { |j| left[i][j] - right[i][j] }
      end
    end

    def multiply(matrix, scalar)
      matrix.map { |row| row.map { |value| value * scalar } }
    end

    def max_abs(matrix)
      matrix.flatten.map(&:abs).max.nonzero? || 1.0
    end

    def frobenius_norm(matrix)
      Math.sqrt(matrix.sum { |row| row.sum { |value| value * value } })
    end

    def zero_matrix
      Array.new(@size) { Array.new(@size, 0.0) }
    end

    def deep_dup(matrix)
      matrix.map(&:dup)
    end
  end
end
